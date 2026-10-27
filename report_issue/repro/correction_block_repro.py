# Reproducer: sdpa `correction_block` (compute_common.hpp) stages 5 DST tiles inside one acquire.
# With fp32_dest_acc_en=True and dst_full_sync_en=False the DST window is 4 fp32 tiles, and the
# result is wrong on one of the two DST halves. Runs the unmodified helper on one core through
# ttnn.generic_op, no C++ build. tt-metal v0.78.0, Wormhole.  Usage: python correction_block_repro.py
import math, struct, torch, ttnn

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
// input tensor = 4*N tiles [prev_max | worker_max | prev_sum | worker_sum]; block b -> CB b
void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    const auto src = TensorAccessor(TensorAccessorArgs<1>(), get_arg_val<uint32_t>(0));
    Noc noc;
    for (uint32_t b = 0; b < 4; ++b) {
        DataflowBuffer dfb(b);
        for (uint32_t t = 0; t < N; ++t) {
            dfb.reserve_back(1);
            noc.async_read(src, dfb, 2048, {.page_id = b * N + t}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb.push_back(1);
        }
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
// CBs 16..19 -> output tensor = 4*N tiles [cur_max | cur_sum | exp_max_diff | exp_max_diff_2]
void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    const auto dst = TensorAccessor(TensorAccessorArgs<1>(), get_arg_val<uint32_t>(0));
    Noc noc;
    for (uint32_t b = 0; b < 4; ++b) {
        DataflowBuffer dfb(16 + b);
        for (uint32_t t = 0; t < N; ++t) {
            dfb.wait_front(1);
            noc.async_write(dfb, dst, 2048, {}, {.page_id = b * N + t});
            noc.async_writes_flushed();
            dfb.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
"""

COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
// The stock decode kernel's child-merge sequence (sdpa_flash_decode.cpp, ~line 555-575), unchanged.
void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(1);
    constexpr uint32_t pre_cycles = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(0, 1, 16);
    for (uint32_t i = 0; i < pre_cycles; ++i) {  // empty acquisitions: each one flips the DST half
        tile_regs_acquire(); tile_regs_commit(); tile_regs_wait(); tile_regs_release();
    }
    move_block<true>(3, 4, N);  // worker_sum -> cb_prev_sum_2, as the stock kernel does
    correction_block<scale_fp32, VectorMode::RC>(
        /*cb_m_in=*/1, /*cb_prev_sum_2=*/4, /*cb_cur_max=*/16, /*cb_prev_max=*/0,
        /*cb_cur_sum=*/17, /*cb_prev_sum=*/2, /*cb_exp_max_diff=*/18, /*cb_exp_max_diff_2=*/19, N);
}
"""


def run(device, N, fp32_acc, full_sync, pre_cycles=0):
    torch.manual_seed(0)
    pm = torch.rand(N, 32, 32) * 2 - 1                    # prev_max
    wm = pm + torch.rand(N, 32, 32) * 2 - 1               # worker_max, above prev_max in ~half the rows
    ps = torch.rand(N, 32, 32) + 1                        # prev_sum
    ws = torch.rand(N, 32, 32) + 1                        # worker_sum
    inp = torch.stack([pm, wm, ps, ws]).reshape(1, 4 * N, 32, 32).to(torch.bfloat16)
    scale = 1 / math.sqrt(128)
    scale_bits = struct.unpack("<I", struct.pack("<f", scale))[0]

    x = ttnn.from_torch(inp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4 * N, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbs = [ttnn.CBDescriptor(total_size=N * 2048, core_ranges=core,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
           for i in (0, 1, 2, 3, 4, 16, 17, 18, 19)]
    src, cta = ttnn.KernelDescriptor.SourceType.SOURCE_CODE, ttnn.TensorAccessorArgs
    ra, wa = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    ra[0][0], wa[0][0] = [x.buffer_address()], [y.buffer_address()]
    kernels = [
        ttnn.KernelDescriptor(kernel_source=READER, source_type=src, core_ranges=core, runtime_args=ra,
                              compile_time_args=[N] + cta(x).get_compile_time_args(), config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=WRITER, source_type=src, core_ranges=core, runtime_args=wa,
                              compile_time_args=[N] + cta(y).get_compile_time_args(), config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=COMPUTE, source_type=src, core_ranges=core, runtime_args=[],
                              compile_time_args=[N, scale_bits, pre_cycles], defines=[("EXP_APPROX_MODE", "0")],
                              config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                                  fp32_dest_acc_en=fp32_acc, dst_full_sync_en=full_sync)),
    ]
    out = ttnn.generic_op([x, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    got = ttnn.to_torch(out).float().reshape(4, N, 32, 32)[..., 0]   # statistics live in column 0 of each tile

    pm, wm, ps, ws = [t.to(torch.bfloat16).float()[..., 0] for t in (pm, wm, ps, ws)]
    cur_max = torch.maximum(pm, wm)                                   # the helper's own comment:
    emd, emd2 = torch.exp((pm - cur_max) * scale), torch.exp((wm - cur_max) * scale)
    ref = torch.stack([cur_max, ps * emd + ws * emd2, emd, emd2])
    err = (got - ref).abs().reshape(4, -1).max(dim=1).values
    aliased = ((got[0] - ws).abs() < 1e-6).float().mean().item()      # cur_max came back equal to worker_sum
    window = (8 if full_sync else 4) if fp32_acc else (16 if full_sync else 8)
    print(f"fp32_acc={str(fp32_acc):5s} full_sync={str(full_sync):5s} fp32 tiles/window={window:2d} "
          f"pre_cycles={pre_cycles} N={N} | max|err| cur_max {err[0]:.3f} cur_sum {err[1]:.3f} "
          f"exp_diff {err[2]:.3f} exp_diff_2 {err[3]:.3f} | cur_max==worker_sum {100 * aliased:5.1f}% "
          f"| {'OK' if err.max() < 0.05 else 'WRONG'}", flush=True)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("--- DST configurations, no acquisitions before the merge")
        for fp32_acc, full_sync in ((True, False), (True, True), (False, False)):
            for N in (1, 2):
                run(device, N, fp32_acc, full_sync)
        print("--- parity probe: empty acquisitions inserted before the merge, 4-tile window")
        for pre in (0, 1, 2, 3):
            run(device, 1, True, False, pre)
    finally:
        ttnn.close_device(device)
