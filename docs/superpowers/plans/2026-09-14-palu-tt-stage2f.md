# Stage 2f Implementation Plan: rank and cache-format allocation under the measured cost model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the quality-per-byte frontier over K rank, V rank and latent cache format (bfp8 / bfp4) for Llama-3.1-8B-Instruct on the n300, including the stock model with a bfp4 KV cache and a unit-variance latent rescale, and recommend one allocation.

**Architecture:** Host-only. A `latent_scale` option in the factor loader moves the singular values into the up-projection; split K/V latent dtypes and a stock `kv_cache_dtype` flow from the CLIs through `build_model`; a `--ranks_json` patch to Palu's `compress.py` produces calibrated uniform-rank dumps; `report --stage 2f` and a frontier figure render the grid. Device time is perplexity screens at 40 windows, then full measurements on five finalists.

**Tech Stack:** PyTorch, safetensors, the Palu submodule (patched at run time by `scripts/patch_palu_cpu.sh`), ttnn / tt-transformers precision settings, matplotlib, pytest.

## Global Constraints

Copied from the spec (`docs/superpowers/specs/2026-09-14-palu-tt-stage2f-design.md`) and the project's standing rules:

- Bytes per token per layer per chip = `4 x (r_k x bpe_k + r_v x bpe_v)`, `bpe` 1.0625 (bfp8) / 0.5625 (bfp4); whole-model KB per token = that x 64 / 1024. Stock bfp8 = 1088 B = 69.6 KB per token; stock bfp4 = 576 B.
- Unit scale: `U'[:, j] = U[:, j] * c_j`, `VT'[j, :] = VT[j, :] / c_j`, `c_j = ||U[:, j]||` taken over the whole group's `U` (`gs x head_dim` rows); components with `c_j == 0` (rank padding) are left alone. The product `U' VT'` equals `U VT` to bf16 precision. Default `latent_scale = "balanced"` reproduces every existing row.
- Row naming: `<dump>[_u][_lk4|_lv4|_l4][_w40]`; `_u` unit scale, `_lk4` bfp4 K latent only, `_lv4` bfp4 V latent only, `_l4` both, `_w40` a 40-window screen; stock bfp4 is `baseline_kv4`. Uniform dumps are `compressed/uni_rk<K>_rv<V>` and rows `uni_rk<K>_rv<V>...`.
- Grids: A = {`palu_full_50`, `palu_full_70`, `glrd_50_f256`, `glrd_70_f256`} x {bfp8, bfp4 both} x {balanced, unit} plus `baseline_kv4`, 13 new screens; B = `uni_rk{32,64,96}_rv{32,64,96,128}` x {bfp8, bfp4} at unit scale, 24 screens. Screens are 40 windows; finalists (four cells plus `baseline_kv4`) get 141-window perplexity, the five tasks, `decode_check`, decode 1k / 4k / 8k (16k where it allocates) and max context at batch 32.
- Success bars: Q1 frontier table and figure; Q2 balanced-vs-unit delta with noise; Q3 stock bfp4 in full; Q4 a recommendation at perplexity <= 9.1 and <= 37 KB per token or an explicit "none on the grid"; Q5 finalists' decode within 10 % of the stage 2d prediction.
- No kernel changes; stage 2e's `--latent-dtype` flag and `_l4` rows are reused (stage 2e Task 6). Whichever stage runs a Grid A cell first keeps its JSON.
- Every device command runs as `scripts/run_device.sh <cmd>` and only after `pgrep -f "palu_tt\."` prints nothing. CPU work niced (`nice -n 10`), <= 12 threads. No sudo, no firmware/driver changes, never touch `/home/user01/tt-data`. `third_party/palu` is patched at run time and never committed. Never commit logs, CSVs, `.superpowers/` or the `compressed/` dumps (only the raw result JSONs under `docs/results/raw/`).
- `git add` by explicit path; commit with `git -c user.name=ryan -c user.email=ryankert01@gmail.com`; every commit message ends with the two trailer lines `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs` (verbatim, no other model name).
- Python env: `source scripts/env.sh`.

---

## File map

| file | responsibility |
|---|---|
| `palu_tt/convert.py` | `unit_scale_lowrank`, `load_palu_dir(..., latent_scale)`, `PaluConfig.latent_scale`, `k_latent_dtype` / `v_latent_dtype`, cache id suffix |
| `palu_tt/model.py` | `build_model(..., latent_scale, k_latent_dtype, v_latent_dtype, kv_cache_dtype)`, stock byte accounting |
| `palu_tt/attention.py` | per-cache latent dtypes (allocation, typecasts, byte accounting, L1 model call) |
| `palu_tt/fused.py` | `palu_l1_cb_bytes(..., k_latent_tile_bytes, v_latent_tile_bytes)` |
| `palu_tt/bench.py`, `eval_ppl.py`, `eval_tasks.py`, `decode_check.py` | `--latent-scale`, `--k-latent-dtype`, `--v-latent-dtype`, `--kv-dtype`, `--latent-dtype` on `eval_tasks` |
| `palu_tt/latent_stats.py` (new) | per-component latent std on wikitext windows through the CPU reference (spec section 3 check) |
| `palu_tt/uniform_ranks.py` (new), `scripts/run_calibration_uniform.sh` (new), `scripts/patch_palu_cpu.sh` | uniform-rank dumps |
| `palu_tt/report.py`, `palu_tt/plots.py` | `--stage 2f` grid table, `--frontier` figure |
| `tests/test_convert.py`, `tests/test_fused.py`, `tests/test_bench_utils.py`, `tests/test_uniform_ranks.py` (new), `tests/test_report.py`, `tests/test_plots.py` | coverage |
| `docs/results/stage2f.md`, `docs/results/frontier_2f.png`, `docs/results/raw/*.json` | results |

---

### Task 1: unit-variance latent scale

**Files:**
- Modify: `palu_tt/convert.py:27-62` (`PaluRuntime`/`PaluConfig`), `:128-146` (`load_palu_dir`), `:235-262` (`add_palu_to_state_dict`)
- Modify: `palu_tt/model.py:13-35`
- Modify: `palu_tt/bench.py`, `palu_tt/eval_ppl.py`, `palu_tt/eval_tasks.py`, `palu_tt/decode_check.py` (flag)
- Create: `palu_tt/latent_stats.py`
- Test: `tests/test_convert.py`

**Interfaces:**
- Produces: `convert.unit_scale_lowrank(lr: LowRank) -> LowRank`; `convert.load_palu_dir(..., latent_scale: str = "balanced")`; `convert.add_palu_to_state_dict(..., latent_scale: str = "balanced")` (cache id gets `-unit` appended when `unit`); `PaluConfig.latent_scale: str`; `build_model(..., latent_scale="balanced")`; CLI `--latent-scale {balanced,unit}` on the four CLIs, recorded in every result JSON as `"latent_scale"`; `python -m palu_tt.latent_stats --palu-dir D --layers 0 15 --windows 2 [--latent-scale unit]` writing `docs/results/raw/latent_stats_<dump>[_u].json`.

- [ ] **Step 1: Failing tests**

Append to `tests/test_convert.py`:

```python
def _balanced_factors(n_kv=2, hd=128, dim=256, r=64, seed=0):
    """Palu's split on synthetic weights: L = U sqrt(S), R = sqrt(S) V, with decaying singular values."""
    g = torch.Generator().manual_seed(seed)
    VT, U = [], []
    for _ in range(n_kv):
        W = torch.randn(hd, dim, generator=g)
        u, s, vt = torch.linalg.svd(W, full_matrices=False)
        s = s[:r] * torch.logspace(0, -2, r)            # make the spectrum decay by 100x across the rank
        VT.append(torch.sqrt(s)[:, None] * vt[:r])
        U.append(u[:, :r] * torch.sqrt(s)[None, :])
    return convert.LowRank(k_VT=torch.stack(VT), k_U=torch.stack(U), v_VT=torch.stack(VT).clone(),
                           v_U=torch.stack(U).clone(), group_size=1)


def test_unit_scale_keeps_the_product_and_flattens_the_latent_variance():
    lr = _balanced_factors()
    unit = convert.unit_scale_lowrank(lr)
    for h in range(2):
        assert torch.allclose(unit.k_U[h] @ unit.k_VT[h], lr.k_U[h] @ lr.k_VT[h], rtol=1e-4, atol=1e-5)
    x = torch.randn(4096, 256)                          # whitened input: identity covariance
    before = (x @ lr.k_VT[0].T).std(0)
    after = (x @ unit.k_VT[0].T).std(0)
    assert before.max() / before.min() > 5              # Palu's split: variance follows the singular values
    assert after.max() / after.min() < 1.3              # unit scale: every component about the same size


def test_unit_scale_leaves_rank_padding_alone():
    lr = _balanced_factors(r=64)
    lr.k_U[:, :, 48:] = 0                               # padded columns
    lr.k_VT[:, 48:, :] = 0
    unit = convert.unit_scale_lowrank(lr)
    assert torch.equal(unit.k_U[:, :, 48:], lr.k_U[:, :, 48:])
    assert torch.isfinite(unit.k_VT).all()


def test_unit_scale_is_group_aware():
    """With group size 4 the group's U is the four heads' hd-row slices stacked, so c_j is the norm over
    all four slices and the (identical) group VT rows are divided by the same c_j on every head."""
    lr = _balanced_factors(n_kv=4)
    lr = convert.LowRank(k_VT=lr.k_VT[:1].repeat(4, 1, 1), k_U=lr.k_U, v_VT=lr.v_VT[:1].repeat(4, 1, 1),
                         v_U=lr.v_U, group_size=4)
    unit = convert.unit_scale_lowrank(lr)
    assert torch.equal(unit.k_VT[0], unit.k_VT[3])
    c = torch.sqrt((lr.k_U ** 2).sum(dim=(0, 1)))
    assert torch.allclose(unit.k_U[2], lr.k_U[2] * c[None, :])


def test_unit_scale_changes_the_cache_id(tmp_path):
    d = tmp_path / "dump"
    d.mkdir()
    (d / "config.json").write_text('{"head_wise_ranks": {}}')
    base = convert.palu_dir_cache_id(d)
    assert convert.palu_cache_id_for(d, "balanced") == base
    assert convert.palu_cache_id_for(d, "unit") == base + "-unit"
```

Run: `source scripts/env.sh && python -m pytest tests/test_convert.py -q -k unit_scale`
Expected: FAIL with `AttributeError: module 'palu_tt.convert' has no attribute 'unit_scale_lowrank'`

- [ ] **Step 2: Implement in `convert.py`**

After `class LowRank` add:

```python
def unit_scale_lowrank(lr: LowRank) -> LowRank:
    """Move each latent component's scale into the up-projection.

    Palu's whitened decomposition splits the singular values evenly (U sqrt(S), sqrt(S) VT), so latent
    component j has variance proportional to sigma_j over the whitening distribution. Scaling column j of
    U by c_j = ||U[:, j]|| (the norm over the whole group's rows, = sqrt(sigma_j)) and row j of VT by 1/c_j
    leaves U @ VT unchanged and makes every latent component unit variance -- what a block-floating-point
    cache wants. Zero columns (rank padding) are left alone.
    """
    def scale(VT, U, gs):
        n_kv = U.shape[0]
        VT2, U2 = VT.clone(), U.clone()
        for g in range(n_kv // gs):
            heads = slice(g * gs, (g + 1) * gs)
            c = torch.sqrt((U[heads] ** 2).sum(dim=(0, 1)))          # [r], over the group's gs*hd rows
            c = torch.where(c > 0, c, torch.ones_like(c))
            U2[heads] = U[heads] * c[None, None, :]
            VT2[heads] = VT[heads] / c[None, :, None]
        return VT2, U2
    k_VT, k_U = scale(lr.k_VT, lr.k_U, lr.group_size)
    v_VT, v_U = scale(lr.v_VT, lr.v_U, lr.group_size)
    return LowRank(k_VT=k_VT, k_U=k_U, v_VT=v_VT, v_U=v_U, group_size=lr.group_size)


def palu_cache_id_for(palu_dir: Path, latent_scale: str) -> str:
    """The weight-cache id for a dump under a latent scale: rescaled factors have the same shapes and
    dtypes as the balanced ones, and the on-disk weight cache is keyed by name/shape/dtype only, so the
    id has to differ (see the ttnn weight-cache note in the README)."""
    base = palu_dir_cache_id(palu_dir)
    return base if latent_scale == "balanced" else f"{base}-{latent_scale}"
```

`load_palu_dir(...)` gains `latent_scale: str = "balanced"` and, before `out[i] = ...`, `lr = LowRank(...)`, `out[i] = unit_scale_lowrank(lr) if latent_scale == "unit" else lr` (assert `latent_scale in ("balanced", "unit")` at the top). `add_palu_to_state_dict(...)` gains `latent_scale: str = "balanced"`, forwards it to `load_palu_dir`, and computes `cache_id = "identity" if palu_dir is None else palu_cache_id_for(palu_dir, latent_scale)`; `PaluConfig` gains `latent_scale: str = "balanced"` set from the argument. `LowRank` needs to be constructible as `LowRank(k_VT=..., k_U=..., v_VT=..., v_U=..., group_size=...)` (it is a dataclass; check the field order at `convert.py:70-90` and use keywords as above).

`palu_tt/model.py` `build_model`: add `latent_scale: str = "balanced"` and pass it to `convert.add_palu_to_state_dict(..., latent_scale=latent_scale)`.

CLIs (`bench.py`, `eval_ppl.py`, `eval_tasks.py`, `decode_check.py`):

```python
    p.add_argument("--latent-scale", choices=["balanced", "unit"], default="balanced",
                   help="stage 2f: 'unit' moves the singular values into the up-projection so every latent "
                        "component has unit variance (what bfp4 caches want); 'balanced' is Palu's split.")
```

forwarded as `latent_scale=a.latent_scale` into every `pm.build_model` call of the Palu path (bench: through the `FUSED` dict) and recorded as `"latent_scale": a.latent_scale` in each result JSON.

Run: `source scripts/env.sh && python -m pytest tests/test_convert.py tests/test_bench_utils.py tests/test_ppl_harness.py -q`
Expected: PASS.

- [ ] **Step 3: The empirical latent-std check**

Create `palu_tt/latent_stats.py`:

```python
"""Per-component standard deviation of the K and V latents on wikitext windows, through the CPU reference
model (stage 2f, spec section 3): shows whether Palu's balanced split leaves a decaying variance profile
and whether the unit rescale flattens it on the real factors.

    python -m palu_tt.latent_stats --palu-dir compressed/palu_full_0.5 --layers 0 15 --windows 2 [--latent-scale unit]
"""
import argparse
import json
from pathlib import Path

import torch

from palu_tt import convert, eval_ppl, hf, reference

RAW = Path("docs/results/raw")


def latent_std(model, ids, layers, windows, seq_len=2048):
    stats = {}
    hooks = []
    captured = {}

    def hook(layer, proj):
        def fn(module, inp, out):
            x = inp[0]
            lat = (x @ module.VT.T).reshape(-1, module.n_kv, module.r)
            captured.setdefault((layer, proj), []).append(lat.float())
        return fn

    for i in layers:
        attn = model.model.layers[i].self_attn
        hooks.append(attn.k_proj.register_forward_hook(hook(i, "k")))
        hooks.append(attn.v_proj.register_forward_hook(hook(i, "v")))
    with torch.no_grad():
        for w in range(windows):
            model(ids[:, w * seq_len:(w + 1) * seq_len])
    for h in hooks:
        h.remove()
    for (i, proj), chunks in captured.items():
        lat = torch.cat(chunks, 0)                       # [tokens, n_kv, r]
        std = lat.std(0)                                 # [n_kv, r]
        stats[f"layer{i}_{proj}"] = {"std_per_component_head0": std[0].tolist(),
                                     "max_over_min": float(std.max(1).values.div(std.min(1).values).mean())}
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--palu-dir", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 15])
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--latent-scale", choices=["balanced", "unit"], default="balanced")
    a = p.parse_args()
    from transformers import AutoTokenizer
    hf_dir = hf.snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    lm = reference.build_cpu_lowrank_llama(hf_dir, a.palu_dir, latent_scale=a.latent_scale)
    ids = eval_ppl.wikitext2_ids(tok)
    r = {"palu_dir": a.palu_dir, "latent_scale": a.latent_scale, "layers": a.layers, "windows": a.windows,
         "stats": latent_std(lm, ids, a.layers, a.windows)}
    name = Path(a.palu_dir).name + ("_u" if a.latent_scale == "unit" else "")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"latent_stats_{name}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v["max_over_min"] for k, v in r["stats"].items()}))


if __name__ == "__main__":
    main()
```

`palu_tt/reference.py` `build_cpu_lowrank_llama(hf_dir, palu_dir, latent_scale="balanced")` forwards `latent_scale` to `convert.load_palu_dir(...)`.

Run (CPU, niced, about 10 minutes each):
`source scripts/env.sh && nice -n 10 python -m palu_tt.latent_stats --palu-dir compressed/palu_full_0.5 --layers 0 15 --windows 2 && nice -n 10 python -m palu_tt.latent_stats --palu-dir compressed/palu_full_0.5 --layers 0 15 --windows 2 --latent-scale unit`
Expected: two JSONs; the balanced `max_over_min` is well above 2 and the unit one is close to 1 (within a factor of two is the spec's expectation). If the unit profile is not flat, the dump's `U` does not have the assumed structure: report DONE_WITH_CONCERNS with both numbers (Task 5 then uses the measured stds as `c_j`; do not silently change the rescale).

- [ ] **Step 4: Commit**

```bash
git add palu_tt/convert.py palu_tt/model.py palu_tt/reference.py palu_tt/bench.py palu_tt/eval_ppl.py palu_tt/eval_tasks.py palu_tt/decode_check.py palu_tt/latent_stats.py tests/test_convert.py docs/results/raw/latent_stats_palu_full_0.5.json docs/results/raw/latent_stats_palu_full_0.5_u.json
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2f: unit-variance latent scale (loader option, cache id, CLI flag) and the latent-std check" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 1b: cache-faithful prefill (the prefill instruments must see the latent format)

**Files:**
- Modify: `palu_tt/convert.py` (`PaluRuntime.latent_quant_in_prefill: bool = True`), `palu_tt/attention.py:396-442` (the prefill reconstruction and cache write)
- Test: `tests/test_attention_device.py`

**Interfaces:**
- Produces: `PaluRuntime.latent_quant_in_prefill` (default True); `build_model(..., latent_quant_in_prefill=True)`; when on, `PaluAttention` prefill round-trips each latent through its cache's dtype before the in-flight reconstruction, so `eval_ppl` and `eval_tasks` score what decode reads back.

- [ ] **Step 1: Failing device test**

Append to `tests/test_attention_device.py` (use its existing `_setup` / `_tt_attention` / `_to_torch` helpers and the `compressed/svd_r64` dump the file already uses; read the neighbouring `test_prefill_matches_reference` for the call shape):

```python
def test_prefill_sees_the_latent_cache_format(mesh_device):
    """With the round trip on, a bfp4 latent cache changes the prefill output (the window attends over the
    quantised latent); with it off, the latent format is invisible to prefill. Both stay close to the
    reference. Layer 0, one 2048-token window, batch 1."""
    import ttnn
    from palu_tt import fused
    outs = {}
    for name, dtype, quant in (("bfp8", ttnn.bfloat8_b, True), ("bfp4", ttnn.bfloat4_b, True), ("bfp4_blind", ttnn.bfloat4_b, False)):
        args, sd, rope, tt_ccl, x, ref = _setup(mesh_device, "compressed/svd_r64", 1, 2048)
        args.palu.latent_dtype = dtype
        args.palu.runtime.latent_quant_in_prefill = quant
        attn = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
        outs[name] = _to_torch(mesh_device, args, attn.forward_prefill(x, rope.get_rot_mats(...), user_id=0))
    assert not torch.equal(outs["bfp8"], outs["bfp4"])           # the format is visible
    assert torch.equal(outs["bfp8"], outs["bfp4_blind"])          # and invisible when the switch is off
    for name in ("bfp8", "bfp4"):
        ok, msg = comp_pcc(ref, outs[name], 0.99)
        assert ok, f"{name}: {msg}"
```

(Adapt the helper calls to the file's real signatures; `_setup` returns what `test_prefill_matches_reference` unpacks. If the file's helpers construct the layer from a fresh `args` each time, set the two attributes on that `args.palu` before constructing.)

Run: `scripts/run_device.sh python -m pytest tests/test_attention_device.py -q -k latent_cache_format`
Expected: FAIL on the first assert (today the outputs are equal: the format is invisible).

- [ ] **Step 2: Implement**

`palu_tt/convert.py` `PaluRuntime`: add

```python
    # stage 2f: round-trip the prefill latents through the cache format before reconstructing K and V, so
    # perplexity and task scores (prefill-only) see what decode will read back. Off = the stage 1..2e
    # behaviour, where the cache format never entered a prefill score.
    latent_quant_in_prefill: bool = True
```

`palu_tt/model.py` `build_model`: add `latent_quant_in_prefill: bool = True` and set `cfg.runtime.latent_quant_in_prefill = latent_quant_in_prefill`.

`palu_tt/attention.py`, in the prefill path, before `k_pre = recon(...)`: obtain `keys, values = self.layer_past` here (move the assignment up from the cache loop) and add

```python
        def _roundtrip(t, dtype):
            """t quantised to the cache format and back to bf16 (a new tensor; caller frees the old one)."""
            if dtype == ttnn.bfloat16:
                return t
            q = ttnn.typecast(t, dtype=dtype)
            out = ttnn.typecast(q, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q)
            return out

        if self.palu.runtime.latent_quant_in_prefill:
            # stage 2f: attend over what decode will read back. The zero padding of the slots stays zero, and
            # the cache write below re-quantises to the same values (block floating point is idempotent on
            # aligned tiles), so prefill and decode see one latent.
            if self.recon_k:
                k_q = _roundtrip(k_slot, keys.dtype)
                if k_q is not k_slot:
                    <free the old slot exactly as the later deallocation code would: k_slot when gs > 1, else k_lat_pad; then alias both names to k_q so the later deallocations stay valid>
            if self.recon_v:
                <same for v_slot / v_lat_pad with values.dtype>
```

Write the freeing so that the existing deallocation lines after the cache loop (`ttnn.deallocate(k_slot)` for `gs > 1`, `ttnn.deallocate(k_lat_pad)` for `recon_k`, and the V counterparts) still free exactly one live tensor each: the simplest correct form is to deallocate the old tensor immediately and rebind both `k_slot` and `k_lat_pad` (they are the same object at `gs == 1`) to the new one. Do not round-trip the fast-path K (`recon_k` False: the cache holds rotated full K, and the SDPA already uses `k8` typecast to the cache dtype) nor a full-width V (`recon_v` False). Keep the cache-write loop unchanged: it typecasts the (already quantised) latent again, which is exact.

- [ ] **Step 3: Run the prefill device tests**

Run: `scripts/run_device.sh python -m pytest tests/test_attention_device.py -q -k "prefill or latent_cache_format"`
Expected: PASS, including `test_identity_prefill_equals_stock_attention` (identity rows have `recon_k`/`recon_v` False, so nothing changes) and `test_prefill_matches_reference` on `svd_r64` (bfp8 round trip; PCC bar 0.99). Then `scripts/run_device.sh python -m pytest tests/test_attention_device.py -q -k "fused_decode_matches_naive"` (decode is untouched; a sanity check that the caches still hold what they held).

- [ ] **Step 4: One perplexity to quantify the shift, then commit**

`scripts/run_device.sh python -m palu_tt.eval_ppl --row palu_full_50_q --kind palu --palu-dir compressed/palu_full_0.5 --windows 40` (bfp8, faithful; about 3 minutes) and compare with the blind 12.93. Commit:

```bash
git add palu_tt/convert.py palu_tt/model.py palu_tt/attention.py tests/test_attention_device.py docs/results/raw/ppl_palu_full_50_q.json
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2f: cache-faithful prefill -- the latent is round-tripped through the cache format before reconstruction, so prefill perplexity and tasks see it" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 2: cache formats: split K/V latent dtypes and the stock bfp4 KV cache

**Files:**
- Modify: `palu_tt/convert.py` (`PaluConfig.k_latent_dtype`, `v_latent_dtype`), `palu_tt/model.py:13-35,52-66`, `palu_tt/attention.py:145-150,255-262,305-309,325-330` (and the typecasts that write the two caches), `palu_tt/fused.py:142-190`
- Modify: the four CLIs
- Test: `tests/test_fused.py`, `tests/test_bench_utils.py`

**Interfaces:**
- Produces: `build_model(..., latent_dtype=None, k_latent_dtype=None, v_latent_dtype=None, kv_cache_dtype=None)` (`k_`/`v_` override `latent_dtype`; `kv_cache_dtype` applies to the stock path and to the Palu path's `kv_cache_dtype` for the naive reconstruction temporaries); `PaluAttention.k_latent_dtype` / `v_latent_dtype`; `fused.palu_l1_cb_bytes(..., latent_tile_bytes=1088, k_latent_tile_bytes=None, v_latent_tile_bytes=None)`; CLI `--k-latent-dtype`, `--v-latent-dtype` (`{bfp8,bfp4}`, default: follow `--latent-dtype`), `--kv-dtype {bfp8,bfp4}` (stock rows), `--latent-dtype` also on `eval_tasks`; result JSONs record `k_latent_dtype`, `v_latent_dtype`, `kv_dtype`.

- [ ] **Step 1: Failing host tests**

Append to `tests/test_fused.py`:

```python
def test_l1_model_takes_separate_k_and_v_latent_tile_sizes():
    both8 = fused.palu_l1_cb_bytes(64, 128, 512, True, 1, cores_per_head=1)
    k4 = fused.palu_l1_cb_bytes(64, 128, 512, True, 1, cores_per_head=1, k_latent_tile_bytes=576)
    v4 = fused.palu_l1_cb_bytes(64, 128, 512, True, 1, cores_per_head=1, v_latent_tile_bytes=576)
    assert both8 - k4 == 16 * 2 * 2 * (1088 - 576)      # c_9: sk * rt * 2 tiles at r_k 64
    assert both8 - v4 == 16 * 4 * 2 * (1088 - 576)      # c_2: sk * rvt * 2 tiles at r_v 128
    assert fused.palu_l1_cb_bytes(64, 128, 512, True, 1, cores_per_head=1, latent_tile_bytes=576) == \
        fused.palu_l1_cb_bytes(64, 128, 512, True, 1, cores_per_head=1, k_latent_tile_bytes=576, v_latent_tile_bytes=576)
```

Append to `tests/test_bench_utils.py`:

```python
def test_latent_dtype_flags_resolve_to_ttnn_formats():
    from palu_tt.bench import resolve_latent_dtypes
    import ttnn
    assert resolve_latent_dtypes("bfp8", None, None) == (ttnn.bfloat8_b, ttnn.bfloat8_b)
    assert resolve_latent_dtypes("bfp8", "bfp4", None) == (ttnn.bfloat4_b, ttnn.bfloat8_b)
    assert resolve_latent_dtypes("bfp4", None, "bfp8") == (ttnn.bfloat4_b, ttnn.bfloat8_b)
```

Run: `source scripts/env.sh && python -m pytest tests/test_fused.py tests/test_bench_utils.py -q -k "latent_tile_sizes or latent_dtype_flags"`
Expected: FAIL (unexpected keyword / no `resolve_latent_dtypes`).

- [ ] **Step 2: L1 model and flag resolver**

`palu_tt/fused.py` `palu_l1_cb_bytes`: add `k_latent_tile_bytes: Optional[int] = None, v_latent_tile_bytes: Optional[int] = None` after `latent_tile_bytes`; at the top `kt = k_latent_tile_bytes or latent_tile_bytes; vt_ = v_latent_tile_bytes or latent_tile_bytes`; use `kt` in the `c_9` line and `vt_` in the `c_2` line. `group_heads_per_core` forwards the two new keywords.

`palu_tt/bench.py` (module level, imported by the other CLIs):

```python
LATENT_DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def resolve_latent_dtypes(latent, k_latent, v_latent):
    """(K, V) cache formats from the three flags: the per-cache flags override the shared one."""
    return LATENT_DTYPES[k_latent or latent], LATENT_DTYPES[v_latent or latent]
```

Run the two host tests: PASS.

- [ ] **Step 3: Model and attention**

`palu_tt/convert.py` `PaluConfig`: add `k_latent_dtype: Optional[object] = None` and `v_latent_dtype: Optional[object] = None` next to `latent_dtype` (None = follow `latent_dtype`).

`palu_tt/model.py` `build_model`: add `k_latent_dtype=None, v_latent_dtype=None, kv_cache_dtype=None`; set `cfg.k_latent_dtype, cfg.v_latent_dtype = k_latent_dtype, v_latent_dtype`; and, right after `args = ModelArgs(...)`:

```python
    if kv_cache_dtype is not None:
        # stage 2f: the stock (and naive-Palu) KV cache format. tt-transformers reads it per layer from
        # the DecodersPrecision object; rebuild that object with the KV_CACHE group overridden.
        from models.tt_transformers.tt.model_config import DecodersPrecision, ModelOptimizations, PrecisionSetting, TensorGroup
        setting = {ttnn.bfloat8_b: PrecisionSetting.BFP8, ttnn.bfloat4_b: PrecisionSetting.BFP4,
                   ttnn.bfloat16: PrecisionSetting.BF16}[kv_cache_dtype]
        conf = ModelOptimizations.accuracy(args.model_name)
        conf.tensor_dtype_settings[TensorGroup.KV_CACHE] = setting
        args.optimizations = DecodersPrecision(args.n_layers, args.model_name, conf)
        args.model_config["DECODERS_OPTIMIZATIONS"] = args.optimizations
```

(`tensor_dtype_settings` is the dict `get_tensor_dtype` reads at `model_config.py:4745-4770`; confirm the attribute name there and in `ModelOptimizations.accuracy` at `:134-200`, and use the name the code has.) `kv_bytes_per_token_per_layer` (line 63): extend the `bpe` map with `ttnn.bfloat4_b: 0.5625`.

`palu_tt/attention.py`: replace the single `self.latent_dtype` (line 147) with

```python
        base = self.palu.latent_dtype if self.palu.latent_dtype is not None else self.kv_cache_dtype
        self.k_latent_dtype = self.palu.k_latent_dtype or base
        self.v_latent_dtype = self.palu.v_latent_dtype or base
        self.latent_dtype = self.k_latent_dtype      # kept for callers that only know one format
```

then `grep -n "latent_dtype" palu_tt/attention.py` and, at every use, pick the K or the V format by what is being allocated or written: the K latent cache allocation and every `typecast` of a K latent use `self.k_latent_dtype`; the V ones `self.v_latent_dtype`; `kv_bytes_per_token` (line 325) becomes `4 heads x (r_k x bpe(k) + r_v x bpe(v))` with `bpe = {bfp8: 1.0625, bf16: 2.0, bfp4: 0.5625}`; the `fused.group_heads_per_core(...)` call passes `k_latent_tile_bytes=LATENT_TILE_BYTES.get(self.k_latent_dtype, 2048), v_latent_tile_bytes=LATENT_TILE_BYTES.get(self.v_latent_dtype, 2048)` instead of `latent_tile_bytes`.

CLIs: `--k-latent-dtype` / `--v-latent-dtype` (`choices=["bfp8", "bfp4"], default=None`) on all four, `--kv-dtype` (`choices=["bfp8", "bfp4"], default="bfp8"`) on all four, `--latent-dtype` added to `eval_tasks.py`; every Palu `build_model` call receives `k_latent_dtype, v_latent_dtype = resolve_latent_dtypes(a.latent_dtype, a.k_latent_dtype, a.v_latent_dtype)` and `kv_cache_dtype=LATENT_DTYPES[a.kv_dtype]`; stock calls receive `kv_cache_dtype=LATENT_DTYPES[a.kv_dtype]`. Each result JSON records `"k_latent_dtype"`, `"v_latent_dtype"`, `"kv_dtype"`.

Run: `source scripts/env.sh && python -m pytest tests/test_convert.py tests/test_fused.py tests/test_bench_utils.py tests/test_ppl_harness.py -q`
Expected: PASS.

- [ ] **Step 4: Device smoke of the two new paths (gated)**

```bash
pgrep -f "palu_tt\." || true
scripts/run_device.sh python -m palu_tt.eval_ppl --row smoke_kv4 --kind stock --kv-dtype bfp4 --windows 2
scripts/run_device.sh python -m palu_tt.eval_ppl --row smoke_lk4 --kind palu --palu-dir compressed/palu_full_0.7 --k-latent-dtype bfp4 --windows 2
```

Expected: both finish and print a perplexity (any value; two windows is a smoke). Delete the two smoke JSONs (`docs/results/raw/ppl_smoke_*.json`) afterwards; they are never committed. If the stock bfp4 run fails inside a tt-metal op, record the error text in the report: Q3 then reads "unavailable" and the rest of the stage proceeds.

- [ ] **Step 5: Commit**

```bash
git add palu_tt/convert.py palu_tt/model.py palu_tt/attention.py palu_tt/fused.py palu_tt/bench.py palu_tt/eval_ppl.py palu_tt/eval_tasks.py palu_tt/decode_check.py tests/test_fused.py tests/test_bench_utils.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2f: per-cache latent formats and a stock bfp4 KV cache, with byte accounting and the L1 model" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 3: uniform-rank calibrated dumps

**Files:**
- Create: `palu_tt/uniform_ranks.py`, `scripts/run_calibration_uniform.sh`
- Modify: `scripts/patch_palu_cpu.sh` (append the `--ranks_json` patch)
- Test: `tests/test_uniform_ranks.py` (new)

**Interfaces:**
- Produces: `uniform_ranks.write(path, n_layers: int, groups_per_matrix: int, r_k: int, r_v: int)` writing `{"model.layers.<i>.self_attn.k_proj": [r_k] * groups, "...v_proj": [r_v] * groups}`; `scripts/run_calibration_uniform.sh <r_k> <r_v> [gs]` -> `compressed/uni_rk<K>_rv<V>` (gs 1) with `palu_lowrank.safetensors` and `config.json` carrying `head_wise_ranks`, like the other dumps.

- [ ] **Step 1: Failing test**

Create `tests/test_uniform_ranks.py`:

```python
import json

from palu_tt import uniform_ranks


def test_writer_emits_one_entry_per_projection_with_the_uniform_rank(tmp_path):
    p = tmp_path / "ranks.json"
    uniform_ranks.write(p, n_layers=3, groups_per_matrix=8, r_k=64, r_v=32)
    d = json.loads(p.read_text())
    assert set(d) == {f"model.layers.{i}.self_attn.{m}" for i in range(3) for m in ("k_proj", "v_proj")}
    assert d["model.layers.1.self_attn.k_proj"] == [64] * 8
    assert d["model.layers.2.self_attn.v_proj"] == [32] * 8
```

Run: `source scripts/env.sh && python -m pytest tests/test_uniform_ranks.py -q` -> FAIL (no module).

- [ ] **Step 2: Writer, patch, script**

`palu_tt/uniform_ranks.py`:

```python
"""Uniform per-group rank lists in the format Palu's compress.py consumes through --ranks_json (stage 2f)."""
import argparse
import json
from pathlib import Path


def write(path, n_layers, groups_per_matrix, r_k, r_v):
    d = {}
    for i in range(n_layers):
        d[f"model.layers.{i}.self_attn.k_proj"] = [int(r_k)] * groups_per_matrix
        d[f"model.layers.{i}.self_attn.v_proj"] = [int(r_v)] * groups_per_matrix
    Path(path).write_text(json.dumps(d, indent=1))
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--groups", type=int, default=8, help="latent groups per matrix: n_kv_heads / group size")
    p.add_argument("--r-k", type=int, required=True)
    p.add_argument("--r-v", type=int, required=True)
    a = p.parse_args()
    write(a.out, a.layers, a.groups, a.r_k, a.r_v)


if __name__ == "__main__":
    main()
```

Append to `scripts/patch_palu_cpu.sh` (before its final "applied cleanly" echo; idempotent like the other patches):

```bash
# compress.py: --ranks_json replaces the rank search with a supplied per-module rank list
# (stage 2f uniform-rank dumps). Keys are module names, values the per-group rank lists.
python3 - <<'PY'
import pathlib
p = pathlib.Path("compress.py"); s = p.read_text()
old = "    search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)\n"
new = ("    if getattr(args, 'ranks_json', None):\n"
       "        import json\n"
       "        search_results = json.load(open(args.ranks_json))\n"
       "        rank_sum = sum(sum(v) for v in search_results.values())\n"
       "        total_rank = 0\n"
       "    else:\n"
       "        search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)\n")
if old in s:
    s = s.replace(old, new)
if '"--ranks_json"' not in s:
    s = s.replace('    parser.add_argument(\n        "--model_id",',
                  '    parser.add_argument("--ranks_json", type=str, default=None, help="per-module rank lists; skips the rank search")\n    parser.add_argument(\n        "--model_id",', 1)
p.write_text(s)
PY
grep -n "ranks_json" compress.py >/dev/null || { echo "ranks_json patch missing"; exit 1; }
```

`scripts/run_calibration_uniform.sh`:

```bash
#!/usr/bin/env bash
# Calibrated (whitened) per-head factors at UNIFORM ranks (stage 2f): no Fisher search, r_k for every
# k_proj group and r_v for every v_proj group. Usage: scripts/run_calibration_uniform.sh <r_k> <r_v> [gs]
# Output: compressed/uni_rk<r_k>_rv<r_v> (gs 1) or compressed/uni_glrd_rk<r_k>_rv<r_v>; factors only.
set -euo pipefail
source "$(dirname "$0")/env.sh"
rk="$1"; rv="$2"; gs="${3:-1}"
scripts/patch_palu_cpu.sh
name="uni_rk${rk}_rv${rv}"; [[ "$gs" != "1" ]] && name="uni_glrd_rk${rk}_rv${rv}"
groups=$((8 / gs))
mkdir -p compressed
python -m palu_tt.uniform_ranks --out "compressed/${name}.ranks.json" --layers 32 --groups "$groups" --r-k "$rk" --r-v "$rv"
cd third_party/palu
HF_DIR="$(python -c 'from palu_tt import hf; print(hf.snapshot_dir())')"
mkdir -p cache
nice -n 10 python compress.py --model_id "$HF_DIR" --device cpu --calib_dataset wikitext2 --calib_seqlen 1024 \
  --param_ratio_target 1.0 --search_method uniform --decompose_method whiten --head_group_size "$gs" \
  --ranks_json "../../compressed/${name}.ranks.json" --use_cache --dump_huggingface_model \
  2>&1 | tee "../../compressed/calibration_${name}.log"
out="$(ls -d *_ratio-1.0_gs-${gs}-uniform-whiten 2>/dev/null | head -1)"
python - "$out" <<'PY'
import sys, os
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
d = Path(sys.argv[1]); tensors = {}
for f in sorted(d.glob("*.safetensors")):
    with safe_open(str(f), framework="pt") as sf:
        for k in sf.keys():
            if ".k_proj." in k or ".v_proj." in k:
                tensors[k] = sf.get_tensor(k).contiguous()
save_file(tensors, str(d / "palu_lowrank.safetensors"), metadata={"format": "pt"})
for f in d.glob("model*.safetensors"):
    os.remove(f)
print("kept", len(tensors), "low-rank tensors, removed the full dump")
PY
mv "$out" "../../compressed/${name}"
echo "done -> compressed/${name}"
```

`chmod +x scripts/run_calibration_uniform.sh`. The whitening scale matrices are cached under `third_party/palu/cache/` from the stage 1 / 2c runs (`--use_cache`), so a dump is the SVDs plus the 30 GB dump-and-trim, about ten minutes.

- [ ] **Step 3: First dump, and check it loads**

Run: `nice -n 10 scripts/run_calibration_uniform.sh 64 64 2>&1 | tail -3` (CPU only, about 10 minutes; leaves `compressed/uni_rk64_rv64/`).
Then: `source scripts/env.sh && python -c "from palu_tt import convert; lr = convert.load_palu_dir('compressed/uni_rk64_rv64', 32, 8, 128); print(lr[0].r_k, lr[0].r_v, lr[31].k_VT.shape)"`
Expected: `64 64 torch.Size([8, 64, 4096])`. Confirm `compressed/uni_rk64_rv64/config.json` has `head_wise_ranks` with 64s throughout.

- [ ] **Step 4: Commit (the dump itself is not committed)**

```bash
git add palu_tt/uniform_ranks.py scripts/run_calibration_uniform.sh scripts/patch_palu_cpu.sh tests/test_uniform_ranks.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2f: uniform-rank calibrated dumps (ranks_json patch, writer, run script)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

Then start Grid B's remaining eleven dumps as a background CPU queue (scratchpad script `stage2f_dumps.sh`: a loop over `rk in 32 64 96; rv in 32 64 96 128` skipping `compressed/uni_rk${rk}_rv${rv}` when it exists, `nohup ... &`). They run while Tasks 4 and 5 proceed; `tail -2` of its log shows progress.

---

### Task 4: the frontier report and figure

**Files:**
- Modify: `palu_tt/report.py` (`ROWS2F`, `_stage2f_lines`, `--stage 2f`), `palu_tt/plots.py` (`frontier_figure`, `--frontier`)
- Test: `tests/test_report.py`, `tests/test_plots.py`

**Interfaces:**
- Consumes: `ppl_<row>.json` (fields `ppl`, `windows`), `tasks_<row>.json`, `decode_check_<row>.json`, `bench_decode_<row>.json`, `bench_maxctx_<row>.json`; the dumps' `config.json` `head_wise_ranks` via `report._per_layer_ranks` (`palu_tt/report.py:187`).
- Produces: `report.frontier_rows(raw) -> list[dict]` with keys `row, dump, r_k_mean, r_v_mean, k_fmt, v_fmt, scale, bytes_layer_chip, kb_token, reduction, ppl, windows, avg5, decode_4k, maxctx`; `plots.frontier_figure(raw_dir, out_png)`.

- [ ] **Step 1: Failing tests**

Append to `tests/test_report.py`:

```python
def test_frontier_rows_parse_the_2f_row_names(tmp_path):
    from palu_tt import report
    raw = tmp_path
    (raw / "ppl_baseline.json").write_text('{"ppl": 7.254, "windows": 141}')
    (raw / "ppl_baseline_kv4.json").write_text('{"ppl": 7.9, "windows": 141}')
    (raw / "ppl_palu_full_50_u_l4_w40.json").write_text('{"ppl": 13.1, "windows": 40}')
    (raw / "ppl_uni_rk64_rv32_lv4_w40.json").write_text('{"ppl": 9.9, "windows": 40}')
    rows = {r["row"]: r for r in report.frontier_rows(raw, ranks={"palu_full_50": (33, 102), "uni_rk64_rv32": (64, 32)})}
    assert rows["baseline"]["bytes_layer_chip"] == 1088 and rows["baseline_kv4"]["bytes_layer_chip"] == 576
    r = rows["palu_full_50_u_l4_w40"]
    assert (r["k_fmt"], r["v_fmt"], r["scale"], r["windows"]) == ("bfp4", "bfp4", "unit", 40)
    assert abs(r["bytes_layer_chip"] - 4 * (33 + 102) * 0.5625) < 1e-6
    u = rows["uni_rk64_rv32_lv4_w40"]
    assert (u["k_fmt"], u["v_fmt"]) == ("bfp8", "bfp4")
    assert abs(u["bytes_layer_chip"] - 4 * (64 * 1.0625 + 32 * 0.5625)) < 1e-6
```

Append to `tests/test_plots.py`:

```python
def test_frontier_figure_renders(tmp_path):
    from palu_tt import plots
    (tmp_path / "ppl_baseline.json").write_text('{"ppl": 7.254, "windows": 141}')
    (tmp_path / "ppl_baseline_kv4.json").write_text('{"ppl": 7.9, "windows": 141}')
    (tmp_path / "ppl_uni_rk64_rv32_l4_w40.json").write_text('{"ppl": 9.9, "windows": 40}')
    out = tmp_path / "frontier.png"
    plots.frontier_figure(tmp_path, out, ranks={"uni_rk64_rv32": (64, 32)})
    assert out.exists() and out.stat().st_size > 1000
```

Run: `source scripts/env.sh && python -m pytest tests/test_report.py tests/test_plots.py -q -k frontier` -> FAIL.

- [ ] **Step 2: Implement**

`palu_tt/report.py`:

```python
import re as _re

FRONTIER_RE = _re.compile(r"^(?P<dump>baseline|palu_full_50|palu_full_70|glrd_50_f256|glrd_70_f256|uni_rk\d+_rv\d+)"
                          r"(?P<kv4>_kv4)?(?P<u>_u)?(?P<fmt>_lk4|_lv4|_l4)?(?P<w40>_w40)?$")
BPE = {"bfp8": 1.0625, "bfp4": 0.5625}


def _dump_ranks(raw, dump, ranks):
    """(mean r_k, mean r_v) for a dump: the caller's override, else head_wise_ranks from the dump."""
    if dump in ranks:
        return ranks[dump]
    if dump == "baseline":
        return 128, 128
    if dump.startswith("uni_"):
        m = _re.match(r"uni_rk(\d+)_rv(\d+)", dump)
        return int(m.group(1)), int(m.group(2))
    palu_dir = {"palu_full_50": "compressed/palu_full_0.5", "palu_full_70": "compressed/palu_full_0.7",
                "glrd_50_f256": "compressed/glrd_0.5_f256", "glrd_70_f256": "compressed/glrd_0.7_f256"}[dump]
    per_layer = _per_layer_ranks(Path(palu_dir), None)
    return per_layer["k_mean"], per_layer["v_mean"]


def frontier_rows(raw, ranks=None):
    raw, ranks = Path(raw), ranks or {}
    out = []
    for p in sorted(raw.glob("ppl_*.json")):
        row = p.stem[len("ppl_"):]
        m = FRONTIER_RE.match(row)
        if not m:
            continue
        dump = m.group("dump")
        stock = dump == "baseline"
        fmt = m.group("fmt") or ""
        k_fmt = "bfp4" if (fmt in ("_lk4", "_l4") or (stock and m.group("kv4"))) else "bfp8"
        v_fmt = "bfp4" if (fmt in ("_lv4", "_l4") or (stock and m.group("kv4"))) else "bfp8"
        r_k, r_v = _dump_ranks(raw, dump, ranks)
        bytes_lc = 4 * (r_k * BPE[k_fmt] + r_v * BPE[v_fmt])
        d = json.loads(p.read_text())
        tsk = _load(raw, f"tasks_{row}")
        dec = _load(raw, f"bench_decode_{row}")
        mx = _load(raw, f"bench_maxctx_{row}")
        out.append({"row": row, "dump": dump, "r_k_mean": r_k, "r_v_mean": r_v, "k_fmt": k_fmt, "v_fmt": v_fmt,
                    "scale": "unit" if m.group("u") else "balanced", "bytes_layer_chip": bytes_lc,
                    "kb_token": bytes_lc * 64 / 1024, "reduction": 1088 / bytes_lc,
                    "ppl": d["ppl"], "windows": d["windows"],
                    "avg5": _avg5(tsk) if tsk else None,
                    "decode_4k": _min_only(_cell(dec, 4096)) if dec else None,
                    "maxctx": mx["data"].get("max_ctx") if mx else None})
    return sorted(out, key=lambda r: (r["kb_token"], r["ppl"]))
```

(`_per_layer_ranks`'s return shape: read it at `palu_tt/report.py:187-209` and take the mean K and V rank from what it returns; the dict keys above are illustrative, use the real ones. `_avg5`, `_min_only`, `_cell`, `_load` are the existing helpers.) Add `_stage2f_lines(raw)` that renders `frontier_rows` as a markdown table with the columns of the spec's section 3 (row, `r_k`, `r_v`, K fmt, V fmt, scale, B per token per layer per chip, KB per token, reduction, perplexity with windows, avg5, decode 4k, max context), wired as `--stage 2f` writing `docs/results/stage2f.md` through `_finish` like the other stages.

`palu_tt/plots.py`:

```python
def frontier_figure(raw_dir, out_png, ranks=None):
    """Perplexity against KB per token for every 2f cell; marker shape = K format, fill = V format,
    colour = dump family; the Pareto front (lowest perplexity at each byte budget) drawn through the points."""
    from palu_tt import report
    rows = report.frontier_rows(raw_dir, ranks=ranks)
    fig, ax = plt.subplots(figsize=(8, 5))
    colour = {"baseline": "#333333", "palu_full_50": "#2f6fdb", "palu_full_70": "#7fa8f2",
              "glrd_50_f256": "#e07b39", "glrd_70_f256": "#f0a06a"}
    for r in rows:
        c = colour.get(r["dump"], "#3c9d6b")
        marker = "s" if r["k_fmt"] == "bfp4" else "o"
        face = c if r["v_fmt"] == "bfp8" else "white"
        ax.scatter(r["kb_token"], r["ppl"], marker=marker, s=48, facecolors=face, edgecolors=c, linewidths=1.5)
        ax.annotate(r["row"], (r["kb_token"], r["ppl"]), textcoords="offset points", xytext=(4, 3), fontsize=6)
    front, best = [], float("inf")
    for r in sorted(rows, key=lambda r: r["kb_token"]):
        if r["ppl"] < best:
            best = r["ppl"]; front.append(r)
    if len(front) > 1:
        ax.plot([r["kb_token"] for r in front], [r["ppl"] for r in front], color="#999999", linewidth=1, linestyle="--")
    ax.set_xlabel("KV cache, KB per token (whole model, both chips)"); ax.set_ylabel("wikitext-2 perplexity")
    ax.set_title("quality per byte: rank x format (filled = bfp8 V, hollow = bfp4 V; square = bfp4 K)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
```

with `--frontier` in `plots.main()` writing `docs/results/frontier_2f.png`.

Run: `source scripts/env.sh && python -m pytest tests/test_report.py tests/test_plots.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add palu_tt/report.py palu_tt/plots.py tests/test_report.py tests/test_plots.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2f: frontier table (report --stage 2f) and figure (plots --frontier)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 5: Grid A (the afternoon)

**Files:**
- Create (scratchpad): `stage2f_gridA.sh`
- Produce: `docs/results/raw/ppl_*_w40.json` for the 12 Palu cells, `docs/results/raw/ppl_baseline_kv4.json` (141 windows)

- [ ] **Step 1: Queue**

```bash
#!/usr/bin/env bash
set -uo pipefail
cd /home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9 && source scripts/env.sh
gate() { while pgrep -f "palu_tt\." >/dev/null; do sleep 30; done; scripts/device_free.sh; }
declare -A DIR=([palu_full_50]=compressed/palu_full_0.5 [palu_full_70]=compressed/palu_full_0.7
                [glrd_50_f256]=compressed/glrd_0.5_f256 [glrd_70_f256]=compressed/glrd_0.7_f256)
for dump in palu_full_50 palu_full_70 glrd_50_f256 glrd_70_f256; do
  for scale in balanced unit; do
    for fmt in bfp8 bfp4; do
      row="$dump"; [[ $scale == unit ]] && row="${row}_u"; [[ $fmt == bfp4 ]] && row="${row}_l4"; row="${row}_w40"
      [[ -f docs/results/raw/ppl_${row}.json ]] && continue
      gate; python -m palu_tt.eval_ppl --row "$row" --kind palu --palu-dir "${DIR[$dump]}" --latent-scale "$scale" --latent-dtype "$fmt" --windows 40
    done
  done
done
[[ -f docs/results/raw/ppl_baseline_kv4.json ]] || { gate; python -m palu_tt.eval_ppl --row baseline_kv4 --kind stock --kv-dtype bfp4; }
echo GRID A FINISHED
```

Sixteen screens (the bfp8 balanced cells are re-measured under the cache-faithful prefill) plus the full stock bfp4 perplexity: about 60 minutes. (`palu_full_50_l4` and `palu_full_70_l4` at 141 windows may already exist from stage 2e Task 6; the screens are separate rows and still run, so balanced bfp4 has both a full and a 40-window value: a same-cell noise check.)

- [ ] **Step 2: Run, render, commit**

`nohup bash $SCRATCH/stage2f_gridA.sh > $SCRATCH/stage2f_gridA.log 2>&1 &`, wait for `GRID A FINISHED`, then `python -m palu_tt.report --stage 2f && python -m palu_tt.plots --frontier`. Commit the JSONs, the table and the figure:

```bash
git add docs/results/raw/ppl_*_w40.json docs/results/raw/ppl_baseline_kv4.json docs/results/stage2f.md docs/results/frontier_2f.png
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2f: Grid A screens (formats x scale on the four calibrated dumps) and the stock bfp4 perplexity" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

Report the Q2 deltas (bfp4 unit minus bfp4 balanced per dump) and the stock bfp4 perplexity in the task report.

---

### Task 6: Grid B (the factorial)

**Files:**
- Create (scratchpad): `stage2f_gridB.sh`
- Produce: `docs/results/raw/ppl_uni_rk*_rv*_u_{,l4_}w40.json` (24 files)

- [ ] **Step 1: Wait for the dumps**

`ls -d compressed/uni_rk*_rv*` must list all twelve (Task 3's background queue); re-run any missing one with `nice -n 10 scripts/run_calibration_uniform.sh <rk> <rv>`.

- [ ] **Step 2: Queue and run**

```bash
#!/usr/bin/env bash
set -uo pipefail
cd /home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9 && source scripts/env.sh
gate() { while pgrep -f "palu_tt\." >/dev/null; do sleep 30; done; scripts/device_free.sh; }
for rk in 32 64 96; do for rv in 32 64 96 128; do for fmt in bfp8 bfp4; do
  row="uni_rk${rk}_rv${rv}_u"; [[ $fmt == bfp4 ]] && row="${row}_l4"; row="${row}_w40"
  [[ -f docs/results/raw/ppl_${row}.json ]] && continue
  gate; python -m palu_tt.eval_ppl --row "$row" --kind palu --palu-dir "compressed/uni_rk${rk}_rv${rv}" --latent-scale unit --latent-dtype "$fmt" --windows 40
done; done; done
echo GRID B FINISHED
```

About 75 minutes. Then `python -m palu_tt.report --stage 2f && python -m palu_tt.plots --frontier`, commit the 24 JSONs, the table and the figure with the same commit form as Task 5 (`Stage 2f: Grid B screens, uniform ranks x format at unit scale`).

- [ ] **Step 3: Decide the split-budget trigger**

From the table: at equal bytes, compare cells that trade `r_k` for `r_v` (e.g. `uni_rk64_rv32` vs `uni_rk32_rv64`, `uni_rk96_rv64` vs `uni_rk64_rv96`). If the perplexity difference between such pairs exceeds 0.3 consistently in one direction, the trigger fires: record it in the report and run the two split-budget dumps as the spec describes (Fisher-uniform on `k_proj` at one ratio and `v_proj` at another is a further `patch_palu_cpu.sh` change to `rank_search`'s `fisher_uniform` branch restricting the module set; it is written only if the trigger fires and is then its own commit). Otherwise record "trigger not fired" with the numbers.

---

### Task 7: finalists, the stock bfp4 baseline in full, and the write-up

**Files:**
- Produce: for each finalist `ppl_<row>.json` (141 windows), `tasks_<row>.json`, `decode_check_<row>_fk2.json`, `bench_decode_<row>_fk2.json`, `bench_maxctx_<row>_fk2.json`; for `baseline_kv4`: `tasks_baseline_kv4.json`, `bench_decode_baseline_kv4.json`, `bench_maxctx_baseline_kv4.json`
- Modify: `docs/results/stage2f.md` (hand-written analysis below the marker)

- [ ] **Step 1: Pick the finalists**

From the frontier table choose four Palu cells: the lowest-perplexity cell at or under 37 KB per token, the lowest-byte cell at or under perplexity 9.1, and the two cells nearest the Pareto front between them; at most two may share a dump. Write the choice and its reasons in the task report before running anything.

- [ ] **Step 2: Measure (gated queue in the scratchpad)**

For each finalist row `R` (name without `_w40`; flags as its name says: `--latent-scale`, `--latent-dtype` / `--k-latent-dtype` / `--v-latent-dtype`, dump dir):

```bash
gate; python -m palu_tt.eval_ppl   --row R --kind palu --palu-dir D <flags>
gate; python -m palu_tt.eval_tasks --row R --kind palu --palu-dir D <flags>
gate; python -m palu_tt.decode_check --row R_fk2 --palu-dir D --fused p2 --group-heads 1 <flags>
gate; python -m palu_tt.bench --row R_fk2 --kind palu --palu-dir D --what decode --fused p2 <flags>
gate; python -m palu_tt.bench --row R_fk2 --kind palu --palu-dir D --what maxctx --fused p2 <flags>
```

and for the stock baseline: `eval_tasks --row baseline_kv4 --kind stock --kv-dtype bfp4`, `bench --row baseline_kv4 --kind stock --what decode --kv-dtype bfp4`, `bench ... --what maxctx --kv-dtype bfp4`. Where a bfp4 finalist's max context at batch 32 reaches 16384, add `PALU_BENCH_CONTEXTS=1024,4096,8192,16384` to its decode bench (the env var `palu_tt.bench` reads at `bench.py:49`). About six hours; run as one background queue and poll.

- [ ] **Step 3: Render and write**

`python -m palu_tt.report --stage 2f && python -m palu_tt.plots --frontier`. Below the marker in `docs/results/stage2f.md`, in this order: (1) Q1 to Q5, one line each, met or missed with the number; (2) the recommendation, or "none on the grid", with bytes, perplexity, tasks and decode time against stock and against `palu_full_70`; (3) Q2 in words: did unit scaling matter, by how much, on which dumps; (4) the stock bfp4 baseline against Palu ratio 0.5 at equal bytes; (5) the factorial read: which of `r_k`, `r_v` perplexity is more sensitive to at equal bytes, and whether the split-budget trigger fired; (6) Q5: the finalists' decode against the stage 2d prediction; (7) caveats from the spec's section 5. Under 1000 words; every number in a generated table or a raw JSON.

- [ ] **Step 4: Commit**

```bash
git add docs/results/raw/ppl_*.json docs/results/raw/tasks_*.json docs/results/raw/decode_check_*.json docs/results/raw/bench_decode_*.json docs/results/raw/bench_maxctx_*.json docs/results/stage2f.md docs/results/frontier_2f.png
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2f: finalists in full, the stock bfp4 baseline, and the results document" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

(`git add` with globs only adds files matching the patterns; confirm with `git status --short` that no smoke or log file is staged.) The controller then runs the final whole-branch review, merges, and publishes the results page.

---

## Self-review

- **Spec coverage.** Section 0's rescale: Task 1 (implementation, synthetic test, group awareness, cache id, empirical check). Section 2 Grid A: Task 5; Grid B: Tasks 3 and 6; the trigger: Task 6 Step 3; screening and finalists: Tasks 5 to 7. Section 3 pipeline items: uniform dumps (Task 3), latent scale (Task 1), cache formats incl. stock bfp4 and byte accounting (Task 2), report and frontier (Task 4). Section 4 time prediction: Task 7 Step 3 item 6 against the stage 2d fit (the prediction is arithmetic on the fit table; no new code). Section 5 report items 1 to 6: Tasks 4 and 7. Section 6 risks: bfp4 on stock (Task 2 Step 4), bfp4 on the fused op (stage 2e Task 6 and Task 2's smoke), the rescale assumption (Task 1 Step 3), uniform vs Fisher (the anchors in the table, Task 6 Step 3), tasks before recommending (Task 7), the K-only cliff (`_lv4` cells are Grid B's format axis; add `_lk4`/`_lv4` cells for the two best Grid B corners in Task 7 Step 1 if the both-bfp4 cell of a corner fails and the bfp8 one passes).
- **Placeholders.** None; helper names referenced from `report.py` are pinned to line ranges with the instruction to use the real return shapes.
- **Type consistency.** `latent_scale` is a `str` (`"balanced"` / `"unit"`) in `load_palu_dir`, `add_palu_to_state_dict`, `PaluConfig`, `build_model`, the CLIs and `build_cpu_lowrank_llama`; `resolve_latent_dtypes(latent, k_latent, v_latent) -> (ttnn dtype, ttnn dtype)`; `palu_l1_cb_bytes(..., k_latent_tile_bytes=None, v_latent_tile_bytes=None)`; `frontier_rows(raw, ranks=None)` and `frontier_figure(raw_dir, out_png, ranks=None)` share the `ranks: dict[dump] -> (r_k, r_v)` override; `uniform_ranks.write(path, n_layers, groups_per_matrix, r_k, r_v)`.
