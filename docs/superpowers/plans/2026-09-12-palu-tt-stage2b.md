# Palu on TT — Stage 2b: G-LRD and Task Accuracy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Palu's G-LRD (group size 4, one group per chip) end to end — converter, prefill, naive decode, fused kernel with in-kernel V reconstruction — and evaluate every study row with lm-eval zero-shot accuracy on five tasks, on the n300, alongside the existing perplexity, memory and decode rows.

**Architecture:** The group latent lives where the four heads' latents lived (the K/V section of each chip's fused QKV weight), the caches become one K latent and one V latent per chip, and every consumer reconstructs a head from the shared latent with that head's slice of the group factor. The fused kernel gets a `heads_per_group` index, a wider latent envelope, and a second on-core build pass that reconstructs V into the K slot after the QK matmul has consumed it. Task accuracy comes from an lm-eval adapter that scores continuations through device prefill and the host lm_head, the path perplexity already uses.

**Tech Stack:** tt-metal v0.78.0 + submodule branch `palu-sdpa`, ttnn Python, torch, lm-eval 0.4.2, pytest. Spec: `docs/superpowers/specs/2026-09-12-palu-tt-stage2b-design.md` (§3 host layout, §4 kernel, §2 harness, §7 budget).

## Global Constraints

- **Shared server.** No sudo. Every device command through `scripts/run_device.sh <cmd>` (refuses when the n300 is busy; wait and retry, never bypass). `source scripts/env.sh && source .venv/bin/activate` first, from the repo root `/home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9`. CPU work `nice -n 10`, ≤ 12 threads. Never touch `/home/user01/tt-data`. One device process at a time; long device jobs run as a background loop with per-run logs, never inside one blocking tool call.
- **Evaluation runs on the device, not the CPU** (user requirement). CPU is for calibration (done) and unit tests. No CPU perplexity for the new rows.
- **Disk.** Root disk is shared and was full once. Calibration dumps are stored factors-only (`palu_lowrank.safetensors`); nothing regenerable goes through the weight cache; latent caches get no `cache_file_name`. `.log` files are never committed.
- **Two repos.** tt-metal changes are committed inside the submodule on branch `palu-sdpa` with `--no-verify` (the clang-format hook reformats copied files): `git -C third_party/tt-metal add <files>; git -C third_party/tt-metal -c user.name=ryan -c user.email=ryankert01@gmail.com commit --no-verify -m ...`; the superproject then records the pointer with `git add third_party/tt-metal`. Never `git add third_party/palu` or `.superpowers/`. Host C++ edits need `scripts/build_ttnn.sh` (10-25 min); kernel edits are JIT.
- **Commit trailers**, verbatim last two lines of every commit in both repos: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs`. Commit as `git -c user.name=ryan -c user.email=ryankert01@gmail.com commit`.
- **Group layout (spec §3).** `group_size = n_kv_heads / n_groups` from the dump (`head_wise_ranks[proj]` has one entry per group); on this model `n_kv = 8`, groups = 2, one group per chip. Latent widths are padded to a multiple of 32: `r_g ∈ 32..512`. Group `g`'s VT occupies rows `[g·gs·hd, g·gs·hd + r_g)` of `wk_lat_pad`/`wv_lat_pad`; head `h` has `u_k[h] = reverse_permute(U_g[(h mod gs)·hd : +hd, :r_g]).T` (`[r_g, hd]`) and `u_v[h] = U_g[slice].T`; `w_rot[h] = [u_k[h] | u_k[h]·T]`. Caches: `[B, n_local_groups, S, r_k_g]`, `[B, n_local_groups, S, r_v_g]`, `n_local_groups = n_local_kv_heads / gs` (= 1 here). Group size 1 is the same code and must reproduce stage 1/2 tensors bit-for-bit.
- **Kernel contract (spec §4).** New op attribute `heads_per_group = w_rot.shape[1] / k_latent.shape[1]` (must divide); reader offsets latent reads by `cur_head / heads_per_group`; `num_kv_heads` for the work layout comes from `w_rot.shape[1]`. `r_k` any multiple of 32 in 32..512; when `r_k > 128` or `u_v` is given the host default chunk is 128 and the per-head weight CB holds one block. New optional input `u_v [1, n_kv, r_v_g, 128]` bf16 selects in-kernel V reconstruction: per chunk, after the QK matmul, `V_h = lat_v @ u_v[h]` is built row-tile by row-tile into `c_1` in row-major tile order (`k·vDHt + n`) and the PV matmul reads in1 from `c_1`; `head_dim_v` is then 128 (the op's output width) and the V latent width is `v_latent.shape[3]`; the per-head weight block in `c_4` is `[w_rot (Rt·8) | u_v (Rvt·4) | T (1, Phase 2)]`, one push per head, CB sized in whole blocks. Without `u_v` the op is bit-identical to stage 2.
- **lm-eval (spec §2).** Version stays 0.4.2. Tasks: `openbookqa, piqa, arc_easy, arc_challenge, winogrande`, full, zero-shot. Dataset overrides: `allenai/openbookqa` (config `main`), `ybisk/piqa` (config `plain_text`), `allenai/winogrande` (config `winogrande_xl`), registered from `palu_tt/lm_eval_tasks/` via `TaskManager(include_path=...)`. Downloads need `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` once; runs are offline afterwards. Output `docs/results/raw/tasks_<row>.json`.
- **Rows.** `baseline` (stock), `palu_full_50`, `palu_full_70` (M-LRD dumps `compressed/palu_full_0.5|0.7`), `glrd_50`, `glrd_70` (dumps `compressed/glrd_0.5|0.7`, group size 4). Decode rows for G-LRD: naive (`--fused off`) and fused Phase 2 (`--fused p2`, row suffix `_fk2`).
- **Accuracy bars carried from stage 2.** Op-level PCC ≥ 0.99 vs the fp32 reference and ≥ 0.995 vs the stock op on host-reconstructed K/V; attention-level fused-vs-naive PCC ≥ 0.99 with byte-identical caches; converter group-size-1 output bit-identical to before.
- **Time budget (spec §7):** ~5 h of device time in total; the task-accuracy block (~3 h) runs as one queued job.

## File structure

- `palu_tt/convert.py` — group-aware `load_palu_dir`, `LowRank.group_size`, `PaluConfig.group_size`, `palu_layer_tensors` group layout.
- `palu_tt/attention.py` — group caches, group latent extraction (`_group_latent`), prefill/naive-decode reconstruction from the group latent, fused path passing `u_v`, no latent cache files.
- `palu_tt/reference.py` — unchanged API; works on the per-head expanded factors.
- `palu_tt/fused.py` — `reference_decode(..., u_v=None)`.
- `palu_tt/lm_eval_tt.py` (new) — `TTLM`, the lm-eval adapter.
- `palu_tt/eval_tasks.py` (new) — CLI writing `tasks_<row>.json`.
- `palu_tt/lm_eval_tasks/*.yaml` (new) — dataset-id overrides.
- `palu_tt/report.py` — `--stage 2b`, `docs/results/stage2b.md`.
- `scripts/run_calibration.sh` — `--group-size`, factors-only output.
- tt-metal submodule `palu_sdpa_decode/` — `heads_per_group`, wide `r_k`, `u_v` mode: types, prim, public op, binding, validation, factory, reader, `palu_dataflow_common.hpp`, compute kernel, `palu_compute_common.hpp`.
- tests: `tests/test_convert.py`, `tests/test_reference.py`, `tests/test_attention_device.py`, `tests/test_palu_sdpa_device.py`, `tests/test_fused.py`, `tests/test_lm_eval_tt.py` (new), `tests/test_report.py`.
- results: `docs/results/stage2b.md`, `docs/results/raw/{tasks_*,ppl_glrd_*,bench_*_glrd_*,decode_check_glrd_*}.json`, `README.md`.

---

### Task 1: Group-aware converter and calibration script

Outcome: `convert.load_palu_dir` reads a group-size-4 dump into per-head factors with `group_size` recorded; `palu_layer_tensors` lays the group latent into the group's head slots and emits per-head `u_k`/`u_v`/`w_rot` of width `r_g`; group size 1 output is unchanged bit-for-bit. `scripts/run_calibration.sh` takes a group size and strips the full dump.

**Files:**
- Modify: `palu_tt/convert.py`, `scripts/run_calibration.sh`, `tests/test_convert.py`, `tests/test_reference.py` (one assertion).

**Interfaces:**
- Produces: `LowRank(k_VT, k_U, v_VT, v_U, group_size: int = 1)`; for `group_size = gs`, `k_VT[h]` is the group's VT (identical for the `gs` heads of a group), `k_U[h]` is `[hd, r_g]` = the head's row slice of the group's `U_g` (`[gs·hd, r_g]`), same for V. `LowRank.r_k`/`.r_v` are the (padded) group ranks.
- Produces: `load_palu_dir(palu_dir, n_layers, n_kv_heads, head_dim, rank_multiple=32) -> Dict[int, LowRank]` accepting both group sizes; `PaluConfig.group_size: int = 1`; `add_palu_to_state_dict(...)` sets it from the dump; `palu_layer_tensors(hf_attn, lr, n_heads, n_kv_heads, head_dim)` returns the same keys with group-layout content; `CONVERT_VERSION = 4`.
- Produces: `scripts/run_calibration.sh <ratio> [group_size]` → `compressed/palu_full_<ratio>` (gs 1) or `compressed/glrd_<ratio>` (gs 4), factors-only.

- [ ] **Step 1: Failing tests**

Append to `tests/test_convert.py` (uses the file's `DIM, NH, NKV, HD = 64, 4, 2, 16` and `_hf_attn()`):

```python
def _group_lowrank(w, gs, r):
    """A synthetic group factorisation: one U_g [gs*HD, r] and VT_g [r, DIM] per group of gs KV heads."""
    n_groups = NKV // gs
    VT = torch.randn(n_groups, r, DIM)
    U = torch.randn(n_groups, gs * HD, r)
    return VT, U


def _write_group_dump(tmp_path, VT, U, gs):
    """Palu's on-disk layout for head_group_size gs: U.{g}.weight [gs*hd, r_g], one stacked VT.weight, one rank per group."""
    from safetensors.torch import save_file
    d = tmp_path / "glrd"; d.mkdir()
    n_groups, r, _ = VT.shape
    tensors, ranks = {}, {}
    for proj, (vt, u) in {"k_proj": (VT, U), "v_proj": (VT * 0.5, U * 2.0)}.items():
        key = f"model.layers.0.self_attn.{proj}"
        tensors[f"{key}.VT.weight"] = vt.reshape(n_groups * r, DIM).contiguous()
        for g in range(n_groups):
            tensors[f"{key}.U.{g}.weight"] = u[g].contiguous()
        ranks[key] = [r] * n_groups
    save_file(tensors, str(d / "palu_lowrank.safetensors"))
    (d / "config.json").write_text(json.dumps({"head_wise_ranks": ranks}))
    return d


def test_load_palu_dir_expands_groups_to_heads(tmp_path):
    gs, r = 2, 8                                   # NKV = 2 -> one group of two heads
    VT, U = _group_lowrank(_hf_attn(), gs, r)
    d = _write_group_dump(tmp_path, VT, U, gs)
    lr = convert.load_palu_dir(d, 1, NKV, HD, rank_multiple=8)[0]
    assert lr.group_size == gs and lr.r_k == r
    x = torch.randn(5, DIM)
    for h in range(NKV):
        g, hh = h // gs, h % gs
        assert torch.equal(lr.k_VT[h], VT[g])                          # the group's VT, shared by its heads
        assert torch.equal(lr.k_U[h], U[g, hh * HD:(hh + 1) * HD])      # the head's slice of the group's U
        k_h = x @ VT[g].T @ U[g, hh * HD:(hh + 1) * HD].T             # what Palu computes for head h
        assert torch.allclose(x @ lr.k_VT[h].T @ lr.k_U[h].T, k_h, atol=1e-5)
        assert torch.allclose(lr.v_U[h], 2.0 * U[g, hh * HD:(hh + 1) * HD]) and torch.allclose(lr.v_VT[h], 0.5 * VT[g])


def test_group_size_one_dump_still_loads_as_before(tmp_path):
    d = _tiny_palu_dir(tmp_path / "p", r=HDR)
    lr = convert.load_palu_dir(d, 1, NKV, HDR)[0]
    assert lr.group_size == 1


def test_layer_tensors_lay_the_group_latent_into_the_groups_head_slots():
    gs, r = 2, 8
    w = _hf_attn()
    VT, U = _group_lowrank(w, gs, r)
    heads = []
    for h in range(NKV):
        g, hh = h // gs, h % gs
        heads.append((VT[g], U[g, hh * HD:(hh + 1) * HD]))
    lr = convert.LowRank(k_VT=torch.stack([a for a, _ in heads]), k_U=torch.stack([b for _, b in heads]),
                         v_VT=torch.stack([a for a, _ in heads]), v_U=torch.stack([b for _, b in heads]), group_size=gs)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    # K block: group g's VT sits at rows [g*gs*HD, g*gs*HD + r), the rest of the group's rows are zero
    assert torch.equal(t["wk_lat_pad"][:r], VT[0]) and torch.count_nonzero(t["wk_lat_pad"][r:gs * HD]) == 0
    assert torch.equal(t["wv_lat_pad"][:r], VT[0])
    # per-head up-projections carry the group rank, in Meta order for K
    assert t["u_k"].shape == (NKV, r, HD) and t["u_v"].shape == (NKV, r, HD) and t["w_rot"].shape == (NKV, r, 2 * HD)
    lat = torch.randn(3, r)
    for h in range(NKV):
        hh = h % gs
        v_h = lat @ U[0, hh * HD:(hh + 1) * HD].T
        assert torch.allclose(lat @ t["u_v"][h], v_h, atol=1e-5)
        k_h_meta = reverse_permute(U[0, hh * HD:(hh + 1) * HD], 1, HD, r).T
        assert torch.allclose(t["u_k"][h], k_h_meta, atol=1e-6)
    # grouped rows never absorb V: wo_fused is wo
    assert torch.equal(t["wo_fused"], w["o"].float())


def test_convert_version_is_4_and_group_size_default_is_1():
    assert convert.CONVERT_VERSION == 4
    cfg = convert.PaluConfig(rank_k=[32], rank_v=[64], cache_id="x")
    assert cfg.group_size == 1
```

`_tiny_palu_dir` and `HDR` already exist at the bottom of the file (the weight-set-identity tests); import `json` at the top if it is not imported. Append to `tests/test_reference.py`:

```python
def test_identity_lowrank_reports_group_size_one():
    from palu_tt import convert
    w = {"k": torch.randn(2 * 16, 64), "v": torch.randn(2 * 16, 64)}
    assert convert.identity_lowrank(w, 2, 16).group_size == 1
```

- [ ] **Step 2: Run, expect failure**

```bash
nice -n 10 pytest tests/test_convert.py tests/test_reference.py -q 2>&1 | tail -3
```
Expected: `TypeError: ... unexpected keyword argument 'group_size'` / `AssertionError` on the group tests; the existing tests pass.

- [ ] **Step 3: Converter**

In `palu_tt/convert.py`:

```python
# v4: group-aware layout (G-LRD): the group latent occupies its heads' slots in the fused QKV weight and
#     u_k/u_v/w_rot carry the group rank per head; group size 1 is unchanged.
CONVERT_VERSION = 4
```

`LowRank` gains a field and the docstring:

```python
@dataclass
class LowRank:
    """Per-head view of the factors: W_h ≈ U_h @ VT_h with k_VT [n_kv, r, dim], k_U [n_kv, hd, r], same for v.

    group_size > 1 (Palu's G-LRD): the gs heads of a group share one latent, so k_VT[h] is the group's
    VT (identical for the group's heads) and k_U[h] is the head's hd-row slice of the group's U
    [gs*hd, r]. Every consumer that reconstructs a head sees the same per-head interface."""
    k_VT: torch.Tensor
    k_U: torch.Tensor
    v_VT: torch.Tensor
    v_U: torch.Tensor
    group_size: int = 1
```

`PaluConfig` gains, after `latent_dtype`:

```python
    # Palu head group size (1 = M-LRD per head, 4 = G-LRD one group per chip on the N300). rank_k/rank_v
    # are the per-layer *group* latent widths; at group size 1 those are the per-head ranks as before.
    group_size: int = 1
```

Replace `_read_palu_layer` and `load_palu_dir`:

```python
def _read_palu_layer(tensors, ranks, layer, proj, n_kv_heads, head_dim, rank_multiple):
    """One projection of one layer -> per-head (VT, U) with the group's VT repeated over its heads.

    Palu stores one U per *group* (`U.{g}.weight`, [gs*hd, r_g]), one stacked VT ([sum r_g, dim]) and one
    rank per group; group size 1 is the head-wise case."""
    key = f"model.layers.{layer}.self_attn.{proj}"
    r_list = ranks[key]
    n_groups = len(r_list)
    assert n_kv_heads % n_groups == 0, f"{key}: {n_groups} groups do not divide {n_kv_heads} KV heads"
    gs = n_kv_heads // n_groups
    r_max = int(math.ceil(max(r_list) / rank_multiple) * rank_multiple)
    VT_all = tensors[f"{key}.VT.weight"]
    dim = VT_all.shape[1]
    VT = torch.zeros(n_kv_heads, r_max, dim)
    U = torch.zeros(n_kv_heads, head_dim, r_max)
    off = 0
    for g, r in enumerate(r_list):
        U_g = tensors[f"{key}.U.{g}.weight"]                       # [gs*hd, r]
        assert U_g.shape == (gs * head_dim, r), f"{key}.U.{g}: {tuple(U_g.shape)} != {(gs * head_dim, r)}"
        for hh in range(gs):
            h = g * gs + hh
            VT[h, :r] = VT_all[off:off + r]
            U[h, :, :r] = U_g[hh * head_dim:(hh + 1) * head_dim]
        off += r
    return VT, U, gs


def load_palu_dir(palu_dir: Path, n_layers: int, n_kv_heads: int, head_dim: int, rank_multiple: int = 32) -> Dict[int, LowRank]:
    """Reads a factors-only file (palu_lowrank.safetensors) or a full Palu HF dump (same key names). Group
    ranks are zero-padded to the layer max rounded up to rank_multiple, which changes nothing numerically."""
    palu_dir = Path(palu_dir)
    ranks = json.loads((palu_dir / "config.json").read_text())["head_wise_ranks"]
    tensors = {}
    for f in sorted(palu_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                if ".k_proj." in k or ".v_proj." in k:
                    tensors[k] = sf.get_tensor(k).float()
    out = {}
    for i in range(n_layers):
        k_VT, k_U, gs_k = _read_palu_layer(tensors, ranks, i, "k_proj", n_kv_heads, head_dim, rank_multiple)
        v_VT, v_U, gs_v = _read_palu_layer(tensors, ranks, i, "v_proj", n_kv_heads, head_dim, rank_multiple)
        assert gs_k == gs_v, f"layer {i}: K group size {gs_k} != V group size {gs_v}"
        out[i] = LowRank(k_VT, k_U, v_VT, v_U, group_size=gs_k)
    return out
```

In `palu_layer_tensors`, the group layout. Replace the body from `fast_v = r_v == head_dim` through the loop with:

```python
    gs = lr.group_size
    # Full-rank collapses (the stock-K and fast-V paths) exist for the per-head case only: a grouped
    # latent is always kept as a latent and reconstructed per head (in prefill, in the naive decode
    # and in the fused kernel), so wo_fused is plain wo there.
    fast_v = gs == 1 and r_v == head_dim
    u_v_eff = torch.eye(head_dim).expand(n_kv_heads, head_dim, head_dim) if fast_v else lr.v_U
    for h in range(n_kv_heads):
        g, hh = h // gs, h % gs
        if gs == 1 and r_k == head_dim:
            wk_lat_pad[h * head_dim:(h + 1) * head_dim] = reverse_permute(lr.k_U[h] @ lr.k_VT[h], 1, head_dim, dim)
        elif hh == 0:
            # the group's VT goes once, into the first r_k rows of the group's gs*head_dim slot
            wk_lat_pad[g * gs * head_dim:g * gs * head_dim + r_k] = lr.k_VT[h]
        if fast_v:
            wv_lat_pad[h * head_dim:(h + 1) * head_dim] = lr.v_U[h] @ lr.v_VT[h]
        elif hh == 0:
            wv_lat_pad[g * gs * head_dim:g * gs * head_dim + r_v] = lr.v_VT[h]
        # K leaves the latent in Meta-permuted head order (RoPE convention of tt_transformers)
        u_k[h] = reverse_permute(lr.k_U[h], 1, head_dim, r_k).T
        u_v[h] = u_v_eff[h].T
    wo = hf_attn["o"].float()                       # [dim, n_heads*hd]
    if gs > 1:
        wo_fused = wo.clone()
    else:
        wo_fused = torch.zeros_like(wo)
        rep = n_heads // n_kv_heads
        for qh in range(n_heads):
            kv = qh // rep
            wo_fused[:, qh * head_dim:qh * head_dim + r_v] = wo[:, qh * head_dim:(qh + 1) * head_dim] @ u_v_eff[kv]
    return {"wk_lat_pad": wk_lat_pad, "wv_lat_pad": wv_lat_pad, "u_k": u_k, "u_v": u_v, "wo_fused": wo_fused,
            "w_rot": fused.w_rot_from_u_k(u_k)}
```

(Keep the existing comments above the loop; they still describe the `gs == 1` branches. `r_k`/`r_v` are `lr.r_k`/`lr.r_v`, the padded group ranks; for `gs > 1` they may exceed `head_dim`, and `u_k`/`u_v` are allocated `[n_kv, r_k, hd]`/`[n_kv, r_v, hd]` as today.) In `add_palu_to_state_dict`, set the group size from layer 0's factors and assert it is constant:

```python
    gs = {lr.group_size for lr in (lrs.values() if lrs else [])} or {1}
    assert len(gs) == 1, f"mixed group sizes across layers: {gs}"
    return PaluConfig(rank_k=rank_k, rank_v=rank_v, cache_id=cache_id, group_size=gs.pop())
```

- [ ] **Step 4: Calibration script**

Replace `scripts/run_calibration.sh` with:

```bash
#!/usr/bin/env bash
# Palu's own recipe on CPU: Fisher-uniform rank search + whitened decomposition (float64 whitening via
# scripts/patch_palu_cpu.sh). Usage: scripts/run_calibration.sh <ratio> [group_size]   (group_size 1 = M-LRD,
# 4 = G-LRD one group per chip). Output: compressed/palu_full_<ratio> (gs 1) or compressed/glrd_<ratio>.
# The full HF dump is 30 GB; only the k_proj/v_proj factors are kept (palu_lowrank.safetensors, ~0.6 GB).
set -euo pipefail
source "$(dirname "$0")/env.sh"
ratio="${1:-0.5}"
gs="${2:-1}"
scripts/patch_palu_cpu.sh
uv pip install loguru click lm-eval==0.4.2 2>/dev/null || true
cd third_party/palu
HF_DIR="$(python -c 'from palu_tt import hf; print(hf.snapshot_dir())')"
mkdir -p cache
name="palu_full_${ratio}"; [[ "$gs" != "1" ]] && name="glrd_${ratio}"
nice -n 10 python compress.py --model_id "$HF_DIR" --device cpu --calib_dataset wikitext2 --calib_seqlen 1024 \
  --param_ratio_target "$ratio" --search_method fisher_uniform --decompose_method whiten --head_group_size "$gs" \
  --use_cache --dump_huggingface_model 2>&1 | tee "../../compressed/calibration_${name}.log"
out="$(ls -d *_ratio-${ratio}_gs-${gs}-fisher_uniform-whiten 2>/dev/null | head -1)"
python - "$out" <<'PY'
import sys, glob, os
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

- [ ] **Step 5: Run and commit**

```bash
nice -n 10 pytest tests/test_convert.py tests/test_reference.py tests/test_attention_tag.py tests/test_fused.py -q 2>&1 | tail -2
```
Expected: all pass (`test_attention_tag` follows `CONVERT_VERSION` automatically; if it hard-codes `palu_v3_`, update it to `palu_v4_`).

```bash
git add palu_tt/convert.py scripts/run_calibration.sh tests/test_convert.py tests/test_reference.py tests/test_attention_tag.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2b Task 1: group-aware converter (G-LRD layout), calibration script with group size and factors-only output

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 2: G-LRD on the host path — group caches, prefill and naive decode

Outcome: `PaluAttention` runs a group-size-4 dump through prefill and the naive decode path against the CPU reference; caches are one K latent and one V latent per chip; latent caches no longer touch the weight cache. Group size 1 behaviour is unchanged (all existing device tests pass).

**Files:**
- Modify: `palu_tt/attention.py`, `tests/test_attention_device.py`.

**Interfaces:**
- Consumes: `PaluConfig.group_size`, `LowRank.group_size`, `palu_layer_tensors` group layout (Task 1).
- Produces: `PaluAttention.gs`, `.n_local_groups`, `.u_v_decode` (naive path, grouped rows), `._group_latent(heads_1BKD, r) -> [1, B, 1, r]` height-sharded, `._group_latent_prefill(heads_1KSD, r) -> [1, 1, S, r]`; caches `[B, n_local_groups, max_seq_len, r]`; `kv_bytes_per_token = n_local_groups * (r_k + r_v) * bpe`. `use_fused_k` stays False for grouped rows until Task 4.

- [ ] **Step 1: Failing device test**

Append to `tests/test_attention_device.py`:

```python
def test_glrd_prefill_and_naive_decode_match_reference(mesh_device):
    """Group size 4: one K latent and one V latent per chip; prefill and the naive decode rebuild every
    head from the shared latent. Reference = the per-head CPU module on the expanded factors."""
    batch = 32
    args, sd, rope, tt_ccl, ref = _setup(mesh_device, "compressed/glrd_0.5", batch=batch, max_seq_len=SEQ)
    args.palu.runtime.check = True
    tt = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    assert tt.gs == 4 and tt.n_local_groups == 1 and not tt.use_fused_k
    assert tuple(tt.layer_past[0].shape)[:2] == (batch, 1) and tuple(tt.layer_past[1].shape)[:2] == (batch, 1)
    assert tt.kv_bytes_per_token() == 1 * (tt.r_k + tt.r_v) * 1.0625
    P = 128
    x_pre = torch.randn(1, P, args.dim) * 0.05
    y_ref = ref.forward(x_pre, start_pos=0)
    rot = [rope.cos_matrix_prefill[:, :, :P, :], rope.sin_matrix_prefill[:, :, :P, :]]
    y = tt.forward(args.prepare_residual_tensor_prefill(x_pre, force_replicated=True), None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    y = _to_torch(mesh_device, args, y)[0, 0, :P, : args.dim]
    ok, msg = comp_pcc(y_ref[0], y, 0.99)
    assert ok, f"prefill {msg}"
    for step in range(4):
        pos = P + step
        x = torch.randn(1, 1, args.dim) * 0.05
        y_ref = ref.forward(x, start_pos=pos)
        current_pos = torch.full((batch,), pos)
        args.palu.runtime.recon_len = ((pos + 1 + 31) // 32) * 32
        x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x.expand(batch, 1, -1).contiguous(), current_pos)
        y = tt.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
        y = _to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim)
        ok, msg = comp_pcc(y_ref, y[0:1], 0.99)
        assert ok, f"pos={pos} {msg}"
    # the group latent is the per-head latent of any head in the group (the reference keeps it per head)
    for i, ref_cache in ((0, ref.cache_k_lat), (1, ref.cache_v_lat)):
        c = ttnn.to_torch(tt.layer_past[i], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))
        got = c[:1, :, : P + 4, :]                                  # [1, n_groups(2), P+4, r]
        want = ref_cache[:, ::tt.gs, : P + 4, :]                    # head 0 of each group
        ok, msg = comp_pcc(want, got, 0.99)
        assert ok, f"cache {i} {msg}"
```

(`_setup` builds the reference with `convert.load_palu_dir(palu_dir, 1, n_kv, hd)` and `reference.PaluRefAttention`, both group-aware after Task 1 because the reference only ever sees per-head factors.)

```bash
scripts/run_device.sh pytest tests/test_attention_device.py -q -k glrd 2>&1 | tail -3
```
Expected: fails in `PaluAttention.__init__` (the `r_k <= head_dim` assert, since the group rank is 64 for K but 512 for V).

- [ ] **Step 2: `attention.py` — group geometry and caches**

In `__init__`, replace the rank asserts and the `recon_*` block:

```python
        self.palu = configuration.palu
        self.gs = int(self.palu.group_size)                         # heads sharing one latent (1 = M-LRD)
        assert self.n_local_kv_heads % self.gs == 0, f"group size {self.gs} must divide {self.n_local_kv_heads} local KV heads"
        self.n_local_groups = self.n_local_kv_heads // self.gs
        self.r_k = self.palu.rank_k[layer_num]                       # group latent widths (per head at gs == 1)
        self.r_v = self.palu.rank_v[layer_num]
        assert self.r_k % 32 == 0 and self.r_v % 32 == 0 and self.r_k <= self.gs * self.head_dim and self.r_v <= self.gs * self.head_dim
        # Full-rank fast paths exist for per-head factors only (see convert.palu_layer_tensors): a grouped
        # latent is always a latent, reconstructed per head.
        self.recon_k = self.gs > 1 or self.r_k < self.head_dim
        self.recon_v = self.gs > 1 or self.r_v < self.head_dim

        rt = self.palu.runtime
        # stage 2: fused K reconstruction. Only where K really is low rank; on the V-only and identity
        # rows the stock K path stays, flag or no flag. Grouped rows join in Task 4 (in-kernel V).
        self.use_fused_k = self.recon_k and bool(rt.use_fused_k) and self.gs == 1
        self.phase2 = self.use_fused_k and bool(rt.cs_recurrence)
```

Up-projections: `pad_u` pads to the width of the latent as the head split delivers it (`gs * head_dim`), and the naive decode needs `u_v_decode` on grouped rows:

```python
        def pad_u(u, r):
            t = torch.zeros(self.n_kv_heads, self.gs * self.head_dim, self.head_dim)
            t[:, :r, :] = u
            return t.unsqueeze(0)
        if self.recon_k:
            u_k = state_dict[f"{P}.u_k"]  # [n_kv, r_k, hd]
            self.u_k_prefill = ttnn.as_tensor(pad_u(u_k, self.r_k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_k_prefill"))
            if self.use_fused_k:
                self.w_rot = ttnn.as_tensor(
                    state_dict[f"{P}.w_rot"].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_w_rot"))
            else:
                self.u_k_decode = ttnn.as_tensor(
                    u_k.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads,
                    cache_file_name=cache_name(f"{tag}_u_k_decode_b{B}"),
                )
        if self.recon_v:
            u_v = state_dict[f"{P}.u_v"]  # [n_kv, r_v, hd]
            self.u_v_prefill = ttnn.as_tensor(pad_u(u_v, self.r_v), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_v_prefill"))
            if self.gs > 1:
                # grouped rows do not absorb V into wo (the group latent is up to gs*hd wide); the naive
                # decode reconstructs V per head like K
                self.u_v_decode = ttnn.as_tensor(
                    u_v.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads,
                    cache_file_name=cache_name(f"{tag}_u_v_decode_b{B}"),
                )
```

`wo_fused`: `self.wo_fused = wo_tensor(...) if (self.recon_v and self.gs == 1) else self.wo`. The `cache_tag` call must carry the group size so a grouped dump never shares entries with a per-head one: change `cache_tag` to `f"palu_v{CONVERT_VERSION}_{cache_id}_rk{r_k}_rv{r_v}"` + (`f"_gs{gs}"` when `gs > 1`), passing `self.gs` (update `tests/test_attention_tag.py` for the new keyword; group size 1 tags are unchanged).

Latent caches (no weight-cache file, one per group):

```python
        self.layer_past = []
        for r in (self.r_k, self.r_v):
            z = torch.zeros(B, self.n_local_groups, self.max_seq_len, r)
            # Never through the weight cache: the content is zeros and every batch/context shape would
            # leave a file of up to 273 MB behind (212 GB found on disk during stage 2b).
            self.layer_past.append(ttnn.from_torch(z, dtype=self.latent_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)))
```

`kv_bytes_per_token` returns `self.n_local_groups * (self.r_k + self.r_v) * bpe`; `_kv_prefill_shard_cfg` uses `self.n_local_groups * seq_len // 64` rows.

- [ ] **Step 3: `attention.py` — group latent extraction**

Add next to `_slice_latent`:

```python
    def _group_latent_prefill(self, heads_1KSD, r):
        """[1, n_local_kv, S, 128] (head-split latent slots) -> the group latent [1, n_local_groups, S, r].

        The group's VT occupies its heads' consecutive slots, so the latent is the concatenation of the
        group's head slices along the last dim; with one group per chip that is nlp_concat_heads."""
        if self.gs == 1:
            return heads_1KSD[:, :, :, :r] if r < self.head_dim else heads_1KSD
        assert self.n_local_groups == 1, "one group per chip is what nlp_concat_heads gives; generalise with a reshape if needed"
        cat = ttnn.experimental.nlp_concat_heads(heads_1KSD, memory_config=ttnn.DRAM_MEMORY_CONFIG)   # [1, 1, S, n_local_kv*128]
        lat = cat[:, :, :, :r] if r < cat.shape[-1] else cat
        if lat is not cat:
            ttnn.deallocate(cat)
        return lat

    def _group_latent(self, heads_1BKD, r):
        """Decode: [1, B, n_local_kv, 128] height-sharded -> [1, B, n_local_groups, r] height-sharded
        (shard [32, r] per batch core), the layout paged_update_cache wants."""
        if self.gs == 1:
            return self._slice_latent(heads_1BKD, r)
        B = self.batch_size_per_device_group
        t = ttnn.to_memory_config(heads_1BKD, self.args.get_attn_sdpa_output_mem_config(Mode.DECODE, B, None))
        cat = ttnn.experimental.nlp_concat_heads_decode(t, num_heads=self.n_local_kv_heads)              # [1, 1, B, n_local_kv*128]
        ttnn.deallocate(t)
        ttnn.deallocate(heads_1BKD)
        cat = ttnn.sharded_to_interleaved(cat, ttnn.L1_MEMORY_CONFIG)
        lat = cat[:, :, :, :r] if r < cat.shape[-1] else cat
        lat = ttnn.reshape(lat, (1, B, self.n_local_groups, r))
        if lat is not cat:
            ttnn.deallocate(cat)
        return ttnn.interleaved_to_sharded(lat, self._latent_shard_cfg(r))

    def _expand_latent_over_heads(self, lat):
        """[B, n_local_groups, S, r] -> [B, n_local_kv, S, r]: each head of a group reads the group's latent.
        ttnn.matmul cannot broadcast a batched in1, so the latent is materialised per head."""
        return lat if self.gs == 1 else ttnn.repeat(lat, ttnn.Shape([1, self.gs, 1, 1]))
```

(`ttnn.repeat` takes the repeat factors per dim; if this ttnn version wants a plain list, pass `[1, self.gs, 1, 1]`.)

- [ ] **Step 4: `attention.py` — prefill and naive decode on the group latent**

Prefill: replace the reconstruction and cache-write block (from `# reconstruct V (and K, ...` through the two `deallocate` guards) with:

```python
        # group latents (the head-split slots, or their concatenation for a grouped row) -----------------
        k_lat = self._group_latent_prefill(k_lat_pad, self.r_k)     # [1, n_local_groups, S, r_k] or the head slots
        v_lat = self._group_latent_prefill(v_lat_pad, self.r_v)
        # reconstruct per head: [1, n_local_kv, S, width] @ [1, n_local_kv, width, 128]; width is the slot
        # width (128 per head, gs*128 per group), matching u_*_prefill's zero padding
        def recon(lat_full_width, u):
            src = lat_full_width if self.gs == 1 else ttnn.repeat(lat_full_width, ttnn.Shape([1, self.gs, 1, 1]))
            out = ttnn.matmul(src, u, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.compute_kernel_config_hifi2)
            if src is not lat_full_width:
                ttnn.deallocate(src)
            return out
        k_slot = k_lat_pad if self.gs == 1 else ttnn.experimental.nlp_concat_heads(k_lat_pad, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_slot = v_lat_pad if self.gs == 1 else ttnn.experimental.nlp_concat_heads(v_lat_pad, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_pre = recon(k_slot, self.u_k_prefill) if self.recon_k else k_lat_pad
        v_full = recon(v_slot, self.u_v_prefill) if self.recon_v else v_lat_pad

        # RoPE on q and K (stock prefill op) ------------------------------------------------------------
        q = ttnn.experimental.rotary_embedding_llama(q_pre, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False)
        k_rot = ttnn.experimental.rotary_embedding_llama(k_pre, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False)
        ttnn.deallocate(q_pre)
        ttnn.deallocate(k_pre)

        # caches: latents, except the per-head fast path's K cache which holds rotated full K like stock --
        keys, values = self.layer_past
        for src, r, cache in (((k_lat if self.recon_k else k_rot), self.r_k, keys), (v_lat, self.r_v, values)):
            lat = ttnn.typecast(src, dtype=cache.dtype)
            if seq_len >= self.min_kv_prefill_shard_seqlen:
                lat = ttnn.interleaved_to_sharded(lat, self._kv_prefill_shard_cfg(seq_len, r))
            ttnn.fill_cache(cache, lat, user_id % self.batch_size_per_device_group)
            ttnn.deallocate(lat)
        for t in {id(k_lat): k_lat, id(v_lat): v_lat, id(k_slot): k_slot, id(v_slot): v_slot}.values():
            if t is not k_lat_pad and t is not v_lat_pad and t is not v_full and t is not k_rot:
                ttnn.deallocate(t)
        if self.recon_k:
            ttnn.deallocate(k_lat_pad)
        if self.recon_v:
            ttnn.deallocate(v_lat_pad)      # otherwise v_lat_pad *is* v_full, freed after the SDPA
```

(At group size 1 `k_lat`/`k_slot` alias `k_lat_pad` and the slice, exactly as before; the `typecast` of an already-cast latent is the existing behaviour.)

Decode: replace `v_lat = self._slice_latent(v_lat_pad, self.r_v)` and `k_new = self._slice_latent(k_lat_pad, self.r_k)` with `self._group_latent(...)` calls. In the naive branch, generalise the K reconstruction and add V for grouped rows:

```python
            if self.recon_k:
                S = self._recon_len(current_pos)
                k_lat_S = keys[:, :, :S, :] if S < self.max_seq_len else keys                 # [B, n_local_groups, S, r_k]
                v_att = values[:, :, :S, :] if S < self.max_seq_len else values               # [B, n_local_groups, S, r_v]
                k_src = self._expand_latent_over_heads(k_lat_S)                              # [B, n_local_kv, S, r_k]
                k_rec = ttnn.matmul(k_src, self.u_k_decode, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    compute_kernel_config=self.compute_kernel_config_hifi2)   # [B, n_local_kv, S, 128]
                if k_src is not k_lat_S:
                    ttnn.deallocate(k_src)
                if S < self.max_seq_len:
                    ttnn.deallocate(k_lat_S)
                ... (RoPE block unchanged) ...
                k_att = ttnn.typecast(k_rot, dtype=self.kv_cache_dtype)
                ttnn.deallocate(k_rot)
                if self.gs > 1:
                    # grouped V is reconstructed too (no absorption): stock decode SDPA on full K and V
                    v_src = self._expand_latent_over_heads(v_att)
                    v_rec = ttnn.matmul(v_src, self.u_v_decode, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                        compute_kernel_config=self.compute_kernel_config_hifi2)   # [B, n_local_kv, S, 128]
                    ttnn.deallocate(v_src)
                    if S < self.max_seq_len:
                        ttnn.deallocate(v_att)
                    v_att = ttnn.typecast(v_rec, dtype=self.kv_cache_dtype)
                    ttnn.deallocate(v_rec)
            else:
                S, k_att, v_att = self.max_seq_len, keys, values

            if self.gs > 1:
                attn = ttnn.transformer.scaled_dot_product_attention_decode(
                    q, k_att, v_att, cur_pos_tensor=current_pos, scale=self.scale,
                    program_config=self.args.get_attn_sdpa_decode_program_config(None),
                    compute_kernel_config=self.sdpa_decode_compute_kernel_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, B, nh, 128]
            else:
                attn = ttnn.transformer.flash_multi_latent_attention_decode(
                    q, k_att, v_att, head_dim_v=self.r_v, cur_pos_tensor=current_pos, scale=self.scale,
                    program_config=self.args.get_attn_sdpa_decode_program_config(None),
                    compute_kernel_config=self.sdpa_decode_compute_kernel_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, B, nh, r_v]
            ttnn.deallocate(q)
            if self.recon_k:
                ttnn.deallocate(k_att)
                if self.gs > 1 or S < self.max_seq_len:
                    ttnn.deallocate(v_att)
```

The padding of `attn` to `head_dim` after the op applies only when `attn.shape[-1] < head_dim` (already true for grouped rows: the width is 128); the stock tail then uses `self.wo_fused`, which is `self.wo` for grouped rows.

- [ ] **Step 5: Run**

```bash
scripts/run_device.sh pytest tests/test_attention_device.py -q 2>&1 | tail -3
nice -n 10 pytest tests/test_attention_tag.py tests/test_convert.py -q 2>&1 | tail -1
```
Expected: all previous device tests pass unchanged (16) plus the G-LRD test; CPU pass. If the prefill PCC is low, check the concat order first (`nlp_concat_heads` output columns are head-major: head 0's 128 slots first, which is the converter's layout).

- [ ] **Step 6: Commit**

```bash
git add palu_tt/attention.py tests/test_attention_device.py tests/test_attention_tag.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2b Task 2: G-LRD host path -- group latent caches, prefill and naive decode from the shared latent; latent caches off the weight cache

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 3: Kernel — grouped latents, wide `r_k`, in-kernel V reconstruction

Outcome: `palu_sdpa_decode` accepts a latent cache with fewer heads than the rotation weight (`heads_per_group`), any `r_k` multiple of 32 up to 512, and an optional `u_v` that makes the kernel rebuild V per head on core; without `u_v` and with `heads_per_group = 1` it is bit-identical to stage 2.

**Files:**
- Modify (submodule `ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/`): `palu_sdpa_decode.{hpp,cpp}`, `palu_sdpa_decode_nanobind.cpp`, `device/palu_sdpa_decode_device_operation_types.hpp`, `device/palu_sdpa_decode_device_operation.{hpp,cpp}`, `device/palu_sdpa_decode_program_factory.cpp`, `device/kernels/dataflow/palu_reader_decode.cpp`, `device/kernels/dataflow/palu_dataflow_common.hpp`, `device/kernels/compute/palu_sdpa_flash_decode.cpp`, `device/kernels/compute/palu_compute_common.hpp`.
- Modify: `palu_tt/fused.py` (`reference_decode(..., u_v=None)`), `tests/test_fused.py`, `tests/test_palu_sdpa_device.py`.

**Interfaces:**
- Produces: Python `ttnn.transformer.palu_sdpa_decode(q, k_latent, w_rot, cs_tab, v_latent, head_dim_v, cur_pos_tensor, *, rot_step=None, trans_mat=None, u_v=None, scale=None, memory_config=None, program_config=None, compute_kernel_config=None)`. With `u_v` (`[1, n_kv, r_v_g, 128]` bf16): `v_latent` is `[B, n_groups, S, r_v_g]`, `head_dim_v` must be 128, output `[1, B, nh, 128]`. `k_latent`/`v_latent` head dim `n_groups` must divide `w_rot.shape[1] = n_kv`; `heads_per_group = n_kv / n_groups` is an op attribute.
- Produces (C++): `PaluSdpaDecodeParams{..., cs_recurrence, heads_per_group, recon_v}`, `PaluSdpaDecodeInputs{..., u_v}`; prim `palu_sdpa_decode(q, k_latent, w_rot, cs_tab, v_latent, cur_pos_tensor, rot_step, trans_mat, u_v, head_dim_v, scale, output_mem_config, program_config, compute_kernel_config, k_chunk_size)`.
- Produces (kernels): reader compile-time args 39 = `heads_per_group`, 40 = `recon_v`, 41 = `Rvt` (V latent width in tiles; equals `vDHt` when `!recon_v`), accessors from 42 with `u_v` appended after `trans_mat`; reader runtime arg `u_v_addr` after `trans_mat_addr` (idle count 25); compute compile-time args 33 = `recon_v`, 34 = `Rvt`; define `PALU_RECON_V`; per-head weight block `[w_rot (Rt·Wt) | u_v (Rvt·DHt, if recon_v) | T (1, if phase2)]`; helper `palu_build_v_chunk<cb_vlat, cb_w, cb_v_out, Rvt, vDHt, dst_tiles>(Sk_chunk_t, uv_tile_base)`.
- Produces: `fused.reference_decode(q, k_lat, u_k, v_lat, cur_pos, cos, sin, scale, u_v=None)`: with `u_v` (`[nkv, r_v, hd]`), `V = einsum("bhsr,hrd->bhsd", v_lat, u_v)`.

- [ ] **Step 1: Failing tests**

`tests/test_fused.py`, append:

```python
def test_reference_decode_reconstructs_v_when_u_v_is_given():
    B, nkv, nh, S, r_k, r_v = 1, 2, 4, 16, 32, 48
    q = torch.randn(1, B, nh, HD); k = torch.randn(B, nkv, S, r_k); u_k = torch.randn(nkv, r_k, HD)
    v_lat = torch.randn(B, nkv, S, r_v); u_v = torch.randn(nkv, r_v, HD)
    cos, sin = fused.cos_sin_tables(HD, S, 500000.0, None)
    out = fused.reference_decode(q, k, u_k, v_lat, torch.tensor([S - 1]), cos, sin, 1.0, u_v=u_v)
    v_full = torch.einsum("bhsr,hrd->bhsd", v_lat, u_v)
    exp = fused.reference_decode(q, k, u_k, v_full, torch.tensor([S - 1]), cos, sin, 1.0)
    assert out.shape == (1, B, nh, HD)
    torch.testing.assert_close(out, exp, atol=1e-5, rtol=1e-4)
```

`tests/test_palu_sdpa_device.py`, append (uses the file's `_dev`, `_pos`, `_host`, `_bf`, `_cfg`, `HD, NH, NKV, THETA`):

```python
def _inputs_group(batch, S, r_k, r_v, gs):
    """Grouped latents: n_groups = NKV // gs share one K latent and one V latent; per-head factors of width r_g."""
    ng = NKV // gs
    q = _bf(torch.randn(1, batch, NH, HD))
    k_lat = _bf(torch.randn(batch, ng, S, r_k))
    u_k = _bf(torch.randn(NKV, r_k, HD) / r_k ** 0.5)
    v_lat = _bf(torch.randn(batch, ng, S, r_v))
    u_v = _bf(torch.randn(NKV, r_v, HD) / r_v ** 0.5)
    pos = torch.randint(0, S, (batch,))
    pos[0] = S - 1
    if batch >= 4:
        pos[1], pos[2], pos[3] = 0, 127, 128
    cos, sin = fused.cos_sin_tables(HD, S, THETA, None)
    return q, k_lat, u_k, v_lat, u_v, pos, _bf(cos), _bf(sin)


def _expand(lat, gs):
    return lat.repeat_interleave(gs, dim=1)


def _stock_on_reconstructed_kv(md, q, k_full_rot, v_full, pos, k_chunk):
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        _dev(md, q), _dev(md, k_full_rot), _dev(md, v_full), cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5, **_cfg(k_chunk))
    return _host(out)


def _fused_group(md, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, k_chunk, phase2):
    S = k_lat.shape[2]
    extra = dict(u_v=_dev(md, u_v[None]))
    if phase2:
        c = k_chunk or fused.chunk_size(S)
        extra.update(rot_step=_dev(md, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
                     trans_mat=_dev(md, fused.trans_mat_tile()))
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(md, q), _dev(md, k_lat), _dev(md, fused.w_rot_from_u_k(u_k)[None]), _dev(md, fused.cs_table(cos, sin, S)[None, None]),
        _dev(md, v_lat), head_dim_v=HD, cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5, **extra, **_cfg(k_chunk))
    return _host(out)


def _check_group(md, batch, S, r_k, r_v, gs, k_chunk, phase2, pcc_ref=0.99, pcc_stock=0.995):
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(batch, S, r_k, r_v, gs)
    k_h, v_h = _expand(k_lat, gs), _expand(v_lat, gs)
    ref = fused.reference_decode(q, k_h, u_k, v_h, pos, cos, sin, HD ** -0.5, u_v=u_v)
    out = _fused_group(md, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, k_chunk, phase2)
    assert out.shape == (1, batch, NH, HD)
    ok, msg = comp_pcc(ref, out, pcc_ref)
    assert ok, f"vs fp32 reference: {msg}"
    K = torch.einsum("bhsr,hrd->bhsd", k_h, u_k)
    K = _bf(K * cos[None, None, :S] + (K @ fused.rope_transformation_matrix(HD)) * sin[None, None, :S])
    V = _bf(torch.einsum("bhsr,hrd->bhsd", v_h, u_v))
    stock = _stock_on_reconstructed_kv(md, q, K, V, pos, k_chunk)
    ok, msg = comp_pcc(stock, out, pcc_stock)
    assert ok, f"vs stock decode on host-rebuilt K and V: {msg}"


@pytest.mark.parametrize("r_k,r_v", [(128, 256), (256, 512), (384, 512)])
@pytest.mark.parametrize("batch", [1, 32])
def test_grouped_latents_with_in_kernel_v(mesh_device, r_k, r_v, batch):
    _check_group(mesh_device, batch, 1024, r_k, r_v, gs=4, k_chunk=0, phase2=True)   # host default chunk = 128 here


def test_grouped_latents_phase1_and_phase2_agree(mesh_device):
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(32, 1024, 128, 256, 4)
    p1 = _fused_group(mesh_device, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, 128, phase2=False)
    p2 = _fused_group(mesh_device, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, 128, phase2=True)
    ok, msg = comp_pcc(p1, p2, 0.998)
    assert ok, msg


def test_grouped_latents_long_context_batch8(mesh_device):
    _check_group(mesh_device, 8, 8192, 64, 512, gs=4, k_chunk=0, phase2=True)


def test_in_kernel_v_with_per_head_latents(mesh_device):
    """u_v without grouping: the V mode alone, heads_per_group = 1."""
    _check_group(mesh_device, 32, 1024, 64, 64, gs=1, k_chunk=0, phase2=True)


def test_group_envelope_is_enforced(mesh_device):
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(1, 256, 128, 256, 4)
    w = _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None])
    cs = _dev(mesh_device, fused.cs_table(cos, sin, 256)[None, None])
    good = dict(head_dim_v=HD, cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5, **_cfg(128))
    with pytest.raises(RuntimeError, match="heads_per_group"):      # 3 latent heads do not divide 4 KV heads
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, torch.randn(1, 3, 256, 128)), w, cs,
                                          _dev(mesh_device, v_lat), u_v=_dev(mesh_device, u_v[None]), **good)
    with pytest.raises(RuntimeError, match="u_v"):                  # grouped V latent without u_v
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs, _dev(mesh_device, v_lat), **good)
    with pytest.raises(RuntimeError, match="head_dim_v"):           # u_v given but head_dim_v is not 128
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs, _dev(mesh_device, v_lat),
                                          u_v=_dev(mesh_device, u_v[None]), **{**good, "head_dim_v": 256})
```

```bash
nice -n 10 pytest tests/test_fused.py -q -k u_v 2>&1 | tail -2
scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q -k "in_kernel_v_with_per_head" 2>&1 | tail -3
```
Expected: CPU `TypeError: unexpected keyword 'u_v'`; device `TypeError` on the op call.

- [ ] **Step 2: `fused.reference_decode`**

```python
def reference_decode(q, k_lat, u_k, v_lat, cur_pos, cos, sin, scale: float, u_v=None) -> torch.Tensor:
    """... (existing docstring) ...
    u_v [nkv, r_v, hd] (optional): V is reconstructed per head as v_lat @ u_v (the kernel's in-kernel V
    mode); without it v_lat is attended over directly (the absorbed-V path)."""
    ...
    V = v_lat.float()
    if u_v is not None:
        V = torch.einsum("bhsr,hrd->bhsd", V, u_v.float())
    V = V.repeat_interleave(rep, dim=1)
    ...
```

- [ ] **Step 3: Host side of the op**

`device/palu_sdpa_decode_device_operation_types.hpp`: after `cs_recurrence` add

```cpp
    // Palu G-LRD: KV heads per shared latent (k_latent/v_latent have n_kv / heads_per_group heads).
    uint32_t heads_per_group = 1;
    // Palu: V rebuilt on core per head from the V latent and u_v (head_dim_v is then 128).
    bool recon_v = false;
```

and to the inputs, after `trans_mat`: `std::optional<Tensor> u_v;   // [1, n_kv, r_v_g, 128] bf16 (in-kernel V)`.

Prim (`.hpp` declaration and `.cpp` definition): add `const std::optional<const Tensor>& u_v,` right after `trans_mat`; in the body

```cpp
    const uint32_t n_kv = w_rot.padded_shape()[1];
    const uint32_t n_groups = k_latent.padded_shape()[1];
    TT_FATAL(n_groups > 0 && n_kv % n_groups == 0,
             "palu_sdpa_decode: heads_per_group must be an integer: w_rot has {} heads, k_latent {}", n_kv, n_groups);
    ...
        .cs_recurrence = rot_step.has_value(),
        .heads_per_group = n_kv / n_groups,
        .recon_v = u_v.has_value(),
    ...
        .trans_mat = trans_mat,
        .u_v = u_v,
```

Public op (`palu_sdpa_decode.hpp/.cpp`): add `const std::optional<const Tensor>& u_v = std::nullopt,` after `trans_mat`; replace the `v_latent width == head_dim_v` fatal with

```cpp
    if (u_v.has_value()) {
        TT_FATAL(head_dim_v == 128, "palu_sdpa_decode: with u_v the output width head_dim_v must be 128, got {}", head_dim_v);
        TT_FATAL(u_v->padded_shape()[2] == v_latent.padded_shape()[-1],
                 "palu_sdpa_decode: u_v rows ({}) must equal the V latent width ({})", u_v->padded_shape()[2], v_latent.padded_shape()[-1]);
    } else {
        TT_FATAL(v_latent.padded_shape()[-1] == head_dim_v,
                 "palu_sdpa_decode: without u_v the V latent width ({}) must equal head_dim_v ({})", v_latent.padded_shape()[-1], head_dim_v);
    }
```

and, before the prim call, the host L1 rule for the default chunk:

```cpp
    // L1 rule (spec §4): a wide latent (Rt > 4) or the in-kernel V mode does not fit the default 512-position
    // chunk next to the stock CBs; take 128 unless the caller pinned a chunk.
    const bool caller_pinned = program_config.has_value() && program_config.value().k_chunk_size > 0;
    if (!caller_pinned && (k_latent.padded_shape()[-1] > 128 || u_v.has_value())) {
        k_chunk_size = std::min<uint32_t>(k_chunk_size, 128);
    }
```

Binding: `nb::arg("u_v") = nb::none(),` after `trans_mat`, docstring updated.

Validation (`device/palu_sdpa_decode_device_operation.cpp`):
- The `use_mla` block's V check (`v_shape[3] == head_dim_v`) becomes: if `recon_v`, `v_shape[3] == u_v.padded_shape()[2]`, else the stock check.
- The `r_k` check becomes `r_k_logical % 32 == 0 && 32 <= r_k_logical && r_k_logical <= 512 && k_shape[-1] == r_k_logical`, message "palu_sdpa_decode: r_k (latent width) must be a multiple of 32 in 32..512, got {}".
- The stock "K and V must have same number of heads" check stays (both are `n_groups`); the GQA check `q heads % k_shape[1]` stays true.
- Envelope block additions:

```cpp
    const uint32_t n_kv = tensor_args.w_rot.padded_shape()[1];
    const uint32_t hpg = operation_attributes.heads_per_group;
    TT_FATAL(hpg >= 1 && n_kv == hpg * k_shape[1], "palu_sdpa_decode: heads_per_group {} x latent heads {} != w_rot heads {}", hpg, k_shape[1], n_kv);
    expect(tensor_args.w_rot, "w_rot", {1, n_kv, r_k, 2 * D_palu}, DataType::BFLOAT16);
    if (operation_attributes.recon_v) {
        expect(*tensor_args.u_v, "u_v", {1, n_kv, v_shape[3], D_palu}, DataType::BFLOAT16);
        TT_FATAL(operation_attributes.head_dim_v.value() == D_palu, "palu_sdpa_decode: with u_v head_dim_v must be {} (got {})", D_palu, operation_attributes.head_dim_v.value());
        TT_FATAL(v_shape[3] % 32 == 0 && v_shape[3] <= 512, "palu_sdpa_decode: V latent width must be a multiple of 32 up to 512, got {}", v_shape[3]);
    } else {
        TT_FATAL(hpg == 1, "palu_sdpa_decode: grouped latents (heads_per_group {}) need u_v: the absorbed-V path is per head only", hpg);
    }
```

(The existing `w_rot` `expect` is replaced by the one above, which uses `n_kv` from `w_rot` rather than from `k_shape[1]`.)

- [ ] **Step 4: Program factory**

In `create_descriptor`:
- After the Palu inputs block: `const bool recon_v = operation_attributes.recon_v; const uint32_t heads_per_group = operation_attributes.heads_per_group; Buffer* u_v_buffer = recon_v ? tensor_args.u_v->buffer() : nullptr;`
- `num_kv_heads` (line ~112): `uint32_t num_kv_heads = w_rot.padded_shape()[1];  // Palu: the work layout follows the per-head weight; the caches may hold n_kv / heads_per_group group latents`.
- After `Wt`: `const uint32_t r_v_lat = v_shape[3]; const uint32_t Rvt = r_v_lat / TILE_WIDTH;  // V latent width in tiles (== vDHt when !recon_v)`.
- Tile sizes: the V CB holds the latent: `v_tiles = Sk_chunk_t_cb_size * Rvt * 2` (replace the stock `vDHt` there only); `k_tiles` stays (K_rot and the rebuilt V are both `Sk × DHt`/`Sk × vDHt` = 4 wide; assert `!recon_v || vDHt == DHt`).
- Weight block: `w_rot_block = Rt * Wt + (recon_v ? Rvt * DHt : 0) + (phase2 ? 1 : 0)`; `w_rot_tiles = (Rt > 4 || recon_v) ? w_rot_block : 2 * w_rot_block;` (single block above r_k 128 or with u_v; keep the whole-block `TT_FATAL`).
- Reader compile-time args: append `heads_per_group, static_cast<uint32_t>(recon_v), Rvt,` after `phase2`; accessor `tt_metal::TensorAccessorArgs(u_v_buffer).append_to(...)` after `trans_mat_buffer`'s.
- Compute compile-time args: append `static_cast<uint32_t>(recon_v), Rvt,` after `Rt`; define `if (recon_v) compute_defines["PALU_RECON_V"] = "1";`.
- Reader runtime args: `reader_rt_args.push_back(u_v_buffer);` after `trans_mat_buffer`; idle count `24 → 25`.
- Comment the CB `TT_FATAL` list unchanged; the `c_2` (V) CB uses `v_tiles` above with the V latent's dtype.

- [ ] **Step 5: Reader**

`palu_reader_decode.cpp`:
- compile args after `phase2`: `constexpr uint32_t heads_per_group = get_compile_time_arg_val(39); constexpr bool recon_v = get_compile_time_arg_val(40) == 1; constexpr uint32_t Rvt = get_compile_time_arg_val(41);` and `q_args = TensorAccessorArgs<42>()`; `constexpr auto u_v_args = TensorAccessorArgs<trans_mat_args.next_compile_time_args_offset()>();` (move the `static_assert` on the chain to `u_v_args`).
- runtime arg after `trans_mat_addr`: `const uint32_t u_v_addr = get_arg_val<uint32_t>(arg_idx++);`; reader `const auto u_v_reader = TensorAccessor(u_v_args, u_v_addr);`.
- Per-head block: the latent offsets use the group, the V read uses the latent width, and the weight block includes `u_v`:

```cpp
            const uint32_t lat_head = cur_head / heads_per_group;                 // the group's latent
            const uint32_t n_lat_heads = num_kv_heads / heads_per_group;
            const uint32_t lat_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * n_lat_heads * St * Rt;
            const uint32_t lat_head_offset = lat_head * St * Rt;
            const uint32_t lat_start_tile_id = lat_batch_offset + lat_head_offset + k_chunk_start * Sk_chunk_t_dynamic * Rt;
            const uint32_t v_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * n_lat_heads * St * Rvt;
            const uint32_t v_head_offset = lat_head * St * Rvt;
            const uint32_t v_start_tile_id = v_batch_offset + v_head_offset + k_chunk_start * Sk_chunk_t_dynamic * Rvt;
            read_w_rot<cb_w_rot, Rt, Wt, Rvt, DHt, barrier_threshold, phase2, recon_v>(cur_head, w_rot_reader, u_v_reader, trans_mat_reader);
            read_palu_chunks<Rt, Rvt, Wt, barrier_threshold, cb_lat_in, cb_v_in, cb_cs_in, phase2>(
                k_chunk_start, k_chunk_end, lat_start_tile_id, v_start_tile_id, Sk_chunk_t_dynamic,
                k_reader, v_reader, cs_reader);
```

(`read_palu_chunks`'s second template argument is the V width in tiles; passing `Rvt` covers both modes.) `palu_dataflow_common.hpp`'s `read_w_rot` becomes:

```cpp
// Per-head resident weight block: [w_rot (Rt x Wt) | u_v (Rvt x DHt, in-kernel V) | trans_mat (1, Phase 2)],
// ONE push per head (the block is the CB's granule; a push writes linearly and cannot wrap mid-block).
template <uint32_t cb_w, uint32_t Rt, uint32_t Wt, uint32_t Rvt, uint32_t DHt, uint32_t barrier_threshold, bool phase2, bool recon_v,
          typename ReaderT, typename UvReaderT, typename TReaderT>
void read_w_rot(uint32_t head, const ReaderT& reader, const UvReaderT& u_v_reader, const TReaderT& trans_mat_reader) {
    constexpr uint32_t block = Rt * Wt + (recon_v ? Rvt * DHt : 0) + (phase2 ? 1 : 0);
    CircularBuffer cb(cb_w);
    cb.reserve_back(block);
    uint32_t wp = palu_read_tiles_at<cb_w, barrier_threshold>(cb.get_write_ptr(), head * Rt * Wt, Rt * Wt, reader);
    if constexpr (recon_v) {
        wp = palu_read_tiles_at<cb_w, barrier_threshold>(wp, head * Rvt * DHt, Rvt * DHt, u_v_reader);   // u_v [1, n_kv, r_v, 128]: head-major tiles
    }
    if constexpr (phase2) {
        palu_read_tiles_at<cb_w, barrier_threshold>(wp, 0, 1, trans_mat_reader);
    }
    cb.push_back(block);
}
```

- [ ] **Step 6: Compute kernel**

`palu_compute_common.hpp`, append:

```cpp
// In-kernel V (Palu G-LRD): V_h for one chunk from the V latent and this head's u_v slice, row tile by
// row tile, packed into cb_v_out in ROW-MAJOR tile order (k * vDHt + n) -- the layout the PV
// matmul_blocks(..., transpose=false) reads in1 in. u_v tiles sit at uv_tile_base + kt*vDHt + n inside
// the resident weight block (cb_w, not popped); the V latent rows are popped here.
template <uint32_t cb_vlat, uint32_t cb_w, uint32_t cb_v_out, uint32_t Rvt, uint32_t vDHt, uint32_t dst_tiles>
ALWI void palu_build_v_chunk(uint32_t Sk_chunk_t, uint32_t uv_tile_base) {
    constexpr uint32_t sub_w = vDHt > dst_tiles ? dst_tiles : vDHt;
    constexpr uint32_t subblocks = vDHt / sub_w;
    CircularBuffer vlat(cb_vlat), w(cb_w), vo(cb_v_out);
    w.wait_front(uv_tile_base + Rvt * vDHt);
    vo.reserve_back(Sk_chunk_t * vDHt);
    reconfig_data_format(cb_w, cb_vlat);  // matmul: in1 -> SrcA, in0 -> SrcB
    pack_reconfig_data_format(cb_v_out);
    matmul_block_init(cb_vlat, cb_w, false, sub_w, 1, Rvt);
    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        vlat.wait_front((i + 1) * Rvt);
        for (uint32_t sub = 0; sub < subblocks; ++sub) {
            tile_regs_acquire();
            uint32_t in0_index = i * Rvt;
            uint32_t in1_index = uv_tile_base + sub * sub_w;
            for (uint32_t kt = 0; kt < Rvt; ++kt) {
                matmul_block(cb_vlat, cb_w, in0_index, in1_index, 0, false, sub_w, 1, Rvt);
                in0_index++;
                in1_index += vDHt;
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < sub_w; ++c) {
                pack_tile<true>(c, cb_v_out, i * vDHt + sub * sub_w + c);
            }
            tile_regs_release();
        }
    }
    vo.push_back(Sk_chunk_t * vDHt);
    vlat.pop_front(Sk_chunk_t * Rvt);
}
```

`palu_sdpa_flash_decode.cpp`:
- compile args after `Rt`: `constexpr bool recon_v = get_compile_time_arg_val(33) == 1; constexpr uint32_t Rvt = get_compile_time_arg_val(34);`
- constants: replace the `w_rot_cb_tiles` pair with

```cpp
    constexpr uint32_t uv_tile_base = Rt * Wt;                                   // u_v follows w_rot in the block
    constexpr uint32_t t_tile_idx = Rt * Wt + (recon_v ? Rvt * DHt : 0);        // trans_mat is last
    constexpr uint32_t w_rot_cb_tiles = t_tile_idx + (PALU_HAS_PHASE2 ? 1 : 0);
```

with `#ifdef PALU_PHASE2 #define PALU_HAS_PHASE2 1 #else #define PALU_HAS_PHASE2 0 #endif` above, and `palu_init_q_state<..., w_rot_cb_tiles>(t_tile_idx)` instead of `(Rt * Wt)`.

- PV matmul: define `constexpr uint32_t cb_v_src = recon_v ? cb_k_in : cb_v_in;` next to `cb_q_src`, and immediately before the `/* OUT_IM = QK @ V_CHUNK */` block:

```cpp
#ifdef PALU_RECON_V
                // V_h for this chunk: the QK matmul above has consumed cb_k_in, so the rebuilt V takes its slot
                palu_build_v_chunk<cb_v_in, cb_w_rot, cb_k_in, Rvt, vDHt, PALU_DST_TILES>(Sk_chunk_t_dynamic, uv_tile_base);
#endif
```

then `reconfig_data_format(cb_v_src, cb_qk_im);` and `matmul_blocks(cb_qk_im, cb_v_src, cb_out_mm, ...)` (the matmul pops `cb_v_src`, so in the V mode `cb_v_in` is popped by the helper and `cb_k_in` by the matmul).

- [ ] **Step 7: Build, test, commit**

```bash
scripts/build_ttnn.sh 2>&1 | tail -3
nice -n 10 pytest tests/test_fused.py -q 2>&1 | tail -1
scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q 2>&1 | tail -5
```
Expected: the 20 stage-2 tests unchanged plus 6 + 1 + 1 + 1 + 1 new = 30 passed. Debug order if not: (1) `test_in_kernel_v_with_per_head_latents` isolates the V mode from grouping; (2) a hang is a block-size mismatch between `read_w_rot`'s block and `w_rot_cb_tiles`; (3) a wrong V is the tile order (`i * vDHt + n`) or `uv_tile_base`.

```bash
cd third_party/tt-metal && git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode && git -c user.name=ryan -c user.email=ryankert01@gmail.com commit --no-verify -m "palu_sdpa_decode: grouped latents (heads_per_group), r_k up to 512, in-kernel V reconstruction (u_v)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs" && cd ../..
git add third_party/tt-metal palu_tt/fused.py tests/test_fused.py tests/test_palu_sdpa_device.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2b Task 3: kernel support for grouped latents, wide r_k and in-kernel V; op tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 4: Fused decode for grouped rows in `PaluAttention`

Outcome: a G-LRD row with `use_fused_k` decodes through the kernel with in-kernel V; fused equals the naive grouped path at the attention level with byte-identical caches.

**Files:**
- Modify: `palu_tt/attention.py`, `tests/test_attention_device.py`.

**Interfaces:**
- Consumes: Task 3's op (`u_v=`), Task 2's group caches and `_group_latent`.
- Produces: `PaluAttention.u_v_kernel` (`[1, n_kv, r_v, 128]` bf16, head-sharded, cache name `{tag}_u_v_kernel`) on grouped fused rows; `use_fused_k` no longer excludes `gs > 1`; the fused call passes `u_v` and `head_dim_v = head_dim` for grouped rows.

- [ ] **Step 1: Failing test** — append to `tests/test_attention_device.py`:

```python
def test_glrd_fused_decode_matches_naive(mesh_device):
    batch = 32
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, "compressed/glrd_0.5", batch=batch, max_seq_len=SEQ)
    args.palu.runtime.check = True
    naive = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    args.palu.runtime.use_fused_k, args.palu.runtime.cs_recurrence = True, True
    fusedm = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    assert fusedm.gs == 4 and fusedm.use_fused_k and fusedm.phase2 and hasattr(fusedm, "u_v_kernel") and not hasattr(fusedm, "u_v_decode")
    P = 128
    x_pre = torch.randn(1, P, args.dim) * 0.05
    rot = [rope.cos_matrix_prefill[:, :, :P, :], rope.sin_matrix_prefill[:, :, :P, :]]
    for m in (naive, fusedm):
        m.forward(args.prepare_residual_tensor_prefill(x_pre, force_replicated=True), None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    for step in range(6):
        pos = P + step
        x = torch.randn(batch, 1, args.dim) * 0.05
        current_pos = torch.full((batch,), pos)
        args.palu.runtime.recon_len = ((pos + 1 + 31) // 32) * 32
        ys = []
        for m in (naive, fusedm):
            x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x, current_pos)
            y = m.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
            ys.append(_to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim))
        ok, msg = comp_pcc(ys[0], ys[1], 0.99)
        assert ok, f"step={step} {msg}"
    for i in range(2):
        assert torch.equal(_cache_torch(mesh_device, args, naive, i), _cache_torch(mesh_device, args, fusedm, i)), f"cache {i}"
```

Run: `scripts/run_device.sh pytest tests/test_attention_device.py -q -k glrd_fused 2>&1 | tail -3` → fails on the `use_fused_k` assertion (Task 2 pins it False for grouped rows).

- [ ] **Step 2: `attention.py`**

- `self.use_fused_k = self.recon_k and bool(rt.use_fused_k)` (drop `and self.gs == 1`).
- In the `recon_v` block: for `gs > 1`, build `u_v_decode` only when `not self.use_fused_k`, and on the fused path

```python
            if self.gs > 1 and self.use_fused_k:
                # the kernel rebuilds V per head from the group latent: u_v as [1, n_kv, r_v, 128], head-sharded
                self.u_v_kernel = ttnn.as_tensor(u_v.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads,
                                                 cache_file_name=cache_name(f"{tag}_u_v_kernel"))
```

- Chunk rule in the fused-config block: `wide = self.r_k > 128 or self.gs > 1`; Phase 2's default chunk becomes `128 if wide else (model config chunk or fused.chunk_size(max_seq_len))`, matching the op's own rule so `rot_step` is built for the chunk the op uses.
- Fused call:

```python
            grouped = self.gs > 1
            attn = ttnn.transformer.palu_sdpa_decode(
                q, keys, self.w_rot, self.cs_tab, values, self.head_dim if grouped else self.r_v, current_pos,
                rot_step=self.rot_step, trans_mat=self.trans_mat, u_v=self.u_v_kernel if grouped else None, scale=self.scale,
                program_config=self.fused_program_config, compute_kernel_config=self.fused_compute_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)                                        # [1, B, nh, r_v] or [1, B, nh, 128]
```

- [ ] **Step 3: Run and commit**

```bash
scripts/run_device.sh pytest tests/test_attention_device.py -q -k "glrd or fused" 2>&1 | tail -3
```
Expected: the two G-LRD tests and the four stage-2 fused tests pass.

```bash
git add palu_tt/attention.py tests/test_attention_device.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2b Task 4: fused decode for grouped rows (in-kernel V) in PaluAttention

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 5: lm-eval task-accuracy harness on the device

Outcome: `python -m palu_tt.eval_tasks --row R --kind K [--palu-dir D] [--fused ...] [--tasks ...] [--limit N]` scores the five tasks through device prefill and writes `docs/results/raw/tasks_<row>.json`.

**Files:**
- Create: `palu_tt/lm_eval_tt.py`, `palu_tt/eval_tasks.py`, `palu_tt/lm_eval_tasks/{openbookqa_hub,piqa_hub,winogrande_hub}.yaml`, `tests/test_lm_eval_tt.py`.
- Modify: `scripts/env.sh` (documented one-time download env), `README.md`.

**Interfaces:**
- Produces: `lm_eval_tt.TTLM(args, model, hf_dir, max_length=2048)` subclassing `lm_eval.api.model.TemplateLM` with `AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM`, `eot_token_id`, `prefix_token_id` (BOS), `tok_encode`, `_loglikelihood_tokens(requests) -> List[Tuple[float, bool]]`, `loglikelihood_rolling`/`generate_until` raising `NotImplementedError`; a `score_fn` hook (`ids [1, L] -> logits [L, vocab]`) so the CPU test can substitute a torch function for the device.
- Produces: `eval_tasks.TASKS = ("openbookqa", "piqa", "arc_easy", "arc_challenge", "winogrande")`, `eval_tasks.TASK_NAME_MAP = {"openbookqa_hub": "openbookqa", "piqa_hub": "piqa", "winogrande_hub": "winogrande"}`, `eval_tasks.task_manager()`, `eval_tasks.run(lm, tasks, limit) -> dict`, JSON schema `{"row", "kind", "palu_dir", "fused", "limit", "lm_eval_version", "tasks": {name: {"acc", "acc_norm" | None, "n"}}, "results", "versions", "configs", "seconds"}`.

- [ ] **Step 1: Failing CPU test** — `tests/test_lm_eval_tt.py`:

```python
import torch
from lm_eval.api.instance import Instance

from palu_tt import lm_eval_tt, eval_tasks


class _Tok:
    """A toy tokenizer: one token per character, BOS = 1, EOS = 2, vocab 128."""
    bos_token_id, eos_token_id = 1, 2
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 128 for c in s]


def _score_fn(ids):
    """Deterministic logits: the model strongly predicts token (prev + 1) % 128 at every position."""
    L = ids.shape[1]
    logits = torch.full((L, 128), -10.0)
    for t in range(L):
        logits[t, (int(ids[0, t]) + 1) % 128] = 10.0
    return logits


def test_loglikelihood_scores_continuation_tokens_only_and_flags_greedy():
    lm = lm_eval_tt.TTLM.from_parts(_Tok(), _score_fn, max_length=64)
    # context "ab" -> the greedy continuation is "c" (b+1); "x" is not greedy
    reqs = [Instance(request_type="loglikelihood", doc={}, arguments=("ab", "c"), idx=0),
            Instance(request_type="loglikelihood", doc={}, arguments=("ab", "x"), idx=1)]
    (lp_c, greedy_c), (lp_x, greedy_x) = lm.loglikelihood(reqs)
    assert greedy_c and not greedy_x and lp_c > lp_x
    # a continuation of two tokens sums two log-probs
    (lp_cd, _), = lm.loglikelihood([Instance(request_type="loglikelihood", doc={}, arguments=("ab", "cd"), idx=2)])
    assert abs(lp_cd - 2 * lp_c) < 1e-4
    # empty context is scored after the BOS prefix token
    (lp_e, _), = lm.loglikelihood([Instance(request_type="loglikelihood", doc={}, arguments=("", "\x02"), idx=3)])
    assert lp_e > -1.0


def test_task_name_map_and_task_list():
    assert set(eval_tasks.TASK_NAME_MAP.values()) <= set(eval_tasks.TASKS)
    assert eval_tasks.TASKS == ("openbookqa", "piqa", "arc_easy", "arc_challenge", "winogrande")
```

Run: `nice -n 10 pytest tests/test_lm_eval_tt.py -q` → `ModuleNotFoundError`.

- [ ] **Step 2: `palu_tt/lm_eval_tt.py`**

```python
"""lm-evaluation-harness (0.4.2) model adapter for the device: log-likelihoods through prefill.

A multiple-choice task asks for log P(continuation | context) for each candidate. We tokenise with the
model tokenizer (lm-eval's TemplateLM splits context/continuation for us), run the whole sequence
through device prefill (`generate.prefill_hidden_states`), apply the final RMSNorm and lm_head on the
host (the perplexity harness's path), and sum the continuation tokens' log-probabilities. One
sequence per prefill; identical sequences within a run are memoised. Only `loglikelihood` is
implemented -- the five tasks of the study need neither generation nor rolling likelihood.
"""
from typing import Callable, List, Tuple

import torch
import transformers
from lm_eval.api.model import TemplateLM

from palu_tt import eval_ppl, generate


class TTLM(TemplateLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM   # selects TemplateLM's causal context/continuation split

    def __init__(self, tokenizer, score_fn: Callable[[torch.Tensor], torch.Tensor], max_length: int = 2048):
        super().__init__()
        self.tokenizer = tokenizer
        self._score = score_fn            # ids [1, L] -> logits [L, vocab] (fp32, host)
        self._max_length = max_length
        self._memo = {}

    @classmethod
    def from_parts(cls, tokenizer, score_fn, max_length=2048):
        return cls(tokenizer, score_fn, max_length)

    @classmethod
    def for_device(cls, args, model, hf_dir, max_length=2048):
        norm_w, lm_w, eps = eval_ppl._head(hf_dir)
        def score(ids):
            L = ids.shape[1]
            padded = torch.nn.functional.pad(ids, (0, (-L) % 128), value=0)   # prefill wants a multiple of 128
            h = generate.prefill_hidden_states(args, model, padded)[:L]         # [L, dim]
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
            return h @ lm_w.T
        return cls(args.tokenizer, score, max_length)

    # ---- TemplateLM contract ----
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        return self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        out = []
        for _, context_enc, continuation_enc in requests:
            whole = (context_enc + continuation_enc)[-(self._max_length):]
            n_cont = len(continuation_enc)
            key = tuple(whole)
            if key not in self._memo:
                ids = torch.tensor([[self.prefix_token_id] + list(whole)])     # BOS + sequence
                logits = self._score(ids).float()                               # [1 + len(whole), vocab]
                self._memo[key] = torch.log_softmax(logits, dim=-1)
            logp = self._memo[key]
            # position t predicts token t+1: the continuation tokens are the last n_cont of `whole`
            tgt = torch.tensor(whole[-n_cont:])
            rows = logp[len(whole) - n_cont: len(whole)]                        # predictions for those tokens
            ll = float(rows.gather(1, tgt[:, None]).sum())
            greedy = bool((rows.argmax(-1) == tgt).all())
            out.append((ll, greedy))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("palu_tt.lm_eval_tt: rolling log-likelihood is not needed by the study's tasks")

    def generate_until(self, requests):
        raise NotImplementedError("palu_tt.lm_eval_tt: generation is not needed by the study's tasks")
```

(Index check: `ids = [BOS] + whole` has length `1 + len(whole)`; logits row `j` predicts `ids[j+1]`; continuation token `whole[len(whole)-n_cont+m]` is `ids[len(whole)-n_cont+m+1]`, predicted by row `len(whole)-n_cont+m` — the slice above. The empty-context case: TemplateLM passes `context_enc = [prefix_token_id]`, so `whole` starts with BOS and the sequence gets a second BOS in front; lm-eval's HFLM does the same.)

- [ ] **Step 3: task overrides and `eval_tasks.py`**

`palu_tt/lm_eval_tasks/openbookqa_hub.yaml`:

```yaml
include: /home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9/.venv/lib/python3.10/site-packages/lm_eval/tasks/openbookqa/openbookqa.yaml
task: openbookqa_hub
dataset_path: allenai/openbookqa
dataset_name: main
```

`piqa_hub.yaml`: include `.../lm_eval/tasks/piqa/piqa.yaml`, `task: piqa_hub`, `dataset_path: ybisk/piqa`, `dataset_name: plain_text`. `winogrande_hub.yaml`: include `.../lm_eval/tasks/winogrande/default.yaml`, `task: winogrande_hub`, `dataset_path: allenai/winogrande`, `dataset_name: winogrande_xl`. The absolute include path is derived at import time instead if lm-eval refuses it: `eval_tasks.write_task_dir()` renders the three files with the installed lm-eval path into `palu_tt/lm_eval_tasks/` (commit the rendered files; the renderer only re-renders when the path differs).

`palu_tt/eval_tasks.py`:

```python
"""Zero-shot task accuracy on the device (the paper's protocol, five of its six tasks).

Usage: python -m palu_tt.eval_tasks --row glrd_50 --kind palu --palu-dir compressed/glrd_0.5 [--fused p2] [--limit 20]
Writes docs/results/raw/tasks_<row>.json. First run needs the datasets downloaded once (README).
"""
import argparse, json, os, time
from pathlib import Path

import lm_eval
from lm_eval.tasks import TaskManager

from palu_tt import hf, lm_eval_tt

RAW = Path("docs/results/raw")
TASKS = ("openbookqa", "piqa", "arc_easy", "arc_challenge", "winogrande")
# lm-eval 0.4.2's configs point at dataset ids the Hub no longer serves; the *_hub tasks override them
TASK_NAME_MAP = {"openbookqa_hub": "openbookqa", "piqa_hub": "piqa", "winogrande_hub": "winogrande"}
TASK_DIR = Path(__file__).parent / "lm_eval_tasks"


def task_manager():
    return TaskManager(include_path=str(TASK_DIR))


def lm_eval_names(tasks):
    inv = {v: k for k, v in TASK_NAME_MAP.items()}
    return [inv.get(t, t) for t in tasks]


def run(lm, tasks, limit=None):
    t0 = time.time()
    res = lm_eval.simple_evaluate(model=lm, tasks=lm_eval_names(tasks), num_fewshot=0, limit=limit,
                                  task_manager=task_manager(), log_samples=False, bootstrap_iters=0)
    per_task = {}
    for name, r in res["results"].items():
        std = TASK_NAME_MAP.get(name, name)
        per_task[std] = {"acc": r.get("acc,none"), "acc_norm": r.get("acc_norm,none"),
                         "n": res.get("n-samples", {}).get(name, {}).get("effective")}
    return {"tasks": per_task, "results": res["results"], "versions": res.get("versions"), "configs": res.get("configs"),
            "seconds": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--fused", choices=["off", "p1", "p2"], default="off")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    from palu_tt import device as dev, model as pm
    RAW.mkdir(parents=True, exist_ok=True)
    md = dev.open_n300()
    try:
        args, model, _ = pm.build_model(md, a.kind, a.palu_dir, max_batch_size=1, max_seq_len=2048,
                                        use_fused_k=a.fused != "off", cs_recurrence=a.fused == "p2")
        lm = lm_eval_tt.TTLM.for_device(args, model, hf.snapshot_dir())
        r = run(lm, [t for t in a.tasks.split(",") if t], a.limit)
    finally:
        dev.close(md)
    import importlib.metadata as m
    r.update({"row": a.row, "kind": a.kind, "palu_dir": a.palu_dir, "fused": a.fused, "limit": a.limit,
              "lm_eval_version": m.version("lm_eval")})
    (RAW / f"tasks_{a.row}.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r["tasks"]))


if __name__ == "__main__":
    main()
```

(`n-samples` may be named differently in 0.4.2; if it is absent, count `len(task.eval_docs)` via the task dict and record that. The `bootstrap_iters=0` switch skips the stderr bootstrap, which is slow on the host; keep it.)

`scripts/env.sh`: add a comment block with the one-time download command:

```bash
# Task datasets (one-time, needs the proxy's CA): HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
#   SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
#   python -c 'from palu_tt import eval_tasks; import lm_eval; lm_eval.tasks.get_task_dict(eval_tasks.lm_eval_names(eval_tasks.TASKS), eval_tasks.task_manager())'
```

- [ ] **Step 4: Download the datasets and smoke-test on the device**

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  nice -n 10 python -c 'from palu_tt import eval_tasks; import lm_eval; d = lm_eval.tasks.get_task_dict(eval_tasks.lm_eval_names(eval_tasks.TASKS), eval_tasks.task_manager()); print({k: len(list(t.eval_docs)) for k, t in d.items()})'
nice -n 10 pytest tests/test_lm_eval_tt.py -q 2>&1 | tail -1
scripts/run_device.sh python -m palu_tt.eval_tasks --row smoke_stock --kind stock --tasks arc_easy --limit 40 2>&1 | tail -2
```
Expected: five document counts (2376/1172/1267/500/1838); CPU test passes; the smoke run prints an arc_easy accuracy between 0.6 and 1.0 in about a minute and writes `docs/results/raw/tasks_smoke_stock.json` (delete it afterwards; it is not a results row).

- [ ] **Step 5: Commit**

```bash
git add palu_tt/lm_eval_tt.py palu_tt/eval_tasks.py palu_tt/lm_eval_tasks tests/test_lm_eval_tt.py scripts/env.sh README.md
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2b Task 5: lm-eval zero-shot task accuracy through device prefill (five tasks, dataset-id overrides)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 6: Stage 2b report

Outcome: `python -m palu_tt.report --stage 2b` renders `docs/results/stage2b.md` from the JSONs.

**Files:**
- Modify: `palu_tt/report.py`, `tests/test_report.py`.

**Interfaces:**
- Produces: `report.ROWS2B = {"baseline": ..., "palu_full_50": ..., "palu_full_70": ..., "glrd_50": "Palu G-LRD (group size 4), ratio 0.5", "glrd_70": ...}`; `report.PAPER = {...}` (the spec §0 table, per task, for baseline/J-LRD/G-LRD/M-LRD); `report.TASK_COLS = ("openbookqa", "piqa", "arc_easy", "arc_challenge", "winogrande")`; `report.OUT2B = Path("docs/results/stage2b.md")`; `--stage 2b`.

- [ ] **Step 1: Failing test** — append to `tests/test_report.py` (uses its `_write(raw, name, obj)` helper):

```python
def test_stage2b_accuracy_table_and_paper_block(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    _write(raw, "tasks_baseline", {"row": "baseline", "tasks": {"openbookqa": {"acc": 0.43, "acc_norm": 0.45, "n": 500},
                                                                "piqa": {"acc": 0.78, "acc_norm": 0.79, "n": 1838}}})
    _write(raw, "tasks_glrd_50", {"row": "glrd_50", "tasks": {"openbookqa": {"acc": 0.41, "acc_norm": 0.42, "n": 500}}})
    _write(raw, "ppl_glrd_50", {"row": "glrd_50", "ppl": 10.5, "windows": 141, "palu_dir": "compressed/glrd_0.5"})
    out = tmp_path / "stage2b.md"
    from palu_tt import report
    report.main(raw=raw, out=out, rows=report.ROWS2B)
    text = out.read_text()
    assert "| baseline |" in text and "43.0" in text and "78.0" in text
    assert "| glrd_50 |" in text and "41.0" in text and " - |" in text          # missing task cells render as dashes
    assert "Paper (Llama-3-8B-Instruct, 50 %)" in text and "64.45" in text      # the reference block
    assert "10.500" in text
```

- [ ] **Step 2: `report.py`**

- `ROWS2B`, `PAPER` (transcribe spec §0 exactly, per task, with the 6-task average and a 5-task average recomputed without HellaSwag), `TASK_COLS`, `OUT2B`.
- `_row_data` gains an eighth element: `tsk = _load(raw, f"tasks_{row}")`; stage 1/2 ignore it.
- `main`: `stage2b = rows is ROWS2B`. Title "Stage 2b results: G-LRD and zero-shot task accuracy on a Wormhole n300". Sections, in order: (1) **Zero-shot accuracy**: `| row | recipe | ` + one column per task (`acc` as a percentage with one decimal; `acc_norm` in parentheses when present) + `| avg (5) |`; then the paper block as a second table with the same columns for the four paper methods, both averages, and the caveat sentence (model revision, 32 vs 2048 calibration samples, our five tasks). (2) **Quality and memory** as in stage 2 (perplexity device, KV bytes, max ctx). (3) **Decode and attention latency, batch 32** for the rows plus their `_fk2` variants when present (reuse the stage-2 loop with `vs stock`). (4) **Decode fidelity** for the fused rows. (5) **Batch-8 max context** from `bench_maxctx_<row>_b8.json` when present. Hand-written tail via the same `MARKER`.
- CLI: `--stage {1,2,2b}`; `2b` → `main(raw, OUT2B, ROWS2B)`.

- [ ] **Step 3: Run and commit**

```bash
nice -n 10 pytest tests/test_report.py -q 2>&1 | tail -1
python -m palu_tt.report --stage 1 >/dev/null && git diff --stat docs/results/stage1.md; python -m palu_tt.report --stage 2 >/dev/null && git diff --stat docs/results/stage2.md
```
Expected: tests pass; stage 1 and 2 documents unchanged apart from the commit-hash stamp line.

```bash
git checkout docs/results/stage1.md docs/results/stage2.md
git add palu_tt/report.py tests/test_report.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2b Task 6: stage 2b report (task accuracy with the paper reference block)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 7: Evaluation runs and write-up

Outcome: all five rows scored on the five tasks; G-LRD rows measured for perplexity, decode/attention timing (naive and fused), decode fidelity and batch-8 max context; `docs/results/stage2b.md` rendered and written up.

**Files:**
- Create: `docs/results/stage2b.md`, `docs/results/raw/tasks_{baseline,palu_full_50,palu_full_70,glrd_50,glrd_70}.json`, `ppl_glrd_{50,70}.json`, `bench_{decode,attention}_glrd_{50,70}{,_fk2}.json`, `decode_check_glrd_{50,70}_fk2.json`, `bench_maxctx_glrd_{50,70}_fk2_b8.json`.
- Modify: `README.md`, the spec's status line.

- [ ] **Step 1: The queued device job (one background script, sequential, every command gated)**

```bash
# all through scripts/run_device.sh; logs to docs/results/raw/<name>.log (never committed)
# A. G-LRD quality/memory/speed (~1.5 h)
python -m palu_tt.eval_ppl --row glrd_50 --kind palu --palu-dir compressed/glrd_0.5
python -m palu_tt.eval_ppl --row glrd_70 --kind palu --palu-dir compressed/glrd_0.7
for r in glrd_50:compressed/glrd_0.5 glrd_70:compressed/glrd_0.7; do IFS=: read row dir <<< "$r"
  python -m palu_tt.bench --row $row --kind palu --palu-dir $dir --what decode
  python -m palu_tt.bench --row $row --kind palu --palu-dir $dir --what attention
  python -m palu_tt.bench --row ${row}_fk2 --kind palu --palu-dir $dir --what decode --fused p2
  python -m palu_tt.bench --row ${row}_fk2 --kind palu --palu-dir $dir --what attention --fused p2
  python -m palu_tt.decode_check --row ${row}_fk2 --palu-dir $dir --fused p2
  python -m palu_tt.bench --row ${row}_fk2 --kind palu --palu-dir $dir --what maxctx --fused p2 --batch 8
done
# B. task accuracy, five rows (~3 h), overnight
python -m palu_tt.eval_tasks --row baseline --kind stock
python -m palu_tt.eval_tasks --row palu_full_50 --kind palu --palu-dir compressed/palu_full_0.5
python -m palu_tt.eval_tasks --row palu_full_70 --kind palu --palu-dir compressed/palu_full_0.7
python -m palu_tt.eval_tasks --row glrd_50 --kind palu --palu-dir compressed/glrd_0.5
python -m palu_tt.eval_tasks --row glrd_70 --kind palu --palu-dir compressed/glrd_0.7
```

Record start/end timestamps per command. A run that fails is data; note it and continue.

- [ ] **Step 2: Render and write up**

`python -m palu_tt.report --stage 2b`; below the marker: (1) accuracy per row against stock and against the paper's gaps (our G-LRD minus stock vs the paper's G-LRD minus baseline, per task and on the five-task average); (2) perplexity and memory for the G-LRD rows next to the M-LRD rows at the same ratio; (3) G-LRD decode: naive vs fused vs stock at 1k/4k/8k, with the expected reasons from spec §4 checked against the attention-only column; (4) decode fidelity; (5) batch-8 max context; (6) caveats (32-sample Fisher, model revision, 2048-token perplexity windows, the fused-kernel V mode's cost); (7) what was not measured (HellaSwag, J-LRD, the 256-sample sensitivity run unless it was done). README: stage 2b commands (`eval_tasks`, the download step, `report --stage 2b`). Spec status line → "executed".

- [ ] **Step 3: Commit**

```bash
git add docs/results/stage2b.md docs/results/raw/*.json README.md docs/superpowers/specs/2026-09-12-palu-tt-stage2b-design.md
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2b Task 7: task accuracy for all rows, G-LRD quality/speed/fidelity/capacity, stage 2b results document

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

## After the last task

Whole-branch review, then `superpowers:finishing-a-development-branch` (merge to `main` locally as before). The controller republishes the results page with the stage 2b tables.
