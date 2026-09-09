# Palu on Tenstorrent — Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a Palu-compressed (low-rank KV cache) Llama-3.1-8B-Instruct on a Wormhole n300 using existing ttnn ops, and compare it against the uncompressed model on the same device: perplexity, KV-cache bytes, decode/prefill speed.

**Architecture:** A `PaluAttention` class replaces tt_transformers' `Attention` through the existing `Transformer(attention_class=...)` hook. It keeps every activation in stock shape (the latent projections are zero-padded to head_dim inside the fused QKV weight, the attention output is zero-padded back to head_dim before the output projection), so all stock program/memory configs are reused. Only the KV caches store latents: V goes through the multi-latent (MLA) flash-decode op with `head_dim_v = r_v`; K is reconstructed per decode step with a batched matmul + prefill-mode RoPE. Compression (per-head SVD) and the CPU reference live in plain torch.

**Tech Stack:** tt-metal v0.78.0 (submodule, built from source with clang-20), ttnn, torch 2.x CPU, transformers, safetensors, huggingface_hub, datasets, pytest. Palu upstream as a submodule for the optional full calibration recipe.

Spec: `docs/superpowers/specs/2026-09-09-palu-tt-stage1-design.md`.

## Global Constraints

- Shared server: no sudo, no firmware/driver/hugepage changes, never read or modify `/home/user01/tt-data`.
- Everything lives under this repo, the user's HF cache (`~/.cache/huggingface`, read-only) and the session scratch dir.
- Every device run goes through `scripts/device_free.sh` first and fails fast if the n300 is busy.
- CPU-heavy work (build, calibration, CPU perplexity, CPU reference model) runs under `nice -n 10` with at most 12 threads (`OMP_NUM_THREADS=12`, `torch.set_num_threads(12)`).
- All CPU numerics in fp32 (bf16 matmul on this CPU is ~1000x slower).
- Model: `meta-llama/Llama-3.1-8B-Instruct` (32 layers, 32 q heads, 8 KV heads, head_dim 128, hidden 4096, vocab 128256). TP=2 on the n300: 16 q heads and 4 KV heads per chip.
- Ranks are multiples of 32; per-layer ranks may differ, per-head ranks within a layer are padded to the layer max. Study points: r=128 (baseline through the Palu path), r=96, r=64, and r_k=128/r_v=64.
- tt-metal pinned at tag `v0.78.0`; Palu pinned at commit `bb22666`.
- Non-paged KV cache, no prefetcher, no tracing, Meta-style RoPE (`use_hf_rope=False`), `use_qk_fused=False` for both stock and Palu models.
- Run all commands from the repo root with `source scripts/env.sh` active.
- Commit after every task with the attribution trailer:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs
  ```

## Deviation from the spec, recorded here

- Spec section 8 lists decode contexts 1k/4k/16k. A non-paged batch-32 cache at 16k does not fit on one n300 chip for the baseline (32 layers x 2 x 32 x 4 x 16384 x 128 bytes ≈ 17 GB > 12 GB). The bench sweeps 1k/4k/8k and the separate "max context at batch 32" probe reports where each row stops fitting. Update the spec line when Task 10 lands.
- Spec section 4 describes `wo_fused` as 4096 x (32·r_v). It is stored zero-padded to 4096 x 4096 so the stock output-projection configs apply unchanged; numerically identical.
- Compressed weights are stored as a delta file (`palu_attn.safetensors` + `config.json`) in Palu's key naming, not a full 16 GB HF dump; a full Palu HF dump is read by the same loader.

## File structure

```
tt-cadlab/
  .gitignore                      .venv/, model_cache/, compressed/, docs/results/raw/, __pycache__
  .gitmodules                     third_party/tt-metal (v0.78.0), third_party/palu (bb22666)
  requirements.txt                datasets, safetensors, huggingface_hub, pytest (rest come from tt-metal venv)
  scripts/
    setup_env.sh                  submodules, niced build (-j10), venv at .venv, extra deps
    env.sh                        TT_METAL_HOME, PYTHONPATH, MESH_DEVICE, HF_MODEL, activate venv
    device_free.sh                exit 1 when the n300 is not idle (tt-smi telemetry)
    run_device.sh                 device_free.sh && "$@"
    patch_palu_cpu.sh             sed patches making Palu's compress.py run on CPU fp32
    run_calibration.sh            niced Palu full recipe -> compressed/palu_full_r<ratio>/
    run_ppl_cpu.sh                niced CPU HF perplexity sanity run
  palu_tt/
    __init__.py
    hf.py                         HF snapshot dir, selective safetensors loading, config
    compress.py                   per-head SVD export (CLI)
    convert.py                    PaluConfig/PaluRuntime, loader for Palu folders, TT state-dict tensors
    reference.py                  torch CPU reference attention + CPU low-rank Llama
    device.py                     open_n300(), synchronize helper, device-free assertion
    attention.py                  PaluAttention
    model.py                      build_model(kind="stock"|"palu", ...)
    eval_ppl.py                   on-device wikitext2 perplexity (CLI)
    bench.py                      decode/prefill timing, KV bytes, max-context probe (CLI)
    generate.py                   greedy generation samples (CLI)
    report.py                     renders docs/results/stage1.md from docs/results/raw/*.json
  tests/
    conftest.py                   mesh_device fixture (N300), seeds
    test_compress.py              CPU
    test_convert.py               CPU
    test_reference.py             CPU
    test_attention_device.py      device, one layer
    test_model_device.py          device, full model (slow)
    test_ppl_harness.py           device, 2 windows
  docs/results/stage1.md          generated table + notes
```

Interfaces that every task relies on (defined in Task 3 and Task 4):

```python
# palu_tt/convert.py
@dataclass
class PaluRuntime:
    recon_len: int | None = None   # K reconstruction length for the current decode step (multiple of 32)
    check: bool = False            # tests: read current_pos back and assert recon_len covers it

@dataclass
class PaluConfig:
    rank_k: list[int]              # per layer
    rank_v: list[int]              # per layer
    runtime: PaluRuntime

# state_dict keys added per layer i (all torch fp32/bf16 host tensors, HF orientation [out, in]):
#   layers.{i}.attention.palu.wk_lat_pad   [n_kv_heads*head_dim, dim]   VT_k per head, zero-padded rows to head_dim
#   layers.{i}.attention.palu.wv_lat_pad   [n_kv_heads*head_dim, dim]
#   layers.{i}.attention.palu.u_k          [n_kv_heads, r_k, head_dim]  (Meta-permuted output columns)
#   layers.{i}.attention.palu.u_v          [n_kv_heads, r_v, head_dim]
#   layers.{i}.attention.palu.wo_fused     [dim, n_heads*head_dim]       o_proj·U_v per head, zero-padded
```

---

### Task 1: Repository scaffold, submodules, build, environment

**Files:**
- Create: `.gitignore`, `.gitmodules` (via `git submodule add`), `requirements.txt`, `scripts/setup_env.sh`, `scripts/env.sh`, `scripts/device_free.sh`, `scripts/run_device.sh`, `palu_tt/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `palu_tt/device.py`

**Interfaces:**
- Produces: `scripts/env.sh` (sourced by every later command), `palu_tt.device.open_n300() -> ttnn.MeshDevice`, `palu_tt.device.assert_device_free()`, pytest fixture `mesh_device`.

- [ ] **Step 1: .gitignore and requirements**

`.gitignore`:
```
.venv/
model_cache/
compressed/
docs/results/raw/
__pycache__/
*.pyc
.pytest_cache/
generated/
```

`requirements.txt`:
```
datasets>=2.19
safetensors>=0.4
huggingface_hub>=0.23
pytest>=8
```

- [ ] **Step 2: Add submodules (pinned)**

```bash
git submodule add https://github.com/tenstorrent/tt-metal.git third_party/tt-metal
git -C third_party/tt-metal fetch --depth 1 origin tag v0.78.0
git -C third_party/tt-metal checkout v0.78.0
git submodule add https://github.com/shadowpa0327/palu.git third_party/palu
git -C third_party/palu checkout bb22666
git add .gitmodules third_party
```
Expected: `git submodule status` shows `third_party/tt-metal` at the v0.78.0 commit and `third_party/palu` at `bb22666`.

- [ ] **Step 3: scripts/env.sh**

```bash
# source this file; never execute it
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TT_METAL_HOME="$ROOT/third_party/tt-metal"
export PYTHONPATH="$TT_METAL_HOME:$ROOT"
export MESH_DEVICE=N300
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=12
export TT_METAL_LOGGER_LEVEL=${TT_METAL_LOGGER_LEVEL:-ERROR}
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
```

- [ ] **Step 4: scripts/setup_env.sh**

```bash
#!/usr/bin/env bash
# One-time environment setup. Shared server: niced, capped at JOBS cores.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="${JOBS:-10}"
cd "$ROOT"
git submodule update --init --recursive --depth 1 third_party/tt-metal
git submodule update --init third_party/palu

cd "$ROOT/third_party/tt-metal"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
nice -n 10 ./build_metal.sh --release            # clang-20 toolchain is the default; ~1.5-2 h at -j10
nice -n 10 ./create_venv.sh --env-dir "$ROOT/.venv"   # uv venv, dev requirements, `uv pip install -e .`

source "$ROOT/.venv/bin/activate"
uv pip install -r "$ROOT/requirements.txt"
echo "setup done. Next: source scripts/env.sh"
```
`chmod +x scripts/setup_env.sh`.

- [ ] **Step 5: scripts/device_free.sh and scripts/run_device.sh**

`scripts/device_free.sh`:
```bash
#!/usr/bin/env bash
# Exit 0 only when the n300 looks idle. Telemetry words pack (max<<16 | current).
# Idle n300: AICLK current == 500 MHz, TDP current ~13 W. Busy: AICLK 1000 MHz and/or TDP >> 20 W.
set -euo pipefail
json="$(tt-smi -s 2>/dev/null)" || { echo "device_free: tt-smi failed"; exit 1; }
python3 - "$json" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
busy = []
for i, dev in enumerate(d["device_info"]):
    t = dev["smbus_telem"]
    aiclk = int(t["AICLK"], 16) & 0xFFFF
    tdp = int(t["TDP"], 16) & 0xFFFF
    if aiclk > 500 or tdp > 25:
        busy.append(f"chip{i}: AICLK={aiclk}MHz TDP={tdp}W")
if busy:
    print("device_free: n300 appears BUSY -> " + "; ".join(busy)); sys.exit(1)
print("device_free: n300 idle")
PY
# our own processes holding the device (other users' processes are invisible without sudo)
if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q tenstorrent; then
    echo "device_free: one of our processes already holds /dev/tenstorrent"; exit 1
fi
```

`scripts/run_device.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/device_free.sh"
exec "$@"
```
`chmod +x scripts/device_free.sh scripts/run_device.sh`.

- [ ] **Step 6: palu_tt/device.py and tests/conftest.py**

`palu_tt/__init__.py` and `tests/__init__.py`: empty.

`palu_tt/device.py`:
```python
import subprocess
from pathlib import Path

import ttnn

ROOT = Path(__file__).resolve().parents[1]


def assert_device_free():
    r = subprocess.run([str(ROOT / "scripts" / "device_free.sh")], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)


def open_n300():
    """Open the 1x2 mesh with 1D fabric (needed by tt_transformers CCL ops)."""
    assert_device_free()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2))


def close(mesh_device):
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def sync(mesh_device):
    ttnn.synchronize_device(mesh_device)
```
If `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)` raises a signature error, use the full form from `third_party/tt-metal/conftest.py::set_fabric` (reliability mode `ttnn.FabricReliabilityMode.STRICT_INIT`, tensix config from `get_default_fabric_tensix_config()`, `ttnn.FabricUDMMode.DISABLED`, `ttnn.FabricManagerMode.DEFAULT`).

`tests/conftest.py`:
```python
import random

import numpy as np
import pytest
import torch

from palu_tt import device as dev


@pytest.fixture(autouse=True)
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)
    torch.set_num_threads(12)


@pytest.fixture(scope="session")
def mesh_device():
    md = dev.open_n300()
    yield md
    dev.close(md)
```

- [ ] **Step 7: Run setup and verify**

```bash
JOBS=10 scripts/setup_env.sh 2>&1 | tail -20
source scripts/env.sh
python -c "import ttnn, torch, transformers; print(ttnn.get_arch_name(), torch.__version__, transformers.__version__)"
scripts/device_free.sh
python - <<'PY'
from palu_tt import device as d
md = d.open_n300(); print("opened", md.get_num_devices()); d.close(md)
PY
```
Expected: `wormhole_b0 2.x.x 4.5x.x`, `device_free: n300 idle`, `opened 2`.

- [ ] **Step 8: Commit**

```bash
git add .gitignore requirements.txt scripts palu_tt tests
git commit -m "Scaffold: tt-metal v0.78.0 + palu submodules, env and device scripts"
```

---

### Task 2: HF helpers and per-head SVD compression

**Files:**
- Create: `palu_tt/hf.py`, `palu_tt/compress.py`, `tests/test_compress.py`

**Interfaces:**
- Produces:
  - `hf.MODEL_ID`, `hf.snapshot_dir(model_id=MODEL_ID) -> Path`, `hf.load_config(hf_dir) -> dict`, `hf.load_tensors(hf_dir, predicate) -> dict[str, torch.Tensor]` (fp32).
  - `compress.svd_per_head(W, n_heads, rank) -> (U [n_heads, hd, rank], VT [n_heads, rank, dim])`.
  - `compress.compress_attention(hf_dir, out_dir, rank_k, rank_v, layers=None)` writing `out_dir/palu_attn.safetensors` with keys `model.layers.{i}.self_attn.{k,v}_proj.VT.weight` ([n_kv*r, dim]) and `model.layers.{i}.self_attn.{k,v}_proj.U.{h}.weight` ([hd, r]) plus `out_dir/config.json` with `head_wise_ranks`.

- [ ] **Step 1: Write the failing tests**

`tests/test_compress.py`:
```python
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from palu_tt import compress


def test_svd_per_head_full_rank_is_exact():
    W = torch.randn(4 * 32, 64)
    U, VT = compress.svd_per_head(W, n_heads=4, rank=32)
    assert U.shape == (4, 32, 32) and VT.shape == (4, 32, 64)
    rec = torch.cat([U[h] @ VT[h] for h in range(4)], dim=0)
    assert torch.allclose(rec, W, atol=1e-4)


def test_svd_per_head_error_decreases_with_rank():
    W = torch.randn(2 * 64, 256)
    errs = []
    for r in (16, 32, 64):
        U, VT = compress.svd_per_head(W, n_heads=2, rank=r)
        rec = torch.cat([U[h] @ VT[h] for h in range(2)], dim=0)
        errs.append((rec - W).norm().item())
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3


def test_compress_attention_writes_palu_layout(tmp_path):
    # tiny fake HF checkpoint: 2 layers, dim 64, 4 q heads, 2 kv heads, head_dim 16
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    cfg = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2}
    (hf_dir / "config.json").write_text(json.dumps(cfg))
    tensors = {}
    for i in range(2):
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(32, 64)
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(32, 64)
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(64, 64)
    save_file(tensors, str(hf_dir / "model.safetensors"))
    (hf_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    out = tmp_path / "out"
    compress.compress_attention(hf_dir, out, rank_k=8, rank_v=16)
    t = load_file(str(out / "palu_attn.safetensors"))
    assert t["model.layers.0.self_attn.k_proj.VT.weight"].shape == (2 * 8, 64)
    assert t["model.layers.0.self_attn.k_proj.U.1.weight"].shape == (16, 8)
    assert t["model.layers.1.self_attn.v_proj.VT.weight"].shape == (2 * 16, 64)
    assert t["model.layers.1.self_attn.v_proj.U.0.weight"].shape == (16, 16)
    c = json.loads((out / "config.json").read_text())
    assert c["head_wise_ranks"]["model.layers.0.self_attn.k_proj"] == [8, 8]
    assert c["head_wise_ranks"]["model.layers.1.self_attn.v_proj"] == [16, 16]
    # reconstruction of v (full rank 16 == head_dim) is exact
    W = tensors["model.layers.1.self_attn.v_proj.weight"]
    VT = t["model.layers.1.self_attn.v_proj.VT.weight"]
    rec = torch.cat([t[f"model.layers.1.self_attn.v_proj.U.{h}.weight"] @ VT[h * 16:(h + 1) * 16] for h in range(2)])
    assert torch.allclose(rec, W, atol=1e-4)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_compress.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'palu_tt.compress'`.

- [ ] **Step 3: Implement palu_tt/hf.py**

```python
import json
from pathlib import Path
from typing import Callable, Dict

import torch
from safetensors import safe_open

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def snapshot_dir(model_id: str = MODEL_ID) -> Path:
    """Local HF snapshot (read-only); never downloads."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, local_files_only=True))


def load_config(hf_dir: Path) -> dict:
    return json.loads((Path(hf_dir) / "config.json").read_text())


def load_tensors(hf_dir: Path, predicate: Callable[[str], bool]) -> Dict[str, torch.Tensor]:
    """Load only the tensors whose key satisfies predicate, as fp32 host tensors."""
    hf_dir = Path(hf_dir)
    index = hf_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = None
    out = {}
    if weight_map is None:
        shards = {"model.safetensors": None}
    else:
        shards = {}
        for k, f in weight_map.items():
            if predicate(k):
                shards.setdefault(f, []).append(k)
    for shard, keys in shards.items():
        with safe_open(str(hf_dir / shard), framework="pt") as f:
            for k in (keys if keys is not None else f.keys()):
                if predicate(k):
                    out[k] = f.get_tensor(k).float()
    return out
```

- [ ] **Step 4: Implement palu_tt/compress.py**

```python
"""Palu-style per-head SVD of k_proj / v_proj (group size 1, plain SVD, uniform rank)."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from palu_tt import hf


def svd_per_head(W: torch.Tensor, n_heads: int, rank: int):
    """W: [n_heads*hd, dim] -> U [n_heads, hd, rank], VT [n_heads, rank, dim] with W_h ≈ U_h @ VT_h.
    Same balanced split as Palu: U = U_r·sqrt(S), VT = sqrt(S)·Vt_r."""
    hd = W.shape[0] // n_heads
    Wh = W.float().reshape(n_heads, hd, -1)
    Us, VTs = [], []
    for h in range(n_heads):
        U, S, Vt = torch.linalg.svd(Wh[h], full_matrices=False)
        s = torch.sqrt(S[:rank])
        Us.append(U[:, :rank] * s[None, :])
        VTs.append(s[:, None] * Vt[:rank, :])
    return torch.stack(Us), torch.stack(VTs)


def compress_attention(hf_dir: Path, out_dir: Path, rank_k: int, rank_v: int, layers=None):
    hf_dir, out_dir = Path(hf_dir), Path(out_dir)
    cfg = hf.load_config(hf_dir)
    n_kv = cfg["num_key_value_heads"]
    n_layers = cfg["num_hidden_layers"] if layers is None else layers
    tensors = hf.load_tensors(hf_dir, lambda k: ("k_proj.weight" in k or "v_proj.weight" in k) and "layers." in k)
    out, ranks = {}, {}
    for i in range(n_layers):
        for proj, r in (("k_proj", rank_k), ("v_proj", rank_v)):
            key = f"model.layers.{i}.self_attn.{proj}"
            U, VT = svd_per_head(tensors[f"{key}.weight"], n_kv, r)
            out[f"{key}.VT.weight"] = VT.reshape(n_kv * r, -1).contiguous()
            for h in range(n_kv):
                out[f"{key}.U.{h}.weight"] = U[h].contiguous()
            ranks[key] = [r] * n_kv
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "palu_attn.safetensors"))
    (out_dir / "config.json").write_text(
        json.dumps({"head_wise_ranks": ranks, "palu_tt": {"method": "svd", "group_size": 1,
                    "rank_k": rank_k, "rank_v": rank_v, "source": str(hf_dir)}}, indent=1)
    )
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rank-k", type=int, required=True)
    p.add_argument("--rank-v", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=hf.MODEL_ID)
    a = p.parse_args()
    torch.set_num_threads(12)
    print(compress_attention(hf.snapshot_dir(a.model), a.out, a.rank_k, a.rank_v))
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_compress.py -q`
Expected: `3 passed`.

- [ ] **Step 6: Produce the study-point weights (CPU, minutes)**

```bash
nice -n 10 python -m palu_tt.compress --rank-k 64  --rank-v 64  --out compressed/svd_r64
nice -n 10 python -m palu_tt.compress --rank-k 96  --rank-v 96  --out compressed/svd_r96
nice -n 10 python -m palu_tt.compress --rank-k 128 --rank-v 64  --out compressed/svd_rk128_rv64
ls -la compressed/*/
```
Expected: three folders, each with `palu_attn.safetensors` (r64 ≈ 0.8 GB) and `config.json`.

- [ ] **Step 7: Commit**

```bash
git add palu_tt/hf.py palu_tt/compress.py tests/test_compress.py
git commit -m "Per-head SVD compression of k_proj/v_proj in Palu key layout"
```

---

### Task 3: Converter: Palu folder -> TT state-dict tensors

**Files:**
- Create: `palu_tt/convert.py`, `tests/test_convert.py`

**Interfaces:**
- Produces: `PaluRuntime`, `PaluConfig` (see top), `LowRank` (per-layer VT/U in head-major layout), `load_palu_dir(palu_dir, n_layers, n_kv_heads, head_dim) -> dict[int, LowRank]`, `identity_lowrank(hf_attn, n_kv_heads, head_dim) -> LowRank`, `palu_layer_tensors(hf_attn, lr, n_heads, n_kv_heads, head_dim) -> dict`, `add_palu_to_state_dict(state_dict, hf_dir, palu_dir, n_layers, n_heads, n_kv_heads, head_dim, prefix="layers.{i}.attention.palu") -> PaluConfig`.
- Consumes: `hf.load_tensors`, `models.tt_transformers.tt.load_checkpoints.reverse_permute`.

- [ ] **Step 1: Write the failing tests**

`tests/test_convert.py`:
```python
import torch

from models.tt_transformers.tt.load_checkpoints import reverse_permute
from palu_tt import convert

DIM, NH, NKV, HD = 64, 4, 2, 16


def _hf_attn():
    return {
        "q": torch.randn(NH * HD, DIM), "k": torch.randn(NKV * HD, DIM),
        "v": torch.randn(NKV * HD, DIM), "o": torch.randn(DIM, NH * HD),
    }


def test_identity_lowrank_reproduces_stock_k_and_v():
    w = _hf_attn()
    lr = convert.identity_lowrank(w, NKV, HD)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    x = torch.randn(5, DIM)
    lat_k = x @ t["wk_lat_pad"].T                       # [5, NKV*HD] (padded latent == full latent here)
    k_meta = torch.cat([lat_k[:, h * HD:(h + 1) * HD] @ t["u_k"][h] for h in range(NKV)], dim=-1)
    k_stock_meta = x @ reverse_permute(w["k"], NKV, NKV * HD, DIM).T
    assert torch.allclose(k_meta, k_stock_meta, atol=1e-5)
    lat_v = x @ t["wv_lat_pad"].T
    v = torch.cat([lat_v[:, h * HD:(h + 1) * HD] @ t["u_v"][h] for h in range(NKV)], dim=-1)
    assert torch.allclose(v, x @ w["v"].T, atol=1e-5)


def test_lowrank_tensors_shapes_and_padding():
    w = _hf_attn()
    r_k, r_v = 8, 16
    lr = convert.LowRank(
        k_VT=torch.randn(NKV, r_k, DIM), k_U=torch.randn(NKV, HD, r_k),
        v_VT=torch.randn(NKV, r_v, DIM), v_U=torch.randn(NKV, HD, r_v),
    )
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert t["wk_lat_pad"].shape == (NKV * HD, DIM) and t["wv_lat_pad"].shape == (NKV * HD, DIM)
    assert t["u_k"].shape == (NKV, r_k, HD) and t["u_v"].shape == (NKV, r_v, HD)
    assert t["wo_fused"].shape == (DIM, NH * HD)
    # padded rows are zero
    assert torch.count_nonzero(t["wk_lat_pad"][r_k:HD]) == 0
    assert torch.count_nonzero(t["wo_fused"][:, r_v:HD]) == 0


def test_wo_fused_matches_attn_times_v_then_o():
    w = _hf_attn()
    r_v = 16  # full rank so V is exact and the check is tight
    lr = convert.identity_lowrank(w, NKV, HD)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    S, T = 3, 7
    x = torch.randn(T, DIM)
    lat_v = (x @ t["wv_lat_pad"].T).reshape(T, NKV, HD)          # [T, NKV, r_v(=HD)]
    attn = torch.softmax(torch.randn(NH, S, T), dim=-1)
    # Palu fused path: (attn @ latent) padded to HD per head, then wo_fused
    fused_in = torch.zeros(S, NH * HD)
    for h in range(NH):
        fused_in[:, h * HD:h * HD + r_v] = attn[h] @ lat_v[:, h // (NH // NKV), :r_v]
    y_fused = fused_in @ t["wo_fused"].T
    # stock path: reconstruct V, attn @ V, o_proj
    V = (x @ w["v"].T).reshape(T, NKV, HD)
    stock_in = torch.cat([attn[h] @ V[:, h // (NH // NKV)] for h in range(NH)], dim=-1)
    y_stock = stock_in @ w["o"].T
    assert torch.allclose(y_fused, y_stock, atol=1e-4)


def test_load_palu_dir_pads_ranks_to_layer_max(tmp_path):
    import json
    from safetensors.torch import save_file
    r = [8, 4]  # non-uniform across heads -> padded to 8
    tensors = {
        "model.layers.0.self_attn.k_proj.VT.weight": torch.randn(sum(r), DIM),
        "model.layers.0.self_attn.k_proj.U.0.weight": torch.randn(HD, r[0]),
        "model.layers.0.self_attn.k_proj.U.1.weight": torch.randn(HD, r[1]),
        "model.layers.0.self_attn.v_proj.VT.weight": torch.randn(2 * 16, DIM),
        "model.layers.0.self_attn.v_proj.U.0.weight": torch.randn(HD, 16),
        "model.layers.0.self_attn.v_proj.U.1.weight": torch.randn(HD, 16),
    }
    save_file(tensors, str(tmp_path / "palu_attn.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"head_wise_ranks": {
        "model.layers.0.self_attn.k_proj": r, "model.layers.0.self_attn.v_proj": [16, 16]}}))
    lrs = convert.load_palu_dir(tmp_path, n_layers=1, n_kv_heads=NKV, head_dim=HD, rank_multiple=4)
    lr = lrs[0]
    assert lr.k_VT.shape == (NKV, 8, DIM) and lr.k_U.shape == (NKV, HD, 8)
    assert torch.count_nonzero(lr.k_VT[1, 4:]) == 0 and torch.count_nonzero(lr.k_U[1, :, 4:]) == 0
    # head 1 still reconstructs the same product
    x = torch.randn(3, DIM)
    ref = x @ tensors["model.layers.0.self_attn.k_proj.VT.weight"][8:12].T @ tensors["model.layers.0.self_attn.k_proj.U.1.weight"].T
    assert torch.allclose(x @ lr.k_VT[1].T @ lr.k_U[1].T, ref, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_convert.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'palu_tt.convert'`.

- [ ] **Step 3: Implement palu_tt/convert.py**

```python
"""Turn Palu low-rank factors into the tensors PaluAttention loads (see plan header for key layout)."""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

from models.tt_transformers.tt.load_checkpoints import reverse_permute
from palu_tt import hf


@dataclass
class PaluRuntime:
    recon_len: Optional[int] = None
    check: bool = False


@dataclass
class PaluConfig:
    rank_k: List[int]
    rank_v: List[int]
    runtime: PaluRuntime = field(default_factory=PaluRuntime)


@dataclass
class LowRank:
    """Head-major factors: W_h ≈ U_h @ VT_h. Shapes k_VT [n_kv, r_k, dim], k_U [n_kv, hd, r_k], same for v."""
    k_VT: torch.Tensor
    k_U: torch.Tensor
    v_VT: torch.Tensor
    v_U: torch.Tensor

    @property
    def r_k(self):
        return self.k_VT.shape[1]

    @property
    def r_v(self):
        return self.v_VT.shape[1]


def identity_lowrank(hf_attn: Dict[str, torch.Tensor], n_kv_heads: int, head_dim: int) -> LowRank:
    """r == head_dim, VT = W rows of the head, U = I. Makes the Palu path bit-for-bit the stock model."""
    eye = torch.eye(head_dim).expand(n_kv_heads, head_dim, head_dim).contiguous()
    return LowRank(
        k_VT=hf_attn["k"].float().reshape(n_kv_heads, head_dim, -1).contiguous(), k_U=eye.clone(),
        v_VT=hf_attn["v"].float().reshape(n_kv_heads, head_dim, -1).contiguous(), v_U=eye.clone(),
    )


def _read_palu_layer(tensors, ranks, layer, proj, n_kv_heads, head_dim, rank_multiple):
    key = f"model.layers.{layer}.self_attn.{proj}"
    r_list = ranks[key]
    assert len(r_list) == n_kv_heads, f"{key}: expected {n_kv_heads} groups (group size 1), got {len(r_list)}"
    r_max = int(math.ceil(max(r_list) / rank_multiple) * rank_multiple)
    VT_all = tensors[f"{key}.VT.weight"]
    dim = VT_all.shape[1]
    VT = torch.zeros(n_kv_heads, r_max, dim)
    U = torch.zeros(n_kv_heads, head_dim, r_max)
    off = 0
    for h, r in enumerate(r_list):
        VT[h, :r] = VT_all[off:off + r]
        U[h, :, :r] = tensors[f"{key}.U.{h}.weight"]
        off += r
    return VT, U


def load_palu_dir(palu_dir: Path, n_layers: int, n_kv_heads: int, head_dim: int, rank_multiple: int = 32) -> Dict[int, LowRank]:
    """Reads our delta file or a full Palu HF dump (same key names). Per-head ranks are zero-padded to the
    layer max rounded up to rank_multiple, which changes nothing numerically."""
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
        k_VT, k_U = _read_palu_layer(tensors, ranks, i, "k_proj", n_kv_heads, head_dim, rank_multiple)
        v_VT, v_U = _read_palu_layer(tensors, ranks, i, "v_proj", n_kv_heads, head_dim, rank_multiple)
        out[i] = LowRank(k_VT, k_U, v_VT, v_U)
    return out


def palu_layer_tensors(hf_attn: Dict[str, torch.Tensor], lr: LowRank, n_heads: int, n_kv_heads: int, head_dim: int):
    dim = hf_attn["q"].shape[1]
    r_k, r_v = lr.r_k, lr.r_v
    wk_lat_pad = torch.zeros(n_kv_heads * head_dim, dim)
    wv_lat_pad = torch.zeros(n_kv_heads * head_dim, dim)
    u_k = torch.zeros(n_kv_heads, r_k, head_dim)
    u_v = torch.zeros(n_kv_heads, r_v, head_dim)
    for h in range(n_kv_heads):
        wk_lat_pad[h * head_dim:h * head_dim + r_k] = lr.k_VT[h]
        wv_lat_pad[h * head_dim:h * head_dim + r_v] = lr.v_VT[h]
        # K leaves the latent in Meta-permuted head order (RoPE convention of tt_transformers)
        u_k[h] = reverse_permute(lr.k_U[h], 1, head_dim, r_k).T
        u_v[h] = lr.v_U[h].T
    wo = hf_attn["o"].float()                       # [dim, n_heads*hd]
    wo_fused = torch.zeros_like(wo)
    rep = n_heads // n_kv_heads
    for qh in range(n_heads):
        kv = qh // rep
        wo_fused[:, qh * head_dim:qh * head_dim + r_v] = wo[:, qh * head_dim:(qh + 1) * head_dim] @ lr.v_U[kv]
    return {"wk_lat_pad": wk_lat_pad, "wv_lat_pad": wv_lat_pad, "u_k": u_k, "u_v": u_v, "wo_fused": wo_fused}


def hf_attn_weights(hf_dir: Path, layer: int) -> Dict[str, torch.Tensor]:
    p = f"model.layers.{layer}.self_attn."
    t = hf.load_tensors(hf_dir, lambda k: k.startswith(p) and k.endswith("proj.weight"))
    return {"q": t[p + "q_proj.weight"], "k": t[p + "k_proj.weight"], "v": t[p + "v_proj.weight"], "o": t[p + "o_proj.weight"]}


def add_palu_to_state_dict(state_dict: dict, hf_dir: Path, palu_dir: Optional[Path], n_layers: int,
                           n_heads: int, n_kv_heads: int, head_dim: int) -> PaluConfig:
    """palu_dir None -> identity factorisation (stock model through the Palu path)."""
    lrs = None if palu_dir is None else load_palu_dir(palu_dir, n_layers, n_kv_heads, head_dim)
    rank_k, rank_v = [], []
    for i in range(n_layers):
        w = hf_attn_weights(hf_dir, i)
        lr = identity_lowrank(w, n_kv_heads, head_dim) if lrs is None else lrs[i]
        for name, t in palu_layer_tensors(w, lr, n_heads, n_kv_heads, head_dim).items():
            state_dict[f"layers.{i}.attention.palu.{name}"] = t
        rank_k.append(lr.r_k)
        rank_v.append(lr.r_v)
    return PaluConfig(rank_k=rank_k, rank_v=rank_v)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_convert.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add palu_tt/convert.py tests/test_convert.py
git commit -m "Converter: Palu factors -> padded latent projections, permuted U_k, fused wo"
```

---

### Task 4: CPU reference attention and CPU low-rank Llama

**Files:**
- Create: `palu_tt/reference.py`, `tests/test_reference.py`

**Interfaces:**
- Produces:
  - `PaluRefAttention(hf_cfg: dict, w: dict(q,k,v,o), lr: LowRank)`; `.forward(x [B,S,dim], start_pos) -> [B,S,dim]`; attributes `cache_k_lat [B, n_kv, T, r_k]`, `cache_v_lat [B, n_kv, T, r_v]` (HF conventions, fp32).
  - `LowRankProj(nn.Module)` and `build_cpu_lowrank_llama(hf_dir, palu_dir|None) -> transformers.LlamaForCausalLM` (fp32, k_proj/v_proj replaced).
- Consumes: `convert.LowRank`, `convert.identity_lowrank`, `convert.load_palu_dir`, `hf.load_config`.

- [ ] **Step 1: Write the failing tests**

`tests/test_reference.py`:
```python
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from palu_tt import convert, reference

CFG = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
           intermediate_size=128, vocab_size=256, max_position_embeddings=512, rope_theta=10000.0,
           rms_norm_eps=1e-5, tie_word_embeddings=False)


def _tiny_model(tmp_path):
    m = LlamaForCausalLM(LlamaConfig(**CFG)).float().eval()
    m.save_pretrained(tmp_path / "hf", safe_serialization=True)
    return m, tmp_path / "hf"


def test_ref_attention_identity_matches_hf_layer(tmp_path):
    m, hf_dir = _tiny_model(tmp_path)
    attn = m.model.layers[0].self_attn
    w = {"q": attn.q_proj.weight.data, "k": attn.k_proj.weight.data, "v": attn.v_proj.weight.data, "o": attn.o_proj.weight.data}
    lr = convert.identity_lowrank(w, 2, 16)
    ref = reference.PaluRefAttention(CFG, w, lr)
    x = torch.randn(1, 8, 64)
    pos = torch.arange(8)[None]
    cos, sin = m.model.rotary_emb(x, pos)
    y_hf, _ = attn(x, position_embeddings=(cos, sin), attention_mask=None)
    y = ref.forward(x, start_pos=0)
    assert torch.allclose(y, y_hf, atol=1e-4)
    # decode continues from the cache: prefill 8 then one token must equal a fresh 9-token prefill
    x9 = torch.cat([x, torch.randn(1, 1, 64)], dim=1)
    y_last = ref.forward(x9[:, 8:], start_pos=8)
    ref2 = reference.PaluRefAttention(CFG, w, lr)
    y_full = ref2.forward(x9, start_pos=0)
    assert torch.allclose(y_last, y_full[:, 8:], atol=1e-4)
    assert ref.cache_k_lat.shape == (1, 2, 9, 16)


def test_cpu_lowrank_llama_identity_equals_hf(tmp_path):
    m, hf_dir = _tiny_model(tmp_path)
    lm = reference.build_cpu_lowrank_llama(hf_dir, None)
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        a = m(ids).logits
        b = lm(ids).logits
    assert torch.allclose(a, b, atol=1e-4)


def test_cpu_lowrank_llama_uses_palu_factors(tmp_path):
    from palu_tt import compress
    m, hf_dir = _tiny_model(tmp_path)
    palu_dir = compress.compress_attention(hf_dir, tmp_path / "palu", rank_k=8, rank_v=8)
    lm = reference.build_cpu_lowrank_llama(hf_dir, palu_dir)
    kp = lm.model.layers[0].self_attn.k_proj
    assert isinstance(kp, reference.LowRankProj) and kp.VT.shape == (2 * 8, 64)
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        assert torch.isfinite(lm(ids).logits).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_reference.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'palu_tt.reference'`.

- [ ] **Step 3: Implement palu_tt/reference.py**

```python
"""CPU ground truth. HF conventions (rotate_half RoPE, unpermuted weights); fp32."""
import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from palu_tt import convert, hf


class PaluRefAttention:
    def __init__(self, hf_cfg: dict, w: Dict[str, torch.Tensor], lr: convert.LowRank):
        self.cfg = LlamaConfig(**{k: v for k, v in hf_cfg.items() if k != "architectures"})
        self.nh = self.cfg.num_attention_heads
        self.nkv = self.cfg.num_key_value_heads
        self.hd = self.cfg.hidden_size // self.nh
        self.wq, self.wo = w["q"].float(), w["o"].float()
        self.lr = lr
        self.rotary = LlamaRotaryEmbedding(config=self.cfg)
        self.cache_k_lat = None  # [B, nkv, T, r_k]
        self.cache_v_lat = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        B, S, _ = x.shape
        q = (x @ self.wq.T).view(B, S, self.nh, self.hd).transpose(1, 2)              # [B, nh, S, hd]
        k_lat = torch.einsum("bsd,hrd->bhsr", x, self.lr.k_VT)                        # [B, nkv, S, r_k]
        v_lat = torch.einsum("bsd,hrd->bhsr", x, self.lr.v_VT)
        if self.cache_k_lat is None or start_pos == 0:
            self.cache_k_lat, self.cache_v_lat = k_lat, v_lat
        else:
            assert self.cache_k_lat.shape[2] == start_pos
            self.cache_k_lat = torch.cat([self.cache_k_lat, k_lat], dim=2)
            self.cache_v_lat = torch.cat([self.cache_v_lat, v_lat], dim=2)
        T = self.cache_k_lat.shape[2]
        K = torch.einsum("bhtr,hdr->bhtd", self.cache_k_lat, self.lr.k_U)              # [B, nkv, T, hd]
        V = torch.einsum("bhtr,hdr->bhtd", self.cache_v_lat, self.lr.v_U)
        pos_q = torch.arange(start_pos, start_pos + S)[None].expand(B, -1)
        pos_k = torch.arange(0, T)[None].expand(B, -1)
        cos_q, sin_q = self.rotary(x, pos_q)
        cos_k, sin_k = self.rotary(x, pos_k)
        q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
        _, K = apply_rotary_pos_emb(K, K, cos_k, sin_k)
        rep = self.nh // self.nkv
        K = K.repeat_interleave(rep, dim=1)
        V = V.repeat_interleave(rep, dim=1)
        scores = q @ K.transpose(-1, -2) / math.sqrt(self.hd)                          # [B, nh, S, T]
        mask = torch.ones(S, T, dtype=torch.bool).tril(diagonal=start_pos)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, S, self.nh * self.hd)
        return out @ self.wo.T


class LowRankProj(nn.Module):
    """x -> concat_h (x @ VT_h^T @ U_h^T). Drop-in for HF k_proj / v_proj."""

    def __init__(self, VT: torch.Tensor, U: torch.Tensor):
        super().__init__()
        n_kv, r, dim = VT.shape
        self.n_kv, self.r = n_kv, r
        self.VT = nn.Parameter(VT.reshape(n_kv * r, dim).contiguous(), requires_grad=False)  # [n_kv*r, dim]
        self.U = nn.Parameter(U.contiguous(), requires_grad=False)                          # [n_kv, hd, r]

    def forward(self, x):
        lat = (x @ self.VT.T).reshape(*x.shape[:-1], self.n_kv, self.r)
        out = torch.einsum("...hr,hdr->...hd", lat, self.U)
        return out.reshape(*x.shape[:-1], -1)


def build_cpu_lowrank_llama(hf_dir: Path, palu_dir: Optional[Path]) -> LlamaForCausalLM:
    torch.set_num_threads(12)
    model = LlamaForCausalLM.from_pretrained(str(hf_dir), torch_dtype=torch.float32).eval()
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    lrs = None if palu_dir is None else convert.load_palu_dir(palu_dir, cfg.num_hidden_layers, cfg.num_key_value_heads, hd)
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        w = {"q": attn.q_proj.weight.data, "k": attn.k_proj.weight.data, "v": attn.v_proj.weight.data, "o": attn.o_proj.weight.data}
        lr = convert.identity_lowrank(w, cfg.num_key_value_heads, hd) if lrs is None else lrs[i]
        attn.k_proj = LowRankProj(lr.k_VT, lr.k_U)
        attn.v_proj = LowRankProj(lr.v_VT, lr.v_U)
    return model
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_reference.py -q`
Expected: `3 passed`. If `attn(x, position_embeddings=..., attention_mask=None)` in the first test raises on this transformers version, pass `attention_mask=None` positionally as the installed `LlamaAttention.forward` signature requires (check `inspect.signature`), and keep the reference unchanged.

- [ ] **Step 5: Commit**

```bash
git add palu_tt/reference.py tests/test_reference.py
git commit -m "CPU reference: Palu attention with latent cache, low-rank HF Llama"
```

---

### Task 5: PaluAttention — construction and prefill

**Files:**
- Create: `palu_tt/attention.py`, `tests/test_attention_device.py`

**Interfaces:**
- Produces: `PaluAttention(mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num, dtype, transformation_mats, configuration, paged_attention_config=None, use_paged_kv_cache=False, prefetcher=None)` — same signature as stock; requires `configuration.palu: PaluConfig`. `forward(x, current_pos, rot_mats=None, user_id=0, mode=Mode.DECODE, page_table=None, chunk_page_table=None, chunk_start_idx=None, kv_cache=None)`. `layer_past = [k_lat_cache, v_lat_cache]`. `kv_bytes_per_token()`.
- Consumes: state-dict keys from Task 3; tt_transformers ops as in stock `Attention` (v0.78.0).

- [ ] **Step 1: Write the failing prefill test**

`tests/test_attention_device.py` (device; ~2 min for the 16 GB HF load; run under `scripts/run_device.sh`):
```python
import os

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup

from palu_tt import convert, hf, reference
from palu_tt.attention import PaluAttention

SEQ = 256


def _setup(mesh_device, palu_dir, batch, max_seq_len):
    args = ModelArgs(mesh_device, instruct=True, max_batch_size=batch, max_seq_len=max_seq_len, cache_hf=False)
    args.use_qk_fused = False
    args.n_layers = 1
    sd = args.load_state_dict()
    hf_dir = hf.snapshot_dir()
    cfg = convert.add_palu_to_state_dict(sd, hf_dir, palu_dir, 1, args.n_heads, args.n_kv_heads, args.head_dim)
    args.palu = cfg
    rope = RotarySetup(mesh_device, batch, args.head_dim, args.max_seq_len, args.rope_theta, args.rope_scaling, False)
    tt_ccl = TT_CCL(mesh_device)
    w = convert.hf_attn_weights(hf_dir, 0)
    lr = convert.identity_lowrank(w, args.n_kv_heads, args.head_dim) if palu_dir is None else \
        convert.load_palu_dir(palu_dir, 1, args.n_kv_heads, args.head_dim)[0]
    ref = reference.PaluRefAttention(hf.load_config(hf_dir), w, lr)
    return args, sd, rope, tt_ccl, ref


def _tt_attention(cls, mesh_device, args, sd, rope, tt_ccl):
    return cls(mesh_device, tt_ccl, args, sd, weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b), layer_num=0,
               dtype=ttnn.bfloat8_b, transformation_mats=rope.get_both_trans_mats(), configuration=args)


def _to_torch(mesh_device, args, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=args.cluster_shape))


@pytest.mark.parametrize("palu_dir,pcc", [(None, 0.999), ("compressed/svd_r64", 0.99)], ids=["identity", "r64"])
def test_prefill_matches_reference(mesh_device, palu_dir, pcc):
    args, sd, rope, tt_ccl, ref = _setup(mesh_device, palu_dir, batch=1, max_seq_len=SEQ)
    tt = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    x = torch.randn(1, SEQ, args.dim) * 0.05
    y_ref = ref.forward(x, start_pos=0)
    x_tt = args.prepare_residual_tensor_prefill(x, force_replicated=True)
    rot = [rope.cos_matrix_prefill[:, :, :SEQ, :], rope.sin_matrix_prefill[:, :, :SEQ, :]]
    y = tt.forward(x_tt, None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    y = _to_torch(mesh_device, args, y)[:, 0:1, :SEQ, : args.dim].view(1, SEQ, args.dim)
    ok, msg = comp_pcc(y_ref, y, pcc)
    logger.info(msg)
    assert ok, msg
    # latent caches hold what the reference holds
    k_lat = ttnn.to_torch(tt.layer_past[0], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))
    k_lat = k_lat[:1, : args.n_kv_heads, :SEQ, :]
    ok, msg = comp_pcc(ref.cache_k_lat, k_lat, 0.99)
    assert ok, f"k latent cache {msg}"


def test_identity_prefill_equals_stock_attention(mesh_device):
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, None, batch=1, max_seq_len=SEQ)
    stock = _tt_attention(Attention, mesh_device, args, sd, rope, tt_ccl)
    palu = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    x = torch.randn(1, SEQ, args.dim) * 0.05
    rot = [rope.cos_matrix_prefill[:, :, :SEQ, :], rope.sin_matrix_prefill[:, :, :SEQ, :]]
    ys = []
    for m in (stock, palu):
        x_tt = args.prepare_residual_tensor_prefill(x, force_replicated=True)
        y = m.forward(x_tt, None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
        ys.append(_to_torch(mesh_device, args, y)[:, 0:1, :SEQ, : args.dim].view(1, SEQ, args.dim))
    ok, msg = comp_pcc(ys[0], ys[1], 0.999)
    assert ok, msg
```

- [ ] **Step 2: Run to verify failure**

Run: `scripts/run_device.sh pytest tests/test_attention_device.py -q -k prefill`
Expected: FAIL with `ModuleNotFoundError: No module named 'palu_tt.attention'`.

- [ ] **Step 3: Implement palu_tt/attention.py (construction + prefill; decode stub raising NotImplementedError)**

```python
"""PaluAttention for tt_transformers v0.78.0. N150/N300, non-paged, no prefetcher, Meta-style RoPE.
Shapes stay stock everywhere except inside the latent caches and the SDPA call."""
import math

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup, num_to_corerange
from models.tt_transformers.tt.rope import get_rot_mats


class PaluAttention(LightweightModule):
    def __init__(self, mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num, dtype,
                 transformation_mats, configuration, paged_attention_config=None, use_paged_kv_cache=False,
                 prefetcher=None):
        super().__init__()
        assert paged_attention_config is None and not use_paged_kv_cache, "stage 1 is non-paged"
        assert prefetcher is None, "prefetcher not supported"
        assert configuration.num_devices in (1, 2), "N150/N300 only"
        assert not configuration.use_hf_rope and not configuration.use_qk_fused, "set use_qk_fused=False, use_hf_rope=False"
        assert hasattr(configuration, "palu"), "attach a PaluConfig as configuration.palu"

        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices
        self.batch_size_per_device_group = self.max_batch_size
        self.dtype = dtype
        self.transformation_mats = transformation_mats
        self.sliding_window = None
        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.scale = self.head_dim ** -0.5
        self.palu = configuration.palu
        self.r_k = self.palu.rank_k[layer_num]
        self.r_v = self.palu.rank_v[layer_num]
        assert self.r_k % 32 == 0 and self.r_v % 32 == 0 and self.r_k <= self.head_dim and self.r_v <= self.head_dim

        opt = self.args.decoders_optimizations
        self.activation_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.ACTIVATION)
        self.wqkv_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.WQKV)
        self.wo_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.WO)
        self.kv_cache_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.KV_CACHE)
        fid = lambda op: opt.get_math_fidelity(decoder_id=layer_num, op=op, configuration=configuration)
        self.li_qkv_decode_compute_kernel_cfg = fid(OpGroup.LI_QKV_DECODE)
        self.sdpa_decode_compute_kernel_cfg = fid(OpGroup.SDPA_DECODE)
        self.li_o_decode_compute_kernel_cfg = fid(OpGroup.LI_O_DECODE)
        self.sdpa_prefill_compute_kernel_cfg = fid(OpGroup.SDPA_PREFILL)
        self.li_qkv_prefill_compute_kernel_cfg = fid(OpGroup.LI_QKV_PREFILL)
        self.li_o_prefill_compute_kernel_cfg = fid(OpGroup.LI_O_PREFILL)

        layer_name = configuration.get_state_dict_prefix("Attention", layer_num)
        tag = f"palu_rk{self.r_k}_rv{self.r_v}"
        if configuration.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{layer_name}.{name}"
        P = f"{layer_name}.palu"
        B = self.max_batch_size

        # --- fused [wq | wk_lat_pad | wv_lat_pad], per device, exactly like stock ---------------------
        wq = state_dict[f"{layer_name}.wq.weight"]
        wk = state_dict[f"{P}.wk_lat_pad"]
        wv = state_dict[f"{P}.wv_lat_pad"]
        qkv_list = []
        for i in range(self.num_devices):
            q_i = torch.chunk(wq, self.num_devices, dim=0)[i].transpose(-2, -1)
            k_i = torch.chunk(wk, self.num_devices, dim=0)[i].transpose(-2, -1)
            v_i = torch.chunk(wv, self.num_devices, dim=0)[i].transpose(-2, -1)
            qkv_list.append(torch.cat([q_i, k_i, v_i], dim=-1))
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
        self.wqkv = ttnn.as_tensor(
            qkv_cat, dtype=self.wqkv_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
            memory_config=configuration.create_dram_sharded_mem_config(configuration.dim, configuration.qkv_size // self.num_devices),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=configuration.cluster_shape),
            cache_file_name=cache_name(f"{tag}_wqkv_sharded_2d"),
        )

        # --- up-projections -------------------------------------------------------------------------
        u_k = state_dict[f"{P}.u_k"]  # [n_kv, r_k, hd]
        u_v = state_dict[f"{P}.u_v"]  # [n_kv, r_v, hd]
        shard_heads = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        # decode: batched (non-broadcast) matmul needs matching batch dims -> replicate over batch
        self.u_k_decode = ttnn.as_tensor(
            u_k.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads,
            cache_file_name=cache_name(f"{tag}_u_k_decode_b{B}"),
        )
        # prefill: zero-padded to [1, n_kv, hd, hd] so the padded latent heads need no slicing
        def pad_u(u, r):
            t = torch.zeros(self.n_kv_heads, self.head_dim, self.head_dim)
            t[:, :r, :] = u
            return t.unsqueeze(0)
        self.u_k_prefill = ttnn.as_tensor(pad_u(u_k, self.r_k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_k_prefill"))
        self.u_v_prefill = ttnn.as_tensor(pad_u(u_v, self.r_v), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_v_prefill"))

        # --- output projections: stock wo (prefill) and fused wo (decode) ------------------------------
        wo_mem = configuration.create_dram_sharded_mem_config((configuration.n_heads * configuration.head_dim) // self.num_devices, configuration.dim)
        def wo_tensor(w, name):
            return ttnn.as_tensor(w.transpose(-1, -2).unsqueeze(0).unsqueeze(0), dtype=self.wo_dtype, layout=ttnn.TILE_LAYOUT,
                                  device=mesh_device, memory_config=wo_mem, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
                                  cache_file_name=cache_name(name))
        self.wo = wo_tensor(state_dict[f"{layer_name}.wo.weight"], "wo")            # same bytes as stock -> shared cache entry
        self.wo_fused = wo_tensor(state_dict[f"{P}.wo_fused"], f"{tag}_wo_fused")

        # --- latent caches ---------------------------------------------------------------------------
        self.layer_past = []
        for r, nm in ((self.r_k, "k"), (self.r_v, "v")):
            z = torch.zeros(B, self.n_local_kv_heads, self.max_seq_len, r)
            self.layer_past.append(ttnn.as_tensor(
                z, dtype=self.kv_cache_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=cache_name(f"{tag}_latcache_{nm}_{tuple(z.shape)}") if weight_cache_path else None,
            ))

        # --- prefill-mode cos/sin for K reconstruction in decode ---------------------------------------
        self.cos_prefill, self.sin_prefill = get_rot_mats(
            head_dim=self.head_dim, device=mesh_device, seq_len=self.max_seq_len, theta=configuration.rope_theta,
            rope_scaling=configuration.rope_scaling, datatype=ttnn.bfloat16, rot_mats_layout=ttnn.TILE_LAYOUT,
        )

    # ------------------------------------------------------------------------------------------------
    def kv_bytes_per_token(self):
        """Bytes per token per layer on one chip (tile-padded), for the results table."""
        bpe = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0, ttnn.bfloat4_b: 0.5625}[self.kv_cache_dtype]
        return self.n_local_kv_heads * (self.r_k + self.r_v) * bpe

    def _latent_shard_cfg(self, width):
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width), core_grid=ttnn.CoreRangeSet({num_to_corerange(self.max_batch_size)}),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )

    def _kv_prefill_shard_cfg(self, seq_len, width):
        return ttnn.create_sharded_memory_config(
            (self.n_local_kv_heads * seq_len // 64, width), ttnn.CoreGrid(y=8, x=8),
            ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )

    # ------------------------------------------------------------------------------------------------
    def forward_prefill(self, x_11SH, rot_mats, user_id=0, page_table=None, chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        assert page_table is None and chunk_start_idx is None and kv_cache is None
        assert x_11SH.shape[0] == 1, "single-user prefill only"
        seq_len = x_11SH.shape[-2]
        original_seq_len = seq_len
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        # QKV (stock) ---------------------------------------------------------------------------------
        if seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
            padded = ((seq_len + self.MAX_QKV_MM_SEQ_LEN - 1) // self.MAX_QKV_MM_SEQ_LEN) * self.MAX_QKV_MM_SEQ_LEN
            x_11SH = ttnn.pad(x_11SH, padding=[(0, 0), (0, 0), (0, padded - seq_len), (0, 0)], value=0.0)
            seq_len = padded
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])
        if self.args.use_minimal_qkv_prefill_matmul(seq_len):
            xqkv = ttnn.experimental.minimal_matmul(x_11SH, self.wqkv, compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                                                    config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None))
        else:
            xqkv = ttnn.linear(x_11SH, self.wqkv, dtype=self.activation_dtype or ttnn.bfloat16,
                               memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None),
                               compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                               program_config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None))
        xqkv = tt_all_reduce(xqkv, self.mesh_device, self.tt_ccl, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.ccl_dtype)
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv = ttnn.reshape(xqkv, [1, 1, seq_len, -1])
        if original_seq_len != seq_len:
            xqkv = xqkv[:, :, :original_seq_len, :]
            seq_len = original_seq_len
        ttnn.deallocate(x_11SH)

        q_pre, k_lat_pad, v_lat_pad = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xqkv)

        # reconstruct full K and V from the padded latents ([1, nkv, S, 128] @ [1, nkv, 128, 128]) -------
        k_full = ttnn.matmul(k_lat_pad, self.u_k_prefill, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             compute_kernel_config=self.compute_kernel_config_hifi2)
        v_full = ttnn.matmul(v_lat_pad, self.u_v_prefill, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             compute_kernel_config=self.compute_kernel_config_hifi2)

        # RoPE on q and reconstructed K (stock prefill op) --------------------------------------------
        q = ttnn.experimental.rotary_embedding_llama(q_pre, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False)
        k_rot = ttnn.experimental.rotary_embedding_llama(k_full, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False)
        ttnn.deallocate(q_pre)
        ttnn.deallocate(k_full)

        # latents -> caches ------------------------------------------------------------------------
        keys, values = self.layer_past
        for lat_pad, r, cache in ((k_lat_pad, self.r_k, keys), (v_lat_pad, self.r_v, values)):
            lat = lat_pad[:, :, :, :r] if r < self.head_dim else lat_pad
            lat = ttnn.typecast(lat, dtype=cache.dtype)
            if seq_len >= self.min_kv_prefill_shard_seqlen:
                lat = ttnn.interleaved_to_sharded(lat, self._kv_prefill_shard_cfg(seq_len, r))
            ttnn.fill_cache(cache, lat, user_id % self.batch_size_per_device_group)
            ttnn.deallocate(lat)
        ttnn.deallocate(k_lat_pad)
        ttnn.deallocate(v_lat_pad)

        # SDPA (stock, with reconstructed K/V) -----------------------------------------------------------
        q8 = ttnn.typecast(q, dtype=self.activation_dtype or ttnn.bfloat8_b)
        k8 = ttnn.typecast(k_rot, dtype=self.kv_cache_dtype)
        v8 = ttnn.typecast(v_full, dtype=self.kv_cache_dtype)
        for t in (q, k_rot, v_full):
            ttnn.deallocate(t)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q8, k8, v8, is_causal=True, scale=self.scale, compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
            program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, seq_len, None, None))
        for t in (q8, k8, v8):
            ttnn.deallocate(t)

        # output projection (stock tail) ---------------------------------------------------------------
        attn = ttnn.reshape(attn, [1, self.n_local_heads, -1, self.head_dim])
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if seq_len > 1024:
            attn = ttnn.reshape(attn, [1, seq_len // 1024, 1024, -1])
        out = ttnn.linear(attn, self.wo, compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
                          dtype=self.activation_dtype or ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                          program_config=self.args.get_attn_wo_program_config(Mode.PREFILL, seq_len, None))
        if seq_len > 1024:
            out = ttnn.reshape(out, [1, 1, seq_len, -1])
        ttnn.deallocate(attn)
        out = tt_all_reduce(out, self.mesh_device, self.tt_ccl, cluster_axis=0, dim=3, topology=self.ccl_topology,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.ccl_dtype)
        return out

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        raise NotImplementedError("Task 6")

    def forward(self, x, current_pos, rot_mats=None, user_id=0, mode=Mode.DECODE, page_table=None,
                chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        if mode == Mode.PREFILL:
            return self.forward_prefill(x, rot_mats, user_id, page_table=page_table, chunk_page_table=chunk_page_table,
                                        chunk_start_idx=chunk_start_idx, kv_cache=kv_cache)
        return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)
```

- [ ] **Step 4: Run the prefill tests**

Run: `scripts/run_device.sh pytest tests/test_attention_device.py -q -k prefill 2>&1 | tail -20`
Expected: `3 passed` (identity vs reference ≥ 0.999, r64 vs reference ≥ 0.99, identity vs stock ≥ 0.999).

Known failure points and the fix for each:
- `ttnn.fill_cache` rejects the interleaved latent for long sequences: keep the sharded branch; if it rejects the *sharded* one for width r, drop the `interleaved_to_sharded` and pass interleaved.
- `ttnn.matmul` on `[1, nkv, S, 128] @ [1, nkv, 128, 128]` complains about program config: pass `core_grid=ttnn.CoreGrid(y=8, x=8)` explicitly.
- PCC for r64 well below 0.99 while identity passes: compare `k_lat` cache PCC first; if the cache matches, the error is in `u_k` permutation — check `test_convert.py::test_identity_lowrank_reproduces_stock_k_and_v` still passes and that `reverse_permute` is applied to rows of `U_h` (`[hd, r]`), not columns.

- [ ] **Step 5: Commit**

```bash
git add palu_tt/attention.py tests/test_attention_device.py
git commit -m "PaluAttention: weights, latent caches, prefill path"
```

---

### Task 6: PaluAttention — decode

**Files:**
- Modify: `palu_tt/attention.py` (replace `forward_decode` stub)
- Modify: `tests/test_attention_device.py` (add decode tests)

**Interfaces:**
- Consumes: `configuration.palu.runtime.recon_len` (int, multiple of 32, ≥ max(current_pos)+1) set by the caller before each decode step; `None` means reconstruct to `max_seq_len`.

- [ ] **Step 1: Add the failing decode tests**

Append to `tests/test_attention_device.py`:
```python
def _decode_inputs(mesh_device, args, rope, x_bd, current_pos):
    x_tt = args.prepare_residual_tensor_decode(x_bd, args.get_attn_input_mem_config(Mode.DECODE, None), force_replicated=True)
    pos_tt = ttnn.from_torch(current_pos, device=mesh_device, dtype=ttnn.int32,
                             mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=args.cluster_shape))
    return x_tt, pos_tt, rope.get_rot_mats(current_pos)


@pytest.mark.parametrize("palu_dir,pcc", [(None, 0.999), ("compressed/svd_r64", 0.99)], ids=["identity", "r64"])
def test_decode_after_prefill_matches_reference(mesh_device, palu_dir, pcc):
    batch = 32
    args, sd, rope, tt_ccl, ref = _setup(mesh_device, palu_dir, batch=batch, max_seq_len=SEQ)
    args.palu.runtime.check = True
    tt = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    # prefill user 0 with 128 tokens
    P = 128
    x_pre = torch.randn(1, P, args.dim) * 0.05
    ref.forward(x_pre, start_pos=0)
    rot = [rope.cos_matrix_prefill[:, :, :P, :], rope.sin_matrix_prefill[:, :, :P, :]]
    tt.forward(args.prepare_residual_tensor_prefill(x_pre, force_replicated=True), None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    # 8 decode steps, all users get user 0's input so the reference (batch 1) applies to slot 0
    for step in range(8):
        pos = P + step
        x = torch.randn(1, 1, args.dim) * 0.05
        y_ref = ref.forward(x, start_pos=pos)
        current_pos = torch.full((batch,), pos)
        args.palu.runtime.recon_len = ((pos + 1 + 31) // 32) * 32
        x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x.expand(batch, 1, -1).contiguous(), current_pos)
        y = tt.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
        y = _to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim)
        ok, msg = comp_pcc(y_ref, y[0:1], pcc)
        logger.info(f"pos={pos} {msg}")
        assert ok, f"pos={pos} {msg}"
    v_lat = ttnn.to_torch(tt.layer_past[1], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))
    ok, msg = comp_pcc(ref.cache_v_lat, v_lat[:1, : args.n_kv_heads, : P + 8, :], 0.99)
    assert ok, f"v latent cache {msg}"


def test_identity_decode_equals_stock_attention(mesh_device):
    batch = 32
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, None, batch=batch, max_seq_len=SEQ)
    stock = _tt_attention(Attention, mesh_device, args, sd, rope, tt_ccl)
    palu = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    for step in range(4):
        x = torch.randn(batch, 1, args.dim) * 0.05
        current_pos = torch.full((batch,), step)
        args.palu.runtime.recon_len = 32
        ys = []
        for m in (stock, palu):
            x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x, current_pos)
            y = m.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
            ys.append(_to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim))
        ok, msg = comp_pcc(ys[0], ys[1], 0.999)
        assert ok, f"step={step} {msg}"
```

- [ ] **Step 2: Run to verify failure**

Run: `scripts/run_device.sh pytest tests/test_attention_device.py -q -k decode 2>&1 | tail -5`
Expected: FAIL with `NotImplementedError: Task 6`.

- [ ] **Step 3: Implement forward_decode**

Replace the stub in `palu_tt/attention.py`:
```python
    def _recon_len(self, current_pos):
        S = self.palu.runtime.recon_len or self.max_seq_len
        S = min(nearest_32(S), self.max_seq_len)
        if self.palu.runtime.check:
            pos = ttnn.to_torch(ttnn.get_device_tensors(current_pos)[0])
            assert int(pos.max()) < S, f"recon_len {S} does not cover current_pos max {int(pos.max())}"
        return S

    def _slice_latent(self, heads_1BKD, r):
        """[1, B, nkv, 128] height-sharded -> [1, B, nkv, r] height-sharded (shard [32, r] per batch core)."""
        t = ttnn.sharded_to_interleaved(heads_1BKD, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(heads_1BKD)
        if r < self.head_dim:
            t2 = t[:, :, :, :r]
            ttnn.deallocate(t)
            t = t2
        return ttnn.interleaved_to_sharded(t, self._latent_shard_cfg(r))

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        assert page_table is None and kv_cache is None
        B = self.batch_size_per_device_group

        # QKV matmul + head split (stock) -------------------------------------------------------------
        xqkv_sharded = ttnn.linear(
            x, self.wqkv, memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.DECODE, None),
            program_config=self.args.get_attn_qkv_program_config(Mode.DECODE, 1, None),
            compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg, dtype=self.activation_dtype or ttnn.bfloat16)
        ttnn.deallocate(x)
        mem = self.args.get_attn_qkv_all_reduce_output_mem_config(Mode.DECODE, list(self.mesh_device.shape)[1], None)
        tt_all_reduce(xqkv_sharded, self.mesh_device, self.tt_ccl, cluster_axis=1,
                      memory_config=mem if mem is not None else xqkv_sharded.memory_config(),
                      sharded=True, dtype=self.ccl_dtype, topology=self.ccl_topology)
        xqkv = ttnn.sharded_to_interleaved(xqkv_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        ttnn.deallocate(xqkv_sharded)
        xqkv = ttnn.reshape(xqkv, (1, 1, B, xqkv.shape[3]), (1, 1, 32, xqkv.shape[3]))
        q_pre, k_lat_pad, v_lat_pad = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads,
            memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, None))
        ttnn.deallocate(xqkv)

        # RoPE on q only ---------------------------------------------------------------------------------
        q = ttnn.experimental.rotary_embedding_llama(q_pre, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True)
        ttnn.deallocate(q_pre)

        # latents -> caches -----------------------------------------------------------------------------
        keys, values = self.layer_past
        k_lat = self._slice_latent(k_lat_pad, self.r_k)
        v_lat = self._slice_latent(v_lat_pad, self.r_v)
        ttnn.experimental.paged_update_cache(keys, k_lat, update_idxs_tensor=current_pos)
        ttnn.experimental.paged_update_cache(values, v_lat, update_idxs_tensor=current_pos)
        ttnn.deallocate(k_lat)
        ttnn.deallocate(v_lat)

        # K reconstruction (the naive Palu path; stage 2 fuses this) -------------------------------------
        S = self._recon_len(current_pos)
        k_lat_S = keys[:, :, :S, :] if S < self.max_seq_len else keys                     # [B, nkv, S, r_k]
        v_lat_S = values[:, :, :S, :] if S < self.max_seq_len else values                 # [B, nkv, S, r_v]
        k_rec = ttnn.matmul(k_lat_S, self.u_k_decode, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=self.compute_kernel_config_hifi2)       # [B, nkv, S, 128]
        if S < self.max_seq_len:
            ttnn.deallocate(k_lat_S)
        k_rec = ttnn.reshape(k_rec, (1, B * self.n_local_kv_heads, S, self.head_dim))
        cos = self.cos_prefill[:, :, :S, :]
        sin = self.sin_prefill[:, :, :S, :]
        k_rot = ttnn.experimental.rotary_embedding_llama(k_rec, cos, sin, self.transformation_mats["prefill"], is_decode_mode=False)
        for t in (k_rec, cos, sin):
            ttnn.deallocate(t)
        k_rot = ttnn.reshape(k_rot, (B, self.n_local_kv_heads, S, self.head_dim))
        k_rot8 = ttnn.typecast(k_rot, dtype=self.kv_cache_dtype)                          # same precision as a stock K cache
        ttnn.deallocate(k_rot)

        # attention on reconstructed K and the V latent --------------------------------------------------
        attn = ttnn.transformer.flash_multi_latent_attention_decode(
            q, k_rot8, v_lat_S, head_dim_v=self.r_v, cur_pos_tensor=current_pos, scale=self.scale,
            program_config=self.args.get_attn_sdpa_decode_program_config(None),
            compute_kernel_config=self.sdpa_decode_compute_kernel_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, B, nh, r_v]
        ttnn.deallocate(q)
        ttnn.deallocate(k_rot8)
        if S < self.max_seq_len:
            ttnn.deallocate(v_lat_S)
        if self.r_v < self.head_dim:
            attn_p = ttnn.pad(attn, padding=[(0, 0), (0, 0), (0, 0), (0, self.head_dim - self.r_v)], value=0.0)
            ttnn.deallocate(attn)
            attn = attn_p

        # stock tail: concat heads, gather, fused wo, all-reduce -----------------------------------------
        attn_11BH = ttnn.to_memory_config(attn, memory_config=self.args.get_attn_sdpa_output_mem_config(Mode.DECODE, B, None))
        attn_cat = ttnn.experimental.nlp_concat_heads_decode(attn_11BH, num_heads=self.n_local_heads)
        ttnn.deallocate(attn_11BH)
        ttnn.deallocate(attn)
        attn_g = tt_all_gather(attn_cat, self.mesh_device, self.tt_ccl, dim=2, cluster_axis=1,
                               memory_config=self.args.get_attn_gather_users_mem_config(Mode.DECODE, list(self.mesh_device.shape)[1], None),
                               sharded=True)
        dense = ttnn.linear(attn_g, self.wo_fused, program_config=self.args.get_attn_wo_program_config(Mode.DECODE, 1, None),
                            memory_config=self.args.get_attn_wo_output_mem_config(Mode.DECODE, None),
                            compute_kernel_config=self.li_o_decode_compute_kernel_cfg)
        ttnn.deallocate(attn_cat)
        dense = tt_all_reduce(dense, self.mesh_device, self.tt_ccl, cluster_axis=0, dim=3, topology=self.ccl_topology,
                              memory_config=self.args.get_attn_all_reduce_output_mem_config(Mode.DECODE, self.hidden_size, list(self.mesh_device.shape)[0], None),
                              sharded=True, dtype=self.ccl_dtype, use_composite=False)
        return ttnn.to_memory_config(dense, self.args.get_attn_dense_output_mem_config(Mode.DECODE, None))
```

- [ ] **Step 4: Run the decode tests**

Run: `scripts/run_device.sh pytest tests/test_attention_device.py -q -k decode 2>&1 | tail -20`
Expected: `3 passed`.

Known failure points and the fix for each (record whichever applied in the commit message):
- `paged_update_cache` rejects the re-sharded `[32, r]` latent: pass the interleaved tensor instead (drop `interleaved_to_sharded` in `_slice_latent`).
- `flash_multi_latent_attention_decode` fails validation with 4 KV heads (GQA + MLA untested upstream) or gives low PCC while `identity` passes: this is the spec's first risk. Fallback: reconstruct V like K (`v_rec = matmul(v_lat_S, self.u_v_decode)`, add `self.u_v_decode` built like `u_k_decode` with `u_v`), then call stock `ttnn.transformer.scaled_dot_product_attention_decode(q, k_rot8, v_rec8, cur_pos_tensor=..., scale=..., program_config=..., compute_kernel_config=..., memory_config=DRAM)` and use the *original* `self.wo` for decode. Keep the MLA path behind `self.palu.use_mla_decode` (default True) so the bench can report both.
- `ttnn.matmul` batched `[B, 4, S, r] @ [B, 4, r, 128]` rejected: fall back to a per-head loop (`keys[:, h:h+1]` @ `u_k_decode[:1, h:h+1]`, then `ttnn.concat(..., dim=1)`), which uses the batch-broadcast form.
- `rotary_embedding_llama` prefill mode rejects `S` not a multiple of 32: already guaranteed by `nearest_32`.

- [ ] **Step 5: Run the whole attention file once more and commit**

Run: `scripts/run_device.sh pytest tests/test_attention_device.py -q 2>&1 | tail -5`
Expected: `6 passed`.

```bash
git add palu_tt/attention.py tests/test_attention_device.py
git commit -m "PaluAttention decode: latent caches, per-step K reconstruction, MLA decode on V latent"
```

---

### Task 7: Model builder and full-model parity/fidelity tests

**Files:**
- Create: `palu_tt/model.py`, `palu_tt/generate.py`, `tests/test_model_device.py`

**Interfaces:**
- Produces:
  - `model.build_model(mesh_device, kind, palu_dir=None, max_batch_size=1, max_seq_len=2048, n_layers=None, dtype=ttnn.bfloat8_b) -> (model_args, model, palu_cfg|None)`; `kind` in `{"stock", "palu"}`; `kind="palu", palu_dir=None` is the identity (r=128) row.
  - `model.set_recon_len(palu_cfg, current_pos: torch.Tensor)`.
  - `generate.greedy(model_args, model, palu_cfg, prompt_ids: list[int], max_new: int) -> list[int]` (batch-1 model).
  - `generate.prefill_last_logits(model_args, model, ids: torch.Tensor[1, L]) -> torch.Tensor[vocab]`, `generate.decode_step(model_args, model, palu_cfg, token: int, pos: int) -> torch.Tensor[vocab]`.
- Consumes: `PaluAttention`, `convert.add_palu_to_state_dict`, `reference.build_cpu_lowrank_llama`.

- [ ] **Step 1: Write the failing tests**

`tests/test_model_device.py`:
```python
import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from palu_tt import generate, hf, model as pm, reference

PROMPT = "The Tenstorrent Wormhole chip is"
NEW = 32


def _ids(model_args, text):
    return model_args.tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def _cpu_greedy(lm, ids, max_new):
    ids = torch.tensor([ids])
    with torch.no_grad():
        out = lm.generate(ids, max_new_tokens=max_new, do_sample=False)
    return out[0, ids.shape[1]:].tolist()


def test_identity_path_matches_stock_model(mesh_device):
    args_s, stock, _ = pm.build_model(mesh_device, "stock", max_batch_size=1, max_seq_len=2048)
    args_p, palu, cfg = pm.build_model(mesh_device, "palu", palu_dir=None, max_batch_size=1, max_seq_len=2048)
    ids = _ids(args_s, PROMPT)
    ls = generate.prefill_last_logits(args_s, stock, torch.tensor([ids]))
    lp = generate.prefill_last_logits(args_p, palu, torch.tensor([ids]))
    ok, msg = comp_pcc(ls, lp, 0.999)
    assert ok, msg
    ts = generate.greedy(args_s, stock, None, ids, NEW)
    tp = generate.greedy(args_p, palu, cfg, ids, NEW)
    logger.info(f"stock: {args_s.tokenizer.decode(ts)}\npalu-identity: {args_p.tokenizer.decode(tp)}")
    assert ts == tp


@pytest.mark.slow
def test_r64_model_tracks_cpu_lowrank_model(mesh_device):
    args, palu, cfg = pm.build_model(mesh_device, "palu", palu_dir="compressed/svd_r64", max_batch_size=1, max_seq_len=2048)
    ids = _ids(args, PROMPT)
    lm = reference.build_cpu_lowrank_llama(hf.snapshot_dir(), "compressed/svd_r64")
    with torch.no_grad():
        cpu_logits = lm(torch.tensor([ids])).logits[0, -1]
    tt_logits = generate.prefill_last_logits(args, palu, torch.tensor([ids]))
    ok, msg = comp_pcc(cpu_logits, tt_logits, 0.99)
    logger.info(msg)
    assert ok, msg
    cpu_tokens = _cpu_greedy(lm, ids, NEW)
    tt_tokens = generate.greedy(args, palu, cfg, ids, NEW)
    match = sum(a == b for a, b in zip(cpu_tokens, tt_tokens)) / NEW
    logger.info(f"cpu: {args.tokenizer.decode(cpu_tokens)}\ntt : {args.tokenizer.decode(tt_tokens)}\nmatch={match:.2f}")
    assert match >= 0.75  # same bar tt_transformers' own tests accept for bfp8 models
```
Add to `pytest.ini` at repo root:
```
[pytest]
markers =
    slow: needs the 32 GB CPU reference model
```

- [ ] **Step 2: Run to verify failure**

Run: `scripts/run_device.sh pytest tests/test_model_device.py -q -k identity 2>&1 | tail -5`
Expected: FAIL with `ModuleNotFoundError: No module named 'palu_tt.model'`.

- [ ] **Step 3: Implement palu_tt/model.py**

```python
from typing import Optional

import torch
import ttnn
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs

from palu_tt import convert, hf
from palu_tt.attention import PaluAttention


def build_model(mesh_device, kind: str, palu_dir: Optional[str] = None, max_batch_size: int = 1, max_seq_len: int = 2048,
                n_layers: Optional[int] = None, dtype=ttnn.bfloat8_b):
    assert kind in ("stock", "palu")
    args = ModelArgs(mesh_device, instruct=True, max_batch_size=max_batch_size, max_seq_len=max_seq_len, cache_hf=False)
    args.use_qk_fused = False          # same RoPE path for both kinds
    if n_layers is not None:
        args.n_layers = n_layers
    state_dict = args.load_state_dict()
    cfg = None
    if kind == "palu":
        cfg = convert.add_palu_to_state_dict(state_dict, hf.snapshot_dir(), palu_dir, args.n_layers,
                                             args.n_heads, args.n_kv_heads, args.head_dim)
        args.palu = cfg
    model = Transformer(args=args, mesh_device=mesh_device, dtype=dtype, state_dict=state_dict,
                        weight_cache_path=args.weight_cache_path(dtype),
                        attention_class=PaluAttention if kind == "palu" else None)
    return args, model, cfg


def set_recon_len(cfg: Optional[convert.PaluConfig], current_pos: torch.Tensor):
    if cfg is not None:
        cfg.runtime.recon_len = ((int(current_pos.max()) + 1 + 31) // 32) * 32


def kv_bytes_per_token_per_layer(model, args) -> float:
    """Per chip. Stock: K and V full head_dim; Palu: from the attention module."""
    attn = model.layers[0].attention
    if hasattr(attn, "kv_bytes_per_token"):
        return attn.kv_bytes_per_token()
    bpe = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0}[attn.kv_cache_dtype]
    return attn.n_local_kv_heads * 2 * attn.head_dim * bpe
```

- [ ] **Step 4: Implement palu_tt/generate.py**

```python
"""Greedy generation and single-step helpers on top of tt_transformers' Transformer (batch-1 models)."""
import torch
import ttnn
from models.tt_transformers.tt.common import copy_host_to_device

from palu_tt import model as pm


def _pad_to(ids: torch.Tensor, multiple: int) -> torch.Tensor:
    L = ids.shape[1]
    P = ((L + multiple - 1) // multiple) * multiple
    return torch.nn.functional.pad(ids, (0, P - L), value=0)


def prefill_last_logits(args, model, ids: torch.Tensor) -> torch.Tensor:
    """ids [1, L] -> logits of position L-1 (fills user 0's cache with positions < L)."""
    L = ids.shape[1]
    padded = _pad_to(ids, 128)
    x, rot_g, rot_l, *_ = model.prepare_inputs_prefill(padded, start_pos=0)
    tt_logits = model.ttnn_prefill_forward(x, rot_mats_global=rot_g, rot_mats_local=rot_l, user_id=0,
                                          get_last_token=((L - 1) // 32) * 32)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
    ttnn.deallocate(tt_logits)
    return out[0, 0, (L - 1) % 32, : args.vocab_size]


def prefill_hidden_states(args, model, ids: torch.Tensor) -> torch.Tensor:
    """ids [1, L] (L multiple of 128) -> pre-final-norm hidden states [L, dim]."""
    x, rot_g, rot_l, *_ = model.prepare_inputs_prefill(ids, start_pos=0)
    h = model.ttnn_prefill_forward(x, rot_mats_global=rot_g, rot_mats_local=rot_l, user_id=0, get_last_token=-1)
    out = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float()[0, 0, : ids.shape[1], : args.dim]
    ttnn.deallocate(h)
    return out


def decode_step(args, model, cfg, token: int, pos: int) -> torch.Tensor:
    B = args.max_batch_size
    tokens = torch.full((B,), token, dtype=torch.long)
    current_pos = torch.full((B,), pos, dtype=torch.long)
    pm.set_recon_len(cfg, current_pos)
    host = model.prepare_decode_inputs_host(tokens, current_pos)
    tok_d, pos_d, rope_d, _ = copy_host_to_device(host, mesh_device=model.mesh_device)
    logits, _ = model.ttnn_decode_forward(tok_d, pos_d, rot_mat_idxs=rope_d)
    out = model.process_output_decode(logits, B=B)[0, 0]
    ttnn.deallocate(logits)
    return out


def greedy(args, model, cfg, prompt_ids, max_new: int):
    ids = torch.tensor([prompt_ids])
    nxt = int(prefill_last_logits(args, model, ids).argmax())
    out = [nxt]
    pos = ids.shape[1]
    for _ in range(max_new - 1):
        nxt = int(decode_step(args, model, cfg, nxt, pos).argmax())
        out.append(nxt)
        pos += 1
    return out
```

- [ ] **Step 5: Run the identity test, then the slow test**

Run: `scripts/run_device.sh pytest tests/test_model_device.py -q -k identity 2>&1 | tail -8`
Expected: `1 passed`; the log shows identical stock and Palu-identity continuations.

Run: `scripts/run_device.sh nice -n 10 pytest tests/test_model_device.py -q -k r64 2>&1 | tail -8` (loads the 32 GB fp32 CPU model; ~10 min)
Expected: `1 passed`; log shows both continuations and `match=…` ≥ 0.75.

If `process_output_decode` returns a tensor with a leading batch dim of 32 rather than `B`, index `[0, 0]` still selects user 0; if `model.prepare_decode_inputs_host` asserts `B == max_batch_size`, the model was built with the wrong `max_batch_size` (must be 1 here).

- [ ] **Step 6: Commit**

```bash
git add palu_tt/model.py palu_tt/generate.py tests/test_model_device.py pytest.ini
git commit -m "Model builder with attention_class injection; greedy generation; parity and fidelity tests"
```

---

### Task 8: On-device perplexity harness

**Files:**
- Create: `palu_tt/eval_ppl.py`, `tests/test_ppl_harness.py`, `scripts/run_ppl_cpu.sh`

**Interfaces:**
- Produces: `eval_ppl.wikitext2_ids(tokenizer) -> torch.Tensor[1, N]`, `eval_ppl.device_ppl(args, model, hf_dir, ids, seq_len=2048, n_windows=None) -> dict(ppl, nll, tokens, windows)`, `eval_ppl.cpu_ppl(lm, ids, seq_len, n_windows) -> dict`, CLI writing `docs/results/raw/ppl_<row>.json`.
- Consumes: `generate.prefill_hidden_states`, `hf.load_tensors` (final norm + lm_head), `reference.build_cpu_lowrank_llama`.

- [ ] **Step 1: Write the failing test**

`tests/test_ppl_harness.py`:
```python
import torch

from palu_tt import eval_ppl, hf, model as pm


def test_device_ppl_two_windows_is_sane(mesh_device):
    args, model, _ = pm.build_model(mesh_device, "stock", max_batch_size=1, max_seq_len=2048)
    ids = eval_ppl.wikitext2_ids(args.tokenizer)
    assert ids.shape[1] > 200_000
    r = eval_ppl.device_ppl(args, model, hf.snapshot_dir(), ids, seq_len=2048, n_windows=2)
    assert r["windows"] == 2 and r["tokens"] == 2 * 2047
    assert 3.0 < r["ppl"] < 20.0, r   # Llama-3.1-8B-Instruct wikitext2 ppl is ~7-9 at 2k context
```

- [ ] **Step 2: Run to verify failure**

Run: `scripts/run_device.sh pytest tests/test_ppl_harness.py -q 2>&1 | tail -5`
Expected: FAIL with `ModuleNotFoundError: No module named 'palu_tt.eval_ppl'`.

- [ ] **Step 3: Implement palu_tt/eval_ppl.py**

```python
"""Wikitext-2 perplexity. Device rows: prefill 2048-token windows, all-position hidden states, final norm +
lm_head + NLL on host in fp32. CPU rows use the HF/low-rank model directly (sanity check only)."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset

from palu_tt import generate, hf

RAW = Path("docs/results/raw")


def wikitext2_ids(tokenizer) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids


def _head(hf_dir):
    t = hf.load_tensors(hf_dir, lambda k: k in ("model.norm.weight", "lm_head.weight"))
    eps = hf.load_config(hf_dir)["rms_norm_eps"]
    return t["model.norm.weight"], t["lm_head.weight"], eps


def _nll_from_hidden(h, norm_w, lm_w, eps, targets):
    """h [S, dim] pre-norm hidden states; targets [S-1] next tokens. Returns summed NLL over S-1 positions."""
    h = h.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
    nll = 0.0
    for s in range(0, h.shape[0] - 1, 512):            # chunk the [S, vocab] logits
        logits = h[s:min(s + 512, h.shape[0] - 1)] @ lm_w.T
        nll += torch.nn.functional.cross_entropy(logits, targets[s:s + logits.shape[0]], reduction="sum").item()
    return nll


def device_ppl(args, model, hf_dir, ids, seq_len=2048, n_windows=None):
    torch.set_num_threads(12)
    norm_w, lm_w, eps = _head(hf_dir)
    N = ids.shape[1]
    windows = N // seq_len if n_windows is None else min(n_windows, N // seq_len)
    nll, tokens, t0 = 0.0, 0, time.time()
    for w in range(windows):
        chunk = ids[:, w * seq_len:(w + 1) * seq_len]
        h = generate.prefill_hidden_states(args, model, chunk)
        nll += _nll_from_hidden(h, norm_w, lm_w, eps, chunk[0, 1:])
        tokens += seq_len - 1
        if w % 10 == 0:
            print(f"window {w}/{windows} running ppl {math.exp(nll / tokens):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {"ppl": math.exp(nll / tokens), "nll": nll, "tokens": tokens, "windows": windows, "seq_len": seq_len}


@torch.no_grad()
def cpu_ppl(lm, ids, seq_len=2048, n_windows=None):
    torch.set_num_threads(12)
    N = ids.shape[1]
    windows = N // seq_len if n_windows is None else min(n_windows, N // seq_len)
    nll, tokens = 0.0, 0
    for w in range(windows):
        chunk = ids[:, w * seq_len:(w + 1) * seq_len]
        logits = lm(chunk).logits[0, :-1]
        nll += torch.nn.functional.cross_entropy(logits, chunk[0, 1:], reduction="sum").item()
        tokens += seq_len - 1
    return {"ppl": math.exp(nll / tokens), "nll": nll, "tokens": tokens, "windows": windows, "seq_len": seq_len}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True, help="results row name, e.g. baseline, svd_r64")
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--windows", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--cpu", action="store_true", help="CPU HF/low-rank model instead of the device")
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.cpu:
        from palu_tt import reference
        from transformers import AutoTokenizer
        hf_dir = hf.snapshot_dir()
        tok = AutoTokenizer.from_pretrained(str(hf_dir))
        lm = reference.build_cpu_lowrank_llama(hf_dir, a.palu_dir if a.kind == "palu" else None)
        r = cpu_ppl(lm, wikitext2_ids(tok), a.seq_len, a.windows)
        out = RAW / f"ppl_cpu_{a.row}.json"
    else:
        import ttnn
        from palu_tt import device as dev, model as pm
        md = dev.open_n300()
        try:
            args, model, _ = pm.build_model(md, a.kind, a.palu_dir, max_batch_size=1, max_seq_len=a.seq_len)
            r = device_ppl(args, model, hf.snapshot_dir(), wikitext2_ids(args.tokenizer), a.seq_len, a.windows)
        finally:
            dev.close(md)
        out = RAW / f"ppl_{a.row}.json"
    r.update({"row": a.row, "kind": a.kind, "palu_dir": a.palu_dir})
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
```

`scripts/run_ppl_cpu.sh`:
```bash
#!/usr/bin/env bash
# CPU sanity perplexity (fp32 8B, ~1 window/min at 12 threads). Usage: run_ppl_cpu.sh <row> <kind> [palu_dir] [windows]
set -euo pipefail
source "$(dirname "$0")/env.sh"
row="$1"; kind="$2"; palu_dir="${3:-}"; windows="${4:-20}"
extra=(); [ -n "$palu_dir" ] && extra=(--palu-dir "$palu_dir")
nice -n 10 python -m palu_tt.eval_ppl --cpu --row "$row" --kind "$kind" --windows "$windows" "${extra[@]}"
```

- [ ] **Step 4: Run the test**

Run: `scripts/run_device.sh pytest tests/test_ppl_harness.py -q 2>&1 | tail -5`
Expected: `1 passed`.

If `prefill_hidden_states` returns a tensor whose last dim is `2 * dim` (concatenated replicas), the `get_device_tensors(...)[0]` indexing was bypassed; keep device 0 only. If the device runs out of memory at `get_last_token=-1` (hidden states are `[1,1,2048,4096]` bf16, ~16 MB, so it should not), lower `seq_len` to 1024 for all rows and record it.

- [ ] **Step 5: Baseline device vs CPU agreement (20 windows)**

```bash
scripts/run_device.sh python -m palu_tt.eval_ppl --row baseline --kind stock --windows 20
scripts/run_ppl_cpu.sh baseline stock "" 20
python - <<'PY'
import json; d=json.load(open("docs/results/raw/ppl_baseline.json")); c=json.load(open("docs/results/raw/ppl_cpu_baseline.json"))
print(d["ppl"], c["ppl"], abs(d["ppl"]-c["ppl"])/c["ppl"])
PY
```
Expected: relative difference under 5% (bfp8 weights on device vs fp32 on CPU). Record both numbers in the Task 10 doc.

- [ ] **Step 6: Commit**

```bash
git add palu_tt/eval_ppl.py tests/test_ppl_harness.py scripts/run_ppl_cpu.sh
git commit -m "On-device wikitext2 perplexity harness with CPU sanity path"
```

---

### Task 9: Benchmarks: decode step, attention-only, prefill, KV bytes, max context

**Files:**
- Create: `palu_tt/bench.py`

**Interfaces:**
- Produces CLI `python -m palu_tt.bench --row <name> --kind stock|palu [--palu-dir D] --what decode|attention|prefill|maxctx` writing `docs/results/raw/bench_<what>_<row>.json`.
- Consumes: `model.build_model`, `model.set_recon_len`, `model.kv_bytes_per_token_per_layer`, `PaluAttention`, stock `Attention`, `tests/test_attention_device.py::_setup` logic (re-implemented here, not imported).

- [ ] **Step 1: Implement palu_tt/bench.py**

```python
"""Timing and memory numbers for the results table. All device work is untraced (stage 1)."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup

from palu_tt import convert, device as dev, hf, model as pm
from palu_tt.attention import PaluAttention

RAW = Path("docs/results/raw")
CONTEXTS = (1024, 4096, 8192)
BATCH = 32


def _time(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"p50_ms": statistics.median(ts) * 1e3, "min_ms": min(ts) * 1e3, "iters": iters}


def bench_decode(md, kind, palu_dir):
    """Whole decode step at batch 32 for each context; the model is rebuilt per context (max_seq_len == context)."""
    rows = {}
    for ctx in CONTEXTS:
        try:
            args, model, cfg = pm.build_model(md, kind, palu_dir, max_batch_size=BATCH, max_seq_len=ctx)
        except Exception as e:  # allocation failure -> report and continue
            rows[ctx] = {"error": str(e)[:200]}
            continue
        tokens = torch.randint(0, args.vocab_size, (BATCH,))
        current_pos = torch.full((BATCH,), ctx - 1)
        pm.set_recon_len(cfg, current_pos)
        host = model.prepare_decode_inputs_host(tokens, current_pos)

        def step():
            tok_d, pos_d, rope_d, _ = copy_host_to_device(host, mesh_device=md)
            logits, _ = model.ttnn_decode_forward(tok_d, pos_d, rot_mat_idxs=rope_d)
            dev.sync(md)
            ttnn.deallocate(logits)

        rows[ctx] = _time(step)
        rows[ctx]["kv_bytes_per_token_per_layer_per_chip"] = pm.kv_bytes_per_token_per_layer(model, args)
        del model
    return rows


def bench_attention(md, kind, palu_dir):
    """One attention layer, decode, batch 32, per context. Isolates the K-reconstruction cost."""
    rows = {}
    for ctx in CONTEXTS:
        args = ModelArgs(md, instruct=True, max_batch_size=BATCH, max_seq_len=ctx, cache_hf=False)
        args.use_qk_fused = False
        args.n_layers = 1
        sd = args.load_state_dict()
        cls = Attention
        if kind == "palu":
            args.palu = convert.add_palu_to_state_dict(sd, hf.snapshot_dir(), palu_dir, 1, args.n_heads, args.n_kv_heads, args.head_dim)
            cls = PaluAttention
        rope = RotarySetup(md, BATCH, args.head_dim, ctx, args.rope_theta, args.rope_scaling, False)
        attn = cls(md, TT_CCL(md), args, sd, weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b), layer_num=0,
                   dtype=ttnn.bfloat8_b, transformation_mats=rope.get_both_trans_mats(), configuration=args)
        current_pos = torch.full((BATCH,), ctx - 1)
        if kind == "palu":
            pm.set_recon_len(args.palu, current_pos)
        pos_tt = ttnn.from_torch(current_pos, device=md, dtype=ttnn.int32,
                                 mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, None), mesh_shape=args.cluster_shape))
        rot = rope.get_rot_mats(current_pos)
        x = torch.randn(BATCH, 1, args.dim) * 0.05

        def step():
            x_tt = args.prepare_residual_tensor_decode(x, args.get_attn_input_mem_config(Mode.DECODE, None), force_replicated=True)
            y = attn.forward(x_tt, pos_tt, rot_mats=rot, mode=Mode.DECODE)
            dev.sync(md)
            ttnn.deallocate(y)

        rows[ctx] = _time(step, iters=20)
        del attn
    return rows


def bench_prefill(md, kind, palu_dir):
    rows = {}
    args, model, _ = pm.build_model(md, kind, palu_dir, max_batch_size=1, max_seq_len=8192)
    for L in (2048, 8192):
        ids = torch.randint(0, args.vocab_size, (1, L))

        def step():
            x, rot_g, rot_l, *_ = model.prepare_inputs_prefill(ids, start_pos=0)
            out = model.ttnn_prefill_forward(x, rot_mats_global=rot_g, rot_mats_local=rot_l, user_id=0, get_last_token=((L - 1) // 32) * 32)
            dev.sync(md)
            ttnn.deallocate(out)

        rows[L] = _time(step, iters=5, warmup=2)
    return rows


def bench_maxctx(md, kind, palu_dir):
    """Largest max_seq_len (power of two from 1k) at batch 32 that allocates."""
    ok = None
    for ctx in (1024, 2048, 4096, 8192, 16384, 32768):
        try:
            args, model, _ = pm.build_model(md, kind, palu_dir, max_batch_size=BATCH, max_seq_len=ctx)
            ok = ctx
            del model
        except Exception as e:
            return {"max_context_batch32": ok, "failed_at": ctx, "error": str(e)[:200]}
    return {"max_context_batch32": ok}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--what", choices=["decode", "attention", "prefill", "maxctx"], required=True)
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    md = dev.open_n300()
    try:
        r = {"decode": bench_decode, "attention": bench_attention, "prefill": bench_prefill, "maxctx": bench_maxctx}[a.what](md, a.kind, a.palu_dir)
    finally:
        dev.close(md)
    r = {"row": a.row, "kind": a.kind, "palu_dir": a.palu_dir, "what": a.what, "data": r}
    (RAW / f"bench_{a.what}_{a.row}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the attention benchmark on both kinds**

```bash
scripts/run_device.sh python -m palu_tt.bench --row baseline --kind stock --what attention
scripts/run_device.sh python -m palu_tt.bench --row svd_r64 --kind palu --palu-dir compressed/svd_r64 --what attention
cat docs/results/raw/bench_attention_*.json
```
Expected: JSON with `p50_ms` per context for both; Palu attention slower than stock at 4k/8k (reconstruction cost), the ratio growing with context. If `build_model` for `ctx=8192` fails with an allocation error, the row records `error` and the run continues.

- [ ] **Step 3: Commit**

```bash
git add palu_tt/bench.py
git commit -m "Benchmarks: decode step, attention-only, prefill, KV bytes, max-context probe"
```

---

### Task 10: Run the study rows and write the results document

**Files:**
- Create: `palu_tt/report.py`, `docs/results/stage1.md` (generated then hand-annotated)
- Modify: `docs/superpowers/specs/2026-09-09-palu-tt-stage1-design.md` (section 8 contexts line, wo_fused padding note, delta-file note)

**Interfaces:**
- Consumes every `docs/results/raw/*.json` produced by Tasks 8 and 9.

- [ ] **Step 1: Run all rows (device, sequential; each row is minutes to tens of minutes)**

```bash
for spec in "baseline stock" "identity palu" "svd_r96 palu compressed/svd_r96" "svd_r64 palu compressed/svd_r64" "svd_rk128_rv64 palu compressed/svd_rk128_rv64"; do
  set -- $spec; row=$1; kind=$2; pd=${3:-}
  extra=(); [ -n "$pd" ] && extra=(--palu-dir "$pd")
  scripts/run_device.sh python -m palu_tt.eval_ppl --row $row --kind $kind "${extra[@]}"
  for what in decode attention prefill maxctx; do
    scripts/run_device.sh python -m palu_tt.bench --row $row --kind $kind --what $what "${extra[@]}"
  done
done
# CPU sanity for two rows (20 windows each, niced)
scripts/run_ppl_cpu.sh baseline stock "" 20
scripts/run_ppl_cpu.sh svd_r64 palu compressed/svd_r64 20
# generation samples
scripts/run_device.sh python - <<'PY'
import json, torch
from palu_tt import device as dev, generate, model as pm
prompts = ["The Tenstorrent Wormhole chip is", "Explain KV-cache compression in two sentences.", "Write a haiku about matrices."]
md = dev.open_n300(); out = {}
for row, kind, pd in [("baseline","stock",None),("svd_r64","palu","compressed/svd_r64"),("svd_r96","palu","compressed/svd_r96")]:
    args, model, cfg = pm.build_model(md, kind, pd, max_batch_size=1, max_seq_len=2048)
    out[row] = {p: args.tokenizer.decode(generate.greedy(args, model, cfg, args.tokenizer(p).input_ids, 48)) for p in prompts}
    del model
dev.close(md)
json.dump(out, open("docs/results/raw/samples.json","w"), indent=1)
PY
```
Expected: `docs/results/raw/` contains `ppl_*.json`, `bench_{decode,attention,prefill,maxctx}_*.json`, `ppl_cpu_*.json`, `samples.json`.

- [ ] **Step 2: Implement palu_tt/report.py**

```python
"""Render docs/results/stage1.md from docs/results/raw/*.json."""
import json
import subprocess
from pathlib import Path

RAW = Path("docs/results/raw")
ROWS = ["baseline", "identity", "svd_r96", "svd_r64", "svd_rk128_rv64"]
CONTEXTS = ("1024", "4096", "8192")


def _load(name):
    p = RAW / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    lines = ["# Stage 1 results: Llama-3.1-8B-Instruct vs Palu(Llama-3.1-8B-Instruct) on Wormhole n300", "",
             f"Commit `{commit}`, tt-metal v0.78.0, TP=2, bfp8 weights and KV, non-paged, untraced decode.", ""]
    lines += ["| row | r_k | r_v | ppl (device) | ppl (cpu) | KV B/token/layer/chip | max ctx @b32 | decode p50 ms 1k/4k/8k | attn p50 ms 1k/4k/8k | TTFT ms 2k/8k |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in ROWS:
        ppl, cpu = _load(f"ppl_{row}"), _load(f"ppl_cpu_{row}")
        dec, att, pre, mx = _load(f"bench_decode_{row}"), _load(f"bench_attention_{row}"), _load(f"bench_prefill_{row}"), _load(f"bench_maxctx_{row}")
        cfg = json.loads(Path(ppl["palu_dir"], "config.json").read_text())["palu_tt"] if ppl and ppl.get("palu_dir") else {}
        rk, rv = cfg.get("rank_k", 128), cfg.get("rank_v", 128)
        kvb = next((d.get("kv_bytes_per_token_per_layer_per_chip") for d in dec["data"].values() if "p50_ms" in d), None) if dec else None
        f = lambda b, k: "/".join(_fmt(b["data"].get(c, {}).get(k), 1) if b else "-" for c in CONTEXTS)
        ttft = "/".join(_fmt(pre["data"].get(L, {}).get("p50_ms"), 0) if pre else "-" for L in ("2048", "8192"))
        lines.append(f"| {row} | {rk} | {rv} | {_fmt(ppl and ppl['ppl'])} | {_fmt(cpu and cpu['ppl'])} | {_fmt(kvb, 0)} | "
                     f"{mx['data'].get('max_context_batch32') if mx else '-'} | {f(dec, 'p50_ms')} | {f(att, 'p50_ms')} | {ttft} |")
    s = _load("samples")
    if s:
        lines += ["", "## Generation samples (greedy, 48 tokens)", ""]
        for prompt in next(iter(s.values())):
            lines.append(f"**{prompt}**\n")
            for row, d in s.items():
                lines.append(f"- {row}: {d[prompt].strip()}")
            lines.append("")
    Path("docs/results/stage1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Render, annotate, update the spec**

```bash
python -m palu_tt.report
```
Then append to `docs/results/stage1.md` by hand:
- which fallback (if any) from Task 6 is active (MLA decode on V latent vs V reconstruction), with the reason;
- the device-vs-CPU perplexity agreement for `baseline` and `svd_r64`;
- one paragraph interpreting decode latency: V-only row vs baseline (should be within noise) and the full-compression rows (reconstruction cost, the stage 2 target);
- device state at the time of the runs (`tt-smi` version, driver, other users logged in).

Edit the spec: section 8 contexts to "1k/4k/8k plus the max-context probe", section 4's `wo_fused` sentence to mention zero padding to 4096 x 4096, and the export note about the delta file.

- [ ] **Step 4: Commit**

```bash
git add palu_tt/report.py docs/results/stage1.md docs/superpowers/specs/2026-09-09-palu-tt-stage1-design.md
git commit -m "Stage 1 results table and spec deviations"
```

---

### Task 11 (background, independent of Tasks 5–10): Palu full calibration recipe on CPU

**Files:**
- Create: `scripts/patch_palu_cpu.sh`, `scripts/run_calibration.sh`

**Interfaces:**
- Produces: `compressed/palu_full_<ratio>/` — a full Palu HF dump readable by `convert.load_palu_dir` (same key names; `config.json` carries `head_wise_ranks`). Then Tasks 8–10 can be re-run for the row `palu_full_50` with `--palu-dir compressed/palu_full_0.5`.

- [ ] **Step 1: scripts/patch_palu_cpu.sh** (idempotent sed patches on the submodule working tree; never committed into the submodule)

```bash
#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)/third_party/palu"
cd "$P"
git checkout -- palu/decomposition.py palu/rank_search.py utils.py
# decomposition.py: three .cuda() sites -> stay on the working device
sed -i 's/subset\[name\].scaling_diag_matrix.double().cuda()/subset[name].scaling_diag_matrix.double().to(dev)/' palu/decomposition.py
sed -i 's/torch.eye(raw_scaling_diag_matrix.shape\[0\]).cuda()/torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)/' palu/decomposition.py
sed -i 's/torch.eye(scaling_diag_matrix.shape\[0\]).cuda()/torch.eye(scaling_diag_matrix.shape[0]).to(dev)/' palu/decomposition.py
# rank_search.py: fp32 on CPU, grads only for attention linears (memory), no .half()
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("palu/rank_search.py"); s = p.read_text()
s = s.replace("    model.half()\n    model.to(device)\n",
              "    model.float() if device.type == 'cpu' else model.half()\n    model.to(device)\n"
              "    for n, p_ in model.named_parameters():\n        p_.requires_grad_('attn' in n)\n")
p.write_text(s)
p = pathlib.Path("utils.py"); s = p.read_text()
s = s.replace('torch_dtype=torch.float16,', 'torch_dtype=torch.float32,').replace('device_map="auto",', '')
p.write_text(s)
PY
grep -n "cuda()" palu/*.py utils.py compress.py || echo "no .cuda() left"
```

- [ ] **Step 2: scripts/run_calibration.sh**

```bash
#!/usr/bin/env bash
# Full Palu recipe (Fisher-uniform rank search + whitening) on CPU fp32. Hours; niced. Usage: run_calibration.sh 0.5
set -euo pipefail
source "$(dirname "$0")/env.sh"
ratio="${1:-0.5}"
scripts/patch_palu_cpu.sh
uv pip install loguru click lm-eval==0.4.2 2>/dev/null || true
cd third_party/palu
HF_DIR="$(python -c 'from palu_tt import hf; print(hf.snapshot_dir())')"
mkdir -p cache
nice -n 10 python compress.py --model_id "$HF_DIR" --device cpu --calib_dataset wikitext2 --calib_seqlen 1024 \
  --param_ratio_target "$ratio" --search_method fisher_uniform --decompose_method whiten --head_group_size 1 \
  --use_cache --dump_huggingface_model 2>&1 | tee "../../compressed/calibration_${ratio}.log"
out="$(ls -d Llama-3.1-8B-Instruct_ratio-${ratio}_gs-1-fisher_uniform-whiten 2>/dev/null | head -1)"
mv "$out" "../../compressed/palu_full_${ratio}"
echo "done -> compressed/palu_full_${ratio}"
```
Notes: `fisher_uniform` gives one rank per layer shared by all heads, which `convert.load_palu_dir` accepts directly. `compress.py` imports `run_lm_eval`, which needs `lm_eval`; if the pip install fails, stub the import by creating `third_party/palu/run_lm_eval.py` with `def run_lm_eval_zero_shot(*a, **k): raise NotImplementedError` (working-tree only). The dump is a full fp32 model (~32 GB on disk) — fine, 394 GB free; only the k/v keys are read.

- [ ] **Step 3: Launch in the background and check memory**

```bash
nohup scripts/run_calibration.sh 0.5 > compressed/calibration_0.5.nohup 2>&1 &
sleep 600; free -g | head -2; tail -3 compressed/calibration_0.5.log
```
Expected: memory under ~90 GB during Fisher; the log advancing through 32 Fisher samples, then whitening layer by layer. When it finishes, validate:
```bash
python - <<'PY'
from palu_tt import convert
lrs = convert.load_palu_dir("compressed/palu_full_0.5", 32, 8, 128)
print([lrs[i].r_k for i in range(32)]); print([lrs[i].r_v for i in range(32)])
PY
```
Expected: per-layer ranks, multiples of 32, averaging about 64.

- [ ] **Step 4: Add the calibrated row and commit**

Re-run Task 10 Step 1's loop for `"palu_full_50 palu compressed/palu_full_0.5"` and `python -m palu_tt.report` (add `palu_full_50` to `ROWS`).
```bash
git add scripts/patch_palu_cpu.sh scripts/run_calibration.sh palu_tt/report.py docs/results/stage1.md
git commit -m "Palu full calibration recipe on CPU; calibrated row in results"
```

---

## Self-review

**Spec coverage.** Environment and shared-server rules → Task 1. Compression and export, delta format, loader for Palu dumps, rank padding → Tasks 2, 3, 11. Device data flow (padded latents, V via MLA op, K reconstruction, prefill reconstruction, caches) → Tasks 5, 6. Loading and injection via `attention_class` → Task 7. Tests 1–4 of spec section 7 → Tasks 2–4 (CPU), 5–6 (attention PCC, identity parity), 7 (model fidelity, baseline parity). Evaluation table (device perplexity for every row, CPU sanity, KV bytes, max context, decode/prefill timing, samples) → Tasks 8, 9, 10. Fallbacks for the two spec risks → Task 6 Step 4. Success criteria → Task 10 annotations.

**Placeholder scan.** No TBD/TODO. Every code step has full code; fallbacks are spelled out with the exact call to switch to.

**Type consistency.** `PaluConfig.rank_k/rank_v` are per-layer lists everywhere; `configuration.palu` is read by `PaluAttention` and set by `build_model` and the attention test/bench. `PaluRuntime.recon_len` is set through `pm.set_recon_len` (model/generate/bench) or directly in tests. State-dict key names `layers.{i}.attention.palu.{wk_lat_pad,wv_lat_pad,u_k,u_v,wo_fused}` match between `convert.add_palu_to_state_dict` and `PaluAttention.__init__`. `LowRank` field names `k_VT/k_U/v_VT/v_U` match across convert, reference and tests. `generate.prefill_hidden_states` (Task 7) is what `eval_ppl.device_ppl` (Task 8) consumes.
