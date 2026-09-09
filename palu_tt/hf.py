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
