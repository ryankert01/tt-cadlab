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
