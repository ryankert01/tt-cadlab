"""ttnn.fill_cache into a bfp4 KV cache (stage 2e / 2f): the stock op only whitelisted fp32 / bf16 / bfp8.

fill_cache is a dataflow tile copy (reader + writer, no compute kernel) whose page size follows the tensor's
tile size, so a bfp4 input into a bfp4 cache must land byte-for-byte; the palu-sdpa branch relaxes the
dtype check for exactly that pairing. Runs on the N300 mesh fixture; chip 0 is read back."""
import pytest
import torch
import ttnn


def _dev(md, t, dtype):
    return ttnn.from_torch(t, device=md, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(md))


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


@pytest.mark.parametrize("width", [64, 128])
def test_fill_cache_accepts_bfp4_and_copies_the_tiles(mesh_device, width):
    S, B, NKV = 2048, 8, 4
    cache = _dev(mesh_device, torch.zeros(B, NKV, S, width), ttnn.bfloat4_b)
    lat = _dev(mesh_device, torch.randn(1, NKV, S, width), ttnn.bfloat4_b)
    ttnn.fill_cache(cache, lat, 3)
    got = _host(cache)
    want = _host(lat)[0]
    assert torch.equal(got[3], want)                       # the filled user is the input's bfp4 tiles, exactly
    assert torch.equal(got[2], torch.zeros_like(got[2]))   # neighbours untouched
    assert torch.equal(got[4], torch.zeros_like(got[4]))


def test_fill_cache_still_rejects_a_bfp4_input_into_a_bfp8_cache(mesh_device):
    cache = _dev(mesh_device, torch.zeros(2, 4, 256, 64), ttnn.bfloat8_b)
    lat = _dev(mesh_device, torch.randn(1, 4, 256, 64), ttnn.bfloat4_b)
    with pytest.raises(RuntimeError, match="dtype"):
        ttnn.fill_cache(cache, lat, 0)
