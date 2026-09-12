"""Cache-name tag construction (needs ttnn on the host, but never opens the device)."""
import pytest

pytest.importorskip("ttnn", reason="ttnn not importable on this host")

from palu_tt.attention import cache_tag
from palu_tt.convert import CONVERT_VERSION


def test_tag_carries_cache_id_and_ranks():
    assert cache_tag("identity", 128, 128) == "palu_v4_identity_rk128_rv128"
    assert cache_tag("palu_full_0.5-3fa2b1c0", 32, 128) == "palu_v4_palu_full_0.5-3fa2b1c0_rk32_rv128"


def test_group_size_widens_the_tag_but_leaves_group_size_one_alone():
    """A grouped dump's u_*/latent tensors have different shapes *and* different content from a
    per-head dump's at the same ranks, so they must never share weight-cache entries."""
    assert cache_tag("glrd_0.5-1234abcd", 64, 512, 1) == cache_tag("glrd_0.5-1234abcd", 64, 512)
    assert cache_tag("glrd_0.5-1234abcd", 64, 512, 4) == "palu_v4_glrd_0.5-1234abcd_rk64_rv512_gs4"
    assert cache_tag("d", 32, 64, 4) != cache_tag("d", 32, 64)


def test_tag_carries_the_converter_version():
    """A converter fix that changes tensor *content* without changing shape/dtype/weight set (the
    v1 -> v2 full-rank-V collapse) must not be served the old cache entries."""
    assert CONVERT_VERSION >= 2
    for tag in (cache_tag("identity", 128, 128), cache_tag("dump-deadbeef", 32, 128)):
        assert tag.startswith(f"palu_v{CONVERT_VERSION}_"), tag


def test_same_ranks_different_weight_sets_get_different_tags():
    """The bug this guards: two weight sets with identical per-layer ranks used to share cache files."""
    assert cache_tag("palu_full_0.5-aaaaaaaa", 32, 128) != cache_tag("svd_rk32_rv128-bbbbbbbb", 32, 128)


def test_empty_cache_id_is_rejected():
    for bad in ("", None, 3):
        with pytest.raises(AssertionError):
            cache_tag(bad, 32, 32)
