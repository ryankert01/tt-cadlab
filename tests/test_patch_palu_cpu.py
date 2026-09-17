"""Host-only pin for ``scripts/patch_rank_search_split.py``, the split-budget patch
``scripts/patch_palu_cpu.sh`` applies to ``palu/rank_search.py``'s ``fisher_uniform`` branch
(2f Task 6's conditional third grid; review Important 2).

Everything here runs against a pristine copy of ``third_party/palu``'s *committed*
``rank_search.py`` (``git -C third_party/palu show HEAD:palu/rank_search.py``), copied into
``tmp_path`` -- never against the submodule's live working tree, which ``patch_palu_cpu.sh`` has
already patched for CPU calibration, and nothing here writes under ``third_party/``. Skipped
outright if ``third_party/palu`` is not checked out (e.g. a shallow clone with no submodules).

Three things are pinned:
  (a) applying the patch a second time is a fixed point (byte-identical to the first application);
  (b) with neither ``--k_ratio_target`` nor ``--v_ratio_target``, the patched ``fisher_uniform``
      allocation block executes identically to the pristine one over synthetic Fisher data -- same
      ``select_result``, same ``total_rank``;
  (c) with both ratios set, the split path hits each matrix family's target rank exactly.

(b) and (c) exec the branch body directly (the same technique the patch's own anchors rely on:
bracketing text between ``HEAD`` and ``TAIL``) against lightweight fakes, rather than driving the
whole ``rank_search()`` function -- there is no need to stub ``get_calib_data``/``calib_fisher_info``
or build a fake HF model when only the allocation arithmetic is under test.
"""
import math
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import patch_rank_search_split as prs  # noqa: E402


def _pristine_rank_search_source():
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT / "third_party" / "palu"), "show", "HEAD:palu/rank_search.py"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout:
        pytest.skip(f"third_party/palu not checked out ({result.stderr.strip() or 'empty output'})")
    return result.stdout


@pytest.fixture(scope="module")
def pristine_source():
    return _pristine_rank_search_source()


def test_patch_applies_to_a_pristine_copy_and_a_second_application_is_a_fixed_point(tmp_path, pristine_source):
    copy = tmp_path / "rank_search.py"
    copy.write_text(pristine_source)

    prs.main([str(copy)])
    once = copy.read_text()
    assert "k_ratio_target" in once and "v_ratio_target" in once
    compile(once, str(copy), "exec")  # still valid Python

    prs.main([str(copy)])
    twice = copy.read_text()
    assert twice == once, "a second application of the patch changed the file -- not a fixed point"


# ---- (b)/(c): exec the fisher_uniform allocation block over synthetic Fisher data -------------

N_LAYERS, NUM_LR_GROUPS, LR_GROUP_DIMS = 32, 1, 1024
# One family's total rank across every layer -- matches the model's real group shape (project
# review: num_lr_groups 1, lr_group_dims 1024, 32 layers) and reproduces the exact figures the
# implementer's calibration logs recorded (8192/32768 at ratio 0.25, 24576/32768 at 0.75, etc).
PER_FAMILY_TOTAL = N_LAYERS * NUM_LR_GROUPS * LR_GROUP_DIMS


class _Info:
    num_lr_groups = NUM_LR_GROUPS
    lr_group_dims = LR_GROUP_DIMS


class _TargetModelClass:
    @staticmethod
    def get_kv_info(model, x):
        return _Info()


class _Module:
    """Just enough of a k_proj/v_proj Linear for the allocation block: ``in_features`` and a
    ``fisher_info`` tensor shaped so ``.reshape(num_lr_groups, -1, in_features)`` succeeds."""

    def __init__(self, fisher_value, in_features=4096):
        import torch
        self.in_features = in_features
        self.fisher_info = torch.full((NUM_LR_GROUPS, in_features), float(fisher_value))


def _fake_model(seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    modules = []
    for layer in range(N_LAYERS):
        for name, scale in (("k_proj", 1.0), ("v_proj", 3.0)):  # V's Fisher is larger, as measured
            value = (1.0 + torch.rand(1, generator=g).item()) * scale
            modules.append((f"model.layers.{layer}.self_attn.{name}", _Module(value)))
    return types.SimpleNamespace(
        named_modules=lambda: iter(modules),
        config=types.SimpleNamespace(num_key_value_heads=8),
    )


def _extract_block(source, head, tail):
    """The allocation block between ``head`` and ``tail``, dedented so it execs as top-level code."""
    i, j = source.index(head), source.index(tail)
    return textwrap.dedent(source[i:j])


def _run_block(body, args, seed=0):
    import torch
    g = {
        "torch": torch, "math": math,
        "target_model_class": _TargetModelClass(),
        "model": _fake_model(seed),
        "args": args,
        "total_rank": 0,
        "fisher_sum": 0.0,
        "fisher_info_dict": {},
        "select_result": {},
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
    }
    exec(compile(body, "<fisher_uniform-block>", "exec"), g)
    return g["select_result"], g["total_rank"]


def _family_sum(select_result, family):
    return sum(sum(v) for name, v in select_result.items() if family in name)


@pytest.fixture(scope="module")
def patched_body(pristine_source, tmp_path_factory):
    copy = tmp_path_factory.mktemp("patch_body") / "rank_search.py"
    copy.write_text(pristine_source)
    prs.main([str(copy)])
    return _extract_block(copy.read_text(), prs.HEAD, prs.TAIL)


@pytest.fixture(scope="module")
def pristine_body(pristine_source):
    return _extract_block(pristine_source, prs.HEAD, prs.TAIL)


def test_no_flags_path_is_byte_identical_to_pristine_upstream(pristine_body, patched_body):
    args = types.SimpleNamespace(param_ratio_target=0.5)  # no k_ratio_target/v_ratio_target at all
    for seed in range(4):
        pristine_sel, pristine_total = _run_block(pristine_body, args, seed=seed)
        patched_sel, patched_total = _run_block(patched_body, args, seed=seed)
        assert patched_sel == pristine_sel, f"select_result diverged at seed {seed}"
        assert patched_total == pristine_total, f"total_rank diverged at seed {seed}"


def test_split_path_hits_each_familys_target_exactly(patched_body):
    args = types.SimpleNamespace(param_ratio_target=0.5, k_ratio_target=0.25, v_ratio_target=0.75)
    select_result, total_rank = _run_block(patched_body, args, seed=1)

    assert total_rank == 2 * PER_FAMILY_TOTAL
    assert _family_sum(select_result, "k_proj") == 0.25 * PER_FAMILY_TOTAL == 8192
    assert _family_sum(select_result, "v_proj") == 0.75 * PER_FAMILY_TOTAL == 24576

    args2 = types.SimpleNamespace(param_ratio_target=0.5, k_ratio_target=0.5, v_ratio_target=0.75)
    select_result2, _ = _run_block(patched_body, args2, seed=1)
    assert _family_sum(select_result2, "k_proj") == 0.5 * PER_FAMILY_TOTAL == 16384
    assert _family_sum(select_result2, "v_proj") == 0.75 * PER_FAMILY_TOTAL == 24576


def test_one_flag_only_falls_back_to_param_ratio_target_for_the_other_family(patched_body):
    """``getattr(args, ..., None)`` is the whole point of the flags being optional: setting only
    ``--k_ratio_target`` must still run v_proj's own pass, at ``--param_ratio_target``, rather than
    silently pooling it back in with k_proj."""
    args = types.SimpleNamespace(param_ratio_target=0.5, k_ratio_target=0.25)
    select_result, total_rank = _run_block(patched_body, args, seed=1)
    assert total_rank == 2 * PER_FAMILY_TOTAL
    assert _family_sum(select_result, "k_proj") == 0.25 * PER_FAMILY_TOTAL == 8192
    assert _family_sum(select_result, "v_proj") == 0.5 * PER_FAMILY_TOTAL == 16384


def test_patch_script_runs_from_the_repo_root_by_relative_path(repo_root=None):
    """scripts/patch_palu_cpu.sh must work when invoked as `scripts/patch_palu_cpu.sh` (the way every
    calibration script calls it): it changes into third_party/palu before locating its helper file, so the
    helper path has to be resolved first. Regression for the 2f Task 6 fix-wave refactor."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if not (root / "third_party" / "palu" / "compress.py").exists():
        import pytest
        pytest.skip("third_party/palu is not checked out")
    r = subprocess.run(["bash", "scripts/patch_palu_cpu.sh"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert "applied cleanly" in r.stdout
