#!/usr/bin/env python3
"""``palu/rank_search.py``'s split-budget patch (stage 2f Task 6's conditional third grid),
factored out of ``scripts/patch_palu_cpu.sh`` into its own file so
``tests/test_patch_palu_cpu.py`` can import ``patch_source`` and run it against a pristine copy of
``rank_search.py`` (``git -C third_party/palu show HEAD:palu/rank_search.py``) without mutating
``third_party/palu`` -- see that test for the pinned behaviour (the no-flags path is byte-identical
to upstream's own ``fisher_uniform``; the split path hits each matrix family's target exactly).

Upstream's ``fisher_uniform`` branch pools every ``k_proj`` and ``v_proj`` module into one
Fisher-weighted budget, so the K/V byte split is whatever Fisher happens to pick. This patch adds
``--k_ratio_target``/``--v_ratio_target``: with neither set, there is exactly one pass over both
families at ``--param_ratio_target`` and the arithmetic is upstream's, unchanged; with either set,
the allocation runs once per matrix family, each restricted to its own module set with its own
total rank, its own Fisher sum and its own ratio. ``dif`` (the greedy top-up remainder) is summed
over the current pass's modules only, since ``select_result`` also holds the previous pass's
entries.

Run standalone, from within ``third_party/palu`` (as ``scripts/patch_palu_cpu.sh`` does):
    python3 patch_rank_search_split.py [path/to/rank_search.py]
"""
import sys
from pathlib import Path

# Anchors: the fisher_uniform branch is the only one that calls get_kv_info with
# num_key_value_heads (the "fisher"/"uniform" branches pass args.head_group_size) and the only one
# that calls split_values, so this pair brackets its allocation block and nothing else.
HEAD = "        info = target_model_class.get_kv_info(model, model.config.num_key_value_heads)\n"
TAIL = "        select_result = split_values(select_result, model.config.num_key_value_heads//args.head_group_size)\n"

# NEW_BLOCK's own text starts with HEAD and never contains TAIL, so splicing it in place of the
# pristine [HEAD, TAIL) span leaves both anchors present exactly once in the result -- applying the
# patch a second time finds the same two anchors at the same relative position and reproduces the
# same text: a fixed point (pinned by tests/test_patch_palu_cpu.py).
NEW_BLOCK = '''        info = target_model_class.get_kv_info(model, model.config.num_key_value_heads)

        k_ratio = getattr(args, "k_ratio_target", None)
        v_ratio = getattr(args, "v_ratio_target", None)
        if k_ratio is None and v_ratio is None:
            passes = [(("k_proj", "v_proj"), args.param_ratio_target)]
        else:
            passes = [(("k_proj",), args.param_ratio_target if k_ratio is None else k_ratio),
                      (("v_proj",), args.param_ratio_target if v_ratio is None else v_ratio)]

        for matrices, ratio in passes:
            pass_total_rank = 0
            fisher_sum = 0.0
            fisher_info_dict = {}
            for name, module in model.named_modules():
                if any(m in name for m in matrices):
                    module_rank = info.num_lr_groups * info.lr_group_dims
                    pass_total_rank += module_rank
                    total_rank += module_rank

                    select_result.update({name: [info.lr_group_dims] * info.num_lr_groups})
                    fisher = module.fisher_info.reshape(info.num_lr_groups, -1, module.in_features)

                    fisher_list = [torch.mean(fisher[i]).item() for i in range(info.num_lr_groups)]
                    fisher_info_dict.update({name: fisher_list})
                    fisher_sum += sum(fisher_list)

            target_rank = pass_total_rank * ratio

            indexes = []
            select_result_float = {}

            for name, fisher in fisher_info_dict.items():
                ranks = []
                for i in range(len(fisher)):
                    rank_float = target_rank * fisher[i] / fisher_sum
                    ranks.append(rank_float)
                    indexes.append((name, i))
                    select_result[name][i] = min(select_result[name][i], math.floor(rank_float))

                select_result_float.update({name: ranks})

            indexes = sorted(indexes, key=lambda x: select_result_float[x[0]][x[1]] - select_result[x[0]][x[1]])
            # this pass's modules only -- select_result also holds the previous pass's entries
            dif = target_rank - sum(sum(select_result[name]) for name in fisher_info_dict)

            while dif > 0:
                for i in range(len(indexes)):
                    if select_result[indexes[i][0]][indexes[i][1]] == info.lr_group_dims:
                        continue
                    select_result[indexes[i][0]][indexes[i][1]] += 1
                    dif -= 1

                    if dif == 0:
                        break

            logger.info(f"[Rank Search] {'/'.join(matrices)} target ratio {ratio}: "
                        f"{sum(sum(select_result[name]) for name in fisher_info_dict)} "
                        f"of {pass_total_rank} rank")

'''


def patch_source(s: str) -> str:
    """Return ``s`` (``palu/rank_search.py``'s text) with the split-budget block spliced in place
    of the pooled ``fisher_uniform`` allocation between ``HEAD`` and ``TAIL``."""
    assert s.count(HEAD) == 1, "anchor missing/ambiguous: fisher_uniform get_kv_info"
    assert s.count(TAIL) == 1, "anchor missing/ambiguous: fisher_uniform split_values"
    i, j = s.index(HEAD), s.index(TAIL)
    return s[:i] + NEW_BLOCK + s[j:]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else Path("palu/rank_search.py")
    path.write_text(patch_source(path.read_text()))


if __name__ == "__main__":
    main()
