from pathlib import Path

import torch
from lm_eval.api.instance import Instance

from palu_tt import lm_eval_tt, eval_tasks


class _Tok:
    """A toy tokenizer: one token per character, BOS = 1, EOS = 2, vocab 128."""
    bos_token_id, eos_token_id = 1, 2
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 128 for c in s]


def _score_fn(ids, n_last):
    """Deterministic logits for exactly the n_last rows the real score_fn contract returns: the model
    strongly predicts token (prev + 1) % 128 at every position. Row j of the output predicts ids[j+1],
    so the n_last rows needed are the n_last before the last row of `ids` (see TTLM.for_device)."""
    L = ids.shape[1]
    logits = torch.full((n_last, 128), -10.0)
    for row, t in enumerate(range(L - 1 - n_last, L - 1)):
        logits[row, (int(ids[0, t]) + 1) % 128] = 10.0
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


def test_loglikelihood_clips_overlong_continuation_from_the_left():
    # "abcd" (4 tokens) as the continuation is longer than max_length=3, so it must be clipped from
    # the left (matching lm-eval's HF model) to its last 3 tokens -- the same effective window as
    # scoring "bcd" directly -- rather than let n_cont exceed len(whole) and wrap the row slice.
    lm_long = lm_eval_tt.TTLM.from_parts(_Tok(), _score_fn, max_length=3)
    lm_clipped = lm_eval_tt.TTLM.from_parts(_Tok(), _score_fn, max_length=3)
    (lp_long, greedy_long), = lm_long.loglikelihood(
        [Instance(request_type="loglikelihood", doc={}, arguments=("xy", "abcd"), idx=0)])
    (lp_clipped, greedy_clipped), = lm_clipped.loglikelihood(
        [Instance(request_type="loglikelihood", doc={}, arguments=("x", "bcd"), idx=1)])
    assert lp_long == lp_clipped and greedy_long == greedy_clipped


def test_task_name_map_and_task_list():
    assert set(eval_tasks.TASK_NAME_MAP.values()) <= set(eval_tasks.TASKS)
    assert eval_tasks.TASKS == ("openbookqa", "piqa", "arc_easy", "arc_challenge", "winogrande")


def test_lm_eval_names_maps_hub_overrides():
    assert eval_tasks.lm_eval_names(["openbookqa", "arc_easy"]) == ["openbookqa_hub", "arc_easy"]


class _FakeTask:
    def __init__(self, n_docs):
        self.eval_docs = list(range(n_docs))


def test_effective_n_uses_eval_docs_clipped_to_limit():
    assert eval_tasks._effective_n(_FakeTask(500), None) == 500
    assert eval_tasks._effective_n(_FakeTask(500), 40) == 40
    assert eval_tasks._effective_n(_FakeTask(10), 40) == 10


def test_task_yaml_templates_have_no_hardcoded_worktree_path():
    """The committed templates carry an {LM_EVAL_TASKS} placeholder instead of a literal install
    path, so the branch works when checked out into a different worktree or venv location."""
    for path in eval_tasks.TEMPLATE_DIR.glob("*.yaml"):
        text = path.read_text()
        assert "{LM_EVAL_TASKS}" in text
        assert "worktrees" not in text
        assert "site-packages" not in text


def test_write_task_dir_renders_includes_that_exist_on_disk():
    """write_task_dir() substitutes the installed lm-eval location for the placeholder, into a cache
    dir outside the repo tree; the resulting include: path must resolve to a real file."""
    out_dir = eval_tasks.write_task_dir()
    repo_root = Path(__file__).resolve().parents[1]
    assert repo_root not in out_dir.parents and out_dir != repo_root
    for name in ("openbookqa_hub.yaml", "piqa_hub.yaml", "winogrande_hub.yaml"):
        text = (out_dir / name).read_text()
        include_line = next(line for line in text.splitlines() if line.startswith("include:"))
        include_path = Path(include_line.split(":", 1)[1].strip())
        assert include_path.is_file(), include_path
        assert "{LM_EVAL_TASKS}" not in str(include_path)
