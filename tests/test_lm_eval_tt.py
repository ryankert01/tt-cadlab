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
