"""lm-evaluation-harness (0.4.2) model adapter for the device: log-likelihoods through prefill.

A multiple-choice task asks for log P(continuation | context) for each candidate. We tokenise with the
model tokenizer (lm-eval's TemplateLM splits context/continuation for us), run the whole sequence
through device prefill (`generate.prefill_hidden_states`), and apply the final RMSNorm and lm_head on
the host (the perplexity harness's path) -- but only over the trailing rows that predict the
continuation tokens, since those are the only ones ever read. One sequence per prefill; identical
(sequence, n_last) pairs within a run are memoised. Only `loglikelihood` is implemented -- the five
tasks of the study need neither generation nor rolling likelihood.
"""
from typing import Callable, List, Tuple

import torch
import transformers
from lm_eval.api.model import TemplateLM


class TTLM(TemplateLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM   # selects TemplateLM's causal context/continuation split

    def __init__(self, tokenizer, score_fn: Callable[[torch.Tensor, int], torch.Tensor], max_length: int = 2048):
        super().__init__()
        self.tokenizer = tokenizer
        self._score = score_fn            # ids [1, L], n_last -> logits [n_last, vocab] (fp32, host)
        self._max_length = max_length
        self._memo = {}

    @classmethod
    def from_parts(cls, tokenizer, score_fn, max_length=2048):
        return cls(tokenizer, score_fn, max_length)

    @classmethod
    def for_device(cls, args, model, hf_dir, max_length=2048):
        from palu_tt import eval_ppl, generate   # imports ttnn; kept lazy so CPU-only callers never load it
        torch.set_num_threads(12)   # matches eval_ppl.py's host lm_head/log_softmax convention
        norm_w, lm_w, eps = eval_ppl._head(hf_dir)
        def score(ids, n_last):
            L = ids.shape[1]
            padded = torch.nn.functional.pad(ids, (0, (-L) % 128), value=0)   # prefill wants a multiple of 128
            h = generate.prefill_hidden_states(args, model, padded)[:L]         # [L, dim]
            # row j predicts ids[j+1]; the last row (predicting one past the sequence) is never read,
            # so the n_last rows we need are the n_last before it: indices L-1-n_last .. L-2.
            h = h[-(n_last + 1):-1]                                            # [n_last, dim]
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
            return h @ lm_w.T                                                   # [n_last, vocab]
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
        for (context, _continuation), context_enc, continuation_enc in requests:
            # TemplateLM.loglikelihood substitutes context_enc = [eot_token_id] for an empty context (a
            # sentinel "start of sequence" marker, not real context); we supply our own BOS below, so drop
            # that sentinel here rather than double up two special tokens in front of the continuation.
            ctx_enc = [] if context == "" else context_enc
            # Length cap: the slice below drops context from the left once ctx+continuation exceeds
            # max_length (as lm-eval's HF model does); a continuation longer than max_length on its own
            # is clipped from the left too (lm-eval's HF model asserts instead), so the row slice below
            # can never wrap. None of the five tasks comes near the cap.
            n_cont = min(len(continuation_enc), self._max_length)
            whole = (ctx_enc + continuation_enc)[-(self._max_length):]
            key = (tuple(whole), n_cont)                 # n_cont must be in the key: two requests can
            if key not in self._memo:                    # share `whole` while splitting context/continuation differently
                ids = torch.tensor([[self.prefix_token_id] + list(whole)])     # BOS + sequence
                logp = torch.log_softmax(self._score(ids, n_cont).float(), dim=-1)  # [n_cont, vocab]
                self._memo[key] = logp
            logp = self._memo[key]
            # the continuation tokens are the last n_cont of `whole`; score_fn already returns exactly
            # the n_cont rows that predict them (see TTLM.for_device's `score`)
            tgt = torch.tensor(whole[-n_cont:])
            ll = float(logp.gather(1, tgt[:, None]).sum())
            greedy = bool((logp.argmax(-1) == tgt).all())
            out.append((ll, greedy))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("palu_tt.lm_eval_tt: rolling log-likelihood is not needed by the study's tasks")

    def generate_until(self, requests):
        raise NotImplementedError("palu_tt.lm_eval_tt: generation is not needed by the study's tasks")
