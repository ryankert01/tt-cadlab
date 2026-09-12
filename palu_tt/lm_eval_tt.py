"""lm-evaluation-harness (0.4.2) model adapter for the device: log-likelihoods through prefill.

A multiple-choice task asks for log P(continuation | context) for each candidate. We tokenise with the
model tokenizer (lm-eval's TemplateLM splits context/continuation for us), run the whole sequence
through device prefill (`generate.prefill_hidden_states`), apply the final RMSNorm and lm_head on the
host (the perplexity harness's path), and sum the continuation tokens' log-probabilities. One
sequence per prefill; identical sequences within a run are memoised. Only `loglikelihood` is
implemented -- the five tasks of the study need neither generation nor rolling likelihood.
"""
from typing import Callable, List, Tuple

import torch
import transformers
from lm_eval.api.model import TemplateLM

from palu_tt import eval_ppl, generate


class TTLM(TemplateLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM   # selects TemplateLM's causal context/continuation split

    def __init__(self, tokenizer, score_fn: Callable[[torch.Tensor], torch.Tensor], max_length: int = 2048):
        super().__init__()
        self.tokenizer = tokenizer
        self._score = score_fn            # ids [1, L] -> logits [L, vocab] (fp32, host)
        self._max_length = max_length
        self._memo = {}

    @classmethod
    def from_parts(cls, tokenizer, score_fn, max_length=2048):
        return cls(tokenizer, score_fn, max_length)

    @classmethod
    def for_device(cls, args, model, hf_dir, max_length=2048):
        norm_w, lm_w, eps = eval_ppl._head(hf_dir)
        def score(ids):
            L = ids.shape[1]
            padded = torch.nn.functional.pad(ids, (0, (-L) % 128), value=0)   # prefill wants a multiple of 128
            h = generate.prefill_hidden_states(args, model, padded)[:L]         # [L, dim]
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
            return h @ lm_w.T
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
            whole = (ctx_enc + continuation_enc)[-(self._max_length):]
            n_cont = len(continuation_enc)
            key = tuple(whole)
            if key not in self._memo:
                ids = torch.tensor([[self.prefix_token_id] + list(whole)])     # BOS + sequence
                logits = self._score(ids).float()                               # [1 + len(whole), vocab]
                self._memo[key] = torch.log_softmax(logits, dim=-1)
            logp = self._memo[key]
            # position t predicts token t+1: the continuation tokens are the last n_cont of `whole`
            tgt = torch.tensor(whole[-n_cont:])
            rows = logp[len(whole) - n_cont: len(whole)]                        # predictions for those tokens
            ll = float(rows.gather(1, tgt[:, None]).sum())
            greedy = bool((rows.argmax(-1) == tgt).all())
            out.append((ll, greedy))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("palu_tt.lm_eval_tt: rolling log-likelihood is not needed by the study's tasks")

    def generate_until(self, requests):
        raise NotImplementedError("palu_tt.lm_eval_tt: generation is not needed by the study's tasks")
