"""Audit check: does palu_tt.lm_eval_tt.TTLM score exactly like lm-eval 0.4.2's HFLM on the same model?
CPU fp32 HF model; TTLM gets a score_fn built from that model; HFLM is run with add_bos_token True and False.
Compares per-choice log-likelihoods and accuracies on a limited slice of three tasks."""
import json, sys, time, torch, lm_eval
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
from palu_tt import hf, lm_eval_tt, eval_tasks

OUT = sys.argv[1]; LIMIT = int(sys.argv[2]); TASKS = sys.argv[3].split(",")
torch.set_num_threads(12)
hf_dir = hf.snapshot_dir()
tok = AutoTokenizer.from_pretrained(str(hf_dir))
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(str(hf_dir), torch_dtype=torch.float32).eval()
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

@torch.no_grad()
def score(ids, n_last):
    logits = model(ids).logits[0]            # [L, vocab] fp32
    return logits[-(n_last + 1):-1]          # the rows TTLM.for_device returns (pre-log_softmax)

lms = {
    "ttlm_cpu": lm_eval_tt.TTLM.from_parts(tok, score),
    "hflm_bos": HFLM(pretrained=model, tokenizer=tok, add_bos_token=True, batch_size=1),
    "hflm_nobos": HFLM(pretrained=model, tokenizer=tok, add_bos_token=False, batch_size=1),
}
tm = eval_tasks.task_manager()
names = eval_tasks.lm_eval_names(TASKS)
results = {}
for lm_name, lm in lms.items():
    for task in names:
        t1 = time.time()
        res = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=0, limit=LIMIT, task_manager=tm,
                                      log_samples=True, bootstrap_iters=0)
        r = res["results"][task]
        samples = [{"doc_id": s["doc_id"], "resps": s["filtered_resps"], "acc": s.get("acc"), "acc_norm": s.get("acc_norm")}
                   for s in res["samples"][task]]
        results.setdefault(task, {})[lm_name] = {"acc": r.get("acc,none"), "acc_norm": r.get("acc_norm,none"),
                                                  "seconds": time.time() - t1, "samples": samples}
        print(f"{lm_name:11s} {task:16s} acc {r.get('acc,none'):.4f} acc_norm {r.get('acc_norm,none')} ({time.time()-t1:.0f}s)", flush=True)
        json.dump(results, open(OUT, "w"), indent=1, default=str)

# per-choice comparison ttlm vs hflm_bos (should be identical up to fp32 noise) and vs hflm_nobos
def flat(entry):
    out = []
    for s in entry["samples"]:
        for r in s["resps"]:
            ll = r[0] if isinstance(r, (list, tuple)) else r
            out.append(float(ll[0]) if isinstance(ll, (list, tuple)) else float(ll))
    return out
summary = {}
for task, by_lm in results.items():
    a = flat(by_lm["ttlm_cpu"]); b = flat(by_lm["hflm_bos"]); c = flat(by_lm["hflm_nobos"])
    n = min(len(a), len(b), len(c))
    d_bos = [abs(x - y) for x, y in zip(a[:n], b[:n])]; d_nobos = [abs(x - y) for x, y in zip(a[:n], c[:n])]
    summary[task] = {"n_choices": n, "max_abs_diff_vs_hflm_bos": max(d_bos), "mean_abs_diff_vs_hflm_bos": sum(d_bos) / n,
                     "max_abs_diff_vs_hflm_nobos": max(d_nobos),
                     "acc": {k: v["acc"] for k, v in by_lm.items()}, "acc_norm": {k: v["acc_norm"] for k, v in by_lm.items()}}
results["_summary"] = summary
json.dump(results, open(OUT, "w"), indent=1, default=str)
print("SUMMARY", json.dumps(summary, indent=1))
print("ADAPTER EQUIV FINISHED")
