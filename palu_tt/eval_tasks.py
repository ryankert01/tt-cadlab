"""Zero-shot task accuracy on the device (the paper's protocol, five of its six tasks).

Usage: python -m palu_tt.eval_tasks --row glrd_50 --kind palu --palu-dir compressed/glrd_0.5 [--fused p2] [--limit 20]
Writes docs/results/raw/tasks_<row>.json. First run needs the datasets downloaded once (README).
"""
import argparse, json, os, time
from pathlib import Path

import lm_eval
from lm_eval.tasks import TaskManager

from palu_tt import hf, lm_eval_tt

RAW = Path("docs/results/raw")
TASKS = ("openbookqa", "piqa", "arc_easy", "arc_challenge", "winogrande")
# lm-eval 0.4.2's configs point at dataset ids the Hub no longer serves; the *_hub tasks override them
TASK_NAME_MAP = {"openbookqa_hub": "openbookqa", "piqa_hub": "piqa", "winogrande_hub": "winogrande"}
TASK_DIR = Path(__file__).parent / "lm_eval_tasks"


def task_manager():
    return TaskManager(include_path=str(TASK_DIR))


def lm_eval_names(tasks):
    inv = {v: k for k, v in TASK_NAME_MAP.items()}
    return [inv.get(t, t) for t in tasks]


def run(lm, tasks, limit=None):
    t0 = time.time()
    res = lm_eval.simple_evaluate(model=lm, tasks=lm_eval_names(tasks), num_fewshot=0, limit=limit,
                                  task_manager=task_manager(), log_samples=False, bootstrap_iters=0)
    per_task = {}
    for name, r in res["results"].items():
        std = TASK_NAME_MAP.get(name, name)
        per_task[std] = {"acc": r.get("acc,none"), "acc_norm": r.get("acc_norm,none"),
                         "n": res.get("n-samples", {}).get(name, {}).get("effective")}
    return {"tasks": per_task, "results": res["results"], "versions": res.get("versions"), "configs": res.get("configs"),
            "seconds": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--fused", choices=["off", "p1", "p2"], default="off")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    from palu_tt import device as dev, model as pm
    RAW.mkdir(parents=True, exist_ok=True)
    md = dev.open_n300()
    try:
        args, model, _ = pm.build_model(md, a.kind, a.palu_dir, max_batch_size=1, max_seq_len=2048,
                                        use_fused_k=a.fused != "off", cs_recurrence=a.fused == "p2")
        lm = lm_eval_tt.TTLM.for_device(args, model, hf.snapshot_dir())
        r = run(lm, [t for t in a.tasks.split(",") if t], a.limit)
    finally:
        dev.close(md)
    import importlib.metadata as m
    r.update({"row": a.row, "kind": a.kind, "palu_dir": a.palu_dir, "fused": a.fused, "limit": a.limit,
              "lm_eval_version": m.version("lm_eval")})
    (RAW / f"tasks_{a.row}.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r["tasks"]))


if __name__ == "__main__":
    main()
