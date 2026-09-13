"""Zero-shot task accuracy on the device (the paper's protocol, five of its six tasks).

Usage: python -m palu_tt.eval_tasks --row glrd_50 --kind palu --palu-dir compressed/glrd_0.5 [--fused p2] [--limit 20]
Writes docs/results/raw/tasks_<row>.json. First run needs the datasets downloaded once (README).
"""
import argparse, hashlib, importlib.util, json, os, time
from pathlib import Path

import lm_eval
from lm_eval.tasks import TaskManager

from palu_tt import hf, lm_eval_tt

RAW = Path("docs/results/raw")
TASKS = ("openbookqa", "piqa", "arc_easy", "arc_challenge", "winogrande")
# lm-eval 0.4.2's configs point at dataset ids the Hub no longer serves; the *_hub tasks override them
TASK_NAME_MAP = {"openbookqa_hub": "openbookqa", "piqa_hub": "piqa", "winogrande_hub": "winogrande"}
# Committed templates: the `include:` path depends on where lm-eval is installed (a venv under a
# worktree that may not exist elsewhere), so it is never committed literally -- write_task_dir()
# substitutes it in at runtime, into a cache directory outside the repo tree, one per lm-eval install
# (two checkouts with their own venvs render different paths; a shared directory would race).
TEMPLATE_DIR = Path(__file__).parent / "lm_eval_tasks"
CACHE_ROOT = Path.home() / ".cache" / "palu_tt" / "lm_eval_tasks"


def _lm_eval_tasks_root():
    """Directory holding the installed lm-eval package's built-in task configs."""
    spec = importlib.util.find_spec("lm_eval")
    return Path(spec.submodule_search_locations[0]) / "tasks"


def write_task_dir():
    """Render the *_hub.yaml templates (their `include:` uses a {LM_EVAL_TASKS} placeholder) with
    the installed lm-eval location, into a per-user cache dir outside the repo tree. Re-renders only
    when the content actually changes, so it is safe (and fast) to call on every task_manager()."""
    tasks_root = _lm_eval_tasks_root()
    cache_dir = CACHE_ROOT / hashlib.sha1(str(tasks_root).encode()).hexdigest()[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    for template in sorted(TEMPLATE_DIR.glob("*.yaml")):
        rendered = template.read_text().replace("{LM_EVAL_TASKS}", str(tasks_root))
        out = cache_dir / template.name
        if not out.exists() or out.read_text() != rendered:
            tmp = out.with_suffix(f".{os.getpid()}.tmp")   # atomic replace: a concurrent reader never sees a partial file
            tmp.write_text(rendered)
            os.replace(tmp, out)
    return cache_dir


def task_manager():
    return TaskManager(include_path=str(write_task_dir()))


def lm_eval_names(tasks):
    inv = {v: k for k, v in TASK_NAME_MAP.items()}
    return [inv.get(t, t) for t in tasks]


def _effective_n(task, limit):
    """Fallback doc count when lm-eval's result dict doesn't report a per-task n (0.4.2 has no
    'n-samples' key): the number of docs actually evaluated, i.e. all eval_docs clipped to --limit."""
    n_docs = len(list(task.eval_docs))
    return min(limit, n_docs) if limit is not None else n_docs


def run(lm, tasks, limit=None):
    t0 = time.time()
    names = lm_eval_names(tasks)
    tm = task_manager()
    res = lm_eval.simple_evaluate(model=lm, tasks=names, num_fewshot=0, limit=limit,
                                  task_manager=tm, log_samples=False, bootstrap_iters=0)
    task_dict = None
    per_task = {}
    for name, r in res["results"].items():
        std = TASK_NAME_MAP.get(name, name)
        n = res.get("n-samples", {}).get(name, {}).get("effective")
        if n is None:
            if task_dict is None:
                task_dict = lm_eval.tasks.get_task_dict(names, tm)
            task = task_dict.get(name)
            n = _effective_n(task, limit) if task is not None else None
        per_task[std] = {"acc": r.get("acc,none"), "acc_norm": r.get("acc_norm,none"), "n": n}
    return {"tasks": per_task, "results": res["results"], "versions": res.get("versions"), "configs": res.get("configs"),
            "seconds": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--fused", choices=["off", "p1", "p2"], default="off")
    p.add_argument("--group-heads", choices=["auto", "1", "2", "4"], default="auto",
                   help="stage-2c group-shared mode: KV heads per core in the fused op (auto = the "
                        "largest that fits L1 per layer, 1 = the stage 2b per-head kernel). lm-eval "
                        "scores these tasks by scoring whole continuations in prefill, so the decode "
                        "kernel -- and therefore this flag -- does not affect the numbers here.")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--latent-scale", choices=["balanced", "unit"], default="balanced",
                   help="stage 2f: 'unit' moves the singular values into the up-projection so every latent "
                        "component has unit variance (what bfp4 caches want); 'balanced' is Palu's split.")
    a = p.parse_args()
    from palu_tt import device as dev, model as pm
    RAW.mkdir(parents=True, exist_ok=True)
    md = dev.open_n300()
    try:
        args, model, _ = pm.build_model(md, a.kind, a.palu_dir, max_batch_size=1, max_seq_len=2048,
                                        use_fused_k=a.fused != "off", cs_recurrence=a.fused == "p2",
                                        group_heads=None if a.group_heads == "auto" else int(a.group_heads),
                                        latent_scale=a.latent_scale)
        lm = lm_eval_tt.TTLM.for_device(args, model, hf.snapshot_dir())
        r = run(lm, [t for t in a.tasks.split(",") if t], a.limit)
    finally:
        dev.close(md)
    import importlib.metadata as m
    r.update({"row": a.row, "kind": a.kind, "palu_dir": a.palu_dir, "fused": a.fused,
              "group_heads": a.group_heads, "limit": a.limit, "latent_scale": a.latent_scale,
              "lm_eval_version": m.version("lm_eval")})
    (RAW / f"tasks_{a.row}.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r["tasks"]))


if __name__ == "__main__":
    main()
