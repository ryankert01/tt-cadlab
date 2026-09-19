# Audit B: task evaluation

Repo: /home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9 @ 40b9a15. lm_eval 0.4.2 at
`.venv/lib/python3.10/site-packages/lm_eval` (abbreviated `LE/` below). Nothing was executed on the
device; CPU checks were the allowed pytest file, the HF tokenizer, and lm-eval's task registry /
request builder driven by a dummy score function.

## Scope read

- palu_tt/lm_eval_tt.py: 1-93 (all).
- palu_tt/eval_tasks.py: 1-156 (all).
- palu_tt/lm_eval_tasks/{openbookqa,piqa,winogrande}_hub.yaml: all 4 lines each.
- tests/test_lm_eval_tt.py: 1-96 (all); tests/conftest.py: 1-22; grep of tests/ and scripts/ for lm_eval/TTLM/for_device/eval_tasks.
- palu_tt/generate.py: 28-37 (`prefill_hidden_states`); palu_tt/eval_ppl.py: 26-52 (`_head`, `_nll_from_hidden`, `device_ppl` head); palu_tt/hf.py: 1-43; palu_tt/__init__.py, palu_tt/device.py imports.
- palu_tt/report.py: 380-445 (`_task_display`, `_avg5`, `_avg5_paper`), 828-870 (stage-2b table), 1165-1181 and 1267-1278 (`_row_is_blind`), 1355-1375 (frontier `avg5`). Rest of report.py not read.
- LE/api/model.py: 1-60, 250-339 (TemplateLM: `_encode_pair`, `loglikelihood`).
- LE/models/huggingface.py: 94-101 (ctor defaults), 255-280, 355-375, 660-682 (`tok_encode`), 720-750 (`_model_call`), 775-797 (`_select_cont_toks`), 874-1088 (`_loglikelihood_tokens`, all). `Collator` itself not read (only its use).
- LE/api/task.py: 636-657 (`eval_docs`, `doc_iterator`), 836-880 (`multiple_input` detection, whitespace check), 925-995 (`fewshot_context`), 1109-1135 (`construct_requests`), 1168-1270 (`process_results`).
- LE/evaluator.py: 117-128, 150-215, 355-440, 455-470; LE/evaluator_utils.py: 62, 94-109, 145-150; LE/utils.py: `load_yaml_config` include-merge lines 331-355.
- LE/tasks/openbookqa/openbookqa.yaml, piqa/piqa.yaml, winogrande/default.yaml + preprocess_winogrande.py, arc/arc_easy.yaml, arc/arc_challenge.yaml: all.
- third_party/tt-metal/models/tt_transformers/tt/model.py: 406-475 (`prepare_inputs_prefill`), 625-660 (`concat_host_output`), 694-740 (`ttnn_prefill_forward`), 935-1035 (`forward`). model_config.py: 3776-3850 (`create_tokenizer`).
- third_party/palu/run_lm_eval.py: 1-50.
- docs/report_palu_on_tt.md: 1-19 (Setup, protocols) and 40-59 (table, quality paragraph); rest skimmed by grep only.
- docs/results/stage2b.md: 1-27, 95-135, 186-200 in full; the rest by grep. docs/results/stage2c.md: 728-795 in full, rest by grep. docs/results/stage2f.md: grep only (task-number transcription).
- docs/superpowers/specs/2026-09-12-palu-tt-stage2b-design.md: 12-28, 46-75, 170-180; rest by grep. README.md: 10-18, 135-162, 285-300; rest by grep.
- docs/remote/goal.md: 235-260; docs/remote/results/gpu_report_2026-09-18.md: 1-30 (rank dumps skimmed, quality table read).
- Raw: all 14 docs/results/raw/tasks_*.json (every top-level field, `tasks`, `results`, `configs` of tasks_baseline in full); tasks_baseline.log head/tail/grep; `latent_quant_in_prefill`/`kv_dtype`/`latent_dtype` fields of 7 ppl_*.json.
- Not read: palu_tt/model.py, attention.py, fused.py (out of area; the device prefill's internal correctness is assumed from the perplexity harness sharing `prefill_hidden_states`); dataset contents beyond document counts.

## Findings

### [Important] Adapter prepends BOS to every request; Palu's harness and the GPU cross-check score with no BOS, and no document says so

- File:line: palu_tt/lm_eval_tt.py:53-55 (`prefix_token_id` = `bos_token_id` = 128000), :79 (`ids = [prefix] + whole`); LE/models/huggingface.py:101 (`add_bos_token=False` default), :663-680 (`tok_encode` -> `add_special_tokens=False or self.add_bos_token`), :964-969 (input = `(ctx+cont)[-(max_length+1):][:-1]`, no prefix); third_party/palu/run_lm_eval.py:17 (`HFLM(..., add_bos_token=False, ...)`); docs/remote/goal.md:243-245 (the GPU 64.25 was produced with Palu's `run_lm_eval.py`).
- What: `TTLM` is numerically equivalent to `HFLM(add_bos_token=True)`: every scored sequence is `<|begin_of_text|>` + context + continuation (confirmed on CPU with the real tokenizer: ids[0] = 128000 for all five tasks, e.g. ARC-c request L=38 where HFLM would feed 36 tokens with no BOS). Palu's reference script, which the paper's Table 1 and the GPU comparison run both use, scores with no BOS. The BOS choice is mentioned only as a design note in the spec (§2 "BOS once"); report §1, stage2b, stage2c §9.3 and README never state it.
- Why it matters: (1) stage2c §9.3 reports the n300 stock row 1.19 points above the fp16 GPU run on the same model, higher on all five tasks (OBQA +2.6, ARC-c +2.1) and files it under "the two harnesses ... nothing below about 1.5 points separates" and "the Palu numbers of this study are the paper's numbers for this model to within the port cost". A bfp8 device beating fp16 on every task is the signature of a deterministic protocol difference, not of port cost or noise, and the code identifies one. (2) stage2b §1 last paragraph infers "this is not a task-harness artefact ... the harness, the datasets and the device scoring path are healthy" from the stock row sitting 0.84 above the paper's (no-BOS, Llama-3) baseline; the +0.84 is not evidence of anything once the BOS difference is on the table. (3) README:15 "scores every row on five zero-shot lm-evaluation-harness tasks the way the paper does" and report §1's "the five-task means of the two harnesses differ by at most 1.5 points in either direction" both present an unexplained offset as harness noise. Within-study comparisons (all n300 rows share the protocol) are unaffected; every cross-harness and paper comparison inherits an unquantified systematic offset of order one point.
- How to verify: `grep -n add_bos_token third_party/palu/run_lm_eval.py .venv/lib/python3.10/site-packages/lm_eval/models/huggingface.py | head`; `sed -n 53,55p\;79p palu_tt/lm_eval_tt.py`. Cheapest empirical check: OBQA alone (2000 sequences, ~5 min of device time) with the prefix disabled; if it lands near the GPU's 33.8 rather than 36.4 the mechanism is confirmed.
- Suggested fix: add an `add_bos_token` argument to `TTLM` (and `--bos/--no-bos` on `eval_tasks`), record it in the JSON, state it in report §1 and stage2c §9.3, and either re-score the stock row without BOS (52 min) or re-run the GPU baseline with `add_bos_token=True`; reword the "paper's protocol" / "not a harness artefact" sentences accordingly.

### [Important] The device numeric path of the adapter has no test; the spec's device smoke test was never written

- File:line: palu_tt/lm_eval_tt.py:37-45 (`for_device.score`: `h[-(n_last+1):-1]`, host RMSNorm, `h @ lm_w.T`); tests/test_lm_eval_tt.py:16-24, 28, 46-47 (only `from_parts` with a synthetic score function); docs/superpowers/specs/2026-09-12-palu-tt-stage2b-design.md:177-178 (promised "a 20-example lm-eval run on arc_easy for stock asserting accuracy in a plausible range and the JSON schema").
- What: `grep -rn "for_device\|TTLM\|eval_tasks" tests/` finds nothing outside the CPU file; no device test imports the adapter. The CPU test's `_score_fn` re-implements the same "row j predicts ids[j+1]" contract that `for_device.score` assumes, so the two can only agree by construction; the slice in `for_device.score`, the norm/lm_head, and `eval_tasks.run()` (the `acc`/`acc_norm`/`n` extraction that feeds every table) never execute under test.
- Why it matters: I verified the arithmetic by reading (rows `L-1-n .. L-2` of a position-aligned pre-norm residual predict `ids[L-n .. L-1]` = `whole[-n:]`; `forward` returns `x` before the norm when `get_last_token == -1`, model.py:1010-1011; `prepare_inputs_prefill` embeds the tokens as given, model.py:430-452), and the plausibility of 65.44 against the GPU's 64.25 rules out a gross shift. But the only guard on every accuracy number in the study is that reading; a future edit to `score` (or to `prefill_hidden_states`, which the perplexity harness shares) would not be caught by `tests/`.
- How to verify: `grep -rn "for_device\|TTLM" tests/`.
- Suggested fix: (a) CPU: `eval_tasks.run(TTLM.from_parts(<real tokenizer>, <deterministic score>), ["arc_easy"], limit=3)` end to end, asserting the JSON schema and `tasks[t]["acc"] == results[name]["acc,none"]`; (b) device, gated: for three real requests compare `TTLM.loglikelihood` against a full-sequence host reference (`prefill_hidden_states` over the whole sequence, norm + lm_head on all rows, gather) to 1e-3, plus the spec's 20-document plausibility check.

### [Minor] No uncertainty anywhere: `bootstrap_iters=0` leaves `acc_stderr,none: []`, and the docs headline sub-noise differences

- File:line: palu_tt/eval_tasks.py:69-70; LE/evaluator_utils.py:62 (`agg_metrics = defaultdict(list)`) and :100-109 (stderr skipped when `bootstrap_iters` is 0, so the key materialises as an empty list when read); docs/results/stage2b.md:138-139, docs/results/stage2c.md:112, 745-746.
- What: binomial SE per task at the observed rates: OBQA (n=500) 2.15 pt, ARC-c (1172) 1.46, WinoGrande (1267) 1.23, PIQA (1838) 0.93, ARC-e (2376) 0.78; five-task mean about 0.6 pt. stage2b calls out "the only compressed row here that beats stock on any task (OpenBookQA acc 36.8 vs 36.4, +0.40)" (two documents of 500); stage2c reports "-0.16 and -0.02 points" and "0.233 points below the stock model" as results. stage2c:452 does mention "~0.5-point bootstrap noise" once.
- Why it matters: readers of the external report cannot tell a 0.4-point cell from noise; the raw files carry nothing to compute it from.
- How to verify: `python3 -c "import math;print(math.sqrt(.364*.636/500)*100)"` -> 2.15.
- Suggested fix: `bootstrap_iters=1000` (cost negligible), render `+-` in the tables, or paired tests from per-document outputs (next finding).

### [Minor] `log_samples=False`: no per-document predictions survive a 52-minute run

- File:line: palu_tt/eval_tasks.py:70.
- What: only aggregates are written; the per-document `acc` vectors that would allow paired comparisons between rows, or an audit of which documents moved, are discarded.
- Why it matters: reproducibility/auditability hole for every task number; a re-run is the only way to inspect anything.
- Suggested fix: `log_samples=True` and write `res["samples"]` (or just the per-doc `acc`/`acc_norm` lists) beside the JSON.

### [Minor] stage2c §9.2 compares blind and faithful prefill rows while stating "same harness and settings"

- File:line: docs/results/stage2c.md:732; docs/results/raw/tasks_glrd_70_f256.json, tasks_palu_full_70.json, tasks_glrd_50_f256.json (no `latent_quant_in_prefill` field: pre-2f, prefill blind to the latent format) vs tasks_glrd_70_f2048.json, tasks_palu_full_70_f2048.json (`latent_quant_in_prefill: true`); the same split holds for the corresponding ppl_*.json.
- What: the quoted deltas (-0.16 / -0.02 task points, -0.022 / 0.000 ppl) are calibration change plus prefill round-trip change (measured at about 0.03 ppl in stage 2f) added together.
- Why it matters: small, and the "calibration ladder is flat" conclusion survives, but the sentence claiming identical settings is contradicted by the row provenance fields the study itself added for this purpose.
- How to verify: `python3 -c "import json;[print(r,json.load(open(f'docs/results/raw/tasks_{r}.json')).get('latent_quant_in_prefill','absent')) for r in ('glrd_70_f256','glrd_70_f2048')]"`.
- Suggested fix: say so in §9.2, or re-score one f256 row faithfully.

### [Minor] report §3 "within a quarter point of the paper's own G-LRD number" is not supported by any comparable figure

- File:line: docs/report_palu_on_tt.md:59; docs/results/stage2b.md:21-26 (paper table, "avg (5) ... is the comparable column").
- What: `glrd_70_f256` scores 64.53 (plain acc) / 67.03 (paper protocol); the paper's G-LRD (ratio 0.5) five-task average is 63.27 (+1.26 / +3.76). The only paper figure within a quarter point is the six-task avg(6) 64.45, which includes HellaSwag and is a different compression ratio; stage2b itself says avg(6) is not comparable.
- Suggested fix: restate as "-0.23 points below our own stock row on the paper protocol" or delete.

### Mechanisms checked and ruled out (question 3)

All by reading the code paths named; none needs the device.

- Logit/target shift or double counting of the first continuation token: ruled out. `for_device.score` returns rows `L-1-n .. L-2`; `_loglikelihood_tokens` gathers `whole[-n:]`, `n` rows, `n` targets, summed once (lm_eval_tt.py:43, 80-85). Identical selection to HFLM's `logits[inplen-contlen:inplen]` (huggingface.py:786) on an input that ends one token before the last continuation token.
- Prefill hidden states one position off: ruled out for the code (model.py:430-452 embeds the tokens as given; :1010-1011 returns the pre-norm residual; generate.py:35 slices `[:, :L]`) and cross-checked by the perplexity harness using the same function with `targets = chunk[0, 1:]` against `h[:-1]` (eval_ppl.py:38, 52), which agrees with an fp32 CPU run.
- Wrong context length / truncation: ruled out; `max_length=2048`, the longest five-task sequence is a few hundred tokens; the left-clip at :75-76 never fires.
- Padding: one sequence per prefill, right-padded with token 0 to a multiple of 128, causal attention, padded rows discarded by `[:L]` (lm_eval_tt.py:39-40) -- never scored. HFLM right-pads batches the same way.
- `acc_norm` reported where `acc` is claimed: ruled out. `report._task_display`/`_avg5` read `tasks[t]["acc"]` (report.py:403-420); for all 14 JSONs `tasks[t]["acc"] == results[name]["acc,none"]`; the five-task means recompute to the rendered values (65.44, 55.84, 61.50, 59.48, 64.09, 60.78, 64.53, 61.34, 64.51, 60.52, 58.84, 61.08, 57.21, 62.65) and the paper-protocol column to 67.27 / 58.88 / 62.18 / 63.86 / 66.12.
- Dataset split: ruled out. `configs` in the JSONs show the stock splits (OBQA test, ARC test, PIQA validation, WinoGrande validation); `n` = 500 / 2376 / 1172 / 1838 / 1267, the standard sizes; lm-eval built 22,398 requests on CPU here (2000 / 9501 / 4687 / 3676 / 2534), matching stage2b:103-104.
- `_hub` yamls changing anything else: ruled out (next section).
- Tokenizer: `args.tokenizer` is `transformers.AutoTokenizer` (model_config.py:3777-3822); bos 128000, eos 128009; `tok_encode` uses `add_special_tokens=False` exactly as HFLM (lm_eval_tt.py:62 vs huggingface.py:669-675). `_encode_pair` is inherited unchanged, including the trailing-whitespace move (api/model.py:294-311).
- Empty-context sentinel: TTLM drops lm-eval's `[eot]` sentinel and uses BOS instead (lm_eval_tt.py:70); no request of the five tasks has an empty context (winogrande's `multiple_input` path puts the sentence prefix plus option in the context, task.py:1119-1124, 967-968).
- Request/result misalignment: ruled out; sequential loop, no Collator, results appended in request order; evaluator zips `resps` with `cloned_reqs` (evaluator.py:373-377).
- Memo collisions: key is `(tuple(whole), n_cont)` (lm_eval_tt.py:77); two requests with equal key have equal answers.
- `limit`: passed straight to `simple_evaluate`; `doc_iterator` takes the first N docs (task.py:647-657); all 14 reported rows have `limit: null`.
- dtype: TTLM log-softmaxes fp32 host logits (`.float()`, :80); HFLM log-softmaxes the model's logits (fp16 on the GPU run). Negligible.
- `is_greedy`: computed as HFLM does (argmax over all continuation positions); unused by the five tasks' metrics.
- Not ruled out: the BOS difference (finding 1) and, secondarily, bfp8-vs-fp16 numerics (which can go either way and is the study's declared "port cost").

## Tests in this area

tests/test_lm_eval_tt.py (7 tests, all pass in 3.7 s: `OMP_NUM_THREADS=4 nice -n 10 python -m pytest tests/test_lm_eval_tt.py -q -p no:cacheprovider -m "not slow"`).

- `test_loglikelihood_scores_continuation_tokens_only_and_flags_greedy` (28-39): drives `TTLM.from_parts` with a one-char-per-token tokenizer and a score function that hands back peaked logits for rows `L-1-n .. L-2`. Asserts greedy/non-greedy flags, `lp("ab","cd") == 2*lp("ab","c")`, and that an empty context is scored after a BOS. It would catch an off-by-one in `_loglikelihood_tokens` (a shifted `tgt` gather or a dropped BOS makes the greedy assertion or `lp_e > -1` fail; I traced both). It would NOT catch an off-by-one in `for_device.score` (never executed) nor a shift inside `prefill_hidden_states`, because the fake score function encodes the same contract the production code assumes rather than an independent reference.
- `test_loglikelihood_clips_overlong_continuation_from_the_left` (42-52): compares two `TTLM` instances on inputs that reduce to the same window; self-consistency only, no reference value. Guards the wrap-around bug it names.
- `test_task_name_map_and_task_list` (55-57), `test_lm_eval_names_maps_hub_overrides` (60-61): assert constants and a one-line dict inversion; cannot fail unless the constants are edited.
- `test_effective_n_uses_eval_docs_clipped_to_limit` (69-72): the fallback `n` with a fake task; fine, trivial.
- `test_task_yaml_templates_have_no_hardcoded_worktree_path` (75-82), `test_write_task_dir_renders_includes_that_exist_on_disk` (85-95): placeholder present, rendered include path exists and is outside the repo. Neither loads the rendered yaml through lm-eval, so a change in lm-eval's `include:` merge order (utils.py:331-355, including-file keys win) or a stock-yaml edit would go unnoticed; the only evidence the `_hub` tasks equal stock is the `configs` block in each raw JSON.
- Gaps: `eval_tasks.run()` (the function that produces every `acc`/`acc_norm`/`n` in the tables) has no test; `TTLM.for_device` has no test on either CPU or device; no test compares the adapter's log-likelihood against a full-sequence reference.

## Claims traced

| Documented claim | Code / data | Status |
|---|---|---|
| "lm-eval 0.4.2, zero-shot (`num_fewshot=0`)" (report §1) | eval_tasks.py:69; JSON `lm_eval_version: 0.4.2`, `configs.*.num_fewshot: 0`; log "Overwriting default num_fewshot ... to 0" | verified |
| Five tasks OBQA, PIQA, ARC-e, ARC-c, WinoGrande, full sets, `--limit` unset (stage2b:102-104) | `TASKS`, JSON `limit: null`, `n` = 500/1838/2376/1172/1267; 22,398 requests rebuilt on CPU | verified |
| "the reported number is `acc` averaged over the five tasks (the per-task `acc_norm` is in the results documents)" (report §1) | report.py:403-420; all 14 JSONs recomputed | verified |
| "paper protocol": `acc_norm` for OBQA and ARC-c, `acc` elsewhere (stage2b:125-127) | report.py:423-437; recomputed 67.27 / 58.88 / 62.18 / 63.86 / 66.12 | verified (as a computation; whether the paper's appendix says this was not checked) |
| `_hub` tasks differ from stock only in dataset id (README:148-152, spec §2) | yaml files (4 lines each); utils.py:331-355 merge order; JSON `configs` show stock `doc_to_text`/`doc_to_target`/`doc_to_choice`/`metric_list`/splits/`target_delimiter` | verified |
| arc_easy / arc_challenge are stock tasks | `lm_eval_names` maps only the three; JSON `configs.arc_*.dataset_path: allenai/ai2_arc` (stock) | verified |
| Datasets: `allenai/openbookqa`, `ybisk/piqa`, `allenai/winogrande` (README) | JSON configs; log cache paths; document counts equal the canonical sizes | verified (counts); byte-identity with the ids the stock yamls name not checked |
| Adapter "scores each answer continuation from the prefill hidden states with a continuation-only LM head" (report §1) | lm_eval_tt.py:37-45, 80-85 | verified |
| "tokenise ... (BOS once)" (spec §2) | lm_eval_tt.py:79 | verified -- and undisclosed in the report/results docs; differs from Palu's `add_bos_token=False` (finding 1) |
| "scores every row on five zero-shot lm-evaluation-harness tasks the way the paper does" (README:15); "Those numbers are the paper's protocol on our model" (stage2c §9.3) | run_lm_eval.py:17 vs lm_eval_tt.py:79 | contradicted in the BOS detail |
| "this is not a task-harness artefact ... the harness ... healthy" (stage2b:192-195) | +0.84 vs paper computed correctly; inference rests on a BOS-inclusive number vs a BOS-free one | not verified (unsupported inference) |
| n300 vs GPU: 65.44 vs 64.25 (+1.19), per-task +2.6/+0.2/+0.5/+2.1/+0.4, other rows -0.33/-0.72/+1.50/+0.34 (stage2c §9.3) | gpu_report table vs JSONs | verified (arithmetic); attribution to "harness noise/port cost" not verified (finding 1) |
| "same harness and settings" for f256 vs f2048 task rows (stage2c:732) | `latent_quant_in_prefill` absent vs `true` | contradicted (finding 5) |
| Per-row `seconds` 52.1-53.1 min (stage2b:104-105) | tasks_baseline.json `seconds: 3139.2` = 52.3 min | verified for baseline (others not checked) |
| Task rows are decode-kernel independent (README:185, eval_tasks.py:92-96) | adapter calls only `prefill_hidden_states` | verified |
| Report §3 table "task acc, mean of 5" (65.4, 55.8, 61.5, 61.3, 59.5, 60.8, 64.1, 64.5, 64.5) | JSONs | verified |
| "within a quarter point of the paper's own G-LRD number" (report:59) | 64.53 / 67.03 vs paper 63.27 (avg5) | contradicted (finding 6) |
| Spec §6 device test: 20-example arc_easy run asserting plausible accuracy and JSON schema | tests/ | contradicted -- not present (finding 2) |

## Verdict

The adapter's log-likelihood arithmetic is correct and matches HFLM position for position; the `_hub` task configs are stock apart from the dataset id; splits, document counts, request counts and every task number in the stage2b/2c/2f documents and the external report reproduce from the raw JSONs, so the within-study rows are comparable with each other. The one substantive problem is protocol, not arithmetic: `TTLM` scores every sequence with a BOS token where Palu's own harness (and therefore the paper and the GPU cross-check) uses none, which is nowhere disclosed and is the most likely cause of the bfp8 device outscoring fp16 on all five tasks; the cross-harness and paper comparisons, and the "harness is healthy" inference, should not be presented externally until that offset is measured or the BOS choice is stated and aligned. Test coverage of the numeric path is thin: the only device-side scoring function is untested, `eval_tasks.run()` is untested, and no uncertainty is computed for numbers that are quoted to a tenth of a point.
