---
name: ttnn-weight-cache-trap
description: ttnn.as_tensor(cache_file_name=...) reuses on-disk weights by name/shape/dtype only — changed weight contents under the same name are silently ignored
metadata:
  type: project
---

tt_transformers' weight cache (model_cache/, via `ttnn.as_tensor(..., cache_file_name=...)`) is keyed by file name + shape + dtype, never by content. Two lessons paid for in this project: (1) changing what a tensor holds without changing its cache name loads the old bytes (a regenerated calibrated dump measured bit-identical perplexity); (2) rank-only tags collide across weight sets. The fix in palu_tt is `PaluConfig.cache_id` (dir name + content hash) in every Palu cache tag.

**How to apply:** whenever a device weight's content changes, bump its cache name (or wipe model_cache/); never reuse a tag across different weight sets. Related: [[palu-tt-project]].
