---
name: ttnn-sdpa-decode-chunk32-bug
description: tt-metal SDPA/MLA decode op returns garbage with fp32_dest_acc_en when the auto K chunk is 32 rows and batch < 32 (multi-core per user)
metadata:
  type: project
---

`ttnn.transformer.scaled_dot_product_attention_decode` / `flash_multi_latent_attention_decode` with the stock tt_transformers SDPA_DECODE compute config (HiFi2, fp32_dest_acc_en=True) mis-reduces when the K chunk (largest power of two dividing S, capped 512) is 32 and a user's chunks are spread over several cores (batch < 32). Measured on the n300 (2026-09-12): PCC 0.17 at S=160, 0.19 at S=544, 0.22 at S=1056 for batch 1; 0.53 at batch 8; exact (0.9997) at chunk 64/128/512, at batch 32, or with fp32_dest_acc_en=False. Probe scripts: session scratchpad probe_ck*.py.

**Why:** Palu's naive decode reconstructs K/V for the first `recon_len` rows; `((pos+1+31)//32)*32` in tests gave S=160 and the batch-1 outputs were wrong while caches were right.

**How to apply:** keep any K/V handed to the decode attention op at a multiple of 64 rows (palu_tt `_recon_len` rounds to 64; `model.set_recon_len` rounds to 256), or pass an explicit `k_chunk_size >= 64`. Also: `nlp_concat_heads_decode` pads users to 32 rows, slice `[:B]` before reshaping. See [[palu-tt-project]].

**Update 2026-09-23:** same root cause as [[correction-block-dst-fix]]: the cross-core softmax merge (`correction_block`) overflowed the four-tile fp32 half-sync DEST window. Upstream tracked it as tenstorrent/tt-metal#56171 (chunk 32, second chunk in range) and fixed it in #56225 (2026-09-14, after v0.78.0). Our submodule branch `palu-sdpa` carries its own fix (a41a23f), so the S-multiple-of-64 rule is no longer needed for correctness there; keep it only where reported numbers must stay comparable.
