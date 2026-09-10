"""PaluAttention for tt_transformers v0.78.0. N150/N300, non-paged, no prefetcher, Meta-style RoPE.
Shapes stay stock everywhere except inside the latent caches and the SDPA call."""
import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup, num_to_corerange
from models.tt_transformers.tt.rope import get_rot_mats

from palu_tt import fused
from palu_tt.convert import CONVERT_VERSION


def cache_tag(cache_id: str, r_k: int, r_v: int) -> str:
    """Prefix for every Palu-specific weight-cache entry.

    ttnn.as_tensor reuses an on-disk cache file whenever name/shape/dtype match -- it never looks at
    the content -- so the ranks alone are not enough: two weight sets with the same per-layer ranks
    (a re-generated calibration dump and the dump it replaced, say) would share entries and one
    would silently load the other's tensors. PaluConfig.cache_id identifies the weight set, and
    convert.CONVERT_VERSION the converter that built the tensors -- a converter fix that changes a
    tensor's content but neither its shape nor the weight set it came from (v1 -> v2 collapsed the
    full-rank V factorisation into the QKV weight) must not be served an entry from the old scheme.
    """
    assert isinstance(cache_id, str) and cache_id, f"PaluConfig.cache_id must be a non-empty string, got {cache_id!r}"
    return f"palu_v{CONVERT_VERSION}_{cache_id}_rk{r_k}_rv{r_v}"


def _fused_tables(configuration, mesh_device, chunk):
    """Device tensors the fused kernel shares across layers: the [cos | sin] table (bf16) and, for
    Phase 2, the R_{-chunk} step constants (fp32) and the 32x32 pair-rotation tile (bf16). Built once
    per model and kept on PaluConfig.device_tables; a second layer with the same key reuses them.

    The cache (and the ``id(mesh_device)`` term in the key) assumes it lives on the same
    ``configuration``/model for its whole lifetime -- head_dim, rope_theta and rope_scaling are part
    of the key precisely so a second, differently-configured model reusing the same mesh handle
    cannot be served stale tables; it is not meant to survive a config change on the same model.
    """
    rt = configuration.palu.runtime
    key = ("fused", id(mesh_device), configuration.max_seq_len, int(chunk), bool(rt.cs_recurrence),
           configuration.head_dim, configuration.rope_theta, repr(configuration.rope_scaling))
    tabs = configuration.palu.device_tables
    if key not in tabs:
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
        cos, sin = fused.cos_sin_tables(configuration.head_dim, configuration.max_seq_len, configuration.rope_theta,
                                        configuration.rope_scaling)
        cs = up(fused.cs_table(cos, sin, configuration.max_seq_len)[None, None], ttnn.bfloat16)
        step = tm = None
        if rt.cs_recurrence:
            step = up(fused.rot_step_table(cos, sin, chunk)[None, None], ttnn.float32)
            tm = up(fused.trans_mat_tile(), ttnn.bfloat16)
        tabs[key] = (cs, step, tm)
    return tabs[key]


class PaluAttention(LightweightModule):
    def __init__(self, mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num, dtype,
                 transformation_mats, configuration, paged_attention_config=None, use_paged_kv_cache=False,
                 prefetcher=None):
        super().__init__()
        assert paged_attention_config is None and not use_paged_kv_cache, "stage 1 is non-paged"
        assert prefetcher is None, "prefetcher not supported"
        assert configuration.num_devices in (1, 2), "N150/N300 only"
        assert not configuration.use_hf_rope and not configuration.use_qk_fused, "set use_qk_fused=False, use_hf_rope=False"
        assert hasattr(configuration, "palu"), "attach a PaluConfig as configuration.palu"
        # stock features this class silently drops -- refuse the model rather than lose them quietly
        _ln = configuration.get_state_dict_prefix("Attention", layer_num)
        assert f"{_ln}.q_norm.weight" not in state_dict, "q_norm not supported"
        assert f"{_ln}.k_norm.weight" not in state_dict, "k_norm not supported"
        assert f"{_ln}.wq.bias" not in state_dict, "QKV bias not supported"
        assert configuration.query_pre_attn_scalar is None, "query_pre_attn_scalar not supported"
        assert not configuration.layer_types or configuration.layer_types[layer_num] != "sliding_attention", \
            "sliding-window (local) attention layers not supported"

        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices
        self.batch_size_per_device_group = self.max_batch_size
        self.dtype = dtype
        self.transformation_mats = transformation_mats
        self.sliding_window = None
        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.scale = self.head_dim ** -0.5
        self.palu = configuration.palu
        self.r_k = self.palu.rank_k[layer_num]
        self.r_v = self.palu.rank_v[layer_num]
        assert self.r_k % 32 == 0 and self.r_v % 32 == 0 and self.r_k <= self.head_dim and self.r_v <= self.head_dim
        # r_k == head_dim (the V-only row and the identity row) buys nothing on the K side: the
        # converter then puts the plain Meta-permuted wk in the fused QKV weight, so the "latent"
        # already *is* K in the order the stock RoPE and SDPA want. Run the stock K path in that
        # case -- no per-step reconstruction, no prefill-mode RoPE over the whole context.
        self.recon_k = self.r_k < self.head_dim
        # Same argument on the V side: at r_v == head_dim (only the identity row) the prefill
        # up-projection is an identity matmul, and routing the latent through it costs a
        # bfp8 -> bf16 -> bfp8 round trip that stock does not do.
        self.recon_v = self.r_v < self.head_dim

        rt = self.palu.runtime
        # stage 2: fused K reconstruction. Only where K really is low rank; on the V-only and identity
        # rows the stock K path stays, flag or no flag.
        self.use_fused_k = self.recon_k and bool(rt.use_fused_k)
        self.phase2 = self.use_fused_k and bool(rt.cs_recurrence)

        opt = self.args.decoders_optimizations
        self.activation_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.ACTIVATION)
        self.wqkv_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.WQKV)
        self.wo_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.WO)
        self.kv_cache_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.KV_CACHE)
        # latent caches may be kept wider than the stock KV cache dtype (spec 5, "Latent dtype")
        self.latent_dtype = self.palu.latent_dtype if self.palu.latent_dtype is not None else self.kv_cache_dtype
        fid = lambda op: opt.get_math_fidelity(decoder_id=layer_num, op=op, configuration=configuration)
        self.li_qkv_decode_compute_kernel_cfg = fid(OpGroup.LI_QKV_DECODE)
        self.sdpa_decode_compute_kernel_cfg = fid(OpGroup.SDPA_DECODE)
        self.li_o_decode_compute_kernel_cfg = fid(OpGroup.LI_O_DECODE)
        self.sdpa_prefill_compute_kernel_cfg = fid(OpGroup.SDPA_PREFILL)
        self.li_qkv_prefill_compute_kernel_cfg = fid(OpGroup.LI_QKV_PREFILL)
        self.li_o_prefill_compute_kernel_cfg = fid(OpGroup.LI_O_PREFILL)

        layer_name = configuration.get_state_dict_prefix("Attention", layer_num)
        tag = cache_tag(self.palu.cache_id, self.r_k, self.r_v)
        # "_kstock2" marks the V-only fast path: same ranks, *different* K block in the fused QKV
        # weight (the full per-head K weight rebuilt from the factors, not the padded latent). The
        # on-disk weight cache is keyed by name/shape/dtype only, so without this an existing cache
        # would be silently reused. It applies to the fused wqkv alone -- every other cached tensor
        # either has identical content on both paths (the latent caches, wo) or is not built at all
        # on the fast path (u_k_*, u_v_prefill, wo_fused).
        wqkv_tag = tag + ("" if self.recon_k else "_kstock2")
        if configuration.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{layer_name}.{name}"
        P = f"{layer_name}.palu"
        B = self.max_batch_size

        # --- fused [wq | wk_lat_pad | wv_lat_pad], per device, exactly like stock ---------------------
        wq = state_dict[f"{layer_name}.wq.weight"]
        wk = state_dict[f"{P}.wk_lat_pad"]
        wv = state_dict[f"{P}.wv_lat_pad"]
        qkv_list = []
        for i in range(self.num_devices):
            q_i = torch.chunk(wq, self.num_devices, dim=0)[i].transpose(-2, -1)
            k_i = torch.chunk(wk, self.num_devices, dim=0)[i].transpose(-2, -1)
            v_i = torch.chunk(wv, self.num_devices, dim=0)[i].transpose(-2, -1)
            qkv_list.append(torch.cat([q_i, k_i, v_i], dim=-1))
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
        self.wqkv = ttnn.as_tensor(
            qkv_cat, dtype=self.wqkv_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
            memory_config=configuration.create_dram_sharded_mem_config(configuration.dim, configuration.qkv_size // self.num_devices),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=configuration.cluster_shape),
            cache_file_name=cache_name(f"{wqkv_tag}_wqkv_sharded_2d"),
        )

        # --- up-projections -------------------------------------------------------------------------
        # Only the *active* paths get a device tensor. On a full-rank fast path the converter has
        # already folded the factor into the QKV weight and emits an identity `u_*` that nothing
        # reads, so uploading it would burn DRAM (u_k_decode alone is 2-4 MiB/chip per layer at batch 32, 64-128 MiB/chip over 32 layers) on a
        # tensor no forward touches. The attribute is simply absent there -- every use below is
        # guarded by the same `recon_*` flag that decides whether it is built.
        shard_heads = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        # prefill: zero-padded to [1, n_kv, hd, hd] so the padded latent heads need no slicing
        def pad_u(u, r):
            t = torch.zeros(self.n_kv_heads, self.head_dim, self.head_dim)
            t[:, :r, :] = u
            return t.unsqueeze(0)
        if self.recon_k:
            u_k = state_dict[f"{P}.u_k"]  # [n_kv, r_k, hd]
            self.u_k_prefill = ttnn.as_tensor(pad_u(u_k, self.r_k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_k_prefill"))
            if self.use_fused_k:
                # decode: the kernel's packed rotation weight [1, n_kv, r_k, 2*hd] = [u_k | u_k T], head-sharded like u_k
                self.w_rot = ttnn.as_tensor(
                    state_dict[f"{P}.w_rot"].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_w_rot"))
            else:
                # decode: batched (non-broadcast) matmul needs matching batch dims -> replicate over batch
                self.u_k_decode = ttnn.as_tensor(
                    u_k.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads,
                    cache_file_name=cache_name(f"{tag}_u_k_decode_b{B}"),
                )
        if self.recon_v:
            u_v = state_dict[f"{P}.u_v"]  # [n_kv, r_v, hd]
            self.u_v_prefill = ttnn.as_tensor(pad_u(u_v, self.r_v), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_v_prefill"))

        # --- output projections: stock wo (prefill) and fused wo (decode) ------------------------------
        wo_mem = configuration.create_dram_sharded_mem_config((configuration.n_heads * configuration.head_dim) // self.num_devices, configuration.dim)
        def wo_tensor(w, name):
            return ttnn.as_tensor(w.transpose(-1, -2).unsqueeze(0).unsqueeze(0), dtype=self.wo_dtype, layout=ttnn.TILE_LAYOUT,
                                  device=mesh_device, memory_config=wo_mem, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
                                  cache_file_name=cache_name(name))
        self.wo = wo_tensor(state_dict[f"{layer_name}.wo.weight"], "wo")            # same bytes as stock -> shared cache entry
        # r_v == head_dim: the converter emits wo_fused == wo (U_v is folded into the QKV weight and
        # the identity handed downstream), so a second upload would be a byte-identical duplicate of
        # a ~300 MB/device tensor. Alias it instead.
        self.wo_fused = wo_tensor(state_dict[f"{P}.wo_fused"], f"{tag}_wo_fused") if self.recon_v else self.wo

        # --- latent caches ---------------------------------------------------------------------------
        self.layer_past = []
        for r, nm in ((self.r_k, "k"), (self.r_v, "v")):
            z = torch.zeros(B, self.n_local_kv_heads, self.max_seq_len, r)
            self.layer_past.append(ttnn.as_tensor(
                z, dtype=self.latent_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=cache_name(f"{tag}_latcache_{nm}_{tuple(z.shape)}") if weight_cache_path else None,
            ))

        if self.recon_k and not self.use_fused_k:
            # prefill-mode cos/sin for the naive K reconstruction in decode
            self.cos_prefill, self.sin_prefill = get_rot_mats(
                head_dim=self.head_dim, device=mesh_device, seq_len=self.max_seq_len, theta=configuration.rope_theta,
                rope_scaling=configuration.rope_scaling, datatype=ttnn.bfloat16, rot_mats_layout=ttnn.TILE_LAYOUT,
            )
        if self.use_fused_k:
            if self.phase2:
                # Phase 2: an explicit override, else the model's stock decode config's chunk (falling
                # back to the op's own default if that config leaves it at 0 too)
                chunk = int(rt.fused_k_chunk) if rt.fused_k_chunk else (
                    self.args.get_attn_sdpa_decode_program_config(None).k_chunk_size or fused.chunk_size(self.max_seq_len))
            else:
                # Phase 1: the op's streamed cos/sin table does not fit L1 above k_chunk_size 128
                chunk = int(rt.fused_k_chunk or 128)
            # Always pin the chunk explicitly (never 0) so the op takes the program-config override
            # path instead of deriving its own chunk a second time -- a mismatch between this host
            # derivation and the op's internal get_chunk_size would otherwise be silently wrong.
            pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                        q_chunk_size=0, k_chunk_size=chunk)
            self.fused_program_config = pc
            self.cs_tab, self.rot_step, self.trans_mat = _fused_tables(configuration, mesh_device, chunk)
            cfg = self.sdpa_decode_compute_kernel_cfg
            # Phase 2 keeps the rotated query in fp32 and needs an fp32 dest; the HiFi4 preset has it
            self.fused_compute_cfg = cfg if (not self.phase2 or cfg.fp32_dest_acc_en) else configuration.compute_kernel_config_hifi4

    # ------------------------------------------------------------------------------------------------
    def kv_bytes_per_token(self):
        """Bytes per token per layer on one chip (tile-padded), for the results table."""
        bpe = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0, ttnn.bfloat4_b: 0.5625}[self.latent_dtype]
        return self.n_local_kv_heads * (self.r_k + self.r_v) * bpe

    def _latent_shard_cfg(self, width):
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width), core_grid=ttnn.CoreRangeSet({num_to_corerange(self.max_batch_size)}),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )

    def _kv_prefill_shard_cfg(self, seq_len, width):
        return ttnn.create_sharded_memory_config(
            (self.n_local_kv_heads * seq_len // 64, width), ttnn.CoreGrid(y=8, x=8),
            ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )

    # ------------------------------------------------------------------------------------------------
    def forward_prefill(self, x_11SH, rot_mats, user_id=0, page_table=None, chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        assert page_table is None and chunk_start_idx is None and kv_cache is None
        assert x_11SH.shape[0] == 1, "single-user prefill only"
        seq_len = x_11SH.shape[-2]
        original_seq_len = seq_len
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        # QKV (stock) ---------------------------------------------------------------------------------
        if seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
            padded = ((seq_len + self.MAX_QKV_MM_SEQ_LEN - 1) // self.MAX_QKV_MM_SEQ_LEN) * self.MAX_QKV_MM_SEQ_LEN
            x_11SH = ttnn.pad(x_11SH, padding=[(0, 0), (0, 0), (0, padded - seq_len), (0, 0)], value=0.0)
            seq_len = padded
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])
        if self.args.use_minimal_qkv_prefill_matmul(seq_len):
            xqkv = ttnn.experimental.minimal_matmul(x_11SH, self.wqkv, compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                                                    config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None))
        else:
            xqkv = ttnn.linear(x_11SH, self.wqkv, dtype=self.activation_dtype or ttnn.bfloat16,
                               memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None),
                               compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                               program_config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None))
        xqkv = tt_all_reduce(xqkv, self.mesh_device, self.tt_ccl, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.ccl_dtype)
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv = ttnn.reshape(xqkv, [1, 1, seq_len, -1])
        if original_seq_len != seq_len:
            xqkv = xqkv[:, :, :original_seq_len, :]
            seq_len = original_seq_len
        ttnn.deallocate(x_11SH)

        q_pre, k_lat_pad, v_lat_pad = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xqkv)

        # reconstruct V (and K, unless the V-only fast path applies) from the padded latents ---------
        # ([1, nkv, S, 128] @ [1, nkv, 128, 128]); at r_k == head_dim k_lat_pad is already full K.
        k_pre = ttnn.matmul(k_lat_pad, self.u_k_prefill, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=self.compute_kernel_config_hifi2) if self.recon_k else k_lat_pad
        v_full = ttnn.matmul(v_lat_pad, self.u_v_prefill, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             compute_kernel_config=self.compute_kernel_config_hifi2) if self.recon_v else v_lat_pad

        # RoPE on q and K (stock prefill op) ------------------------------------------------------------
        q = ttnn.experimental.rotary_embedding_llama(q_pre, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False)
        k_rot = ttnn.experimental.rotary_embedding_llama(k_pre, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False)
        ttnn.deallocate(q_pre)
        ttnn.deallocate(k_pre)          # the fast path's k_pre *is* k_lat_pad; the RoPE has consumed it

        # caches: latents, except the fast path's K cache which holds rotated full K like stock --------
        keys, values = self.layer_past
        for src, r, cache in (((k_lat_pad if self.recon_k else k_rot), self.r_k, keys), (v_lat_pad, self.r_v, values)):
            lat = src[:, :, :, :r] if r < self.head_dim else src
            lat = ttnn.typecast(lat, dtype=cache.dtype)
            if seq_len >= self.min_kv_prefill_shard_seqlen:
                lat = ttnn.interleaved_to_sharded(lat, self._kv_prefill_shard_cfg(seq_len, r))
            ttnn.fill_cache(cache, lat, user_id % self.batch_size_per_device_group)
            ttnn.deallocate(lat)
        if self.recon_k:
            ttnn.deallocate(k_lat_pad)
        if self.recon_v:
            ttnn.deallocate(v_lat_pad)      # otherwise v_lat_pad *is* v_full, freed after the SDPA

        # SDPA (stock, with reconstructed K/V) -----------------------------------------------------------
        q8 = ttnn.typecast(q, dtype=self.activation_dtype or ttnn.bfloat8_b)
        k8 = ttnn.typecast(k_rot, dtype=self.kv_cache_dtype)
        v8 = ttnn.typecast(v_full, dtype=self.kv_cache_dtype)
        for t in (q, k_rot, v_full):
            ttnn.deallocate(t)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q8, k8, v8, is_causal=True, scale=self.scale, compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
            program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, seq_len, None, None))
        for t in (q8, k8, v8):
            ttnn.deallocate(t)

        # output projection (stock tail) ---------------------------------------------------------------
        attn = ttnn.reshape(attn, [1, self.n_local_heads, -1, self.head_dim])
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if seq_len > 1024:
            attn = ttnn.reshape(attn, [1, seq_len // 1024, 1024, -1])
        out = ttnn.linear(attn, self.wo, compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
                          dtype=self.activation_dtype or ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                          program_config=self.args.get_attn_wo_program_config(Mode.PREFILL, seq_len, None))
        if seq_len > 1024:
            out = ttnn.reshape(out, [1, 1, seq_len, -1])
        ttnn.deallocate(attn)
        out = tt_all_reduce(out, self.mesh_device, self.tt_ccl, cluster_axis=0, dim=3, topology=self.ccl_topology,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.ccl_dtype)
        return out

    def _recon_len(self, current_pos):
        S = self.palu.runtime.recon_len or self.max_seq_len
        S = min(nearest_32(S), self.max_seq_len)
        if self.palu.runtime.check:
            pos = ttnn.to_torch(ttnn.get_device_tensors(current_pos)[0])
            assert int(pos.max()) < S, f"recon_len {S} does not cover current_pos max {int(pos.max())}"
        return S

    def _slice_latent(self, heads_1BKD, r):
        """[1, B, nkv, 128] height-sharded -> [1, B, nkv, r] height-sharded (shard [32, r] per batch core).

        At r == head_dim there is nothing to slice: the head-split output already *is* the latent,
        in the layout `paged_update_cache` wants, so the sharded -> interleaved -> sharded round
        trip would only copy it twice. Return the input untouched.
        """
        if r == self.head_dim:
            return heads_1BKD
        t = ttnn.sharded_to_interleaved(heads_1BKD, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(heads_1BKD)
        t2 = t[:, :, :, :r]
        ttnn.deallocate(t)
        return ttnn.interleaved_to_sharded(t2, self._latent_shard_cfg(r))

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        assert page_table is None and kv_cache is None
        B = self.batch_size_per_device_group

        # QKV matmul + head split (stock) -------------------------------------------------------------
        xqkv_sharded = ttnn.linear(
            x, self.wqkv, memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.DECODE, None),
            program_config=self.args.get_attn_qkv_program_config(Mode.DECODE, 1, None),
            compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg, dtype=self.activation_dtype or ttnn.bfloat16)
        ttnn.deallocate(x)
        mem = self.args.get_attn_qkv_all_reduce_output_mem_config(Mode.DECODE, list(self.mesh_device.shape)[1], None)
        tt_all_reduce(xqkv_sharded, self.mesh_device, self.tt_ccl, cluster_axis=1,
                      memory_config=mem if mem is not None else xqkv_sharded.memory_config(),
                      sharded=True, dtype=self.ccl_dtype, topology=self.ccl_topology)
        xqkv = ttnn.sharded_to_interleaved(xqkv_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        ttnn.deallocate(xqkv_sharded)
        xqkv = ttnn.reshape(xqkv, (1, 1, B, xqkv.shape[3]), (1, 1, 32, xqkv.shape[3]))
        q_pre, k_lat_pad, v_lat_pad = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads,
            memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, None))
        ttnn.deallocate(xqkv)

        # RoPE on q only ---------------------------------------------------------------------------------
        q = ttnn.experimental.rotary_embedding_llama(q_pre, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True)
        ttnn.deallocate(q_pre)

        # latents -> caches -----------------------------------------------------------------------------
        keys, values = self.layer_past
        v_lat = self._slice_latent(v_lat_pad, self.r_v)
        if self.recon_k:
            k_new = self._slice_latent(k_lat_pad, self.r_k)
        else:
            # V-only fast path: the K "latent" is full K in Meta order, so this is the stock K path
            # -- decode-mode RoPE straight into a head_dim cache.
            k_new = ttnn.experimental.rotary_embedding_llama(
                k_lat_pad, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True)
            ttnn.deallocate(k_lat_pad)
        ttnn.experimental.paged_update_cache(keys, k_new, update_idxs_tensor=current_pos)
        ttnn.experimental.paged_update_cache(values, v_lat, update_idxs_tensor=current_pos)
        ttnn.deallocate(k_new)
        ttnn.deallocate(v_lat)

        if self.use_fused_k:
            # stage 2: K rebuilt and rotated inside the kernel; the caches go in as they are
            attn = ttnn.transformer.palu_sdpa_decode(
                q, keys, self.w_rot, self.cs_tab, values, self.r_v, current_pos,
                rot_step=self.rot_step, trans_mat=self.trans_mat, scale=self.scale,
                program_config=self.fused_program_config, compute_kernel_config=self.fused_compute_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)                                        # [1, B, nh, r_v]
            ttnn.deallocate(q)
        else:
            if self.recon_k:
                # K reconstruction (the naive Palu path; stage 2 fuses this) ---------------------------------
                S = self._recon_len(current_pos)
                k_lat_S = keys[:, :, :S, :] if S < self.max_seq_len else keys                 # [B, nkv, S, r_k]
                v_att = values[:, :, :S, :] if S < self.max_seq_len else values               # [B, nkv, S, r_v]
                k_rec = ttnn.matmul(k_lat_S, self.u_k_decode, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    compute_kernel_config=self.compute_kernel_config_hifi2)   # [B, nkv, S, 128]
                if S < self.max_seq_len:
                    ttnn.deallocate(k_lat_S)
                k_rec = ttnn.reshape(k_rec, (1, B * self.n_local_kv_heads, S, self.head_dim))
                # A full-range ttnn slice is a no-op that returns the *same* tensor (see the "No-op check"
                # in ttnn::slice), so at S == max_seq_len -- the case a decode at the end of the context
                # window hits -- these alias the persistent prefill cos/sin and must not be deallocated.
                sliced = S < self.max_seq_len
                cos = self.cos_prefill[:, :, :S, :] if sliced else self.cos_prefill
                sin = self.sin_prefill[:, :, :S, :] if sliced else self.sin_prefill
                k_rot = ttnn.experimental.rotary_embedding_llama(k_rec, cos, sin, self.transformation_mats["prefill"], is_decode_mode=False)
                ttnn.deallocate(k_rec)
                if sliced:
                    ttnn.deallocate(cos)
                    ttnn.deallocate(sin)
                k_rot = ttnn.reshape(k_rot, (B, self.n_local_kv_heads, S, self.head_dim))
                k_att = ttnn.typecast(k_rot, dtype=self.kv_cache_dtype)                       # same precision as a stock K cache
                ttnn.deallocate(k_rot)
            else:
                # Both caches carry seq dim max_seq_len, which is what the MLA op's equal-seq-len check
                # wants -- exactly how stock hands its full caches to SDPA decode.
                S, k_att, v_att = self.max_seq_len, keys, values

            # attention on K (reconstructed, or the cache itself) and the V latent ---------------------------
            attn = ttnn.transformer.flash_multi_latent_attention_decode(
                q, k_att, v_att, head_dim_v=self.r_v, cur_pos_tensor=current_pos, scale=self.scale,
                program_config=self.args.get_attn_sdpa_decode_program_config(None),
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, B, nh, r_v]
            ttnn.deallocate(q)
            if self.recon_k:
                ttnn.deallocate(k_att)
                if S < self.max_seq_len:
                    ttnn.deallocate(v_att)
        if self.r_v < self.head_dim:
            attn_p = ttnn.pad(attn, padding=[(0, 0), (0, 0), (0, 0), (0, self.head_dim - self.r_v)], value=0.0)
            ttnn.deallocate(attn)
            attn = attn_p

        # stock tail: concat heads, gather, fused wo, all-reduce -----------------------------------------
        attn_11BH = ttnn.to_memory_config(attn, memory_config=self.args.get_attn_sdpa_output_mem_config(Mode.DECODE, B, None))
        attn_cat = ttnn.experimental.nlp_concat_heads_decode(attn_11BH, num_heads=self.n_local_heads)
        ttnn.deallocate(attn_11BH)
        ttnn.deallocate(attn)
        attn_g = tt_all_gather(attn_cat, self.mesh_device, self.tt_ccl, dim=2, cluster_axis=1,
                               memory_config=self.args.get_attn_gather_users_mem_config(Mode.DECODE, list(self.mesh_device.shape)[1], None),
                               sharded=True)
        dense = ttnn.linear(attn_g, self.wo_fused, program_config=self.args.get_attn_wo_program_config(Mode.DECODE, 1, None),
                            memory_config=self.args.get_attn_wo_output_mem_config(Mode.DECODE, None),
                            compute_kernel_config=self.li_o_decode_compute_kernel_cfg)
        ttnn.deallocate(attn_cat)
        dense = tt_all_reduce(dense, self.mesh_device, self.tt_ccl, cluster_axis=0, dim=3, topology=self.ccl_topology,
                              memory_config=self.args.get_attn_all_reduce_output_mem_config(Mode.DECODE, self.hidden_size, list(self.mesh_device.shape)[0], None),
                              sharded=True, dtype=self.ccl_dtype, use_composite=False)
        return ttnn.to_memory_config(dense, self.args.get_attn_dense_output_mem_config(Mode.DECODE, None))

    def forward(self, x, current_pos, rot_mats=None, user_id=0, mode=Mode.DECODE, page_table=None,
                chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        if mode == Mode.PREFILL:
            return self.forward_prefill(x, rot_mats, user_id, page_table=page_table, chunk_page_table=chunk_page_table,
                                        chunk_start_idx=chunk_start_idx, kv_cache=kv_cache)
        return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)
