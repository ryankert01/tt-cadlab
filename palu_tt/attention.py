"""PaluAttention for tt_transformers v0.78.0. N150/N300, non-paged, no prefetcher, Meta-style RoPE.
Shapes stay stock everywhere except inside the latent caches and the SDPA call."""
import math

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup, num_to_corerange
from models.tt_transformers.tt.rope import get_rot_mats


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

        opt = self.args.decoders_optimizations
        self.activation_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.ACTIVATION)
        self.wqkv_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.WQKV)
        self.wo_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.WO)
        self.kv_cache_dtype = opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.KV_CACHE)
        fid = lambda op: opt.get_math_fidelity(decoder_id=layer_num, op=op, configuration=configuration)
        self.li_qkv_decode_compute_kernel_cfg = fid(OpGroup.LI_QKV_DECODE)
        self.sdpa_decode_compute_kernel_cfg = fid(OpGroup.SDPA_DECODE)
        self.li_o_decode_compute_kernel_cfg = fid(OpGroup.LI_O_DECODE)
        self.sdpa_prefill_compute_kernel_cfg = fid(OpGroup.SDPA_PREFILL)
        self.li_qkv_prefill_compute_kernel_cfg = fid(OpGroup.LI_QKV_PREFILL)
        self.li_o_prefill_compute_kernel_cfg = fid(OpGroup.LI_O_PREFILL)

        layer_name = configuration.get_state_dict_prefix("Attention", layer_num)
        tag = f"palu_rk{self.r_k}_rv{self.r_v}"
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
            cache_file_name=cache_name(f"{tag}_wqkv_sharded_2d"),
        )

        # --- up-projections -------------------------------------------------------------------------
        u_k = state_dict[f"{P}.u_k"]  # [n_kv, r_k, hd]
        u_v = state_dict[f"{P}.u_v"]  # [n_kv, r_v, hd]
        shard_heads = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        # decode: batched (non-broadcast) matmul needs matching batch dims -> replicate over batch
        self.u_k_decode = ttnn.as_tensor(
            u_k.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads,
            cache_file_name=cache_name(f"{tag}_u_k_decode_b{B}"),
        )
        # prefill: zero-padded to [1, n_kv, hd, hd] so the padded latent heads need no slicing
        def pad_u(u, r):
            t = torch.zeros(self.n_kv_heads, self.head_dim, self.head_dim)
            t[:, :r, :] = u
            return t.unsqueeze(0)
        self.u_k_prefill = ttnn.as_tensor(pad_u(u_k, self.r_k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_k_prefill"))
        self.u_v_prefill = ttnn.as_tensor(pad_u(u_v, self.r_v), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_v_prefill"))

        # --- output projections: stock wo (prefill) and fused wo (decode) ------------------------------
        wo_mem = configuration.create_dram_sharded_mem_config((configuration.n_heads * configuration.head_dim) // self.num_devices, configuration.dim)
        def wo_tensor(w, name):
            return ttnn.as_tensor(w.transpose(-1, -2).unsqueeze(0).unsqueeze(0), dtype=self.wo_dtype, layout=ttnn.TILE_LAYOUT,
                                  device=mesh_device, memory_config=wo_mem, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
                                  cache_file_name=cache_name(name))
        self.wo = wo_tensor(state_dict[f"{layer_name}.wo.weight"], "wo")            # same bytes as stock -> shared cache entry
        self.wo_fused = wo_tensor(state_dict[f"{P}.wo_fused"], f"{tag}_wo_fused")

        # --- latent caches ---------------------------------------------------------------------------
        self.layer_past = []
        for r, nm in ((self.r_k, "k"), (self.r_v, "v")):
            z = torch.zeros(B, self.n_local_kv_heads, self.max_seq_len, r)
            self.layer_past.append(ttnn.as_tensor(
                z, dtype=self.kv_cache_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=cache_name(f"{tag}_latcache_{nm}_{tuple(z.shape)}") if weight_cache_path else None,
            ))

        # --- prefill-mode cos/sin for K reconstruction in decode ---------------------------------------
        self.cos_prefill, self.sin_prefill = get_rot_mats(
            head_dim=self.head_dim, device=mesh_device, seq_len=self.max_seq_len, theta=configuration.rope_theta,
            rope_scaling=configuration.rope_scaling, datatype=ttnn.bfloat16, rot_mats_layout=ttnn.TILE_LAYOUT,
        )

    # ------------------------------------------------------------------------------------------------
    def kv_bytes_per_token(self):
        """Bytes per token per layer on one chip (tile-padded), for the results table."""
        bpe = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0, ttnn.bfloat4_b: 0.5625}[self.kv_cache_dtype]
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

        # reconstruct full K and V from the padded latents ([1, nkv, S, 128] @ [1, nkv, 128, 128]) -------
        k_full = ttnn.matmul(k_lat_pad, self.u_k_prefill, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             compute_kernel_config=self.compute_kernel_config_hifi2)
        v_full = ttnn.matmul(v_lat_pad, self.u_v_prefill, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             compute_kernel_config=self.compute_kernel_config_hifi2)

        # RoPE on q and reconstructed K (stock prefill op) --------------------------------------------
        q = ttnn.experimental.rotary_embedding_llama(q_pre, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False)
        k_rot = ttnn.experimental.rotary_embedding_llama(k_full, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], is_decode_mode=False)
        ttnn.deallocate(q_pre)
        ttnn.deallocate(k_full)

        # latents -> caches ------------------------------------------------------------------------
        keys, values = self.layer_past
        for lat_pad, r, cache in ((k_lat_pad, self.r_k, keys), (v_lat_pad, self.r_v, values)):
            lat = lat_pad[:, :, :, :r] if r < self.head_dim else lat_pad
            lat = ttnn.typecast(lat, dtype=cache.dtype)
            if seq_len >= self.min_kv_prefill_shard_seqlen:
                lat = ttnn.interleaved_to_sharded(lat, self._kv_prefill_shard_cfg(seq_len, r))
            ttnn.fill_cache(cache, lat, user_id % self.batch_size_per_device_group)
            ttnn.deallocate(lat)
        ttnn.deallocate(k_lat_pad)
        ttnn.deallocate(v_lat_pad)

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

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        raise NotImplementedError("Task 6")

    def forward(self, x, current_pos, rot_mats=None, user_id=0, mode=Mode.DECODE, page_table=None,
                chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        if mode == Mode.PREFILL:
            return self.forward_prefill(x, rot_mats, user_id, page_table=page_table, chunk_page_table=chunk_page_table,
                                        chunk_start_idx=chunk_start_idx, kv_cache=kv_cache)
        return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)
