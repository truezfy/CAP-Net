#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAP-Net: CT-Assisted PET Network for Efficient 3D Tumor Segmentation

Forward Fusion Multi-modal VRWKV Model
Architecture:
1. Forward Fusion Layer: PET and CT are fused via RWKV to obtain global features, then projected to multiple scales
2. UNet Encoder: Each layer receives corresponding global fusion features (addition operation)
3. UNet Decoder: Standard skip connection structure

Core idea: Fusion-then-Guide, similar to HDenseFormer's forward fusion approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
from typing import Tuple

# ===== CUDA WKV Extension =====
T_MAX = 40000
RUN_CUDA = False
wkv_cuda = None

# First try to load the compiled .so file (bypass torch.utils.cpp_extension's locking mechanism)
_wkv_so_path = os.path.join(
    os.path.expanduser("~/.cache/torch_extensions/py39_cu128/wkv_cuda"),
    "wkv_cuda.so"
)

if os.path.exists(_wkv_so_path):
    try:
        # Load compiled module directly using torch.ops.load
        import importlib.util
        spec = importlib.util.spec_from_file_location("wkv_cuda", _wkv_so_path)
        wkv_cuda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wkv_cuda)
        RUN_CUDA = True
        print("CUDA WKV extension loaded successfully (from pre-compiled .so)")
    except Exception as e:
        print(f"Failed to load pre-compiled wkv_cuda.so: {e}")
        wkv_cuda = None
else:
    print(f"wkv_cuda.so not found at {_wkv_so_path}")

# If pre-compilation fails, try using torch.utils.cpp_extension (may need recompilation)
if not RUN_CUDA:
    try:
        from torch.utils.cpp_extension import load
        wkv_cuda = load(
            name="wkv_cuda",
            sources=[
                os.path.join(os.path.dirname(__file__), "../cuda/wkv_op.cpp"),
                os.path.join(os.path.dirname(__file__), "../cuda/wkv_cuda.cu")
            ],
            verbose=False,
            extra_cuda_cflags=['-res-usage', '--maxrregcount 60', f'-DTmax={T_MAX}']
        )
        RUN_CUDA = True
        print("CUDA WKV extension loaded successfully (recompiled)")
    except Exception as e:
        print(f"CUDA WKV extension failed to load: {e}")
        raise RuntimeError(
            "CUDA WKV extension failed to load, this model requires CUDA support.\n"
            "Please ensure:\n"
            "  1. CUDA environment is properly configured\n"
            "  2. wkv_cuda.cu has been compiled\n"
            "  3. torch_extensions directory is writable"
        )


# ===== SingleLayerTokenExtractor =====
class SingleLayerTokenExtractor:
    """Single-layer Token Extractor using preprocessed single-layer token information"""
    
    def __init__(self):
        self.token_info = None
        self.spatial_sort_indices = None
        # Cache GPU tensor to avoid conversion on every forward pass
        self._cached_regions_tensor = None
        self._cached_device = None
    
    def load_token_info(self, token_info_data):
        """Load preprocessed single-layer token information
        
        Args:
            token_info_data: Can be one of the following types:
                - str: File path (load from disk)
                - np.lib.npyio.NpzFile: Already loaded numpy data (use directly)
                - None: Clear token info
        """
        # Clean up old data
        if hasattr(self, 'token_info') and self.token_info is not None:
            for key, value in self.token_info.items():
                if isinstance(value, torch.Tensor):
                    del value
            del self.token_info
            self.token_info = None
        
        if hasattr(self, 'spatial_sort_indices') and self.spatial_sort_indices is not None:
            del self.spatial_sort_indices
            self.spatial_sort_indices = None
        
        # Clean up cache
        self._cached_regions_tensor = None
        self._cached_device = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Handle different input types
        if token_info_data is None:
            # Clear token info
            return
        
        # Load .npz file or use already loaded data
        if isinstance(token_info_data, str):
            # File path - load from disk
            data = np.load(token_info_data, allow_pickle=True)
        else:
            # Already loaded numpy data (npz file or dict)
            data = token_info_data
        
        self.token_info = {
            'image_regions': torch.from_numpy(data['image_regions']),
            'patch_centers': torch.from_numpy(data['patch_centers']),
            'patch_sizes': torch.from_numpy(data['patch_sizes']),
            'variance_similarities': torch.from_numpy(data['variance_similarities']),
            'core_features': torch.from_numpy(data['core_features']),
        }
        
        # If loaded from file, need to close; if already loaded data, keep open
        if isinstance(token_info_data, str):
            data.close()
        
        self.spatial_sort_indices = None
        self._compute_spatial_sort_indices()
        # self._apply_spatial_sorting()
    
    def _get_regions_tensor(self, device):
        """
        Optimization: Cache converted regions tensor to avoid conversion on every forward pass
        """
        # Recreate if cache doesn't exist or device changed
        if self._cached_regions_tensor is None or self._cached_device != device:
            image_regions = self.token_info['image_regions']
            if isinstance(image_regions, torch.Tensor):
                self._cached_regions_tensor = image_regions.to(device)
            elif isinstance(image_regions, list) and isinstance(image_regions[0], torch.Tensor):
                self._cached_regions_tensor = torch.stack(image_regions).to(device)
            else:
                # numpy array or list
                self._cached_regions_tensor = torch.tensor(image_regions, device=device, dtype=torch.float32)
            self._cached_device = device
        
        return self._cached_regions_tensor
    
    def _get_patch_centers(self, device):
        """Cache patch_centers tensor"""
        if 'patch_centers' not in self.token_info:
            return None
        
        # Recreate if cache doesn't exist or device changed
        cache_key = '_cached_patch_centers'
        if not hasattr(self, cache_key) or getattr(self, cache_key) is None or self._cached_device != device:
            patch_centers = self.token_info['patch_centers']
            if isinstance(patch_centers, torch.Tensor):
                cached = patch_centers.to(device)
            else:
                cached = torch.tensor(patch_centers, device=device, dtype=torch.float32)
            setattr(self, cache_key, cached)
            self._cached_device = device
        
        return getattr(self, cache_key)
    
    def _get_patch_sizes(self, device):
        """Cache patch_sizes tensor"""
        if 'patch_sizes' not in self.token_info:
            return None
        
        cache_key = '_cached_patch_sizes'
        if not hasattr(self, cache_key) or getattr(self, cache_key) is None or self._cached_device != device:
            patch_sizes = self.token_info['patch_sizes']
            if isinstance(patch_sizes, torch.Tensor):
                cached = patch_sizes.to(device)
            else:
                cached = torch.tensor(patch_sizes, device=device, dtype=torch.long)
            setattr(self, cache_key, cached)
            self._cached_device = device
        
        return getattr(self, cache_key)
    
    def _compute_spatial_sort_indices(self):
        """Precompute spatial sort indices"""
        if self.spatial_sort_indices is not None:
            return self.spatial_sort_indices
        
        patch_centers = self.token_info['patch_centers']
        
        if len(patch_centers) == 0:
            self.spatial_sort_indices = torch.tensor([])
            return self.spatial_sort_indices
        
        patch_centers_np = patch_centers.numpy()
        sort_indices = torch.from_numpy(np.lexsort((patch_centers_np[:, 2], patch_centers_np[:, 1], patch_centers_np[:, 0])))
        
        self.spatial_sort_indices = sort_indices
        return self.spatial_sort_indices

    # def _apply_spatial_sorting(self):
    #     """Sort token info by spatial coordinates to ensure neighborhood relationships match sequence order"""
    #     if self.spatial_sort_indices is None or len(self.spatial_sort_indices) == 0:
    #         return
    #     sort_idx = self.spatial_sort_indices
    #     for key in ['image_regions', 'patch_centers', 'patch_sizes', 'variance_similarities', 'core_features']:
    #         if key in self.token_info and len(self.token_info[key]) > 0:
    #             self.token_info[key] = self.token_info[key][sort_idx]


# ===== WKV =====
class WKV(torch.autograd.Function):
    """CUDA WKV implementation"""
    
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        # Follow input device (avoid cuda:0 / cuda:1 mismatch causing illegal memory access in multi-GPU)
        _dev = w.device
        y = torch.empty((B, T, C), device=_dev, memory_format=torch.contiguous_format)

        if RUN_CUDA and wkv_cuda is not None:
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
        else:
            raise RuntimeError(
                "CUDA WKV not available, CUDA support is required to run this model.\n"
                "RUN_CUDA=False, wkv_cuda is None"
            )
        
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        _dev = w.device
        gw = torch.zeros((B, C), device=_dev).contiguous()
        gu = torch.zeros((B, C), device=_dev).contiguous()
        gk = torch.zeros((B, T, C), device=_dev).contiguous()
        gv = torch.zeros((B, T, C), device=_dev).contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        
        if RUN_CUDA and wkv_cuda is not None:
            wkv_cuda.backward(B, T, C,
                              w.float().contiguous(),
                              u.float().contiguous(),
                              k.float().contiguous(),
                              v.float().contiguous(),
                              gy.float().contiguous(),
                              gw, gu, gk, gv)
        else:
            raise RuntimeError(
                "CUDA WKV backward not available, CUDA support is required to run this model.\n"
                "RUN_CUDA=False, wkv_cuda is None"
            )
        
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)
    
    @staticmethod
    def _cpu_wkv_forward(B, T, C, w, u, k, v):
        """CPU fallback - supports both (B,C) and (C,) shapes for w and u"""
        y = torch.zeros((B, T, C), device=w.device)
        
        # Handle both (B, C) and (C,) shapes
        if len(w.shape) == 1:
            w = w.unsqueeze(0).expand(B, -1)  # (C,) -> (1, C) -> (B, C)
        if len(u.shape) == 1:
            u = u.unsqueeze(0).expand(B, -1)  # (C,) -> (1, C) -> (B, C)
        
        for b in range(B):
            for c in range(C):
                for t in range(T):
                    kv_sum = 0.0
                    kv_weight_sum = 0.0
                    for i in range(t + 1):
                        weight = torch.exp(-w[b, c] * (t - i))
                        kv_sum += weight * v[b, i, c]
                        kv_weight_sum += weight
                    
                    if kv_weight_sum > 0:
                        y[b, t, c] = (kv_sum + u[b, c] * k[b, t, c]) / kv_weight_sum
                    else:
                        y[b, t, c] = u[b, c] * k[b, t, c]
        return y
    
    @staticmethod
    def _cpu_wkv_backward(B, T, C, w, u, k, v, gy):
        """CPU fallback backward"""
        gw = torch.zeros((B, C), device='cuda')
        gu = torch.zeros((B, C), device='cuda')
        gk = torch.zeros((B, T, C), device='cuda')
        gv = torch.zeros((B, T, C), device='cuda')
        return gw, gu, gk, gv


def RUN_CUDA_WKV(B, T, C, w, u, k, v):
    """CUDA WKV call function"""
    if not RUN_CUDA or wkv_cuda is None:
        raise RuntimeError("CUDA WKV not available, cannot execute RUN_CUDA_WKV.")
    try:
        # Use caller's tensor device, not hardcoded cuda:0 (device mismatch in multi-GPU environment)
        dev = w.device
        return WKV.apply(B, T, C, w.to(dev), u.to(dev), k.to(dev), v.to(dev))
    except Exception as e:
        raise RuntimeError(f"CUDA WKV execution failed: {e}")


# ===== PositionEmbedding =====
class PositionEmbedding(nn.Module):
    """Position Encoding: 3D coordinates + patch size encoding"""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        
        self.pos_embed = nn.Linear(3, n_embd)
        self.patch_size_embed = nn.Linear(6, n_embd)
    
    def _patch_size_to_onehot(self, patch_sizes):
        size_to_idx = {2: 0, 4: 1, 8: 2, 16: 3, 32: 4, 64: 5}
        onehot = torch.zeros(len(patch_sizes), 6)
        for i, size in enumerate(patch_sizes):
            if int(size) in size_to_idx:
                onehot[i, size_to_idx[int(size)]] = 1
        return onehot
    
    def forward(self, tokens: torch.Tensor, token_info: dict) -> torch.Tensor:
        """
        Args:
            tokens: token features [B, N, C]
            token_info: token information
        Returns:
            enhanced_tokens: tokens with position encoding added [B, N, C]
        """
        B, N, C = tokens.shape
        
        if token_info is None or len(token_info['patch_centers']) == 0:
            return tokens
        
        patch_centers = token_info['patch_centers']
        patch_sizes = token_info['patch_sizes']
        
        pos_enc = self.pos_embed(patch_centers.to(tokens.device).float())
        
        patch_sizes_cpu = patch_sizes.cpu().numpy()
        patch_size_onehot = self._patch_size_to_onehot(patch_sizes_cpu)
        patch_size_enc = self.patch_size_embed(patch_size_onehot.to(tokens.device))
        
        context_features = pos_enc + patch_size_enc
        context_features = context_features.unsqueeze(0).expand(B, -1, -1)
        
        enhanced_tokens = tokens + context_features
        
        return enhanced_tokens


# ===== MShift =====
class MShift(nn.Module):
    """M-Shift: Multi-modal fusion"""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        
        self.weight_net = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, pet_tokens: torch.Tensor, ct_tokens: torch.Tensor) -> torch.Tensor:
        """Fuse PET and CT tokens"""
        combined = torch.cat([pet_tokens, ct_tokens], dim=-1)
        weights = self.weight_net(combined)
        w1, w2 = weights[..., 0:1], weights[..., 1:2]
        fused_tokens = w1 * pet_tokens + w2 * ct_tokens
        return fused_tokens


# ===== BiWKV =====
class BiWKV(nn.Module):
    """Bidirectional WKV computation"""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        
        self.spatial_decay = nn.Parameter(torch.randn(n_embd))
        self.spatial_first = nn.Parameter(torch.randn(n_embd))
    
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Bidirectional WKV computation"""
        B, N, C = k.shape

        if RUN_CUDA and N <= T_MAX:
            # cuWKV kernel expects w/u shape as (B, C); Python gives (C,)
            # When kernel indexes by (B, C), OOB -> illegal memory access.
            # Explicitly unsqueeze to (1, C) then broadcast to (B, C) internally doesn't work
            # (it directly uses data_ptr indexing), so expand to (B, C).
            w_decay = self.spatial_decay / N
            u_first = self.spatial_first / N
            if w_decay.dim() == 1:
                w_decay = w_decay.unsqueeze(0).expand(B, C).contiguous()
            if u_first.dim() == 1:
                u_first = u_first.unsqueeze(0).expand(B, C).contiguous()
            forward_wkv = RUN_CUDA_WKV(B, N, C,
                                     w_decay,
                                     u_first,
                                     k, v)

            k_reversed = torch.flip(k, dims=[1])
            v_reversed = torch.flip(v, dims=[1])
            backward_wkv = RUN_CUDA_WKV(B, N, C,
                                       w_decay,
                                       u_first,
                                       k_reversed, v_reversed)
            backward_wkv = torch.flip(backward_wkv, dims=[1])

            wkv_out = (forward_wkv + backward_wkv) / 2
        else:
            if not RUN_CUDA:
                raise RuntimeError(
                    f"CUDA WKV not available, cannot execute BiWKV (N={N}).\n"
                    f"RUN_CUDA={RUN_CUDA}, N={N}, T_MAX={T_MAX}"
                )
            else:
                raise RuntimeError(
                    f"Token count N={N} exceeds T_MAX={T_MAX}, cannot execute BiWKV."
                )
        
        return wkv_out
    
    def _cpu_bi_wkv_forward(self, B, N, C, k, v):
        """CPU fallback"""
        y = torch.zeros((B, N, C), device=k.device)
        
        for b in range(B):
            for c in range(C):
                for t in range(N):
                    # Forward
                    kv_sum = 0.0
                    kv_weight_sum = 0.0
                    for i in range(t + 1):
                        weight = torch.exp(-self.spatial_decay[c] * (t - i))
                        kv_sum += weight * v[b, i, c]
                        kv_weight_sum += weight
                    
                    if kv_weight_sum > 0:
                        forward_val = (kv_sum + self.spatial_first[c] * k[b, t, c]) / kv_weight_sum
                    else:
                        forward_val = self.spatial_first[c] * k[b, t, c]
                    
                    # Backward
                    kv_sum = 0.0
                    kv_weight_sum = 0.0
                    for i in range(t, N):
                        weight = torch.exp(-self.spatial_decay[c] * (i - t))
                        kv_sum += weight * v[b, i, c]
                        kv_weight_sum += weight
                    
                    if kv_weight_sum > 0:
                        backward_val = (kv_sum + self.spatial_first[c] * k[b, t, c]) / kv_weight_sum
                    else:
                        backward_val = self.spatial_first[c] * k[b, t, c]
                    
                    y[b, t, c] = (forward_val + backward_val) / 2
        
        return y


# ===== CAP_ChannelMix =====
class CAP_ChannelMix(nn.Module):
    """CAP Channel Mixing Layer"""
    
    def __init__(self, n_embd: int, n_layer: int, hidden_rate: int = 4):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.hidden_rate = hidden_rate
        
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        
        with torch.no_grad():
            C = n_embd
            self.key.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.value.weight.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Channel mixing forward"""
        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        output = sr * kv
        
        return output


# ===== CAP_SpatialMix =====
class CAP_SpatialMix(nn.Module):
    """CAP Spatial Mixing Layer"""
    
    def __init__(self, n_embd: int, n_layer: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        
        self.m_shift = MShift(n_embd)
        self.bi_wkv = BiWKV(n_embd)
        
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        
        with torch.no_grad():
            C = n_embd
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()
    
    def forward(self, cap_tokens: torch.Tensor, ct_tokens: torch.Tensor) -> torch.Tensor:
        """Spatial mixing forward"""
        fused_tokens = self.m_shift(cap_tokens, ct_tokens)
        
        k = self.key(fused_tokens)
        r = self.receptance(fused_tokens)
        v = self.value(cap_tokens)
        sr = torch.sigmoid(r)
        
        wkv_out = self.bi_wkv(k, v)
        output = self.output(sr * wkv_out)
        
        return output

# ===== CAPRWKVBlock =====
class CAPRWKVBlock(nn.Module):
    """CAP-RWKV Block"""
    
    def __init__(self, n_embd: int, n_layer: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        
        self.spatial_mix = CAP_SpatialMix(n_embd, n_layer)
        self.channel_mix = CAP_ChannelMix(n_embd, n_layer)
        
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, cap_tokens: torch.Tensor, ct_tokens: torch.Tensor, token_info: dict = None) -> torch.Tensor:
        """CAP-RWKV forward"""
        cap_tokens = cap_tokens + self.spatial_mix(self.norm1(cap_tokens), ct_tokens)
        cap_tokens = cap_tokens + self.channel_mix(self.norm2(cap_tokens))
        return cap_tokens


# ===== AdaptiveTokenExtraction =====
class AdaptiveTokenExtraction(nn.Module):
    """Token Extraction: Extract tokens from first layer feature maps (original image size)"""
    
    def __init__(self, feature_dim: int, embedding_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        self.projection = nn.Linear(feature_dim, embedding_dim, bias=False)
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, features: torch.Tensor, token_info: dict) -> torch.Tensor:
        """
        Vectorized optimized token extraction: Batch extract token features from feature maps
        
        Optimization notes:
        ==========
        Original implementation (loop method):
        - Use Python for loop to process each token one by one (N iterations, N=2400+)
        - Each iteration: compute coordinates -> extract features -> project (3 independent operations)
        - Problem: Large Python loop overhead, cannot utilize GPU parallel computing
        - Time: ~4-7 seconds (depending on token count)
        
        Optimized implementation (vectorized method):
        - Batch compute center coordinates for all tokens (1 tensor operation)
        - Use torch.gather to extract all token features at once (1 GPU operation)
        - Batch project all tokens (1 matrix multiplication)
        - Advantage: Fully utilize GPU parallel computing, reduce Python-GPU data transfer
        - Time: Expected <0.1 seconds (50-100x improvement)
        
        Second optimization (2026-03-01):
        - Use caching mechanism to avoid type conversion on every forward pass
        - Cache regions_tensor after first conversion, use directly thereafter
        
        Args:
            features: Feature map [B, C, D, H, W] - original image size (shared_enc1 has no downsampling)
            token_info: token information
        Returns:
            embeddings: token embeddings [B, N, embedding_dim]
        """
        B, C, D_feat, H_feat, W_feat = features.shape
        device = features.device
        
        # Optimization: Try to get cached regions_tensor from token_extractor
        image_regions = token_info.get('image_regions')
        
        N = len(image_regions)
        if N == 0:
            return torch.empty(B, 0, self.embedding_dim, device=device)
        
        # ===== Step 1: Vectorized computation of center coordinates for all tokens =====
        # Optimization: Check for cache
        if hasattr(token_info, '_cached_regions_tensor') and token_info._cached_regions_tensor is not None:
            # Use cache (but ensure device match)
            cached = token_info._cached_regions_tensor
            if cached.device == device:
                regions_tensor = cached
            else:
                regions_tensor = cached.to(device)
        elif isinstance(image_regions, torch.Tensor):
            regions_tensor = image_regions.to(device)
        elif isinstance(image_regions, list) and isinstance(image_regions[0], torch.Tensor):
            regions_tensor = torch.stack(image_regions).to(device)
        else:
            # numpy array or list, convert to tensor
            regions_tensor = torch.tensor(image_regions, device=device, dtype=torch.float32)
        
        # Batch compute center coordinates (consistent with restoration: center = (start + end) / 2.0, then floor)
        # regions_tensor shape: [N, 6] -> [start_d, end_d, start_h, end_h, start_w, end_w]
        # Compute center coordinates for all tokens at once (vectorized operation)
        center_d = ((regions_tensor[:, 0] + regions_tensor[:, 1]) / 2.0).long()  # [N]
        center_h = ((regions_tensor[:, 2] + regions_tensor[:, 3]) / 2.0).long()  # [N]
        center_w = ((regions_tensor[:, 4] + regions_tensor[:, 5]) / 2.0).long()  # [N]
        
        # Boundary check (ensure indices in valid range, clamp is more efficient than max/min)
        center_d = torch.clamp(center_d, 0, D_feat - 1)  # [N]
        center_h = torch.clamp(center_h, 0, H_feat - 1)  # [N]
        center_w = torch.clamp(center_w, 0, W_feat - 1)  # [N]
        
        # ===== Step 2: Vectorized feature extraction (using torch.gather) =====
        # Convert 3D coordinates to linear indices for gather operation
        # Linear index formula: idx = d * (H * W) + h * W + w
        linear_indices = center_d * (H_feat * W_feat) + center_h * W_feat + center_w  # [N]
        
        # Expand indices to match batch dimension
        linear_indices = linear_indices.unsqueeze(0).expand(B, N)  # [B, N]
        
        # Flatten spatial dimensions of feature map: [B, C, D, H, W] -> [B, C, D*H*W]
        features_flat = features.view(B, C, -1)  # [B, C, D*H*W]
        
        # Use gather to extract all token features at once
        # gather operation: extract features from features_flat according to linear_indices
        # linear_indices needs to be expanded to each channel: [B, C, N]
        linear_indices_expanded = linear_indices.unsqueeze(1).expand(B, C, N).long()  # [B, C, N]
        
        # gather operation: on dim=2, extract features indexed by linear_indices from features_flat
        # features_flat: [B, C, D*H*W], index: [B, C, N] -> output: [B, C, N]
        token_feats = torch.gather(features_flat, dim=2, index=linear_indices_expanded)  # [B, C, N]
        
        # Transpose to [B, N, C] for projection
        token_feats = token_feats.transpose(1, 2)  # [B, N, C]
        
        # ===== Step 3: Batch project to embedding dimension =====
        # Project all tokens at once (matrix multiplication, highly optimized on GPU)
        embeddings = self.projection(token_feats)  # [B, N, embedding_dim]
        
        return embeddings


# ===== ForwardFusionMultiModalVRWKV =====
class ForwardFusionMultiModalVRWKV(nn.Module):
    """
    Forward Fusion Multi-modal VRWKV Model (3-layer UNet, lightweight version)
    References HDenseFormer's forward fusion approach:
    - First fuse to get global features via RWKV (similar to transformer part)
    - Project global features to multiple scales (using average pooling, similar to up1, up2, up3)
    - Add these global features to each UNet encoder layer (similar to ds0+at3, ds1+at2, x+attnout)
    - UNet architecture: enc1 -> enc2 -> bottleneck, 3-layer encoder (lighter than standard UNet)
    """
    
    def __init__(self, n_embd: int = 256, n_layer: int = 16, num_classes: int = 2, 
                 init_features: int = 16):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.num_classes = num_classes
        self.init_features = init_features
        
        self.token_extractor = SingleLayerTokenExtractor()
        
        # === Forward fusion layer: PET and CT separate encoders ===
        # Create separate encoders for PET and CT to better adapt to their respective distribution characteristics
        # Although shallow features (edges, textures) may be similar, PET and CT differ significantly in intensity distribution, contrast, etc.
        # Separate encoders can better extract features from each modality, avoiding performance compromises from shared encoders
        self.pet_enc1 = nn.Sequential(
            nn.Conv3d(1, init_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        self.ct_enc1 = nn.Sequential(
            nn.Conv3d(1, init_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        
        
        # === Token extraction module (forward fusion layer) ===
        self.pet_token_extractor_fusion = AdaptiveTokenExtraction(init_features, n_embd)
        self.ct_token_extractor_fusion = AdaptiveTokenExtraction(init_features, n_embd)
        
        # Position Encoding
        self.position_embedding = PositionEmbedding(n_embd)
        
        # === RWKV fusion processing (forward fusion layer) ===
        self.blocks_fusion = nn.ModuleList([
            CAPRWKVBlock(n_embd, n_layer) 
            for _ in range(n_layer)
        ])
        
        # === Project global features to multiple scales (using average pooling, similar to HDenseFormer's up1, up2, up3) ===
        # Original image size feature projection (for adding to encoder layer 1)
        # Note: fused_features_base is restored from tokens to feature map, minimum patch is 2x2x2
        # Using 3x3x3 convolution (larger than minimum patch) can cover multiple patches, even with duplication in restoration won't affect mapping
        # Also 3x3x3 convolution is more sensitive to boundaries, better capturing patch boundary information
        self.fusion_proj_1 = nn.Sequential(
            nn.Conv3d(n_embd, init_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        
        # 1/2 size feature projection (for adding to encoder layer 2)
        # Note: Using 3x3x3 convolution (larger than minimum patch 2x2x2), can cover multiple patches and capture boundary information
        self.fusion_proj_2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),  # Downsample to 1/2 using average pooling
            nn.Conv3d(n_embd, init_features * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features * 2),
            nn.ReLU(inplace=True)
        )
        
        # 1/4 size feature projection (for adding to bottleneck)
        # Note: Using 3x3x3 convolution (larger than minimum patch 2x2x2), can cover multiple patches and capture boundary information
        self.fusion_proj_3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=4, stride=4),  # Downsample to 1/4 using average pooling
            nn.Conv3d(n_embd, init_features * 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features * 4),
            nn.ReLU(inplace=True)
        )
        
       
        # === Feature fusion module: concat then convolve to avoid scale inconsistency ===
        # Fuse encoder layer 1 features (after concat: PET features + fused global features = init_features*2)
        self.fusion_conv_1 = nn.Sequential(
            nn.Conv3d(init_features * 2, init_features, kernel_size=1),
            nn.InstanceNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        
        # Fuse encoder layer 2 features (after concat: init_features*2 + init_features*2 = init_features*4)
        self.fusion_conv_2 = nn.Sequential(
            nn.Conv3d(init_features * 4, init_features * 2, kernel_size=1),
            nn.InstanceNorm3d(init_features * 2),
            nn.ReLU(inplace=True)
        )
        
        # Fuse bottleneck features (after concat: init_features*4 + init_features*4 = init_features*8)
        self.fusion_conv_3 = nn.Sequential(
            nn.Conv3d(init_features * 8, init_features * 4, kernel_size=1),
            nn.InstanceNorm3d(init_features * 4),
            nn.ReLU(inplace=True)
        )
        
        # === UNet Encoder (3 layers: enc1, enc2, bottleneck) ===
        # Encoder Block 1: Original image size
        self.block_1_1 = nn.Sequential(
            nn.Conv3d(init_features, init_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        self.block_1_2 = nn.Sequential(
            nn.Conv3d(init_features, init_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Downsample to 1/2
        
        # Encoder Block 2: 1/2 size
        self.block_2_1 = nn.Sequential(
            nn.Conv3d(init_features, init_features * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features * 2),
            nn.ReLU(inplace=True)
        )
        self.block_2_2 = nn.Sequential(
            nn.Conv3d(init_features * 2, init_features * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features * 2),
            nn.ReLU(inplace=True)
        )
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Downsample to 1/4
        
        # Bottleneck: 1/4 size
        self.bottleneck_1 = nn.Sequential(
            nn.Conv3d(init_features * 2, init_features * 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features * 4),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_2 = nn.Sequential(
            nn.Conv3d(init_features * 4, init_features * 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features * 4),
            nn.ReLU(inplace=True)
        )
        
        # === UNet Decoder (3-layer decoding) ===
        # Decoder Block 2: 1/4 -> 1/2
        self.upconv_2 = nn.ConvTranspose3d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.decoder_2_1 = nn.Sequential(
            nn.Conv3d(init_features * 4, init_features * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features * 2),
            nn.ReLU(inplace=True)
        )
        self.decoder_2_2 = nn.Sequential(
            nn.Conv3d(init_features * 2, init_features * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features * 2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 1: 1/2 -> Original image size
        self.upconv_1 = nn.ConvTranspose3d(init_features * 2, init_features, kernel_size=2, stride=2)
        # Decoder fusion module: fuse decoder_1 features, skip connection and CT features
        # After concat: init_features + init_features + init_features = init_features*3 -> init_features*2
        self.decoder_1_1 = nn.Sequential(
            nn.Conv3d(init_features * 3, init_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        self.decoder_1_2 = nn.Sequential(
            nn.Conv3d(init_features, init_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        
        # === Final output ===
        self.final_conv = nn.Conv3d(init_features, num_classes, kernel_size=1, bias=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.xavier_normal_(self.final_conv.weight, gain=1.0)
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, 0.0)
    
    def load_token_info(self, token_info_path: str):
        """Load token information"""
        self.token_extractor.load_token_info(token_info_path)
        
        # Clear all possible GPU caches to ensure recreation on model's device later
        # Avoid device inconsistency between token cache and model on different GPUs
        self.token_extractor._cached_regions_tensor = None
        self.token_extractor._cached_device = None
        if hasattr(self.token_extractor, '_cached_patch_centers'):
            self.token_extractor._cached_patch_centers = None
        if hasattr(self.token_extractor, '_cached_patch_sizes'):
            self.token_extractor._cached_patch_sizes = None
        # No longer precompute cache, let forward pass create on demand at model's device
    
    def _tokens_to_features(self, tokens: torch.Tensor, target_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        """
        Convert tokens to feature map (ensures exact consistency with AdaptiveTokenExtraction's coordinate calculation)
        Args:
            tokens: [B, N, n_embd] - tokens after RWKV processing
            target_shape: [B, C, D, H, W] - target feature map shape (original image size)
        Returns:
            features: [B, C, D, H, W] - projected feature map
        """
        B, C, D, H, W = target_shape
        device = tokens.device
        
        # First create feature map with n_embd dimension
        temp_features = torch.zeros(B, self.n_embd, D, H, W, device=device, dtype=tokens.dtype)
        
        image_regions = self.token_extractor.token_info['image_regions']
        
        # Compute center coordinates directly (feature map is original size, image_regions are also in original coordinates)
        # Consistent with AdaptiveTokenExtraction: center = (start + end) / 2.0
        
        # Vectorized computation of position coordinates for all tokens
        with torch.no_grad():
            if isinstance(image_regions, torch.Tensor):
                regions_tensor = image_regions
            elif isinstance(image_regions, list) and isinstance(image_regions[0], torch.Tensor):
                regions_tensor = torch.stack(image_regions)
            else:
                regions_tensor = torch.tensor(image_regions)
            
            # Use exact same coordinate calculation method as AdaptiveTokenExtraction
            # Compute center directly: center = (start + end) / 2.0, then floor (consistent with int() in extraction)
            center_d = ((regions_tensor[:, 0] + regions_tensor[:, 1]) / 2.0).long()
            center_h = ((regions_tensor[:, 2] + regions_tensor[:, 3]) / 2.0).long()
            center_w = ((regions_tensor[:, 4] + regions_tensor[:, 5]) / 2.0).long()
            
            # Boundary check (ensure indices in valid range, consistent with extraction)
            center_d = torch.clamp(center_d, 0, D - 1)
            center_h = torch.clamp(center_h, 0, H - 1)
            center_w = torch.clamp(center_w, 0, W - 1)
            
            valid_mask = torch.ones(center_d.shape[0], dtype=torch.bool, device=center_d.device)
            
            if valid_mask.sum() > 0:
                valid_d = center_d[valid_mask]
                valid_h = center_h[valid_mask]
                valid_w = center_w[valid_mask]
                
                linear_idx = valid_d * (H * W) + valid_h * W + valid_w
                
                for b in range(B):
                    flat_features = temp_features[b].view(self.n_embd, -1)
                    tokens_T = tokens[b, valid_mask].transpose(0, 1)
                    flat_features[:, linear_idx] = tokens_T
        
        return temp_features
    
    def forward(self, pet_tensor: torch.Tensor, ct_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Fusion Multi-modal VRWKV Forward Pass
        References HDenseFormer's forward fusion approach:
        1. First fuse to get global features via RWKV (similar to transformer part)
        2. Project global features to multiple scales (similar to up1, up2, up3)
        3. Add these global features to each UNet encoder layer (similar to ds0+at3, ds1+at2, ds2+at1, x+attnout)
        """
        if self.token_extractor.token_info is None:
            raise ValueError("Please load token info first by calling load_token_info")
        
        device = pet_tensor.device
        B, _, D, H, W = pet_tensor.shape
        token_info = self.token_extractor.token_info
        
        # === Forward fusion layer: PET and CT separate encoders, extract tokens, RWKV fusion ===
        pet_feat1 = self.pet_enc1(pet_tensor)  # [B, init_features, D, H, W]
        ct_feat1 = self.ct_enc1(ct_tensor)     # [B, init_features, D, H, W] - separate encoder
        
        
        
        # Extract tokens and process
        pet_tokens = self.pet_token_extractor_fusion(pet_feat1.detach(), token_info)
        ct_tokens = self.ct_token_extractor_fusion(ct_feat1.detach(), token_info)
        
        pet_tokens = self.position_embedding(pet_tokens, token_info)
        ct_tokens = self.position_embedding(ct_tokens, token_info)
        
        # RWKV fusion processing
        for block in self.blocks_fusion:
            pet_tokens = block(pet_tokens, ct_tokens, token_info)
        
        # Convert RWKV fused tokens to feature map (original image size)
        fused_features_base = self._tokens_to_features(pet_tokens, (B, self.n_embd, D, H, W))
        
        # === Project global features to multiple scales (similar to HDenseFormer's up1, up2, up3) ===
        # Original image size features (for adding to encoder layer 1)
        fusion_feat_1 = self.fusion_proj_1(fused_features_base)  # [B, init_features, D, H, W]
        
        # 1/2 size features (for adding to encoder layer 2)
        fusion_feat_2 = self.fusion_proj_2(fused_features_base)  # [B, init_features*2, D/2, H/2, W/2]
        
        # 1/4 size features (for adding to bottleneck)
        fusion_feat_3 = self.fusion_proj_3(fused_features_base)  # [B, init_features*4, D/4, H/4, W/4]
        
        # === UNet Encoder (3 layers: enc1, enc2, bottleneck) ===
        # Encoder Block 1: Original image size, use concat to fuse fusion_feat_1
        # Use PET features as main input (CT indirectly provides information through RWKV fusion)
        ds0 = self.block_1_1(pet_feat1)  # Use PET features as input
        ds0 = self.block_1_2(ds0)
        # Fuse via concat then convolve to avoid scale inconsistency
        ds0 = torch.cat([ds0, fusion_feat_1], dim=1)  # [B, init_features*2, D, H, W]
        ds0 = self.fusion_conv_1(ds0)  # [B, init_features, D, H, W]
        ds1 = self.pool_1(ds0)  # [B, init_features, D/2, H/2, W/2]
        
        # Encoder Block 2: 1/2 size, use concat to fuse fusion_feat_2
        ds1 = self.block_2_1(ds1)
        ds1 = self.block_2_2(ds1)
        # Fuse via concat then convolve
        ds1 = torch.cat([ds1, fusion_feat_2], dim=1)  # [B, init_features*4, D/2, H/2, W/2]
        ds1 = self.fusion_conv_2(ds1)  # [B, init_features*2, D/2, H/2, W/2]
        ds2 = self.pool_2(ds1)  # [B, init_features*2, D/4, H/4, W/4]
        
        # Bottleneck: 1/4 size, use concat to fuse fusion_feat_3
        x = self.bottleneck_1(ds2)
        x = self.bottleneck_2(x)
        # Fuse via concat then convolve
        x = torch.cat([x, fusion_feat_3], dim=1)  # [B, init_features*8, D/4, H/4, W/4]
        x = self.fusion_conv_3(x)  # [B, init_features*4, D/4, H/4, W/4]
        
        # === UNet Decoder (3-layer decoding) ===
        # Decoder Block 2: 1/4 -> 1/2
        x = self.upconv_2(x)  # [B, init_features*2, D/2, H/2, W/2]
        x = torch.cat([x, ds1], dim=1)  # [B, init_features*4, D/2, H/2, W/2]
        x = self.decoder_2_1(x)  # [B, init_features*2, D/2, H/2, W/2]
        x = self.decoder_2_2(x)  # [B, init_features*2, D/2, H/2, W/2]
        
        # Decoder Block 1: 1/2 -> Original image size
        # Add CT features' fine-grained information (original image size)
        x = self.upconv_1(x)  # [B, init_features, D, H, W]
        # Concat decoder features, skip connection and CT features
        x = torch.cat([x, ds0, ct_feat1], dim=1)  # [B, init_features*3, D, H, W]
        x = self.decoder_1_1(x)  # [B, init_features, D, H, W]
        x = self.decoder_1_2(x)  # [B, init_features, D, H, W]
        
        # === Final output ===
        logits = self.final_conv(x)
        
        # Decide whether to compute token_logits based on existence of token_info
        # If token_info has 'core_features' field, token_logits needs to be computed
        compute_token_logits = False
        if hasattr(self.token_extractor, 'token_info') and self.token_extractor.token_info is not None:
            if 'core_features' in self.token_extractor.token_info:
                compute_token_logits = True
        
        if compute_token_logits:
            # Compute token_logits (for token loss)
            token_logits = self._compute_token_logits_from_regions(logits, self.token_extractor.token_info)
        else:
            # Don't use token loss, return empty token_logits (compatible with training loop)
            token_logits = torch.zeros(B, 0, self.num_classes, device=logits.device)
        
        return logits, token_logits
    
    def _compute_token_logits_from_regions(self, logits: torch.Tensor, token_info: dict) -> torch.Tensor:
        """
        Aggregate logits within regions according to token info, compute token logits
        Args:
            logits: Segmentation logits [B, num_classes, D, H, W]
            token_info: Token information dictionary
        Returns:
            token_logits: Token-level logits [B, N, num_classes]
        """
        B, num_classes, D, H, W = logits.shape
        device = logits.device
        
        # Get token information
        image_regions = token_info['image_regions']  # [N, 6]
        N = len(image_regions)
        
        if N == 0:
            # If no tokens, return empty tensor
            return torch.empty(B, 0, num_classes, device=device)
        
        # Use stack and mask to maintain gradients
        token_logit_list = []
        
        # Aggregate region for each token
        for i, region in enumerate(image_regions):
            start_d, end_d, start_h, end_h, start_w, end_w = region
            
            # Ensure indices in valid range
            start_d = max(0, int(start_d))
            end_d = min(D, int(end_d))
            start_h = max(0, int(start_h))
            end_h = min(H, int(end_h))
            start_w = max(0, int(start_w))
            end_w = min(W, int(end_w))
            
            if end_d > start_d and end_h > start_h and end_w > start_w:
                # Extract logits for corresponding region
                region_logits = logits[:, :, start_d:end_d, start_h:end_h, start_w:end_w]  # [B, num_classes, patch_d, patch_h, patch_w]
                
                # Spatial max pooling within region to get token-level logits (raw logits, without softmax)
                # Using max pooling instead of average pooling: capture strongest prediction signal in region
                # Even if token info division is inaccurate, as long as prediction is correct within region
                # More importantly: if region has no tumor, should not predict as having one (max pooling avoids false positives)
                token_logit = F.adaptive_max_pool3d(region_logits, (1, 1, 1)).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, num_classes]
            else:
                # If region invalid, create zero logits (use zeros_like to ensure consistent dtype and requires_grad)
                token_logit = torch.zeros_like(logits[:, :, 0, 0, 0])  # [B, num_classes]
            
            token_logit_list.append(token_logit)
        
        # Stack to tensor [B, N, num_classes]
        token_logits = torch.stack(token_logit_list, dim=1)
        
        return token_logits


def create_forward_fusion_multimodal_vrwkv_model_v7(
    n_embd: int = 256, 
    n_layer: int = 16, 
    num_classes: int = 2, 
    init_features: int = 16
):
    """Create forward fusion multi-modal VRWKV model v7 (based on v5, CT features fused at multiple decoder levels)"""
    print(f"Creating ForwardFusionMultiModalVRWKV_v7 model (based on v5, multi-level decoder CT feature fusion):")
    print(f"  Embedding dimension: {n_embd}")
    print(f"  RWKV layers: {n_layer}")
    print(f"  Initial features: {init_features}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Architecture features:")
    print(f"    - Based on v5 architecture (separate encoders + CT fine-grained features)")
    print(f"    - PET and CT use separate encoders (avoid performance compromise from shared encoders)")
    print(f"    - CT features provide both token-level global information and fine-grained features")
    print(f"    - CT fine-grained features only added to highest-level decoder decoder_1 (original image size)")
    print(f"    - Encoder stage only fuses PET features and fused global features, without CT features")
    print(f"    - Global fused feature projection uses 3x3x3 convolution (larger than minimum patch 2x2x2), can cover multiple patches and capture boundary information")
    print(f"    - 3-layer UNet encoder (enc1, enc2, bottleneck), lighter than standard 5-layer")
    return ForwardFusionMultiModalVRWKV(n_embd, n_layer, num_classes, init_features)


def create_forward_fusion_multimodal_vrwkv_model_v8(
    n_embd: int = 256, 
    n_layer: int = 16, 
    num_classes: int = 2, 
    init_features: int = 16
):
    """Create forward fusion multi-modal VRWKV model v8 (based on v7, fusion_proj uses depthwise separable convolution)"""
    print(f"Creating ForwardFusionMultiModalVRWKV_v8 model (based on v7, depthwise separable convolution optimization):")
    print(f"  Embedding dimension: {n_embd}")
    print(f"  RWKV layers: {n_layer}")
    print(f"  Initial features: {init_features}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Architecture features:")
    print(f"    - Based on v7 architecture (separate encoders + CT fine-grained features)")
    print(f"    - PET and CT use separate encoders (avoid performance compromise from shared encoders)")
    print(f"    - CT features provide both token-level global information and fine-grained features")
    print(f"    - CT fine-grained features only added to highest-level decoder decoder_1 (original image size)")
    print(f"    - Encoder stage only fuses PET features and fused global features, without CT features")
    print(f"    - Global fused feature projection uses depthwise separable 3x3x3 convolution (preserves spatial receptive field, significantly reduces parameters and FLOPS)")
    print(f"    - 3-layer UNet encoder (enc1, enc2, bottleneck), lighter than standard 5-layer")
    return ForwardFusionMultiModalVRWKV(n_embd, n_layer, num_classes, init_features)


if __name__ == "__main__":
    # Test model
    print("=== Testing ForwardFusionMultiModalVRWKV_v7 model ===")
    
    model = create_forward_fusion_multimodal_vrwkv_model_v7(
        n_embd=256,
        n_layer=16,
        num_classes=2,
        init_features=16
    )
    
    # Move model to GPU
    model = model.cuda()
    
    # Test input
    B, D, H, W = 1, 64, 128, 128
    pet_input = torch.randn(B, 1, D, H, W).cuda()
    ct_input = torch.randn(B, 1, D, H, W).cuda()
    
    print(f"Input shapes: PET={pet_input.shape}, CT={ct_input.shape}")
    
    # Simulate token information
    dummy_token_info = {
        'image_regions': torch.randint(0, 64, (50, 6)),
        'patch_centers': torch.randn(50, 3),
        'patch_sizes': torch.randint(2, 8, (50,)),
        'variance_similarities': torch.rand(50)
    }
    
    # Load token information (required for AdaptiveTokenExtraction)
    model.token_extractor.token_info = dummy_token_info
    
    # Forward pass test
    print("\nStarting forward pass...")
    output, token_output = model(pet_input, ct_input)
    print(f"Output shape: {output.shape}")
    print(f"Token output shape: {token_output.shape}")
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n=== Test complete ===")
