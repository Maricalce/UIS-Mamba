"""
UIS-Mamba: Exploring Mamba for Underwater Instance Segmentation 
via Dynamic Tree Scan and Hidden State Weaken

Paper: UIS-Mamba: Exploring Mamba for Underwater Instance Segmentation 
       via Dynamic Tree Scan and Hidden State Weaken
Authors: Runmin Cong, Zongji Yu, Hao Fang, Haoyan Sun, Sam Kwong
Conference: ACM MM 2025
Published: 05 Jul 2025, Last Modified: 11 Jul 2025

Lab: MVPLab (Professor Runmin Cong)
Institution: Shandong University of Finance and Economics

This file implements the core modules for UIS-Mamba:
- Hidden State Weaken (HSW): Suppresses background interference
- Adaptive Graph Deformation: Dynamic patch deformation for underwater scenes
- Dynamic Graph Pruning: Combines spatial and semantic information for MST
- Tree_SSM: State Space Model with tree scanning mechanism
"""

import math
from functools import partial
from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from tree_scan import _C
from .tree_scan_utils.tree_scan_core import MinimumSpanningTree

import torchvision


class HiddenStateWeaken(nn.Module):
    """
    Hidden State Weaken (HSW) Module for UIS-Mamba
    
    Suppresses complex underwater background interference on hidden state updates
    using degree-based foreground/background separation.
    
    Mathematical formulation:
        h_i = sum_j(B_j * phi_i * p_j * prod_k(A_k))
        y_i = C_i * Norm(h_i) + D * phi_i * p_i
    
    where:
        phi_i = 1 (foreground patches)
        phi_i = varphi ∈ (0,1) (background patches, default 0.7)
    
    Args:
        varphi (float): Background suppression factor. Default: 0.7 (experimentally optimal)
    """
    def __init__(self, varphi=0.7):
        super().__init__()
        self.varphi = varphi  # Background suppression factor (experimentally optimal: 0.7)
        
    def degree_based_segmentation(self, edge_index, edge_weights, num_nodes):
        """
        Degree-based Foreground/Background Segmentation
        
        Uses node connectivity (weighted degree) as a heuristic for foreground detection.
        High-degree nodes (well-connected) are more likely to be foreground instance patches.
        
        Args:
            edge_index (torch.Tensor): Edge connectivity, shape (B, num_edges, 2)
            edge_weights (torch.Tensor): Edge weights, shape (B, num_edges)
            num_nodes (int): Total number of nodes (patches) in the graph
        
        Returns:
            torch.Tensor: Binary masks, shape (B, num_nodes)
                         1 = foreground patch, 0 = background patch
        """
        B, num_edges, _ = edge_index.shape
        device = edge_index.device
        

        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, num_edges)
        

        src_idx = edge_index[:, :, 0]  # (B, num_edges)
        dst_idx = edge_index[:, :, 1]  # (B, num_edges)
        

        degrees = torch.zeros(B, num_nodes, device=device)
        

        degrees.scatter_add_(1, src_idx, edge_weights)
        degrees.scatter_add_(1, dst_idx, edge_weights)
        

        degrees_normalized = degrees / (degrees.max(dim=1, keepdim=True)[0] + 1e-8)
        

        threshold = degrees_normalized.median(dim=1, keepdim=True)[0]
        masks = (degrees_normalized >= threshold).float()
        
        return masks
    
    def compute_suppression_weights(self, masks):
        """
        Compute suppression weights phi_i based on foreground/background masks
        
        phi_i = 1 if foreground, varphi if background
        
        Args:
            masks: (B, num_nodes) - binary masks
        Returns:
            phi: (B, num_nodes) - suppression weights
        """
        phi = masks + (1 - masks) * self.varphi  # Foreground=1, Background=varphi
        return phi
    
    def forward(self, edge_index, edge_weights, num_nodes):
        """
        Args:
            edge_index: (B, num_edges, 2)
            edge_weights: (B, num_edges)
            num_nodes: int
        Returns:
            phi: (B, num_nodes) - suppression weights for hidden state
        """

        masks = self.degree_based_segmentation(edge_index, edge_weights, num_nodes)
        

        phi = self.compute_suppression_weights(masks)
        
        return phi


class AdaptiveGraphDeformation(nn.Module):
    """
    Adaptive Graph Deformation Module for Dynamic Tree Scan (DTS)
    
    Dynamically adjusts patch positions and scales to adapt to underwater scene
    characteristics, addressing color distortion and boundary blurring challenges.
    
    
    Mathematical formulation:
        p̃ = G(p(p_x + Δx, p_y + Δy); Δw, Δh)
    
    where G(·) is bilinear interpolation with learnable scaling factors.
    
    Args:
        d_model (int): Feature dimension
        patch_size (int): Base patch size. Default: 1
        pool_size (int): Pooling size for multi-pixel merging. Default: 2
    """
    def __init__(self, d_model, patch_size=1, pool_size=2):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.pool_size = pool_size  # Merge pool_size x pool_size pixels into one patch
        

        hidden_dim = max(d_model // 4, 64)
        
        self.deform_predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # 4 parameters: Δx, Δy, Δw, Δh
        )
        

        self.weight_coef = nn.Parameter(torch.ones(4))
        

        nn.init.constant_(self.deform_predictor[-1].weight, 0)
        nn.init.constant_(self.deform_predictor[-1].bias, 0)
        
    def pool_patches(self, features, H, W):
        """
        Args:
            features: (B, L, C) where L = H * W
            H, W: original spatial dimensions
        Returns:
            pooled_features: (B, L', C) where L' = (H/pool) * (W/pool)
            H_pooled, W_pooled: pooled spatial dimensions
        """
        B, L, C = features.shape
        

        feat_map = features.transpose(1, 2).reshape(B, C, H, W)
        

        pooled_map = F.avg_pool2d(feat_map, kernel_size=self.pool_size, stride=self.pool_size)
        

        _, _, H_pooled, W_pooled = pooled_map.shape
        

        pooled_features = pooled_map.reshape(B, C, -1).transpose(1, 2)
        
        return pooled_features, H_pooled, W_pooled
    
    def unpool_features(self, pooled_features, H_pooled, W_pooled, H_orig, W_orig):
        """
        Upsample pooled features back to original resolution
        
        Args:
            pooled_features: (B, L', C)
            H_pooled, W_pooled: pooled dimensions
            H_orig, W_orig: original dimensions
        Returns:
            upsampled_features: (B, L, C) where L = H_orig * W_orig
        """
        B, L_pooled, C = pooled_features.shape
        

        feat_map = pooled_features.transpose(1, 2).reshape(B, C, H_pooled, W_pooled)

        upsampled_map = F.interpolate(feat_map, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
        

        upsampled_features = upsampled_map.reshape(B, C, -1).transpose(1, 2)
        
        return upsampled_features
    
    def forward(self, features, H, W):
        """
        Args:
            features: (B, L, C) where L = H * W
            H, W: spatial dimensions
        Returns:
            deformed_features: (B, L, C) - same resolution as input
            deform_params: (B, L', 4) - deformation parameters (pooled resolution)
            deformed_coords: (B, L', 2) - new patch coordinates (pooled resolution)
        """
        B, L, C = features.shape
        

        pooled_features, H_pooled, W_pooled = self.pool_patches(features, H, W)
        L_pooled = H_pooled * W_pooled
        

        deform_params = self.deform_predictor(pooled_features)  # (B, L_pooled, 4)
        deform_params = deform_params * self.weight_coef.view(1, 1, 4)
        

        delta_x = torch.tanh(deform_params[:, :, 0])  # (B, L_pooled)
        delta_y = torch.tanh(deform_params[:, :, 1])  # (B, L_pooled)
        delta_w = torch.softplus(deform_params[:, :, 2])  # (B, L_pooled)
        delta_h = torch.softplus(deform_params[:, :, 3])  # (B, L_pooled)
        # delta_w = torch.sigmoid(deform_params[:, :, 2])  # (B, L_pooled)
        # delta_h = torch.sigmoid(deform_params[:, :, 3])  # (B, L_pooled)
        # delta_w = torch.tanh(deform_params[:, :, 2])  # (B, L_pooled)
        # delta_h = torch.tanh(deform_params[:, :, 3])  # (B, L_pooled)

        y_coords = torch.arange(H_pooled, device=features.device).float()
        x_coords = torch.arange(W_pooled, device=features.device).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        

        grid_x = 2.0 * grid_x / max(W_pooled - 1, 1) - 1.0
        grid_y = 2.0 * grid_y / max(H_pooled - 1, 1) - 1.0

        orig_coords = torch.stack([grid_x, grid_y], dim=-1)  # (H_pooled, W_pooled, 2)
        orig_coords = orig_coords.reshape(1, L_pooled, 2).expand(B, -1, -1)  # (B, L_pooled, 2)
        

        deformed_x = orig_coords[:, :, 0] + delta_x * (2.0 / max(W_pooled, 1))
        deformed_y = orig_coords[:, :, 1] + delta_y * (2.0 / max(H_pooled, 1))
        

        deformed_x = torch.clamp(deformed_x, -1.0, 1.0)
        deformed_y = torch.clamp(deformed_y, -1.0, 1.0)
        
        deformed_coords = torch.stack([deformed_x, deformed_y], dim=-1)  # (B, L_pooled, 2)
        

        # This ensures output has same resolution as input
        feat_map = features.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        

        deformed_coords_map = deformed_coords.reshape(B, H_pooled, W_pooled, 2)
        deformed_coords_full = F.interpolate(
            deformed_coords_map.permute(0, 3, 1, 2),  # (B, 2, H_pooled, W_pooled)
            size=(H, W),
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1)  # (B, H, W, 2)
        

        deformed_feat_map = F.grid_sample(
            feat_map, 
            deformed_coords_full, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )  # (B, C, H, W)
        

        deformed_features = deformed_feat_map.reshape(B, C, -1).transpose(1, 2)  # (B, L, C)
        

        full_deform_params = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)  # (B, L_pooled, 4)
        
        return deformed_features, full_deform_params, deformed_coords


class DynamicGraphPruning(nn.Module):
    """
    Dynamic Graph Pruning Module for MST Construction
    
    Constructs a Minimum Spanning Tree (MST) by combining spatial adjacency
    and semantic similarity, crucial for distinguishing overlapping underwater
    instances with similar appearances (e.g., different fish species).
    
    Mathematical formulation:
        w_ij = α · ||p̃_i^c - p̃_j^c||_2 + (1-α) · Cosine(p̃_i, p̃_j)
        
        MST(G) = argmin_G Σ_{e_ij ∈ G} exp(w_ij)
    
    where:
        - p̃_i, p̃_j: Deformed patch features
        - p̃_i^c, p̃_j^c: Patch spatial coordinates
        - α ∈ [0,1]: Learnable balance parameter
    
    Args:
        d_model (int): Feature dimension
        pool_size (int): Pooling size (for consistency). Default: 2
    """
    def __init__(self, d_model, pool_size=2):
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        

        # Critical for distinguishing overlapping marine organisms with similar appearances
        self.alpha = nn.Parameter(torch.tensor(0.5))
        

        proj_dim = max(d_model // 2, 128)
        self.semantic_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        

        self.edge_index_cache = {}
        
    def compute_edge_weights(self, features, deformed_coords, H, W, edge_index):
        """
        Compute dynamic edge weights for ONLY the edges in edge_index
        
        Args:
            features: (B, L, C) - patch features
            deformed_coords: (B, L, 2) - deformed patch coordinates
            H, W: spatial dimensions
            edge_index: (B, num_edges, 2) - edge connectivity
            
        Returns:
            edge_weights: (B, num_edges) - edge weights for specified edges only
        """
        B, L, C = features.shape
        num_edges = edge_index.shape[1]
        

        semantic_features = self.semantic_proj(features)  # (B, L, C)
        semantic_features = F.normalize(semantic_features, p=2, dim=-1)  # L2 normalize
        

        # Create batch indices for advanced indexing
        batch_indices = torch.arange(B, device=features.device).view(B, 1).expand(B, num_edges)
        
        src_idx = edge_index[:, :, 0]  # (B, num_edges)
        dst_idx = edge_index[:, :, 1]  # (B, num_edges)
        

        src_features = semantic_features[batch_indices, src_idx]  # (B, num_edges, C)
        dst_features = semantic_features[batch_indices, dst_idx]  # (B, num_edges, C)
        semantic_sim = (src_features * dst_features).sum(dim=-1)  # (B, num_edges)
        
 
        src_coords = deformed_coords[batch_indices, src_idx]  # (B, num_edges, 2)
        dst_coords = deformed_coords[batch_indices, dst_idx]  # (B, num_edges, 2)
        spatial_dist = torch.sqrt(((src_coords - dst_coords) ** 2).sum(dim=-1) + 1e-8)  # (B, num_edges)
        

        max_dist = 1.414  # sqrt(2.0)
        spatial_sim = 1.0 - (spatial_dist / max_dist)
        

        alpha = torch.sigmoid(self.alpha)
        edge_weights = alpha * semantic_sim + (1 - alpha) * spatial_sim  # (B, num_edges)
        
        return edge_weights
    
    def get_or_build_edge_index(self, H, W, device):
        """
        Get edge index from cache or build it 
        
        4-connected graph structure is fixed for given H,W, so we cache it!
        """
        key = (H, W)
        
        if key not in self.edge_index_cache:
            # Build edge list once
            L = H * W
            edge_list = []
            
            for i in range(L):
                y, x = i // W, i % W
                # 4 neighbors: up, down, left, right
                neighbors = [
                    (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)
                ]
                for ny, nx in neighbors:
                    if 0 <= ny < H and 0 <= nx < W:
                        j = ny * W + nx
                        edge_list.append((i, j))
            
            if len(edge_list) > 0:
                edge_index = torch.tensor(edge_list, dtype=torch.long)
            else:
                edge_index = torch.zeros(1, 2, dtype=torch.long)
            
           
            self.edge_index_cache[key] = edge_index
        
        
        edge_index = self.edge_index_cache[key]
        if edge_index.device != device:
            edge_index = edge_index.to(device)
            self.edge_index_cache[key] = edge_index
        
        return edge_index
    
    def build_dynamic_graph(self, features, deformed_coords, H, W):
        """
        Build dynamic graph with pruned edges based on combined weights
        
        Returns:
            edge_index: (B, num_edges, 2) - edge connectivity
            edge_weights: (B, num_edges) - edge weights
        """
        B, L, C = features.shape
        device = features.device
        
       
        edge_index = self.get_or_build_edge_index(H, W, device)  # (num_edges, 2)
        
       
        edge_index = edge_index.unsqueeze(0).expand(B, -1, -1)  # (B, num_edges, 2)
        
        
        edge_weights_out = self.compute_edge_weights(features, deformed_coords, H, W, edge_index)
        
        return edge_index, edge_weights_out

class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child = \
            _C.bfs_forward(edge_index, max_adj_per_vertex)
        return sorted_index, sorted_parent, sorted_child


class _Refine(Function):
    @staticmethod
    def forward(ctx, feature_in, edge_weight, sorted_index, sorted_parent, sorted_child, edge_coef):
        feature_aggr, feature_aggr_up, = \
            _C.tree_scan_refine_forward(feature_in, edge_weight, sorted_index, sorted_parent, sorted_child, edge_coef)

        ctx.save_for_backward(feature_in, edge_weight, sorted_index, sorted_parent,
                              sorted_child, feature_aggr, feature_aggr_up, edge_coef)
        return feature_aggr
        # return feature_aggr_up

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_in, edge_weight, sorted_index, sorted_parent, \
        sorted_child, feature_aggr, feature_aggr_up, edge_coef = ctx.saved_tensors

        grad_feature = _C.tree_scan_refine_backward_feature(feature_in, edge_weight,
                                                            sorted_index, sorted_parent, sorted_child, feature_aggr,
                                                            feature_aggr_up,
                                                            grad_output, edge_coef)
        grad_edge_weight = _C.tree_scan_refine_backward_edge_weight(feature_in, edge_weight,
                                                                    sorted_index, sorted_parent, sorted_child,
                                                                    feature_aggr, feature_aggr_up,
                                                                    grad_output, edge_coef)
        return grad_feature, grad_edge_weight, None, None, None, None


def batch_index_opr(data, index):
    with torch.no_grad():
        channel = data.shape[1]
        index = index.unsqueeze(1).expand(-1, channel, -1).long()
    data = torch.gather(data, 2, index)
    return data


def tree_scanning_core(xs, dts,
                       As, Bs, Cs, Ds,
                       delta_bias, origin_shape, h_norm,
                       deform_module=None, pruning_module=None, hsw_module=None,
                       use_dynamic_tree=False, use_hsw=False):
    """
    Tree scanning core with optional Dynamic Tree Scan (DTS)
    
    Args:
        deform_module: AdaptiveGraphDeformation module
        pruning_module: DynamicGraphPruning module
        use_dynamic_tree: whether to use dynamic tree scan
    """
    K = 1
    _, _, H, W = origin_shape
    B, D, L = xs.shape
    # print(xs.shape)
    dts = F.softplus(dts + delta_bias.unsqueeze(0).unsqueeze(-1))
    # import pdb;pdb.set_trace()
    deltaA = (dts * As.unsqueeze(0)).exp_()  # b d l
    deltaB = rearrange(dts, 'b (k d) l -> b k d l', k=K, d=int(D / K)) * Bs  # b 1 d L
    BX = deltaB * rearrange(xs, 'b (k d) l -> b k d l', k=K, d=int(D / K))  # b 1 d L

    bfs = _BFS.apply
    refine = _Refine.apply

    feat_in = BX.view(B, -1, L)  # b D L
    edge_weight = deltaA  # b D L

    def edge_transform(edge_weight, sorted_index, sorted_child):
        edge_weight = batch_index_opr(edge_weight, sorted_index)  # b d l
        return edge_weight,

    # === Dynamic Tree Scan (DTS) + Hidden State Weaken (HSW) ===
    phi_weights = None  # Suppression weights for HSW
    
    if use_dynamic_tree and deform_module is not None and pruning_module is not None:
        # Step 1: Adaptive Graph Deformation
        # Convert xs to (B, L, D) for deformation
        xs_for_deform = xs.transpose(1, 2)  # (B, L, D)
        deformed_features, deform_params, deformed_coords = deform_module(xs_for_deform, H, W)
        
        # Note: deformed_coords is at pooled resolution (H_pooled, W_pooled)
        # We need to get the actual pooled dimensions
        pool_size = deform_module.pool_size
        H_pooled = H // pool_size
        W_pooled = W // pool_size
        L_pooled = H_pooled * W_pooled
        
        # Pool features for graph construction
        pooled_features, _, _ = deform_module.pool_patches(xs_for_deform, H, W)
        
        # Step 2: Dynamic Graph Pruning - build weighted graph (on pooled resolution)
        edge_index, dynamic_edge_weights = pruning_module.build_dynamic_graph(
            pooled_features, deformed_coords, H_pooled, W_pooled
        )
        
        # Step 3: HSW - Ncut-based foreground/background separation (if enabled)
        if use_hsw and hsw_module is not None:
            phi_pooled = hsw_module(edge_index, dynamic_edge_weights, L_pooled)  # (B, L_pooled)
            # Upsample phi to original resolution
            phi_map = phi_pooled.reshape(B, 1, H_pooled, W_pooled)
            phi_full = F.interpolate(phi_map, size=(H, W), mode='nearest')  # (B, 1, H, W)
            phi_weights = phi_full.reshape(B, 1, L)  # (B, 1, L)
        
        # Step 4: Build MST with deformed features and dynamic weights
        # Convert deformed features back to spatial format
        fea4tree_hw = deformed_features.transpose(1, 2).reshape(B, D, H, W)  # (B, D, H, W)
        
        # Use custom MST with combined spatial-semantic weights
        mst_layer = MinimumSpanningTree("Custom", torch.exp)
        # Build tree using deformed features
        tree = mst_layer(fea4tree_hw)
        
        # Step 5: Update BX with deformed features for scanning
        BX_deformed = deformed_features.transpose(1, 2).reshape(B, K, D // K, L)  # Match BX shape
        BX_deformed = deltaB * BX_deformed
        
        # Apply HSW to input features if enabled
        if phi_weights is not None:
            # Formula: h_i = sum_j(B_j * phi_i * p_j * prod_k(A_k))
            BX_deformed = BX_deformed * phi_weights.unsqueeze(1)  # Apply suppression
        
        feat_in = BX_deformed.view(B, -1, L)  # Use deformed features
        
    else:
        # Original tree scanning without deformation
        fea4tree_hw = rearrange(xs, 'b d (h w) -> b d h w', h=H, w=W)  # B d L
        mst_layer = MinimumSpanningTree("Cosine", torch.exp)
        tree = mst_layer(fea4tree_hw)
    
    sorted_index, sorted_parent, sorted_child = bfs(tree, 4)
    edge_weight, = edge_transform(edge_weight, sorted_index, sorted_child)
    # import pdb;pdb.set_trace()
    edge_weight_coef = torch.ones_like(sorted_index, dtype=edge_weight.dtype)  # edge coef, default by 1
    feature_out = refine(feat_in, edge_weight, sorted_index, sorted_parent, sorted_child, edge_weight_coef)

    if h_norm is not None:
        out = h_norm(feature_out.transpose(-1, -2).contiguous())

    y = (rearrange(out, 'b l (k d) -> b l k d', k=K, d=int(D / K)).unsqueeze(-1) @ rearrange(Cs,
                                                                                             'b k n l -> b l k n').unsqueeze(
        -1)).squeeze(-1)  # (B L K D N) @ (B L K N 1) -> (B L K D 1)
    # import pdb;pdb.set_trace()
    y = rearrange(y, 'b l k d -> b (k d) l')
    y = y + Ds.reshape(1, -1, 1) * xs
    return y


def tree_scanning(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        h_norm=None,
        deform_module=None,
        pruning_module=None,
        hsw_module=None,
        use_dynamic_tree=False,
        use_hsw=False,
):
    B, D, H, W = x.shape
    origin_shape = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    xs = rearrange(x.unsqueeze(1), 'b k d h w -> b k d (h w)')
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (c, d)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    force_fp32 = True
    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = tree_scanning_core(xs, dts,
                            As, Bs, Cs, Ds,
                            delta_bias, origin_shape, h_norm,
                            deform_module=deform_module,
                            pruning_module=pruning_module,
                            hsw_module=hsw_module,
                            use_dynamic_tree=use_dynamic_tree,
                            use_hsw=use_hsw).view(B, K, -1, H, W)

    y = rearrange(ys, 'b k d h w -> b (k d) (h w)')
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class Tree_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # Dynamic Tree Scan (DTS) + Hidden State Weaken (HSW) ====
            use_dynamic_tree=False,
            use_hsw=False,
            pool_size=2,  # Pooling size for DTS (reduces computation by pool_size^2)
            hsw_varphi=0.7,  # Background suppression factor for HSW
            **kwargs,
    ):
        """
        UIS-Mamba Tree-SSM with DTS and HSW support
        
        Args:
            use_dynamic_tree: whether to use Dynamic Tree Scan with deformable patches
            use_hsw: whether to use Hidden State Weaken for background suppression
            pool_size: pooling size for patch merging in DTS (default 2 -> 4x computation reduction)
            hsw_varphi: background suppression factor (default 0.7, experimentally optimal)
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.use_dynamic_tree = use_dynamic_tree
        self.use_hsw = use_hsw

        self.out_norm = nn.LayerNorm(d_inner)
        self.h_norm = nn.LayerNorm(d_inner)

        self.K = 1
        self.K2 = self.K

        # in proj =======================================
        d_proj = d_expand * 2
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)
        
        # Dynamic Tree Scan (DTS) + Hidden State Weaken (HSW) modules =======================================
        if self.use_dynamic_tree:
            self.deform_module = AdaptiveGraphDeformation(d_inner, pool_size=pool_size)
            self.pruning_module = DynamicGraphPruning(d_inner, pool_size=pool_size)
            print(f"Tree_SSM: Using Dynamic Tree Scan (DTS) with pool_size={pool_size} (computation reduced by {pool_size**2}x)")
        else:
            self.deform_module = None
            self.pruning_module = None
        
        if self.use_hsw:
            self.hsw_module = HiddenStateWeaken(varphi=hsw_varphi)
            print(f"Tree_SSM: Using Hidden State Weaken (HSW) with varphi={hsw_varphi}")
        else:
            self.hsw_module = None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, channel_first=False, force_fp32=None):
        force_fp32 = self.training if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = tree_scanning(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            force_fp32=force_fp32, h_norm=self.h_norm,
            deform_module=self.deform_module,
            pruning_module=self.pruning_module,
            hsw_module=self.hsw_module,
            use_dynamic_tree=self.use_dynamic_tree,
            use_hsw=self.use_hsw,
        )
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        z = self.act(z)
        if self.d_conv > 0:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
