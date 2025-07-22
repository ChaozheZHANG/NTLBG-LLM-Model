"""
修复版NTLBG核心算法 - 解决梯度和学习问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math

class FixedNTLBGCore(nn.Module):
    """修复版NTLBG核心算法"""
    
    def __init__(self, d_visual: int, d_query: int, num_representatives: int = 6):
        super().__init__()
        self.d_visual = d_visual
        self.d_query = d_query
        self.num_representatives = num_representatives
        
        # 改进的统计参数估计网络
        self.mu_estimator = nn.Sequential(
            nn.Linear(d_query, d_visual * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_visual * 2, d_visual),
            nn.LayerNorm(d_visual)
        )
        
        # 改进的协方差估计（确保数值稳定）
        self.sigma_estimator = nn.Sequential(
            nn.Linear(d_query, d_visual),
            nn.GELU(),
            nn.Linear(d_visual, d_visual),
            nn.Sigmoid()  # 输出0-1之间，然后加偏移
        )
        
        # 可学习的代表点选择权重
        self.selection_head = nn.Sequential(
            nn.Linear(d_visual + d_query, d_visual),
            nn.GELU(),
            nn.Linear(d_visual, 1)
        )
        
        # 时序位置编码
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1000, d_visual) * 0.02  # 支持1000帧
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, video_features: torch.Tensor, query_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        修复版前向传播
        """
        B, T, d_visual = video_features.shape
        device = video_features.device
        
        # 1. 添加时序位置编码
        pos_encoding = self.temporal_pos_encoding[:T].unsqueeze(0).expand(B, -1, -1).to(device)
        video_features_with_pos = video_features + pos_encoding
        
        # 2. 估计统计参数（改进数值稳定性）
        mu_q = self.mu_estimator(query_embedding)  # [B, d_visual]
        sigma_raw = self.sigma_estimator(query_embedding)  # [B, d_visual]
        sigma_diag = sigma_raw * 2.0 + 0.1  # 范围在[0.1, 2.1]，避免除零
        
        # 3. 计算改进的马氏距离
        mahalanobis_distances = self._compute_stable_mahalanobis_distance(
            video_features_with_pos, mu_q, sigma_diag
        )
        
        # 4. NTLBG代表点选择（改进版）
        representative_indices = self._improved_ntlbg_selection(
            video_features_with_pos, mahalanobis_distances, query_embedding
        )
        
        # 5. 提取代表点特征
        representative_features = self._extract_representative_features(
            video_features_with_pos, representative_indices
        )
        
        return {
            'representative_features': representative_features,
            'representative_indices': representative_indices,
            'mahalanobis_distances': mahalanobis_distances,
            'mu_q': mu_q,
            'sigma_q': sigma_diag,
            'video_features_processed': video_features_with_pos
        }
    
    def _compute_stable_mahalanobis_distance(self, features: torch.Tensor, 
                                           mu: torch.Tensor, sigma_diag: torch.Tensor) -> torch.Tensor:
        """计算数值稳定的马氏距离"""
        # features: [B, T, d], mu: [B, d], sigma_diag: [B, d]
        
        # 中心化特征
        centered = features - mu.unsqueeze(1)  # [B, T, d]
        
        # 计算加权平方距离（数值稳定版本）
        weighted_squared = (centered ** 2) / (sigma_diag.unsqueeze(1) + 1e-8)
        distances = torch.sum(weighted_squared, dim=-1)  # [B, T]
        
        # 添加数值稳定性：确保距离为正数
        distances = torch.clamp(distances, min=1e-8)
        
        return distances
    
    def _improved_ntlbg_selection(self, features: torch.Tensor, distances: torch.Tensor,
                                query_embedding: torch.Tensor) -> torch.Tensor:
        """改进的NTLBG选择算法"""
        B, T, d = features.shape
        K = self.num_representatives
        
        selected_indices = []
        
        for b in range(B):
            batch_features = features[b]  # [T, d]
            batch_distances = distances[b]  # [T]
            batch_query = query_embedding[b:b+1].expand(T, -1)  # [T, d_query]
            
            if T <= K:
                # 如果帧数不够，重复选择
                indices = torch.arange(T, device=features.device)
                if T < K:
                    # 填充策略：重复最后几帧
                    padding = torch.randint(0, T, (K - T,), device=features.device)
                    indices = torch.cat([indices, padding])
                selected_indices.append(indices)
                continue
            
            # 改进的选择策略：
            # 1. 基于距离的粗选
            target_distance = torch.median(batch_distances)
            distance_scores = -torch.abs(batch_distances - target_distance)  # 越接近越好
            
            # 2. 基于查询相关性的精选
            query_features = torch.cat([batch_features, batch_query], dim=-1)
            relevance_scores = self.selection_head(query_features).squeeze(-1)  # [T]
            
            # 3. 综合评分
            combined_scores = distance_scores + 0.5 * relevance_scores
            
            # 4. Top-K选择，然后时序多样化
            _, top_candidates = torch.topk(combined_scores, min(K*2, T), largest=True)
            
            # 5. 时序多样化
            final_indices = self._temporal_diversification_v2(top_candidates, K)
            
            selected_indices.append(final_indices)
        
        return torch.stack(selected_indices)
    
    def _temporal_diversification_v2(self, candidates: torch.Tensor, K: int) -> torch.Tensor:
        """改进的时序多样化算法"""
        if len(candidates) <= K:
            # 填充到K个
            while len(candidates) < K:
                candidates = torch.cat([candidates, candidates[-1:]])
            return candidates[:K]
        
        candidates_sorted, _ = torch.sort(candidates)
        selected = [candidates_sorted[0]]  # 从最早的开始
        
        remaining = candidates_sorted[1:].tolist()
        
        for _ in range(K - 1):
            if not remaining:
                break
            
            # 找到与已选择帧距离最远的候选帧
            max_min_distance = -1
            best_candidate = remaining[0]
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                min_distance = min(abs(candidate - selected_frame) for selected_frame in selected)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
                    best_idx = i
            
            selected.append(best_candidate)
            remaining.pop(best_idx)
        
        # 确保有K个元素
        while len(selected) < K:
            selected.append(selected[-1])
        
        return torch.tensor(selected[:K], device=candidates.device, dtype=torch.long)
    
    def _extract_representative_features(self, features: torch.Tensor, 
                                       indices: torch.Tensor) -> torch.Tensor:
        """安全的特征提取"""
        B, T, d = features.shape
        K = indices.shape[1]
        
        # 确保索引在有效范围内
        indices = torch.clamp(indices, 0, T - 1)
        
        # 扩展索引
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, d)
        
        # 提取特征
        representative_features = torch.gather(features, 1, expanded_indices)
        
        return representative_features
    
    def compute_ntlbg_constraint_loss(self, representative_features: torch.Tensor,
                                    mu_q: torch.Tensor, sigma_q: torch.Tensor) -> torch.Tensor:
        """改进的约束损失计算"""
        B, K, d = representative_features.shape
        
        # 计算代表点的马氏距离
        rep_distances = self._compute_stable_mahalanobis_distance(
            representative_features, mu_q, sigma_q
        )
        
        # 约束1：代表点应该有相似的距离（在同一椭球面上）
        target_distance = rep_distances.mean(dim=1, keepdim=True)
        distance_consistency_loss = torch.mean((rep_distances - target_distance) ** 2)
        
        # 约束2：避免代表点过于集中
        diversity_loss = -torch.mean(torch.std(rep_distances, dim=1))
        
        # 约束3：确保距离合理范围
        distance_range_loss = torch.mean(torch.relu(rep_distances - 10.0)) + \
                             torch.mean(torch.relu(0.1 - rep_distances))
        
        total_loss = distance_consistency_loss + 0.1 * diversity_loss + 0.1 * distance_range_loss
        
        return total_loss


class FixedNTLBGAttention(nn.Module):
    """修复版NTLBG注意力机制"""
    
    def __init__(self, d_model: int, d_query: int, num_representatives: int = 6):
        super().__init__()
        self.ntlbg_core = FixedNTLBGCore(d_model, d_query, num_representatives)
        
        # 改进的注意力机制
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=0.1, batch_first=True
        )
        
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=0.1, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, video_features: torch.Tensor, query_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """改进的前向传播"""
        # 1. NTLBG核心处理
        ntlbg_results = self.ntlbg_core(video_features, query_embedding)
        representative_features = ntlbg_results['representative_features']
        
        # 2. 自注意力（代表点内部交互）
        self_attended, _ = self.self_attention(
            representative_features, representative_features, representative_features
        )
        representative_features = self.norm1(representative_features + self_attended)
        
        # 3. 跨模态注意力
        query_expanded = query_embedding.unsqueeze(1)  # [B, 1, d]
        cross_attended, cross_weights = self.cross_attention(
            query_expanded, representative_features, representative_features
        )
        attended_features = self.norm2(query_expanded + cross_attended)
        
        # 4. 前馈网络
        ffn_output = self.ffn(attended_features)
        final_features = self.norm3(attended_features + ffn_output)
        
        ntlbg_results.update({
            'attended_features': final_features,
            'cross_attention_weights': cross_weights,
            'processed_representatives': representative_features
        })
        
        return ntlbg_results
