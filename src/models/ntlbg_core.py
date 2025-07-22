"""
NTLBG核心算法实现 - 基于统计学理论
Neural Temporal-aware Long-video Benchmark Generative
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class NTLBGCore(nn.Module):
    """NTLBG核心算法：基于统计学的代表点选择"""
    
    def __init__(self, d_visual: int, d_query: int, num_representatives: int = 6):
        super().__init__()
        self.d_visual = d_visual
        self.d_query = d_query
        self.num_representatives = num_representatives
        
        # 统计参数估计网络
        self.mu_estimator = nn.Sequential(
            nn.Linear(d_query, d_visual),
            nn.LayerNorm(d_visual),
            nn.ReLU(),
            nn.Linear(d_visual, d_visual)
        )
        
        # 协方差矩阵估计（简化为对角线）
        self.sigma_estimator = nn.Sequential(
            nn.Linear(d_query, d_visual),
            nn.LayerNorm(d_visual),
            nn.ReLU(),
            nn.Linear(d_visual, d_visual),
            nn.Softplus()  # 确保正数
        )
        
        # 时序权重学习
        self.temporal_weight = nn.Sequential(
            nn.Linear(d_visual + 1, d_visual // 2),  # +1 for temporal position
            nn.ReLU(),
            nn.Linear(d_visual // 2, 1),
            nn.Sigmoid()
        )
        
        # 代表点精炼网络
        self.representative_refiner = nn.MultiheadAttention(
            d_visual, num_heads=8, batch_first=True
        )
        
    def forward(self, video_features: torch.Tensor, query_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        NTLBG前向传播
        
        Args:
            video_features: [B, T, d_visual] 视频特征
            query_embedding: [B, d_query] 查询嵌入
            
        Returns:
            Dict containing:
                - representative_features: [B, K, d_visual] 选择的代表点特征
                - representative_indices: [B, K] 代表点索引
                - mahalanobis_distances: [B, T] 马氏距离
                - mu_q, sigma_q: 统计参数
        """
        B, T, d_visual = video_features.shape
        
        # 1. 基于查询估计统计参数
        mu_q = self.mu_estimator(query_embedding)  # [B, d_visual]
        sigma_diag = self.sigma_estimator(query_embedding) + 1e-6  # [B, d_visual]
        
        # 2. 计算马氏距离
        mahalanobis_distances = self._compute_mahalanobis_distance(
            video_features, mu_q, sigma_diag
        )  # [B, T]
        
        # 3. NTLBG代表点选择
        representative_indices = self._ntlbg_selection(
            video_features, mahalanobis_distances, mu_q, sigma_diag
        )  # [B, K]
        
        # 4. 提取代表点特征
        representative_features = self._extract_representative_features(
            video_features, representative_indices
        )  # [B, K, d_visual]
        
        # 5. 代表点精炼
        refined_features, attention_weights = self.representative_refiner(
            representative_features, representative_features, representative_features
        )
        
        return {
            'representative_features': refined_features,
            'representative_indices': representative_indices,
            'mahalanobis_distances': mahalanobis_distances,
            'mu_q': mu_q,
            'sigma_q': sigma_diag,
            'attention_weights': attention_weights
        }
    
    def _compute_mahalanobis_distance(self, features: torch.Tensor, 
                                    mu: torch.Tensor, sigma_diag: torch.Tensor) -> torch.Tensor:
        """计算马氏距离"""
        # features: [B, T, d], mu: [B, d], sigma_diag: [B, d]
        centered = features - mu.unsqueeze(1)  # [B, T, d]
        
        # 对角协方差矩阵的马氏距离
        weighted_diff = centered ** 2 / sigma_diag.unsqueeze(1)  # [B, T, d]
        distances = torch.sum(weighted_diff, dim=-1)  # [B, T]
        
        return distances
    
    def _ntlbg_selection(self, features: torch.Tensor, distances: torch.Tensor,
                        mu_q: torch.Tensor, sigma_diag: torch.Tensor) -> torch.Tensor:
        """NTLBG统计代表点选择算法"""
        B, T, d = features.shape
        K = self.num_representatives
        
        selected_indices = []
        
        for b in range(B):
            # 当前批次的距离和特征
            batch_distances = distances[b]  # [T]
            batch_features = features[b]  # [T, d]
            
            # 如果帧数少于代表点数，全选
            if T <= K:
                indices = torch.arange(T, device=features.device)
                # 填充到K个
                padding = torch.zeros(K - T, dtype=torch.long, device=features.device)
                indices = torch.cat([indices, padding])
                selected_indices.append(indices)
                continue
            
            # NTLBG核心算法：
            # 1. 找到统计最优的等高椭球面
            target_distance = torch.median(batch_distances)
            
            # 2. 选择距离目标距离最近的候选点
            distance_to_target = torch.abs(batch_distances - target_distance)
            candidate_size = min(K * 3, T)  # 扩大候选池
            _, candidate_indices = torch.topk(
                distance_to_target, candidate_size, largest=False
            )
            
            # 3. 在候选点中进行时序多样化选择
            final_indices = self._temporal_diversification(
                candidate_indices, K, batch_features[candidate_indices]
            )
            
            selected_indices.append(final_indices)
        
        return torch.stack(selected_indices)  # [B, K]
    
    def _temporal_diversification(self, candidates: torch.Tensor, K: int, 
                                candidate_features: torch.Tensor) -> torch.Tensor:
        """时序多样化选择，确保代表点在时间上分布均匀"""
        if len(candidates) <= K:
            # 填充到K个
            padding_size = K - len(candidates)
            padding = candidates[-1].repeat(padding_size)
            return torch.cat([candidates, padding])
        
        # 贪心算法：最大化最小时间间隔
        candidates_np = candidates.cpu().numpy()
        selected = [candidates_np[0]]  # 从第一个开始
        remaining = list(candidates_np[1:])
        
        for _ in range(K - 1):
            if not remaining:
                break
            
            # 计算每个候选点与已选点的最小时间距离
            min_distances = []
            for candidate in remaining:
                min_dist = min(abs(candidate - selected_point) for selected_point in selected)
                min_distances.append(min_dist)
            
            # 选择最小距离最大的点（最远离的点）
            best_idx = np.argmax(min_distances)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        # 填充到K个（如果需要）
        while len(selected) < K:
            selected.append(selected[-1])
        
        return torch.tensor(selected[:K], device=candidates.device, dtype=torch.long)
    
    def _extract_representative_features(self, features: torch.Tensor, 
                                       indices: torch.Tensor) -> torch.Tensor:
        """提取代表点特征"""
        B, T, d = features.shape
        K = indices.shape[1]
        
        # 扩展索引维度
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, d)  # [B, K, d]
        
        # 收集代表点特征
        representative_features = torch.gather(features, 1, expanded_indices)  # [B, K, d]
        
        return representative_features
    
    def compute_ntlbg_constraint_loss(self, representative_features: torch.Tensor,
                                    mu_q: torch.Tensor, sigma_q: torch.Tensor) -> torch.Tensor:
        """计算NTLBG约束损失，确保代表点在同一等高椭球面上"""
        # representative_features: [B, K, d]
        B, K, d = representative_features.shape
        
        # 计算代表点的马氏距离
        rep_distances = self._compute_mahalanobis_distance(
            representative_features, mu_q, sigma_q
        )  # [B, K]
        
        # 约束：所有代表点应该有相似的马氏距离（在同一椭球面上）
        target_distance = rep_distances.mean(dim=1, keepdim=True)  # [B, 1]
        distance_variance = torch.mean((rep_distances - target_distance) ** 2)
        
        return distance_variance


class NTLBGAttention(nn.Module):
    """NTLBG注意力机制，用于视频-文本对齐"""
    
    def __init__(self, d_model: int, d_query: int, num_representatives: int = 6):
        super().__init__()
        self.ntlbg_core = NTLBGCore(d_model, d_query, num_representatives)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, video_features: torch.Tensor, query_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        NTLBG注意力前向传播
        """
        # 1. NTLBG代表点选择
        ntlbg_results = self.ntlbg_core(video_features, query_embedding)
        
        # 2. 跨模态注意力
        query_expanded = query_embedding.unsqueeze(1)  # [B, 1, d]
        attended_features, cross_attention_weights = self.cross_attention(
            query_expanded, 
            ntlbg_results['representative_features'],
            ntlbg_results['representative_features']
        )
        
        # 3. 残差连接和归一化
        output_features = self.norm(attended_features + query_expanded)
        
        ntlbg_results.update({
            'attended_features': output_features,
            'cross_attention_weights': cross_attention_weights
        })
        
        return ntlbg_results
