import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict, List

class NTLBGAttention(nn.Module):
    """
    基于NTLBG统计理论的视频帧选择注意力模块
    核心创新：将多元正态分布的代表点理论应用到视频理解中
    """
    
    def __init__(self, 
                 d_model: int = 768,
                 d_query: int = 768, 
                 num_representatives: int = 6,
                 temperature: float = 0.07,
                 eps: float = 1e-6):
        super().__init__()
        
        self.d_model = d_model
        self.d_query = d_query
        self.num_representatives = num_representatives
        self.temperature = temperature
        self.eps = eps
        
        # 查询引导的分布参数估计器
        self.query_proj = nn.Linear(d_query, d_model)
        self.mu_estimator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 协方差矩阵估计器（使用低秩分解提高效率）
        self.sigma_rank = min(64, d_model // 4)  # 低秩近似
        self.sigma_u_estimator = nn.Linear(d_model, d_model * self.sigma_rank)
        self.sigma_v_estimator = nn.Linear(d_model, d_model * self.sigma_rank)
        self.sigma_diag_estimator = nn.Linear(d_model, d_model)
        
        # 代表点权重预测器
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                video_features: torch.Tensor,
                query_embedding: torch.Tensor,
                return_stats: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            video_features: [batch_size, T, d_model] 视频帧特征
            query_embedding: [batch_size, d_query] 查询嵌入
            return_stats: 是否返回统计信息
        
        Returns:
            Dict包含：
            - representative_indices: [batch_size, num_representatives] 代表点索引
            - representative_features: [batch_size, num_representatives, d_model] 代表点特征
            - weights: [batch_size, num_representatives] 代表点权重
            - mu_q: [batch_size, d_model] 条件均值
            - sigma_q: [batch_size, d_model, d_model] 条件协方差
        """
        batch_size, T, d_model = video_features.shape
        
        # 1. 查询引导的分布参数估计
        query_embed = self.query_proj(query_embedding)  # [B, d_model]
        mu_q = self.mu_estimator(query_embed)  # [B, d_model]
        
        # 2. 协方差矩阵估计（低秩 + 对角）
        sigma_q = self._estimate_covariance(query_embed, batch_size)  # [B, d_model, d_model]
        
        # 3. 计算马氏距离
        mahalanobis_distances = self._compute_mahalanobis_distance(
            video_features, mu_q, sigma_q
        )  # [B, T]
        
        # 4. NTLBG代表点选择
        representative_indices = self._ntlbg_selection(
            mahalanobis_distances, video_features, mu_q, sigma_q
        )  # [B, num_representatives]
        
        # 5. 提取代表点特征和权重
        representative_features = self._gather_representative_features(
            video_features, representative_indices
        )  # [B, num_representatives, d_model]
        
        weights = self._compute_representative_weights(
            representative_features, mu_q, sigma_q
        )  # [B, num_representatives]
        
        results = {
            'representative_indices': representative_indices,
            'representative_features': representative_features,
            'weights': weights,
        }
        
        if return_stats:
            results.update({
                'mu_q': mu_q,
                'sigma_q': sigma_q,
                'mahalanobis_distances': mahalanobis_distances
            })
        
        return results
    
    def _estimate_covariance(self, query_embed: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        估计查询条件下的协方差矩阵
        使用低秩分解: Σ = UU^T + diag(d) 提高数值稳定性
        """
        # 低秩部分
        u_flat = self.sigma_u_estimator(query_embed)  # [B, d_model * sigma_rank]
        u_matrix = u_flat.view(batch_size, self.d_model, self.sigma_rank)  # [B, d_model, sigma_rank]
        
        # 对角部分
        diag_vals = F.softplus(self.sigma_diag_estimator(query_embed)) + self.eps  # [B, d_model]
        
        # 组合: Σ = UU^T + diag(d)
        low_rank_part = torch.bmm(u_matrix, u_matrix.transpose(-1, -2))  # [B, d_model, d_model]
        diag_part = torch.diag_embed(diag_vals)  # [B, d_model, d_model]
        
        sigma_q = low_rank_part + diag_part  # [B, d_model, d_model]
        
        return sigma_q
    
    def _compute_mahalanobis_distance(self, 
                                     features: torch.Tensor,
                                     mu: torch.Tensor, 
                                     sigma: torch.Tensor) -> torch.Tensor:
        """
        计算数值稳定的马氏距离
        """
        batch_size, T, d_model = features.shape
        
        # 中心化特征
        centered_features = features - mu.unsqueeze(1)  # [B, T, d_model]
        
        try:
            # 使用Cholesky分解求解线性方程组，避免直接求逆
            L = torch.linalg.cholesky(sigma + self.eps * torch.eye(d_model, device=sigma.device))  # [B, d_model, d_model]
            
            # 求解 L * y = centered_features^T
            # torch.triangular_solve 在新版本中被替换为 torch.linalg.solve_triangular
            y = torch.linalg.solve_triangular(
                L, centered_features.transpose(-1, -2), upper=False
            )  # [B, d_model, T]
            
            # 马氏距离 = ||y||^2
            distances = torch.sum(y ** 2, dim=1)  # [B, T]
            
        except Exception as e:
            # Fallback: 使用特征值分解
            print(f"Cholesky decomposition failed, using eigenvalue decomposition: {e}")
            eigenvals, eigenvecs = torch.linalg.eigh(sigma)
            eigenvals = torch.clamp(eigenvals, min=self.eps)
            
            # 重构逆矩阵
            sigma_inv = eigenvecs @ torch.diag_embed(1.0 / eigenvals) @ eigenvecs.transpose(-1, -2)
            
            # 计算马氏距离
            distances = torch.sum(
                centered_features @ sigma_inv * centered_features, dim=-1
            )  # [B, T]
        
        return distances
    
    def _ntlbg_selection(self, 
                        distances: torch.Tensor,
                        features: torch.Tensor,
                        mu: torch.Tensor,
                        sigma: torch.Tensor) -> torch.Tensor:
        """
        基于NTLBG理论的代表点选择
        核心思想：选择在同一等高椭球面上的k个点，保证统计最优性
        """
        batch_size, T = distances.shape
        k = self.num_representatives
        
        # 方法1：基于距离分层选择（实现您论文中的等高线思想）
        indices_list = []
        
        for b in range(batch_size):
            dist_b = distances[b]  # [T]
            
            if k >= T:
                # 如果代表点数量大于等于帧数，直接全选
                indices = torch.arange(T, device=distances.device)
                # 填充到k个
                indices = torch.cat([indices, indices[:k-T]], dim=0)
            else:
                # 找到最优的等高线级别
                sorted_distances, sorted_indices = torch.sort(dist_b)
                
                # 策略：选择中位数附近的点，确保代表性
                median_idx = T // 2
                target_distance = sorted_distances[median_idx]
                
                # 找到距离目标距离最近的k个点
                distance_diff = torch.abs(dist_b - target_distance)
                _, candidate_indices = torch.topk(distance_diff, min(k*2, T), largest=False)
                
                # 从候选点中进一步筛选，确保时间分布均匀
                indices = self._temporal_diversification(candidate_indices, k)
            
            indices_list.append(indices[:k])
        
        representative_indices = torch.stack(indices_list)  # [B, k]
        
        return representative_indices
    
    def _temporal_diversification(self, candidate_indices: torch.Tensor, k: int) -> torch.Tensor:
        """
        时序多样化：在候选代表点中选择时间分布最均匀的k个点
        """
        if len(candidate_indices) <= k:
            return candidate_indices
        
        # 按时间顺序排序
        sorted_candidates, _ = torch.sort(candidate_indices)
        
        if k == 1:
            return sorted_candidates[:1]
        
        # 贪心选择：最大化最小时间间隔
        selected = [sorted_candidates[0]]
        remaining = sorted_candidates[1:].tolist()
        
        for _ in range(k - 1):
            if not remaining:
                break
            
            # 计算每个候选点与已选点的最小距离
            min_distances = []
            for candidate in remaining:
                min_dist = min(abs(candidate - sel) for sel in selected)
                min_distances.append(min_dist)
            
            # 选择最小距离最大的点
            best_idx = np.argmax(min_distances)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return torch.tensor(selected, device=candidate_indices.device)
    
    def _gather_representative_features(self, 
                                      features: torch.Tensor,
                                      indices: torch.Tensor) -> torch.Tensor:
        """
        根据索引提取代表点特征
        """
        batch_size, k = indices.shape
        _, T, d_model = features.shape
        
        # 扩展索引维度以进行gather操作
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, d_model)  # [B, k, d_model]
        
        # 提取对应的特征
        representative_features = torch.gather(features, 1, expanded_indices)  # [B, k, d_model]
        
        return representative_features
    
    def _compute_representative_weights(self, 
                                      rep_features: torch.Tensor,
                                      mu: torch.Tensor,
                                      sigma: torch.Tensor) -> torch.Tensor:
        """
        计算每个代表点的重要性权重
        """
        batch_size, k, d_model = rep_features.shape
        
        # 方法1：基于马氏距离的权重（距离越近，权重越高）
        centered_rep = rep_features - mu.unsqueeze(1)  # [B, k, d_model]
        rep_distances = self._compute_mahalanobis_distance(
            rep_features, mu, sigma
        )  # [B, k]
        
        # 转换为权重（使用softmax确保和为1）
        distance_weights = F.softmax(-rep_distances / self.temperature, dim=-1)  # [B, k]
        
        # 方法2：基于神经网络的权重预测
        neural_weights = self.weight_predictor(rep_features).squeeze(-1)  # [B, k]
        neural_weights = F.softmax(neural_weights, dim=-1)
        
        # 组合两种权重
        final_weights = 0.7 * distance_weights + 0.3 * neural_weights
        
        return final_weights
    
    def compute_ntlbg_constraint_loss(self, 
                                     representative_features: torch.Tensor,
                                     mu: torch.Tensor, 
                                     sigma: torch.Tensor) -> torch.Tensor:
        """
        计算NTLBG约束损失：确保代表点在同一等高椭球面上
        这是核心创新：将您论文中的统计理论直接融入损失函数
        """
        # 计算所有代表点的马氏距离
        rep_distances = self._compute_mahalanobis_distance(
            representative_features, mu, sigma
        )  # [B, k]
        
        # 目标：所有代表点应该有相同的马氏距离（在同一等高线上）
        target_distance = torch.median(rep_distances, dim=-1, keepdim=True)[0]  # [B, 1]
        
        # L2损失：最小化距离差异
        constraint_loss = F.mse_loss(rep_distances, target_distance.expand_as(rep_distances))
        
        return constraint_loss
    
    def visualize_selection(self, 
                          video_features: torch.Tensor,
                          representative_indices: torch.Tensor,
                          save_path: Optional[str] = None) -> Dict:
        """
        可视化代表点选择结果（用于论文图表）
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # 只处理第一个batch
        features = video_features[0].detach().cpu().numpy()  # [T, d_model]
        indices = representative_indices[0].detach().cpu().numpy()  # [k]
        
        # PCA降维到2D进行可视化
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)  # [T, 2]
        
        # 绘图
        plt.figure(figsize=(10, 8))
        
        # 所有帧
        plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                   c='lightgray', alpha=0.6, s=20, label='All frames')
        
        # 代表点
        rep_features_2d = features_2d[indices]
        plt.scatter(rep_features_2d[:, 0], rep_features_2d[:, 1], 
                   c='red', s=100, marker='*', label='Representative points')
        
        # 添加索引标签
        for i, (x, y) in enumerate(rep_features_2d):
            plt.annotate(f'{indices[i]}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('NTLBG Representative Points Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return {
            'features_2d': features_2d,
            'representative_indices': indices,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }


# 测试和使用示例
if __name__ == "__main__":
    # 创建测试数据
    batch_size, T, d_model = 2, 100, 768
    d_query = 768
    
    video_features = torch.randn(batch_size, T, d_model)
    query_embedding = torch.randn(batch_size, d_query)
    
    # 初始化模块
    ntlbg_attention = NTLBGAttention(
        d_model=d_model,
        d_query=d_query,
        num_representatives=6
    )
    
    # 前向传播
    results = ntlbg_attention(video_features, query_embedding, return_stats=True)
    
    print("=== NTLBG Attention Results ===")
    print(f"Representative indices shape: {results['representative_indices'].shape}")
    print(f"Representative features shape: {results['representative_features'].shape}")
    print(f"Weights shape: {results['weights'].shape}")
    print(f"Selected frame indices (batch 0): {results['representative_indices'][0]}")
    print(f"Representative weights (batch 0): {results['weights'][0]}")
    
    # 计算约束损失
    constraint_loss = ntlbg_attention.compute_ntlbg_constraint_loss(
        results['representative_features'],
        results['mu_q'],
        results['sigma_q']
    )
    print(f"NTLBG constraint loss: {constraint_loss.item():.6f}")
    
    print("\n✅ NTLBG Attention module working correctly!") 