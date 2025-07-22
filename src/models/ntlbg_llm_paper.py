import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class NTLBGAttentionModule(nn.Module):
    """
    基于您本科论文的NTLBG理论的注意力模块
    核心创新：统计学指导的代表点选择
    """
    
    def __init__(self, d_model: int, num_representatives: int = 6, temperature: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_representatives = num_representatives
        self.temperature = temperature
        
        # 查询条件分布参数估计器
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 统计参数预测器
        self.mu_predictor = nn.Linear(d_model, d_model)
        self.sigma_predictor = nn.Sequential(
            nn.Linear(d_model, d_model * d_model),
            nn.Tanh()  # 确保数值稳定性
        )
        
        # 代表点重要性评估器
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 数值稳定性参数
        self.eps = 1e-6
        
    def forward(self, video_features: torch.Tensor, query_embedding: torch.Tensor):
        """
        NTLBG代表点选择的核心算法
        Args:
            video_features: [B, T, D] 视频帧特征
            query_embedding: [B, D] 查询条件编码
        Returns:
            dict: 包含代表点和统计信息
        """
        B, T, D = video_features.shape
        
        # 1. 查询条件编码
        q_encoded = self.query_encoder(query_embedding)  # [B, D]
        
        # 2. 估计查询条件下的分布参数
        mu_q = self.mu_predictor(q_encoded)  # [B, D] 条件均值
        sigma_flat = self.sigma_predictor(q_encoded)  # [B, D*D]
        
        # 重构协方差矩阵并确保正定性
        sigma_q = sigma_flat.view(B, D, D)  # [B, D, D]
        sigma_q = torch.bmm(sigma_q, sigma_q.transpose(-1, -2))
        sigma_q = sigma_q + self.eps * torch.eye(D, device=video_features.device).unsqueeze(0)
        
        # 3. 计算马氏距离（NTLBG核心）
        mahalanobis_distances = self._compute_mahalanobis_distance(
            video_features, mu_q, sigma_q
        )  # [B, T]
        
        # 4. NTLBG等高线约束选择
        representative_indices, selection_weights = self._ntlbg_selection(
            video_features, q_encoded, mahalanobis_distances
        )
        
        # 5. 构建富代表点特征
        representative_features = self._construct_rich_representatives(
            video_features, representative_indices, selection_weights, q_encoded
        )
        
        # 6. 返回完整信息（用于损失计算和分析）
        return {
            'representative_features': representative_features,
            'representative_indices': representative_indices,
            'selection_weights': selection_weights,
            'mahalanobis_distances': mahalanobis_distances,
            'mu_q': mu_q,
            'sigma_q': sigma_q,
            'query_encoded': q_encoded
        }
    
    def _compute_mahalanobis_distance(self, features, mu, sigma):
        """计算马氏距离：核心统计学算法"""
        B, T, D = features.shape
        
        # 中心化
        centered_features = features - mu.unsqueeze(1)  # [B, T, D]
        
        # 计算协方差矩阵的逆（数值稳定版本）
        try:
            # 使用Cholesky分解提高数值稳定性
            L = torch.linalg.cholesky(sigma)  # [B, D, D]
            sigma_inv = torch.cholesky_inverse(L)  # [B, D, D]
        except:
            # 备用方案：正则化后求逆
            regularized_sigma = sigma + 1e-4 * torch.eye(D, device=sigma.device).unsqueeze(0)
            sigma_inv = torch.linalg.inv(regularized_sigma)
        
        # 马氏距离计算：d² = (x-μ)ᵀ Σ⁻¹ (x-μ)
        distances = torch.einsum('btd,bde,bte->bt', centered_features, sigma_inv, centered_features)
        
        return torch.clamp(distances, min=0)  # 确保非负
    
    def _ntlbg_selection(self, video_features, query_encoded, distances):
        """
        基于NTLBG理论的代表点选择
        核心思想：选择在同一等高椭球面上的代表点
        """
        B, T, D = video_features.shape
        K = min(self.num_representatives, T)
        
        if self.training:
            # 训练时：软选择（可微分）
            return self._soft_ntlbg_selection(video_features, query_encoded, distances, K)
        else:
            # 推理时：硬选择（更精确）
            return self._hard_ntlbg_selection(distances, K)
    
    def _soft_ntlbg_selection(self, video_features, query_encoded, distances, K):
        """软选择：基于Gumbel-Softmax的可微分选择"""
        B, T, D = video_features.shape
        
        # 计算每帧的重要性分数
        query_expanded = query_encoded.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        combined_features = torch.cat([video_features, query_expanded], dim=-1)  # [B, T, 2D]
        
        importance_scores = self.importance_scorer(combined_features).squeeze(-1)  # [B, T]
        
        # 结合马氏距离调整分数（距离小=重要性高）
        distance_scores = torch.exp(-distances / self.temperature)
        final_scores = importance_scores + distance_scores
        
        # Gumbel-Softmax选择
        if self.training:
            # 添加Gumbel噪声
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(final_scores) + 1e-8) + 1e-8)
            noisy_scores = (final_scores + gumbel_noise) / self.temperature
        else:
            noisy_scores = final_scores / self.temperature
        
        # 选择top-K
        softmax_scores = F.softmax(noisy_scores, dim=1)
        top_weights, top_indices = torch.topk(softmax_scores, K, dim=1)
        
        # 重新归一化
        top_weights = top_weights / (top_weights.sum(dim=1, keepdim=True) + self.eps)
        
        return top_indices, top_weights
    
    def _hard_ntlbg_selection(self, distances, K):
        """硬选择：基于等高线约束的确定性选择"""
        B, T = distances.shape
        
        # 寻找最优等高线值
        # 策略：选择使代表点最均匀分布的等高线值
        median_distance = torch.median(distances, dim=1, keepdim=True)[0]  # [B, 1]
        
        # 计算每个点到目标等高线的距离
        contour_deviations = torch.abs(distances - median_distance)  # [B, T]
        
        # 选择最接近目标等高线的K个点
        _, selected_indices = torch.topk(contour_deviations, K, dim=1, largest=False)  # [B, K]
        
        # 等权重（硬选择）
        equal_weights = torch.ones(B, K, device=distances.device) / K
        
        return selected_indices, equal_weights
    
    def _construct_rich_representatives(self, video_features, indices, weights, query_encoded):
        """构建富代表点：不仅包含视觉特征，还包含上下文信息"""
        B, T, D = video_features.shape
        K = indices.shape[1]
        
        # 收集基础代表点特征
        batch_indices = torch.arange(B, device=video_features.device).unsqueeze(1).expand(-1, K)
        base_features = video_features[batch_indices, indices]  # [B, K, D]
        
        # 添加权重信息
        weighted_features = base_features * weights.unsqueeze(-1)  # [B, K, D]
        
        # 添加查询相关性
        query_similarity = torch.bmm(
            base_features, query_encoded.unsqueeze(-1)
        ).squeeze(-1).unsqueeze(-1)  # [B, K, 1]
        
        # 融合多维信息
        rich_features = torch.cat([
            weighted_features,
            query_similarity.expand(-1, -1, D)
        ], dim=-1)  # [B, K, 2D]
        
        # 投影回原始维度
        fusion_layer = nn.Linear(2 * D, D, device=video_features.device)
        rich_representatives = fusion_layer(rich_features)  # [B, K, D]
        
        return rich_representatives

class PaperNTLBGLLM(nn.Module):
    """
    论文版本的完整NTLBG-LLM模型
    集成统计学理论与大语言模型
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 模型维度
        self.d_model = config.get('d_model', 768)
        self.vocab_size = config.get('vocab_size', 50000)
        
        # 视觉编码器
        self.video_projector = nn.Sequential(
            nn.Linear(config.get('video_feature_dim', 768), self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )
        
        # NTLBG注意力模块（核心创新）
        self.ntlbg_attention = NTLBGAttentionModule(
            d_model=self.d_model,
            num_representatives=config.get('num_representatives', 6),
            temperature=config.get('temperature', 0.1)
        )
        
        # 文本处理
        self.text_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = nn.Parameter(
            torch.randn(config.get('max_text_length', 512), self.d_model) * 0.02
        )
        
        # 多模态融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.get('num_heads', 12),
            batch_first=True,
            dropout=0.1
        )
        
        # 输出层
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, video_features, input_ids, attention_mask, labels=None):
        """完整的前向传播"""
        B = video_features.size(0)
        
        # 1. 视频特征处理
        video_projected = self.video_projector(video_features)  # [B, T, D]
        
        # 2. 文本处理
        text_embeddings = self.text_embedding(input_ids)  # [B, L, D]
        seq_length = text_embeddings.size(1)
        text_embeddings = text_embeddings + self.positional_encoding[:seq_length]
        
        # 生成查询表示
        query_embedding = torch.mean(text_embeddings, dim=1)  # [B, D]
        
        # 3. NTLBG代表点选择（核心算法）
        ntlbg_output = self.ntlbg_attention(video_projected, query_embedding)
        representative_features = ntlbg_output['representative_features']  # [B, K, D]
        
        # 4. 多模态融合
        fused_features, cross_attention_weights = self.cross_attention(
            query=text_embeddings,
            key=representative_features,
            value=representative_features
        )  # [B, L, D]
        
        # 5. 输出生成
        fused_features = self.dropout(fused_features)
        output_features = self.output_norm(fused_features)
        logits = self.output_projection(output_features)  # [B, L, vocab_size]
        
        # 6. 构建输出
        outputs = {
            'logits': logits,
            'representative_features': representative_features,
            'representative_indices': ntlbg_output['representative_indices'],
            'selection_weights': ntlbg_output['selection_weights'],
            'cross_attention_weights': cross_attention_weights,
            'ntlbg_stats': {
                'mahalanobis_distances': ntlbg_output['mahalanobis_distances'],
                'mu_q': ntlbg_output['mu_q'],
                'sigma_q': ntlbg_output['sigma_q']
            }
        }
        
        # 7. 计算损失（如果有标签）
        if labels is not None:
            # 主任务损失
            task_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            
            # NTLBG统计约束损失
            ntlbg_loss = self._compute_ntlbg_constraint_loss(ntlbg_output)
            
            # 代表点分布均匀性损失
            diversity_loss = self._compute_diversity_loss(representative_features)
            
            # 总损失
            total_loss = (
                task_loss + 
                0.1 * ntlbg_loss + 
                0.05 * diversity_loss
            )
            
            outputs.update({
                'loss': total_loss,
                'task_loss': task_loss,
                'ntlbg_loss': ntlbg_loss,
                'diversity_loss': diversity_loss
            })
        
        return outputs
    
    def _compute_ntlbg_constraint_loss(self, ntlbg_output):
        """NTLBG等高线约束损失"""
        distances = ntlbg_output['mahalanobis_distances']  # [B, T]
        indices = ntlbg_output['representative_indices']    # [B, K]
        
        # 获取代表点的马氏距离
        B, K = indices.shape
        batch_indices = torch.arange(B, device=distances.device).unsqueeze(1).expand(-1, K)
        representative_distances = distances[batch_indices, indices]  # [B, K]
        
        # 等高线约束：所有代表点应有相同的马氏距离
        target_distance = torch.median(representative_distances, dim=1, keepdim=True)[0]
        constraint_loss = torch.mean((representative_distances - target_distance) ** 2)
        
        return constraint_loss
    
    def _compute_diversity_loss(self, representative_features):
        """代表点多样性损失"""
        B, K, D = representative_features.shape
        
        # 计算代表点间的相似性
        normalized_features = F.normalize(representative_features, dim=-1)  # [B, K, D]
        similarity_matrix = torch.bmm(
            normalized_features, normalized_features.transpose(-1, -2)
        )  # [B, K, K]
        
        # 除了对角线，其他相似性应该较小
        mask = torch.eye(K, device=similarity_matrix.device).unsqueeze(0).expand(B, -1, -1)
        off_diagonal_similarity = similarity_matrix * (1 - mask)
        
        # 多样性损失：减少代表点间的相似性
        diversity_loss = torch.mean(torch.sum(off_diagonal_similarity ** 2, dim=[1, 2]))
        
        return diversity_loss

def create_paper_model(config):
    """创建论文版本的NTLBG-LLM模型"""
    return PaperNTLBGLLM(config)

if __name__ == "__main__":
    # 测试模型
    config = {
        'd_model': 768,
        'video_feature_dim': 768,
        'num_representatives': 6,
        'temperature': 0.1,
        'num_heads': 12,
        'vocab_size': 50000,
        'max_text_length': 512
    }
    
    model = create_paper_model(config)
    
    # 测试前向传播
    batch_size = 2
    video_features = torch.randn(batch_size, 100, 768)  # 100帧视频
    input_ids = torch.randint(1, 1000, (batch_size, 128))  # 128长度文本
    attention_mask = torch.ones(batch_size, 128)
    labels = torch.randint(1, 1000, (batch_size, 128))
    
    with torch.no_grad():
        outputs = model(video_features, input_ids, attention_mask, labels)
        
    print("✅ 论文模型测试成功!")
    print(f"📊 代表点数量: {outputs['representative_indices'].shape[1]}")
    print(f"📊 总损失: {outputs['loss'].item():.4f}")
    print(f"📊 NTLBG损失: {outputs['ntlbg_loss'].item():.4f}")
