"""
NTLBG-LLM多任务损失函数模块
包含损失计算和权重调度功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class LossWeightScheduler:
    """损失权重动态调度器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.initial_weights = config['loss_weights']
        self.schedule_type = config.get('weight_schedule', 'static')
        self.warmup_steps = config.get('weight_warmup_steps', 1000)
        self.total_steps = config.get('total_steps', 10000)
    
    def get_weights(self, step: int, epoch: int) -> Dict[str, float]:
        """根据训练步数返回当前的损失权重"""
        if self.schedule_type == 'static':
            return self.initial_weights
        
        elif self.schedule_type == 'progressive':
            # 渐进式权重调整：早期专注任务损失，后期增加NTLBG约束
            progress = min(1.0, step / self.warmup_steps)
            
            weights = {}
            weights['task'] = self.initial_weights['task']
            weights['ntlbg'] = self.initial_weights['ntlbg'] * progress
            weights['alignment'] = self.initial_weights['alignment'] * (progress ** 0.5)
            weights['temporal'] = self.initial_weights['temporal'] * (progress ** 0.3)
            weights['info'] = self.initial_weights['info'] * (progress ** 0.7)
            
            return weights
        
        elif self.schedule_type == 'curriculum':
            # 课程学习：不同阶段专注不同损失
            if step < self.warmup_steps * 0.3:
                # 第一阶段：专注任务损失
                return {
                    'task': self.initial_weights['task'],
                    'ntlbg': 0.0,
                    'alignment': 0.0,
                    'temporal': 0.0,
                    'info': 0.0
                }
            elif step < self.warmup_steps * 0.7:
                # 第二阶段：引入NTLBG约束
                return {
                    'task': self.initial_weights['task'],
                    'ntlbg': self.initial_weights['ntlbg'],
                    'alignment': 0.0,
                    'temporal': 0.0,
                    'info': 0.0
                }
            else:
                # 第三阶段：全损失训练
                return self.initial_weights
        
        elif self.schedule_type == 'adaptive':
            # 自适应权重调整
            phase = (step % 1000) / 1000
            weights = {}
            for key, value in self.initial_weights.items():
                if key == 'ntlbg':
                    # NTLBG约束使用余弦调度
                    weights[key] = value * (0.5 + 0.5 * np.cos(phase * np.pi))
                else:
                    weights[key] = value
            return weights
        
        else:
            return self.initial_weights
    
    def update_schedule(self, metrics: Dict[str, float], step: int):
        """根据训练指标动态调整权重计划"""
        if self.schedule_type == 'adaptive':
            # 如果NTLBG损失过高，增加其权重
            if metrics.get('ntlbg_loss', 0) > 1.0:
                self.initial_weights['ntlbg'] *= 1.1
            elif metrics.get('ntlbg_loss', 0) < 0.1:
                self.initial_weights['ntlbg'] *= 0.9
            
            # 确保权重在合理范围内
            self.initial_weights['ntlbg'] = np.clip(self.initial_weights['ntlbg'], 0.1, 2.0)


class NTLBGLossComputer:
    """NTLBG损失计算器"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.loss_weights = config.get('loss_weights', {})
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # 损失权重调度器
        self.weight_scheduler = LossWeightScheduler(config)
    
    def compute_task_loss(self, outputs: Dict, labels: torch.Tensor) -> torch.Tensor:
        """计算任务损失（语言建模损失）"""
        if 'logits' in outputs:
            logits = outputs['logits']
            # 展平logits和labels用于计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.ce_loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_ntlbg_loss(self, outputs: Dict) -> torch.Tensor:
        """计算NTLBG约束损失"""
        ntlbg_components = outputs.get('ntlbg_components', {})
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 1. 代表点分布约束
        if 'representative_points' in ntlbg_components:
            rep_points = ntlbg_components['representative_points']
            mean = ntlbg_components.get('distribution_mean')
            cov = ntlbg_components.get('distribution_cov')
            
            if mean is not None and cov is not None:
                # 确保代表点在同一等高椭球面上
                mahal_distances = self._compute_mahalanobis_distances(rep_points, mean, cov)
                # 代表点应该具有相似的马氏距离
                distance_variance = torch.var(mahal_distances, dim=-1).mean()
                total_loss += distance_variance
        
        # 2. 查询相关性约束
        if 'query_relevance' in ntlbg_components:
            query_relevance = ntlbg_components['query_relevance']
            # 确保代表点与查询高度相关
            relevance_loss = -torch.log(query_relevance + 1e-8).mean()
            total_loss += relevance_loss
        
        # 3. 覆盖度约束
        if 'coverage_score' in ntlbg_components:
            coverage = ntlbg_components['coverage_score']
            # 鼓励高覆盖度
            coverage_loss = -torch.log(coverage + 1e-8).mean()
            total_loss += coverage_loss
        
        # 4. 多样性约束
        if 'diversity_score' in ntlbg_components:
            diversity = ntlbg_components['diversity_score']
            # 鼓励代表点多样性
            diversity_loss = -torch.log(diversity + 1e-8).mean()
            total_loss += diversity_loss
        
        return total_loss
    
    def compute_alignment_loss(self, outputs: Dict) -> torch.Tensor:
        """计算特征对齐损失"""
        alignment_components = outputs.get('alignment_components', {})
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 1. 视频-文本对齐损失
        if 'video_features' in alignment_components and 'text_features' in alignment_components:
            video_features = alignment_components['video_features']
            text_features = alignment_components['text_features']
            
            # 余弦相似度损失
            target = torch.ones(video_features.size(0), device=self.device)
            alignment_loss = self.cosine_loss(video_features, text_features, target)
            total_loss += alignment_loss
        
        # 2. 特征空间对齐损失
        if 'aligned_features' in alignment_components and 'target_features' in alignment_components:
            aligned_features = alignment_components['aligned_features']
            target_features = alignment_components['target_features']
            
            # L2损失
            space_alignment_loss = self.mse_loss(aligned_features, target_features)
            total_loss += space_alignment_loss
        
        return total_loss
    
    def compute_temporal_loss(self, outputs: Dict) -> torch.Tensor:
        """计算时序连贯性损失"""
        temporal_components = outputs.get('temporal_components', {})
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 1. 时序一致性损失
        if 'temporal_features' in temporal_components:
            temporal_features = temporal_components['temporal_features']
            # 相邻帧特征应该相似
            if temporal_features.size(1) > 1:
                diff = temporal_features[:, 1:] - temporal_features[:, :-1]
                consistency_loss = torch.mean(torch.norm(diff, dim=-1))
                total_loss += consistency_loss
        
        # 2. 时序顺序损失
        if 'position_embeddings' in temporal_components:
            pos_emb = temporal_components['position_embeddings']
            # 位置嵌入应该保持单调性
            if pos_emb.size(1) > 1:
                pos_diff = pos_emb[:, 1:] - pos_emb[:, :-1]
                order_loss = torch.mean(torch.clamp(-pos_diff, min=0))
                total_loss += order_loss
        
        return total_loss
    
    def compute_info_loss(self, outputs: Dict) -> torch.Tensor:
        """计算信息保持损失"""
        info_components = outputs.get('info_components', {})
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 1. 信息熵损失
        if 'attention_weights' in info_components:
            attention_weights = info_components['attention_weights']
            # 防止注意力权重过于尖锐
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
            entropy_loss = -torch.mean(entropy)
            total_loss += entropy_loss * 0.1
        
        # 2. 特征多样性损失
        if 'feature_diversity' in info_components:
            diversity = info_components['feature_diversity']
            # 鼓励特征多样性
            diversity_loss = -torch.log(diversity + 1e-8).mean()
            total_loss += diversity_loss * 0.1
        
        return total_loss
    
    def compute_loss(self, outputs: Dict, labels: torch.Tensor, step: int, epoch: int) -> Dict[str, torch.Tensor]:
        """计算完整的多任务损失"""
        # 获取当前权重
        weights = self.weight_scheduler.get_weights(step, epoch)
        
        # 计算各项损失
        losses = {}
        
        # 任务损失
        task_loss = self.compute_task_loss(outputs, labels)
        losses['task'] = task_loss
        
        # NTLBG约束损失
        ntlbg_loss = self.compute_ntlbg_loss(outputs)
        losses['ntlbg'] = ntlbg_loss
        
        # 特征对齐损失
        alignment_loss = self.compute_alignment_loss(outputs)
        losses['alignment'] = alignment_loss
        
        # 时序连贯性损失
        temporal_loss = self.compute_temporal_loss(outputs)
        losses['temporal'] = temporal_loss
        
        # 信息保持损失
        info_loss = self.compute_info_loss(outputs)
        losses['info'] = info_loss
        
        # 计算总损失
        total_loss = (
            weights.get('task', 1.0) * losses['task'] +
            weights.get('ntlbg', 0.5) * losses['ntlbg'] +
            weights.get('alignment', 0.3) * losses['alignment'] +
            weights.get('temporal', 0.2) * losses['temporal'] +
            weights.get('info', 0.1) * losses['info']
        )
        losses['total'] = total_loss
        
        # 添加权重信息
        losses['weights'] = weights
        
        return losses
    
    def _compute_mahalanobis_distances(self, points: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """计算马氏距离"""
        # points: (batch_size, n_points, feature_dim)
        # mean: (batch_size, feature_dim)
        # cov: (batch_size, feature_dim, feature_dim)
        
        # 计算点到均值的差
        diff = points - mean.unsqueeze(1)  # (batch_size, n_points, feature_dim)
        
        # 计算协方差矩阵的逆
        try:
            cov_inv = torch.inverse(cov + 1e-6 * torch.eye(cov.size(-1), device=self.device))
        except:
            # 如果协方差矩阵不可逆，使用伪逆
            cov_inv = torch.pinverse(cov)
        
        # 计算马氏距离
        mahal_dist = torch.sqrt(torch.sum(diff * torch.matmul(diff, cov_inv), dim=-1))
        
        return mahal_dist
    
    def get_loss_statistics(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """获取损失统计信息"""
        stats = {}
        for key, value in losses.items():
            if key != 'weights' and torch.is_tensor(value):
                stats[f'{key}_loss'] = value.item()
        
        # 添加权重信息
        if 'weights' in losses:
            for key, value in losses['weights'].items():
                stats[f'{key}_weight'] = value
        
        return stats 