import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional

class RichRepresentativePointConstructor(nn.Module):
    """
    富代表点构造器：为每个代表点补充时空上下文信息
    解决"点图动态变化"的对齐问题
    """
    
    def __init__(self,
                 d_visual: int = 768,
                 d_context: int = 256,
                 d_temporal: int = 64,
                 context_window: int = 5,
                 max_sequence_length: int = 1000):
        super().__init__()
        
        self.d_visual = d_visual
        self.d_context = d_context
        self.d_temporal = d_temporal
        self.context_window = context_window
        self.max_sequence_length = max_sequence_length
        
        # 上下文编码器：编码周围帧的时序信息
        self.context_encoder = nn.LSTM(
            input_size=d_visual,
            hidden_size=d_context // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 全局位置编码器
        self.global_position_encoder = nn.Sequential(
            nn.Linear(1, d_temporal // 2),
            nn.ReLU(),
            nn.Linear(d_temporal // 2, d_temporal)
        )
        
        # 相对位置编码器
        self.relative_position_encoder = nn.Embedding(
            num_embeddings=2 * context_window + 1,
            embedding_dim=d_temporal
        )
        
        # 代表性权重预测器
        self.representativeness_predictor = nn.Sequential(
            nn.Linear(d_visual + d_context, d_visual // 2),
            nn.LayerNorm(d_visual // 2),
            nn.ReLU(),
            nn.Linear(d_visual // 2, 1),
            nn.Sigmoid()
        )
        
        # 覆盖域估计器
        self.coverage_estimator = nn.Sequential(
            nn.Linear(d_visual + d_context, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [start_offset, end_offset]
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # 语义重要性评估器
        self.semantic_importance_estimator = nn.Sequential(
            nn.Linear(d_visual, d_visual // 2),
            nn.ReLU(),
            nn.Linear(d_visual // 2, 1),
            nn.Sigmoid()
        )
        
        # 过渡状态预测器
        self.transition_predictor = nn.Sequential(
            nn.Linear(d_visual * 2, d_visual),
            nn.ReLU(),
            nn.Linear(d_visual, d_visual)
        )
        
        # 富代表点融合器
        self.rich_point_fuser = nn.Sequential(
            nn.Linear(d_visual + d_context + d_temporal * 2 + 3, d_visual),
            nn.LayerNorm(d_visual),
            nn.ReLU(),
            nn.Linear(d_visual, d_visual)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'predictor' in name or 'estimator' in name:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.zeros_(param.data)
    
    def forward(self,
                video_features: torch.Tensor,
                representative_indices: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                query_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        构造富代表点
        
        Args:
            video_features: [B, T, d_visual] 原始视频特征
            representative_indices: [B, k] 代表点索引
            timestamps: [B, T] 时间戳（可选）
            query_embedding: [B, d_query] 查询嵌入（可选）
        
        Returns:
            Dict包含富代表点的各种属性
        """
        batch_size, T, d_visual = video_features.shape
        k = representative_indices.shape[1]
        
        if timestamps is None:
            timestamps = torch.linspace(0, 1, T, device=video_features.device).unsqueeze(0).expand(batch_size, -1)
        
        # 1. 提取基础视觉特征
        base_features = self._extract_representative_features(video_features, representative_indices)
        
        # 2. 构建上下文特征
        context_features = self._build_context_features(video_features, representative_indices)
        
        # 3. 编码时序信息
        temporal_features = self._encode_temporal_information(representative_indices, timestamps)
        
        # 4. 计算代表性权重
        representativeness_weights = self._compute_representativeness_weights(
            base_features, context_features
        )
        
        # 5. 估计覆盖域
        coverage_ranges = self._estimate_coverage_ranges(
            base_features, context_features, representative_indices, T
        )
        
        # 6. 评估语义重要性
        semantic_importance = self._evaluate_semantic_importance(
            base_features, query_embedding
        )
        
        # 7. 生成过渡信息
        transition_features = self._generate_transition_features(
            base_features, representative_indices
        )
        
        # 8. 融合所有信息生成富代表点
        rich_representative_points = self._fuse_rich_features(
            base_features, context_features, temporal_features,
            representativeness_weights, semantic_importance, coverage_ranges
        )
        
        return {
            'rich_features': rich_representative_points,  # [B, k, d_visual]
            'base_features': base_features,
            'context_features': context_features,
            'temporal_features': temporal_features,
            'representativeness_weights': representativeness_weights,
            'coverage_ranges': coverage_ranges,
            'semantic_importance': semantic_importance,
            'transition_features': transition_features
        }
    
    def _extract_representative_features(self,
                                       video_features: torch.Tensor,
                                       indices: torch.Tensor) -> torch.Tensor:
        """提取代表点的基础视觉特征"""
        batch_size, k = indices.shape
        d_visual = video_features.shape[-1]
        
        # 扩展索引进行gather操作
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, d_visual)
        representative_features = torch.gather(video_features, 1, expanded_indices)
        
        return representative_features  # [B, k, d_visual]
    
    def _build_context_features(self,
                               video_features: torch.Tensor,
                               indices: torch.Tensor) -> torch.Tensor:
        """为每个代表点构建上下文特征"""
        batch_size, T, d_visual = video_features.shape
        k = indices.shape[1]
        
        context_features = []
        
        for b in range(batch_size):
            batch_contexts = []
            
            for i in range(k):
                frame_idx = indices[b, i].item()
                
                # 确定上下文窗口范围
                start_idx = max(0, frame_idx - self.context_window)
                end_idx = min(T, frame_idx + self.context_window + 1)
                
                # 提取上下文帧
                context_frames = video_features[b, start_idx:end_idx]  # [window_size, d_visual]
                
                # 如果上下文窗口不够，进行padding
                window_size = end_idx - start_idx
                if window_size < 2 * self.context_window + 1:
                    padding_size = 2 * self.context_window + 1 - window_size
                    padding = torch.zeros(padding_size, d_visual, device=video_features.device)
                    context_frames = torch.cat([context_frames, padding], dim=0)
                
                # LSTM编码上下文
                context_encoded, _ = self.context_encoder(context_frames.unsqueeze(0))
                context_feature = context_encoded[0, -1]  # 取最后一个时间步的输出
                
                batch_contexts.append(context_feature)
            
            context_features.append(torch.stack(batch_contexts))
        
        return torch.stack(context_features)  # [B, k, d_context]
    
    def _encode_temporal_information(self,
                                   indices: torch.Tensor,
                                   timestamps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码时序信息"""
        batch_size, k = indices.shape
        T = timestamps.shape[1]
        
        # 全局位置编码
        global_positions = []
        for b in range(batch_size):
            positions = timestamps[b, indices[b]]  # [k]
            global_positions.append(positions)
        global_positions = torch.stack(global_positions)  # [B, k]
        
        global_temporal_features = self.global_position_encoder(
            global_positions.unsqueeze(-1)
        )  # [B, k, d_temporal]
        
        # 相对位置编码
        relative_positions = []
        for i in range(k):
            if i == 0:
                rel_pos = self.context_window  # 中心位置
            else:
                # 计算相对于前一个代表点的位置
                rel_pos = min(max(0, self.context_window + i - (i-1)), 2 * self.context_window)
            relative_positions.append(rel_pos)
        
        relative_positions = torch.tensor(relative_positions, device=indices.device).unsqueeze(0).expand(batch_size, -1)
        relative_temporal_features = self.relative_position_encoder(relative_positions)  # [B, k, d_temporal]
        
        return {
            'global': global_temporal_features,
            'relative': relative_temporal_features
        }
    
    def _compute_representativeness_weights(self,
                                          base_features: torch.Tensor,
                                          context_features: torch.Tensor) -> torch.Tensor:
        """计算每个代表点的代表性权重"""
        # 融合基础特征和上下文特征
        combined_features = torch.cat([base_features, context_features], dim=-1)  # [B, k, d_visual + d_context]
        
        # 预测代表性权重
        weights = self.representativeness_predictor(combined_features).squeeze(-1)  # [B, k]
        
        # 归一化权重
        weights = F.softmax(weights, dim=-1)
        
        return weights
    
    def _estimate_coverage_ranges(self,
                                base_features: torch.Tensor,
                                context_features: torch.Tensor,
                                indices: torch.Tensor,
                                total_frames: int) -> torch.Tensor:
        """估计每个代表点的覆盖范围"""
        # 融合特征
        combined_features = torch.cat([base_features, context_features], dim=-1)
        
        # 预测覆盖偏移
        coverage_offsets = self.coverage_estimator(combined_features)  # [B, k, 2]
        
        # 转换为实际的覆盖范围
        batch_size, k = indices.shape
        coverage_ranges = torch.zeros(batch_size, k, 2, device=indices.device)
        
        for b in range(batch_size):
            for i in range(k):
                center_frame = indices[b, i].float()
                
                # 计算覆盖范围（基于预测的偏移）
                start_offset = coverage_offsets[b, i, 0] * (total_frames // 4)  # 最大偏移量为总帧数的1/4
                end_offset = coverage_offsets[b, i, 1] * (total_frames // 4)
                
                start_frame = torch.clamp(center_frame + start_offset, 0, total_frames - 1)
                end_frame = torch.clamp(center_frame + end_offset, 0, total_frames - 1)
                
                coverage_ranges[b, i, 0] = start_frame
                coverage_ranges[b, i, 1] = end_frame
        
        return coverage_ranges  # [B, k, 2]
    
    def _evaluate_semantic_importance(self,
                                    base_features: torch.Tensor,
                                    query_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """评估语义重要性"""
        if query_embedding is None:
            # 无查询时，使用特征本身评估重要性
            importance = self.semantic_importance_estimator(base_features).squeeze(-1)  # [B, k]
        else:
            # 有查询时，计算与查询的相关性
            batch_size, k, d_visual = base_features.shape
            query_expanded = query_embedding.unsqueeze(1).expand(-1, k, -1)  # [B, k, d_query]
            
            # 如果查询维度与视觉特征维度不同，需要投影
            if query_expanded.shape[-1] != d_visual:
                query_proj = nn.Linear(query_expanded.shape[-1], d_visual, device=base_features.device)
                query_expanded = query_proj(query_expanded)
            
            # 计算相似度作为重要性
            similarity = F.cosine_similarity(base_features, query_expanded, dim=-1)  # [B, k]
            importance = torch.sigmoid(similarity)  # 转换到 [0, 1] 范围
        
        return importance
    
    def _generate_transition_features(self,
                                    base_features: torch.Tensor,
                                    indices: torch.Tensor) -> torch.Tensor:
        """生成代表点间的过渡特征"""
        batch_size, k, d_visual = base_features.shape
        
        if k < 2:
            # 如果只有一个代表点，返回零向量
            return torch.zeros(batch_size, k-1 if k > 0 else 0, d_visual, device=base_features.device)
        
        transition_features = []
        
        for i in range(k - 1):
            current_feature = base_features[:, i]  # [B, d_visual]
            next_feature = base_features[:, i + 1]  # [B, d_visual]
            
            # 预测过渡特征
            combined = torch.cat([current_feature, next_feature], dim=-1)  # [B, 2*d_visual]
            transition = self.transition_predictor(combined)  # [B, d_visual]
            
            transition_features.append(transition)
        
        if transition_features:
            return torch.stack(transition_features, dim=1)  # [B, k-1, d_visual]
        else:
            return torch.zeros(batch_size, 0, d_visual, device=base_features.device)
    
    def _fuse_rich_features(self,
                          base_features: torch.Tensor,
                          context_features: torch.Tensor,
                          temporal_features: Dict[str, torch.Tensor],
                          weights: torch.Tensor,
                          importance: torch.Tensor,
                          coverage_ranges: torch.Tensor) -> torch.Tensor:
        """融合所有特征生成最终的富代表点"""
        batch_size, k = base_features.shape[:2]
        
        # 组合所有特征
        features_to_fuse = [
            base_features,  # [B, k, d_visual]
            context_features,  # [B, k, d_context]
            temporal_features['global'],  # [B, k, d_temporal]
            temporal_features['relative'],  # [B, k, d_temporal]
            weights.unsqueeze(-1),  # [B, k, 1]
            importance.unsqueeze(-1),  # [B, k, 1]
            coverage_ranges.view(batch_size, k, -1)  # [B, k, 2] -> [B, k, 2]
        ]
        
        # 还需要一个特征来表示时序顺序
        sequence_order = torch.arange(k, device=base_features.device).float().unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1) / k
        features_to_fuse.append(sequence_order)  # [B, k, 1]
        
        # 拼接所有特征
        fused_input = torch.cat(features_to_fuse, dim=-1)  # [B, k, total_dim]
        
        # 通过融合器生成最终特征
        rich_features = self.rich_point_fuser(fused_input)  # [B, k, d_visual]
        
        return rich_features
    
    def create_llm_compatible_tokens(self,
                                   rich_features: torch.Tensor,
                                   base_features: torch.Tensor,
                                   temporal_info: Dict[str, torch.Tensor],
                                   weights: torch.Tensor,
                                   coverage_ranges: torch.Tensor,
                                   original_indices: torch.Tensor,
                                   total_frames: int) -> Dict[str, torch.Tensor]:
        """
        创建LLM兼容的token表示
        这是解决"特征空间对齐"问题的关键
        """
        batch_size, k = rich_features.shape[:2]
        
        # 1. 位置编码：既包含原始位置，也包含压缩后位置
        original_positions = original_indices.float() / total_frames  # 归一化的原始位置
        compressed_positions = torch.arange(k, device=rich_features.device).float().unsqueeze(0).expand(batch_size, -1) / k
        
        # 2. 创建特殊的位置token
        position_tokens = torch.cat([
            original_positions.unsqueeze(-1),
            compressed_positions.unsqueeze(-1)
        ], dim=-1)  # [B, k, 2]
        
        # 3. 权重token
        weight_tokens = weights.unsqueeze(-1)  # [B, k, 1]
        
        # 4. 覆盖范围token
        coverage_tokens = coverage_ranges  # [B, k, 2]
        
        # 5. 组合成最终的LLM输入token
        llm_tokens = torch.cat([
            rich_features,  # 主要特征
            position_tokens,  # 位置信息
            weight_tokens,  # 权重信息
            coverage_tokens  # 覆盖信息
        ], dim=-1)  # [B, k, d_visual + 2 + 1 + 2]
        
        return {
            'tokens': llm_tokens,
            'original_positions': original_positions,
            'compressed_positions': compressed_positions,
            'weights': weights,
            'coverage_ranges': coverage_ranges
        }
    
    def compute_information_preservation_loss(self,
                                            original_features: torch.Tensor,
                                            rich_features: torch.Tensor,
                                            indices: torch.Tensor) -> torch.Tensor:
        """
        计算信息保持损失：确保富代表点保持了原始信息
        """
        # 提取原始代表点特征
        original_repr = self._extract_representative_features(original_features, indices)
        
        # 计算重构损失
        reconstruction_loss = F.mse_loss(rich_features, original_repr)
        
        return reconstruction_loss
    
    def compute_temporal_coherence_loss(self,
                                      rich_features: torch.Tensor,
                                      indices: torch.Tensor) -> torch.Tensor:
        """
        计算时序连贯性损失：确保代表点间的时序关系合理
        """
        batch_size, k = rich_features.shape[:2]
        
        if k < 2:
            return torch.tensor(0.0, device=rich_features.device)
        
        coherence_loss = 0.0
        
        for i in range(k - 1):
            current_idx = indices[:, i]  # [B]
            next_idx = indices[:, i + 1]  # [B]
            
            # 时间间隔
            time_gap = (next_idx - current_idx).float()  # [B]
            
            # 特征差异
            feature_diff = torch.norm(
                rich_features[:, i + 1] - rich_features[:, i], 
                dim=-1
            )  # [B]
            
            # 期望：时间间隔越大，特征差异应该越大
            expected_diff = torch.sigmoid(time_gap / 10.0)  # 归一化
            
            # 损失：实际差异与期望差异的MSE
            coherence_loss += F.mse_loss(feature_diff, expected_diff)
        
        return coherence_loss / (k - 1)


class TemporalAlignment(nn.Module):
    """
    时序对齐模块：处理代表点间的时序关系
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        # 时序关系编码器
        self.temporal_relation_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 时序位置编码
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, 1000, d_model) * 0.02
        )
    
    def forward(self, rich_features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        对富代表点进行时序对齐
        """
        batch_size, k, d_model = rich_features.shape
        
        # 添加时序位置编码
        positions = indices.unsqueeze(-1).expand(-1, -1, d_model)  # [B, k, d_model]
        pos_encoding = torch.gather(
            self.temporal_pos_encoding.expand(batch_size, -1, -1), 
            1, 
            positions
        )
        
        # 加入位置编码
        aligned_features = rich_features + pos_encoding
        
        # 通过Transformer进行时序关系建模
        aligned_features = self.temporal_relation_encoder(aligned_features)
        
        return aligned_features


# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size, T, d_visual = 2, 100, 768
    k = 6
    
    # 创建测试数据
    video_features = torch.randn(batch_size, T, d_visual)
    representative_indices = torch.randint(0, T, (batch_size, k))
    timestamps = torch.linspace(0, 1, T).unsqueeze(0).expand(batch_size, -1)
    query_embedding = torch.randn(batch_size, 768)
    
    # 对索引进行排序（模拟实际情况）
    representative_indices, _ = torch.sort(representative_indices, dim=1)
    
    # 初始化模块
    rich_constructor = RichRepresentativePointConstructor(
        d_visual=d_visual,
        d_context=256,
        d_temporal=64,
        context_window=5
    )
    
    temporal_aligner = TemporalAlignment(d_model=d_visual)
    
    print("=== Testing Rich Representative Point Constructor ===")
    
    # 构造富代表点
    rich_results = rich_constructor(
        video_features=video_features,
        representative_indices=representative_indices,
        timestamps=timestamps,
        query_embedding=query_embedding
    )
    
    print(f"Rich features shape: {rich_results['rich_features'].shape}")
    print(f"Context features shape: {rich_results['context_features'].shape}")
    print(f"Representativeness weights: {rich_results['representativeness_weights'][0]}")
    print(f"Coverage ranges (batch 0): {rich_results['coverage_ranges'][0]}")
    
    # 创建LLM兼容token
    llm_tokens = rich_constructor.create_llm_compatible_tokens(
        rich_features=rich_results['rich_features'],
        base_features=rich_results['base_features'],
        temporal_info=rich_results['temporal_features'],
        weights=rich_results['representativeness_weights'],
        coverage_ranges=rich_results['coverage_ranges'],
        original_indices=representative_indices,
        total_frames=T
    )
    
    print(f"LLM tokens shape: {llm_tokens['tokens'].shape}")
    print(f"Original positions (batch 0): {llm_tokens['original_positions'][0]}")
    
    # 时序对齐
    aligned_features = temporal_aligner(
        rich_results['rich_features'], 
        representative_indices
    )
    print(f"Aligned features shape: {aligned_features.shape}")
    
    # 计算损失
    info_loss = rich_constructor.compute_information_preservation_loss(
        video_features, rich_results['rich_features'], representative_indices
    )
    temporal_loss = rich_constructor.compute_temporal_coherence_loss(
        rich_results['rich_features'], representative_indices
    )
    
    print(f"Information preservation loss: {info_loss.item():.6f}")
    print(f"Temporal coherence loss: {temporal_loss.item():.6f}")
    
    print("\n✅ Rich Representative Point Constructor working correctly!") 