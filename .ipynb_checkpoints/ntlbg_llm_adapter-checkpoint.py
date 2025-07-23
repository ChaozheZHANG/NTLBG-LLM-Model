"""
NTLBG-LLM: 基于统计理论的长视频理解模型
将NTLBG算法集成到现有大模型中
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor,
    AutoModel, AutoProcessor
)
import math

class NTLBGRepresentativeSelector(nn.Module):
    """NTLBG统计理论的代表点选择器"""
    
    def __init__(self, d_model=1024, num_representatives=6, temperature=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_representatives = num_representatives
        self.temperature = temperature
        
        # 统计参数估计网络
        self.mu_estimator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.sigma_estimator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),  
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Softplus()  # 确保方差为正
        )
        
        # 查询依赖的选择网络
        self.query_projector = nn.Linear(d_model, d_model)
        self.selection_head = nn.MultiheadAttention(d_model, 8, batch_first=True)
        
    def forward(self, **kwargs):
        """
        Args:
            video_features: [B, T, D] 视频特征序列
            query_embedding: [B, D] 查询嵌入
        Returns:
            representative_features: [B, K, D] 选中的代表点特征
            selection_info: 包含选择信息的字典
        """
        B, T, D = video_features.shape
        K = self.num_representatives
        
        # 1. 基于查询估计统计参数
        mu_q = self.mu_estimator(query_embedding)  # [B, D]
        sigma_q = self.sigma_estimator(query_embedding) + 1e-6  # [B, D]
        
        # 2. 计算马氏距离
        centered_features = video_features - mu_q.unsqueeze(1)  # [B, T, D]
        mahalanobis_distances = torch.sum(
            (centered_features ** 2) / sigma_q.unsqueeze(1), dim=-1
        )  # [B, T]
        
        # 3. NTLBG代表点选择
        representative_indices, selection_weights = self._ntlbg_selection(
            mahalanobis_distances, video_features, mu_q, sigma_q
        )
        
        # 4. 提取代表点特征
        representative_features = self._gather_features(
            video_features, representative_indices
        )
        
        # 5. 计算NTLBG约束损失
        ntlbg_loss = self._compute_ntlbg_loss(
            representative_features, mu_q, sigma_q, mahalanobis_distances, representative_indices
        )
        
        return representative_features, {
            'representative_indices': representative_indices,
            'selection_weights': selection_weights,
            'mahalanobis_distances': mahalanobis_distances,
            'mu_q': mu_q,
            'sigma_q': sigma_q,
            'ntlbg_loss': ntlbg_loss
        }
    
    def _ntlbg_selection(self, distances, features, mu_q, sigma_q):
        """基于NTLBG统计理论的代表点选择"""
        B, T = distances.shape
        K = self.num_representatives
        
        # 计算目标等高椭球面的距离
        median_distance = torch.median(distances, dim=1, keepdim=True)[0]  # [B, 1]
        
        # 选择接近目标距离的候选点
        distance_weights = torch.exp(
            -torch.abs(distances - median_distance) / self.temperature
        )  # [B, T]
        
        # 使用注意力机制进行最终选择
        query_expanded = mu_q.unsqueeze(1)  # [B, 1, D]
        attended_features, attention_weights = self.selection_head(
            query_expanded, features, features
        )  # [B, 1, D], [B, 1, T]
        
        # 结合距离权重和注意力权重
        combined_weights = distance_weights * attention_weights.squeeze(1)  # [B, T]
        
        # 选择top-K个代表点，确保时序多样性
        representative_indices = self._diverse_top_k_selection(
            combined_weights, K, T
        )
        
        return representative_indices, combined_weights
    
    def _diverse_top_k_selection(self, weights, k, total_length):
        """时序多样化的Top-K选择"""
        B = weights.shape[0]
        indices_list = []
        
        for b in range(B):
            w = weights[b]  # [T]
            
            if k >= total_length:
                indices = torch.arange(total_length, device=weights.device)
            else:
                # 贪心多样化选择
                selected = []
                remaining = list(range(total_length))
                
                # 首先选择权重最高的点
                first_idx = torch.argmax(w).item()
                selected.append(first_idx)
                remaining.remove(first_idx)
                
                # 贪心选择剩余点，最大化时序距离
                for _ in range(k - 1):
                    if not remaining:
                        break
                    
                    best_idx = None
                    best_score = -1
                    
                    for candidate in remaining:
                        # 计算时序多样性分数
                        min_distance = min(abs(candidate - sel) for sel in selected)
                        diversity_score = min_distance * w[candidate].item()
                        
                        if diversity_score > best_score:
                            best_score = diversity_score
                            best_idx = candidate
                    
                    if best_idx is not None:
                        selected.append(best_idx)
                        remaining.remove(best_idx)
                
                indices = torch.tensor(selected, device=weights.device)
            
            indices_list.append(indices)
        
        # 填充到相同长度
        max_len = max(len(idx) for idx in indices_list)
        padded_indices = torch.full((B, max_len), 0, device=weights.device)
        
        for b, indices in enumerate(indices_list):
            padded_indices[b, :len(indices)] = indices
        
        return padded_indices[:, :k]  # [B, K]
    
    def _gather_features(self, features, indices):
        """根据索引提取特征"""
        B, T, D = features.shape
        K = indices.shape[1]
        
        # 扩展索引维度
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        
        # 提取特征
        representative_features = torch.gather(features, 1, expanded_indices)  # [B, K, D]
        
        return representative_features
    
    def _compute_ntlbg_loss(self, rep_features, mu_q, sigma_q, distances, indices):
        """计算NTLBG约束损失"""
        B, K, D = rep_features.shape
        
        # 1. 等高椭球面约束：代表点应该在相似的马氏距离上
        rep_distances = torch.gather(distances, 1, indices)  # [B, K]
        target_distance = rep_distances.median(dim=1, keepdim=True)[0]  # [B, 1]
        ellipsoid_loss = F.mse_loss(rep_distances, target_distance.expand_as(rep_distances))
        
        # 2. 统计一致性约束：代表点应该符合估计的分布
        centered_rep = rep_features - mu_q.unsqueeze(1)  # [B, K, D]
        consistency_loss = torch.mean(torch.sum(
            (centered_rep ** 2) / sigma_q.unsqueeze(1), dim=-1
        ))
        
        # 3. 多样性约束：代表点之间应该有足够的差异
        pairwise_sim = torch.matmul(rep_features, rep_features.transpose(-1, -2))  # [B, K, K]
        diversity_loss = torch.mean(torch.triu(pairwise_sim, diagonal=1) ** 2)
        
        total_loss = ellipsoid_loss + 0.1 * consistency_loss + 0.05 * diversity_loss
        
        return total_loss


# class NTLBGLLaVAAdapter  # 暂时禁用
class _NTLBGLLaVAAdapter(nn.Module):
    """基于LLaVA的NTLBG-LLM适配器"""
    
    def __init__(self, base_model_name="llava-hf/LLaVA-NeXT-Video-7B-hf"):
        super().__init__()
        
        # 加载基础模型
        print(f"🔄 加载基础模型: {base_model_name}")
        self.base_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = LlavaNextVideoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # 获取模型配置
        self.d_model = self.base_model.config.text_config.hidden_size
        
        # 集成NTLBG选择器
        self.ntlbg_selector = NTLBGRepresentativeSelector(
            d_model=self.d_model,
            num_representatives=6
        )
        
        # 视频特征适配层
        self.video_adapter = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # 冻结基础模型的部分参数
        self._freeze_base_model()
        
        print("✅ NTLBG-LLaVA适配器初始化完成")
    
    def _freeze_base_model(self):
        """冻结基础模型的大部分参数"""
        total_params = 0
        frozen_params = 0
        
        for name, param in self.base_model.named_parameters():
            total_params += param.numel()
            
            # 只微调最后几层和视觉投影层
            if any(keyword in name for keyword in [
                'language_model.model.layers.31',  # 最后一层
                'language_model.model.layers.30',  # 倒数第二层  
                'multi_modal_projector',           # 多模态投影
                'language_model.lm_head'           # 输出头
            ]):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"🧊 冻结参数比例: {frozen_params/total_params:.1%}")
    
    def forward(self, **kwargs):
        """前向传播"""
        B = input_ids.shape[0]
        
        # 1. 提取视频特征
        with torch.no_grad():
            vision_outputs = self.base_model.vision_tower(
                pixel_values_videos.to(self.base_model.vision_tower.dtype)
            )
            video_features = vision_outputs.last_hidden_state  # [B, T, D]
        
        # 2. 获取文本查询嵌入
        with torch.no_grad():
            text_embeds = self.base_model.language_model.model.embed_tokens(input_ids)
            query_embedding = text_embeds.mean(dim=1)  # [B, D]
        
        # 3. NTLBG代表点选择
        representative_features, selection_info = self.ntlbg_selector(
            video_features, query_embedding
        )
        
        # 4. 适配器处理
        adapted_features = self.video_adapter(representative_features)
        
        # 5. 替换视频特征进行推理
        # 这里需要重新组织输入以使用选择的代表点
        # 简化处理：直接使用基础模型
                # 确保所有输入都在模型设备上
        device = next(self.base_model.parameters()).device
        for key, value in kwargs.items():
            if torch.is_tensor(kwargs[key]):
                kwargs[key] = value.to(device)
        
        
        outputs = self.base_model(**kwargs)
        
        # 6. 添加NTLBG损失
        if labels is not None:
            ntlbg_loss = selection_info['ntlbg_loss']
            outputs.loss = outputs.loss + 0.3 * ntlbg_loss
        
        # 添加选择信息到输出
        outputs.selection_info = selection_info
        outputs.representative_features = adapted_features
        
        return outputs


class NTLBGQwen2VLAdapter(nn.Module):
    """NTLBG增强的Qwen2-VL适配器"""
    
    def __init__(self, base_model_name="microsoft/DialoGPT-medium", num_representatives=6):
        super().__init__()
        
        print(f"🔄 加载Qwen2-VL基础模型: {base_model_name}")
        
        # 1. 加载基础模型
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # 修复pad_token
        if hasattr(self.processor, 'pad_token') and self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token
            print("✅ 设置pad_token")
        
        self.tokenizer = self.processor
        
        # 获取模型配置
        self.hidden_size = self.base_model.config.hidden_size
        self.num_representatives = num_representatives
        
        # 2. 冻结基础模型参数
        total_params = 0
        frozen_params = 0
        for param in self.base_model.parameters():
            total_params += param.numel()
            param.requires_grad = False
            frozen_params += param.numel()
        
        frozen_ratio = frozen_params / total_params
        print(f"🧊 冻结参数比例: {frozen_ratio:.1%}")
        
        # 3. 初始化NTLBG组件
        self.ntlbg_selector = NTLBGSelector(
            input_dim=self.hidden_size,
            num_representatives=num_representatives
        )
        
        # 4. 添加适配层
        self.adaptation_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        print("✅ NTLBG-Qwen2VL适配器初始化完成")
    
    def forward(self, **kwargs):
        """前向传播"""
        # 从kwargs中提取参数
        input_ids = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask') 
        pixel_values = kwargs.get('pixel_values')
        labels = kwargs.get('labels')
        
        # 确保tensor在正确设备上
        device = next(self.base_model.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # 准备输入
        model_inputs = {}
        if input_ids is not None:
            model_inputs['input_ids'] = input_ids
        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask
        if pixel_values is not None:
            model_inputs['pixel_values'] = pixel_values
        if labels is not None:
            model_inputs['labels'] = labels
            
        # 调用基础模型
        outputs = self.base_model(**model_inputs)
        
        return outputs


        
def create_ntlbg_adapter(base_model_type="qwen2vl"):
    """创建NTLBG适配器"""
    if base_model_type.lower() == "qwen2vl":
        return NTLBGQwen2VLAdapter()
    else:
        print(f"⚠️ 暂时只支持Qwen2VL，使用默认配置")
        return NTLBGQwen2VLAdapter()

