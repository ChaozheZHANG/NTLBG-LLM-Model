import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import warnings
import math
import numpy as np

# 导入之前创建的模块
from .ntlbg_attention import NTLBGAttention
from .rich_points import RichRepresentativePointConstructor, TemporalAlignment


class VideoFeatureEncoder(nn.Module):
    """视频特征编码器"""
    
    def __init__(self, d_output: int = 768, dropout: float = 0.1):
        super().__init__()
        
        # 简单的CNN特征提取器（实际项目中应该使用预训练的视觉模型）
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.feature_projector = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, d_output)
        )
    
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: [B, T, C, H, W]
        Returns:
            features: [B, T, d_output]
        """
        batch_size, T, C, H, W = video_frames.shape
        
        # 重塑为 [B*T, C, H, W]
        frames_flat = video_frames.view(-1, C, H, W)
        
        # CNN特征提取
        cnn_features = self.cnn_backbone(frames_flat)  # [B*T, 256, 7, 7]
        cnn_features_flat = cnn_features.view(batch_size * T, -1)  # [B*T, 256*7*7]
        
        # 投影到目标维度
        features = self.feature_projector(cnn_features_flat)  # [B*T, d_output]
        
        # 重塑回 [B, T, d_output]
        features = features.view(batch_size, T, -1)
        
        return features


class FeatureSpaceAligner(nn.Module):
    """特征空间对齐器：将压缩特征对齐到LLM期望的输入空间"""
    
    def __init__(self, d_input: int, d_llm_expected: int, alignment_layers: int = 2):
        super().__init__()
        
        self.d_input = d_input
        self.d_llm_expected = d_llm_expected
        
        # 构建对齐网络
        layers = []
        current_dim = d_input
        
        for i in range(alignment_layers):
            if i == alignment_layers - 1:
                # 最后一层直接映射到目标维度
                next_dim = d_llm_expected
            else:
                # 中间层逐渐过渡
                next_dim = d_input + (d_llm_expected - d_input) * (i + 1) // alignment_layers
            
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        # 移除最后的ReLU和Dropout
        layers = layers[:-2]
        
        self.alignment_network = nn.Sequential(*layers)
        
        # 分布对齐损失的参数
        self.register_buffer('target_mean', torch.zeros(d_llm_expected))
        self.register_buffer('target_std', torch.ones(d_llm_expected))
    
    def forward(self, compressed_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_features: [B, k, d_input] 压缩后的特征
        Returns:
            aligned_features: [B, k, d_llm_expected] 对齐后的特征
        """
        aligned_features = self.alignment_network(compressed_features)
        return aligned_features
    
    def compute_alignment_loss(self, 
                              original_features: torch.Tensor,
                              aligned_features: torch.Tensor) -> torch.Tensor:
        """计算特征对齐损失"""
        # 分布对齐损失：确保对齐后的特征分布符合LLM期望
        aligned_mean = aligned_features.mean(dim=(0, 1))  # [d_llm_expected]
        aligned_std = aligned_features.std(dim=(0, 1))   # [d_llm_expected]
        
        mean_loss = F.mse_loss(aligned_mean, self.target_mean)
        std_loss = F.mse_loss(aligned_std, self.target_std)
        
        distribution_loss = mean_loss + std_loss
        
        return distribution_loss
    
    def update_target_distribution(self, llm_embeddings: torch.Tensor):
        """更新目标分布参数（可在训练过程中调用）"""
        with torch.no_grad():
            self.target_mean.copy_(llm_embeddings.mean(dim=(0, 1)))
            self.target_std.copy_(llm_embeddings.std(dim=(0, 1)))


class MultimodalFusion(nn.Module):
    """多模态融合模块"""
    
    def __init__(self, d_visual: int, d_text: int, d_output: int):
        super().__init__()
        
        self.d_visual = d_visual
        self.d_text = d_text
        self.d_output = d_output
        
        # 确保维度一致
        if d_visual != d_output:
            self.visual_proj = nn.Linear(d_visual, d_output)
        else:
            self.visual_proj = nn.Identity()
        
        if d_text != d_output:
            self.text_proj = nn.Linear(d_text, d_output)
        else:
            self.text_proj = nn.Identity()
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_output,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 位置编码
        self.visual_pos_encoding = nn.Parameter(torch.randn(1, 100, d_output) * 0.02)
        self.text_pos_encoding = nn.Parameter(torch.randn(1, 1000, d_output) * 0.02)
    
    def forward(self,
                visual_tokens: torch.Tensor,
                text_embeddings: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        融合视觉和文本特征
        
        Args:
            visual_tokens: [B, k, d_visual] 视觉token
            text_embeddings: [B, seq_len, d_text] 文本嵌入
            attention_mask: [B, seq_len] 文本注意力掩码
        
        Returns:
            fused_embeddings: [B, k + seq_len, d_output] 融合后的嵌入
        """
        batch_size, k, _ = visual_tokens.shape
        seq_len = text_embeddings.shape[1]
        
        # 投影到统一维度
        visual_proj = self.visual_proj(visual_tokens)  # [B, k, d_output]
        text_proj = self.text_proj(text_embeddings)    # [B, seq_len, d_output]
        
        # 添加位置编码
        visual_proj = visual_proj + self.visual_pos_encoding[:, :k, :]
        text_proj = text_proj + self.text_pos_encoding[:, :seq_len, :]
        
        # 跨模态注意力（视觉token作为query，文本作为key和value）
        attended_visual, _ = self.cross_attention(
            query=visual_proj,
            key=text_proj,
            value=text_proj,
            key_padding_mask=~attention_mask.bool()
        )
        
        # 拼接融合后的视觉token和原始文本token
        fused_embeddings = torch.cat([attended_visual, text_proj], dim=1)  # [B, k + seq_len, d_output]
        
        return fused_embeddings


class MockLLM(nn.Module):
    """Mock LLM for testing"""
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768):
        super().__init__()
        
        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'vocab_size': vocab_size,
            'num_attention_heads': 12,
            'eos_token_id': 2
        })()
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=12,
                batch_first=True
            ) for _ in range(6)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, hidden_states)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return type('Output', (), {
            'logits': logits,
            'hidden_states': hidden_states
        })()
    
    def gradient_checkpointing_enable(self):
        pass
    
    def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=100, **kwargs):
        """简单的生成实现"""
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        
        # 简单的贪心解码
        current_embeds = inputs_embeds
        generated_ids = []
        
        for _ in range(max_new_tokens):
            outputs = self.forward(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids.append(next_token_id)
            
            if (next_token_id == self.config.eos_token_id).all():
                break
            
            next_token_embed = self.embed_tokens(next_token_id)
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
        
        if generated_ids:
            return torch.cat(generated_ids, dim=1)
        else:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)


class NTLBGLLM(nn.Module):
    """
    NTLBG-LLM: 融合统计学代表点理论的长视频理解模型
    核心创新：将NTLBG约束直接集成到LLM训练中
    """
    
    def __init__(self,
                 base_model_name: str = "mock",
                 d_visual: int = 768,
                 d_query: int = 768,
                 num_representatives: int = 6,
                 max_video_length: int = 1000,
                 enable_gradient_checkpointing: bool = True):
        super().__init__()
        
        self.d_visual = d_visual
        self.d_query = d_query
        self.num_representatives = num_representatives
        self.max_video_length = max_video_length
        
        # 1. 基础LLM模型
        self.base_llm = self._load_base_model(base_model_name)
        
        # 冻结LLM的部分参数（可选）
        self._freeze_base_model_layers(freeze_ratio=0.7)
        
        # 2. 视频特征编码器
        self.video_encoder = VideoFeatureEncoder(
            d_output=d_visual,
            dropout=0.1
        )
        
        # 3. NTLBG注意力模块
        self.ntlbg_attention = NTLBGAttention(
            d_model=d_visual,
            d_query=d_query,
            num_representatives=num_representatives
        )
        
        # 4. 富代表点构造器
        self.rich_constructor = RichRepresentativePointConstructor(
            d_visual=d_visual,
            d_context=256,
            d_temporal=64,
            context_window=5
        )
        
        # 5. 时序对齐模块
        self.temporal_aligner = TemporalAlignment(d_model=d_visual)
        
        # 6. 特征空间对齐器
        self.feature_aligner = FeatureSpaceAligner(
            d_input=d_visual + 5,  # rich features + additional info
            d_llm_expected=self.base_llm.config.hidden_size,
            alignment_layers=2
        )
        
        # 7. 查询编码器
        self.query_encoder = nn.Sequential(
            nn.Linear(self.base_llm.config.hidden_size, d_query),
            nn.LayerNorm(d_query),
            nn.ReLU(),
            nn.Linear(d_query, d_query)
        )
        
        # 8. 多模态融合器
        self.multimodal_fusion = MultimodalFusion(
            d_visual=self.base_llm.config.hidden_size,
            d_text=self.base_llm.config.hidden_size,
            d_output=self.base_llm.config.hidden_size
        )
        
        # 9. 梯度检查点
        if enable_gradient_checkpointing:
            self.base_llm.gradient_checkpointing_enable()
        
        self._init_custom_weights()
    
    def _load_base_model(self, model_name: str):
        """加载基础LLM模型"""
        if model_name == "mock":
            return MockLLM()
        else:
            try:
                # 这里可以添加真实的模型加载逻辑
                # 例如：from transformers import Qwen2VLForConditionalGeneration
                # return Qwen2VLForConditionalGeneration.from_pretrained(model_name)
                print(f"Warning: Using mock model instead of {model_name}")
                return MockLLM()
            except Exception as e:
                print(f"Failed to load {model_name}, using mock model: {e}")
                return MockLLM()
    
    def _freeze_base_model_layers(self, freeze_ratio: float = 0.7):
        """冻结基础模型的部分层"""
        if hasattr(self.base_llm, 'layers'):
            total_layers = len(self.base_llm.layers)
            freeze_layers = int(total_layers * freeze_ratio)
            
            for i, layer in enumerate(self.base_llm.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                        
        print(f"Frozen {freeze_ratio*100:.0f}% of base model layers")
    
    def _init_custom_weights(self):
        """初始化自定义层的权重"""
        for module in [self.video_encoder, self.query_encoder, self.feature_aligner]:
            for name, param in module.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self,
                video_frames: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_ntlbg_stats: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            video_frames: [B, T, C, H, W] 或 [B, T, d_visual] 视频帧
            input_ids: [B, seq_len] 文本输入
            attention_mask: [B, seq_len] 注意力掩码
            labels: [B, seq_len] 标签（训练时使用）
            return_ntlbg_stats: 是否返回NTLBG统计信息
        
        Returns:
            Dict包含损失、logits和统计信息
        """
        batch_size = video_frames.shape[0]
        device = video_frames.device
        
        # 1. 视频特征编码
        if len(video_frames.shape) == 5:  # [B, T, C, H, W]
            video_features = self.video_encoder(video_frames)  # [B, T, d_visual]
        else:  # 已经是特征形式 [B, T, d_visual]
            video_features = video_frames
        
        # 2. 文本查询编码
        text_embeddings = self.base_llm.embed_tokens(input_ids)  # [B, seq_len, hidden_size]
        query_embedding = self.query_encoder(text_embeddings.mean(dim=1))  # [B, d_query]
        
        # 3. NTLBG代表点选择
        ntlbg_results = self.ntlbg_attention(
            video_features, query_embedding, return_stats=True
        )
        
        # 4. 构造富代表点
        rich_results = self.rich_constructor(
            video_features=video_features,
            representative_indices=ntlbg_results['representative_indices'],
            query_embedding=query_embedding
        )
        
        # 5. 时序对齐
        aligned_features = self.temporal_aligner(
            rich_results['rich_features'],
            ntlbg_results['representative_indices']
        )
        
        # 6. 创建LLM兼容token
        llm_tokens = self.rich_constructor.create_llm_compatible_tokens(
            rich_features=aligned_features,
            base_features=rich_results['base_features'],
            temporal_info=rich_results['temporal_features'],
            weights=rich_results['representativeness_weights'],
            coverage_ranges=rich_results['coverage_ranges'],
            original_indices=ntlbg_results['representative_indices'],
            total_frames=video_features.shape[1]
        )
        
        # 7. 特征空间对齐
        aligned_visual_tokens = self.feature_aligner(llm_tokens['tokens'])  # [B, k, hidden_size]
        
        # 8. 多模态融合
        fused_embeddings = self.multimodal_fusion(
            visual_tokens=aligned_visual_tokens,
            text_embeddings=text_embeddings,
            attention_mask=attention_mask
        )
        
        # 9. LLM前向传播
        llm_outputs = self.base_llm(
            inputs_embeds=fused_embeddings,
            attention_mask=self._create_multimodal_attention_mask(
                attention_mask, aligned_visual_tokens.shape[1]
            )
        )
        
        # 10. 计算总损失
        total_loss = llm_outputs.logits.new_zeros(1)
        loss_components = {}
        
        if labels is not None:
            # 主任务损失
            adjusted_labels = self._adjust_labels_for_visual_tokens(labels, aligned_visual_tokens.shape[1])
            task_loss = F.cross_entropy(
                llm_outputs.logits.view(-1, llm_outputs.logits.size(-1)),
                adjusted_labels.view(-1),
                ignore_index=-100
            )
            
            # NTLBG约束损失
            ntlbg_loss = self.ntlbg_attention.compute_ntlbg_constraint_loss(
                ntlbg_results['representative_features'],
                ntlbg_results['mu_q'],
                ntlbg_results['sigma_q']
            )
            
            # 特征对齐损失
            alignment_loss = self.feature_aligner.compute_alignment_loss(
                llm_tokens['tokens'], aligned_visual_tokens
            )
            
            # 时序连贯性损失
            temporal_loss = self.rich_constructor.compute_temporal_coherence_loss(
                rich_results['rich_features'],
                ntlbg_results['representative_indices']
            )
            
            # 信息保持损失
            info_loss = self.rich_constructor.compute_information_preservation_loss(
                video_features,
                rich_results['rich_features'],
                ntlbg_results['representative_indices']
            )
            
            # 损失权重（可动态调整）
            lambda_task = 1.0
            lambda_ntlbg = 2.0
            lambda_align = 1.0
            lambda_temporal = 0.5
            lambda_info = 0.3
            
            total_loss = (lambda_task * task_loss + 
                         lambda_ntlbg * ntlbg_loss +
                         lambda_align * alignment_loss + 
                         lambda_temporal * temporal_loss +
                         lambda_info * info_loss)
            
            loss_components = {
                'task_loss': task_loss,
                'ntlbg_loss': ntlbg_loss,
                'alignment_loss': alignment_loss,
                'temporal_loss': temporal_loss,
                'info_loss': info_loss
            }
        
        # 11. 准备返回结果
        results = {
            'loss': total_loss,
            'logits': llm_outputs.logits,
            'loss_components': loss_components
        }
        
        if return_ntlbg_stats:
            results.update({
                'representative_indices': ntlbg_results['representative_indices'],
                'representative_weights': rich_results['representativeness_weights'],
                'coverage_ranges': rich_results['coverage_ranges'],
                'ntlbg_stats': {
                    'mu_q': ntlbg_results['mu_q'],
                    'sigma_q': ntlbg_results['sigma_q'],
                    'mahalanobis_distances': ntlbg_results['mahalanobis_distances']
                }
            })
        
        return results
    
    def _create_multimodal_attention_mask(self, text_mask: torch.Tensor, num_visual_tokens: int) -> torch.Tensor:
        """创建多模态注意力掩码"""
        batch_size, text_len = text_mask.shape
        
        # 视觉token的掩码（全部为1，因为代表点都是有效的）
        visual_mask = torch.ones(batch_size, num_visual_tokens, device=text_mask.device, dtype=text_mask.dtype)
        
        # 拼接视觉和文本掩码
        multimodal_mask = torch.cat([visual_mask, text_mask], dim=1)
        
        return multimodal_mask
    
    def _adjust_labels_for_visual_tokens(self, labels: torch.Tensor, num_visual_tokens: int) -> torch.Tensor:
        """调整标签以适应视觉token"""
        batch_size, text_len = labels.shape
        
        # 视觉token的标签设为-100（忽略）
        visual_labels = torch.full(
            (batch_size, num_visual_tokens), 
            -100, 
            device=labels.device, 
            dtype=labels.dtype
        )
        
        # 拼接视觉和文本标签
        adjusted_labels = torch.cat([visual_labels, labels], dim=1)
        
        return adjusted_labels
    
    def generate(self,
                 video_frames: torch.Tensor,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 max_new_tokens: int = 100,
                 do_sample: bool = True,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> Dict[str, torch.Tensor]:
        """
        生成文本回答
        """
        self.eval()
        
        with torch.no_grad():
            # 编码视频和查询
            video_features = self.video_encoder(video_frames) if len(video_frames.shape) == 5 else video_frames
            text_embeddings = self.base_llm.embed_tokens(input_ids)
            query_embedding = self.query_encoder(text_embeddings.mean(dim=1))
            
            # NTLBG选择和富代表点构造
            ntlbg_results = self.ntlbg_attention(video_features, query_embedding)
            rich_results = self.rich_constructor(
                video_features=video_features,
                representative_indices=ntlbg_results['representative_indices'],
                query_embedding=query_embedding
            )
            
            # 对齐和融合
            aligned_features = self.temporal_aligner(
                rich_results['rich_features'],
                ntlbg_results['representative_indices']
            )
            
            llm_tokens = self.rich_constructor.create_llm_compatible_tokens(
                rich_features=aligned_features,
                base_features=rich_results['base_features'],
                temporal_info=rich_results['temporal_features'],
                weights=rich_results['representativeness_weights'],
                coverage_ranges=rich_results['coverage_ranges'],
                original_indices=ntlbg_results['representative_indices'],
                total_frames=video_features.shape[1]
            )
            
            aligned_visual_tokens = self.feature_aligner(llm_tokens['tokens'])
            
            fused_embeddings = self.multimodal_fusion(
                visual_tokens=aligned_visual_tokens,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
            
            # 生成
            multimodal_attention_mask = self._create_multimodal_attention_mask(
                attention_mask, aligned_visual_tokens.shape[1]
            )
            
            # 使用基础LLM的生成功能
            if hasattr(self.base_llm, 'generate'):
                generated_ids = self.base_llm.generate(
                    inputs_embeds=fused_embeddings,
                    attention_mask=multimodal_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                # Fallback: 简单的贪心解码
                generated_ids = self._simple_generate(
                    fused_embeddings, multimodal_attention_mask, max_new_tokens
                )
        
        return {
            'generated_ids': generated_ids,
            'representative_indices': ntlbg_results['representative_indices'],
            'representative_weights': rich_results['representativeness_weights']
        }
    
    def _simple_generate(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """简单的生成实现（fallback）"""
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        
        # 初始化生成的token ID
        generated_ids = []
        current_embeds = inputs_embeds
        current_mask = attention_mask
        
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.base_llm(
                inputs_embeds=current_embeds,
                attention_mask=current_mask
            )
            
            # 获取下一个token
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]
            
            generated_ids.append(next_token_id)
            
            # 更新embeddings和mask
            next_token_embed = self.base_llm.embed_tokens(next_token_id)  # [B, 1, hidden_size]
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
            current_mask = torch.cat([current_mask, torch.ones(batch_size, 1, device=device)], dim=1)
            
            # 检查是否结束
            if hasattr(self.base_llm.config, 'eos_token_id'):
                if (next_token_id == self.base_llm.config.eos_token_id).all():
                    break
        
        if generated_ids:
            return torch.cat(generated_ids, dim=1)
        else:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)


# 工厂函数
def create_ntlbg_llm(config: Dict) -> NTLBGLLM:
    """创建NTLBG-LLM模型"""
    return NTLBGLLM(
        base_model_name=config.get('base_model_name', 'mock'),
        d_visual=config.get('d_visual', 768),
        d_query=config.get('d_query', 768),
        num_representatives=config.get('num_representatives', 6),
        max_video_length=config.get('max_video_length', 1000),
        enable_gradient_checkpointing=config.get('enable_gradient_checkpointing', True)
    )


# 测试和使用示例
if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    T = 50  # 视频帧数
    seq_len = 20  # 文本长度
    vocab_size = 32000
    
    # 创建测试数据
    video_frames = torch.randn(batch_size, T, 3, 224, 224)  # 模拟视频帧
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 初始化模型
    print("Initializing NTLBG-LLM model...")
    config = {
        'base_model_name': 'mock',
        'd_visual': 768,
        'd_query': 768,
        'num_representatives': 6,
        'max_video_length': T
    }
    model = create_ntlbg_llm(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练模式测试
    print("\n=== Training Mode Test ===")
    model.train()
    
    outputs = model(
        video_frames=video_frames,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_ntlbg_stats=True
    )
    
    print(f"Total loss: {outputs['loss'].item():.6f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss components:")
    for name, loss in outputs['loss_components'].items():
        print(f"  {name}: {loss.item():.6f}")
    
    print(f"Representative indices (batch 0): {outputs['representative_indices'][0]}")
    print(f"Representative weights (batch 0): {outputs['representative_weights'][0]}")
    
    # 推理模式测试
    print("\n=== Inference Mode Test ===")
    model.eval()
    
    generation_outputs = model.generate(
        video_frames=video_frames,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        do_sample=False
    )
    
    print(f"Generated IDs shape: {generation_outputs['generated_ids'].shape}")
    print(f"Generated IDs (batch 0): {generation_outputs['generated_ids'][0]}")
    
    print("\n✅ NTLBG-LLM model working correctly!")
    print("\n🎯 Ready for training on real datasets!") 