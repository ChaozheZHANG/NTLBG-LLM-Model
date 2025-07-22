"""
修复版NTLBG-LLM主模型
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, 
    CLIPVisionModel, CLIPImageProcessor,
    AutoModelForCausalLM
)
from .ntlbg_core_fixed import FixedNTLBGAttention
import logging

logger = logging.getLogger(__name__)

class FixedNTLBGLLM(nn.Module):
    """修复版NTLBG-LLM"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 基础模型配置
        self.base_model_name = config.get('base_model_name', 'microsoft/DialoGPT-medium')
        
        # 视觉编码器
        self.vision_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
        self.vision_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        
        # 语言模型
        self.language_model = AutoModel.from_pretrained(self.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 获取维度
        vision_dim = self.vision_encoder.config.hidden_size  # 1024
        lang_dim = self.language_model.config.hidden_size    # 1024
        
        # **修复版NTLBG核心**
        self.ntlbg_attention = FixedNTLBGAttention(
            d_model=vision_dim,
            d_query=lang_dim,
            num_representatives=config.get('num_representatives', 6)
        )
        
        # 模态对齐
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, lang_dim),
            nn.LayerNorm(lang_dim),
            nn.GELU()
        )
        
        # 多模态融合
        self.multimodal_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=lang_dim,
                nhead=8,
                dim_feedforward=lang_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 输出层
        vocab_size = len(self.tokenizer)
        self.output_projection = nn.Sequential(
            nn.Linear(lang_dim, lang_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(lang_dim, vocab_size)
        )
        
        # 分类头（用于多选题）
        self.classification_head = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(lang_dim // 2, 4)  # 4选择题
        )
        
        # 冻结大部分参数
        self._freeze_base_models()
        
        logger.info(f"✅ 修复版NTLBG-LLM初始化完成")
        logger.info(f"   视觉编码器: {vision_dim}D -> {lang_dim}D")
        logger.info(f"   NTLBG代表点: {config.get('num_representatives', 6)}个")
    
    def _freeze_base_models(self):
        """智能冻结策略"""
        # 冻结视觉编码器的前面几层
        for name, param in self.vision_encoder.named_parameters():
            if not any(layer in name for layer in ['layer.22', 'layer.23', 'pooler']):
                param.requires_grad = False
        
        # 冻结语言模型的embedding和前面几层
        for name, param in self.language_model.named_parameters():
            if not any(layer in name for layer in ['layer.22', 'layer.23', 'pooler']):
                param.requires_grad = False
        
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"🧊 冻结参数: {frozen/total:.1%}")
    
    def encode_video_frames(self, video_frames):
        """改进的视频编码"""
        if not video_frames or len(video_frames) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.vision_encoder.config.hidden_size, device=device)
        
        try:
            # 限制帧数以避免内存问题
            max_frames = 64
            if len(video_frames) > max_frames:
                # 均匀采样
                indices = torch.linspace(0, len(video_frames)-1, max_frames, dtype=torch.long)
                video_frames = [video_frames[i] for i in indices]
            
            # 预处理
            if hasattr(video_frames[0], 'size'):  # PIL Images
                inputs = self.vision_processor(video_frames, return_tensors="pt")
            else:
                inputs = {'pixel_values': torch.stack(video_frames)}
            
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 编码（允许梯度传播）
            vision_outputs = self.vision_encoder(**inputs)
            
            # 获取特征
            if hasattr(vision_outputs, 'pooler_output'):
                frame_features = vision_outputs.pooler_output  # [T, hidden_size]
            else:
                frame_features = vision_outputs.last_hidden_state.mean(dim=1)
            
            return frame_features.unsqueeze(0)  # [1, T, hidden_size]
            
        except Exception as e:
            logger.warning(f"视频编码失败: {e}")
            device = next(self.parameters()).device
            return torch.randn(1, 8, self.vision_encoder.config.hidden_size, device=device)
    
    def encode_text(self, text_input):
        """改进的文本编码"""
        if not text_input:
            device = next(self.parameters()).device
            return torch.zeros(1, self.language_model.config.hidden_size, device=device)
        
        try:
            tokens = self.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256  # 减少长度
            )
            
            device = next(self.parameters()).device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # 编码（允许梯度传播）
            outputs = self.language_model(**tokens)
            
            # 获取特征
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                text_features = outputs.pooler_output
            else:
                text_features = outputs.last_hidden_state.mean(dim=1)
            
            return text_features  # [1, hidden_size]
            
        except Exception as e:
            logger.warning(f"文本编码失败: {e}")
            device = next(self.parameters()).device
            return torch.randn(1, self.language_model.config.hidden_size, device=device)
    
    def forward(self, video_frames=None, text_input=None, labels=None, return_loss=True):
        """修复版前向传播"""
        device = next(self.parameters()).device
        
        # 1. 编码输入
        video_features = self.encode_video_frames(video_frames)  # [1, T, vision_dim]
        text_features = self.encode_text(text_input)  # [1, lang_dim]
        
        # 2. 视觉特征投影
        video_features_proj = self.vision_projection(video_features)  # [1, T, lang_dim]
        
        # 3. NTLBG处理
        ntlbg_results = self.ntlbg_attention(
            video_features=video_features_proj,
            query_embedding=text_features
        )
        
        # 4. 多模态融合
        representative_features = ntlbg_results['representative_features']  # [1, K, lang_dim]
        attended_features = ntlbg_results['attended_features']  # [1, 1, lang_dim]
        
        # 合并所有特征
        all_features = torch.cat([
            text_features.unsqueeze(1),  # 原始文本
            attended_features,           # 注意力特征
            representative_features      # 代表点特征
        ], dim=1)  # [1, 1+1+K, lang_dim]
        
        # 多模态融合
        fused_features = self.multimodal_fusion(all_features)  # [1, 1+1+K, lang_dim]
        
        # 5. 输出预测
        pooled_features = fused_features.mean(dim=1)  # [1, lang_dim]
        
        # 生成式输出
        generation_logits = self.output_projection(pooled_features)  # [1, vocab_size]
        
        # 分类输出（用于多选题）
        classification_logits = self.classification_head(pooled_features)  # [1, 4]
        
        outputs = {
            'logits': generation_logits,
            'classification_logits': classification_logits,
            'representative_features': representative_features,
            'representative_indices': ntlbg_results['representative_indices'],
            'mahalanobis_distances': ntlbg_results['mahalanobis_distances'],
            'attention_weights': ntlbg_results.get('cross_attention_weights')
        }
        
        # 6. 计算损失
        if return_loss and labels is not None:
            # 处理不同类型的标签
            if isinstance(labels, torch.Tensor) and labels.numel() == 1:
                # 单个标签（分类任务）
                if labels.item() < 4:  # 4选择题
                    loss = nn.CrossEntropyLoss()(classification_logits, labels.view(-1))
                else:  # 生成任务
                    loss = nn.CrossEntropyLoss()(generation_logits, labels.view(-1))
            else:
                # 序列标签或其他格式
                loss = nn.CrossEntropyLoss()(generation_logits, labels.view(-1))
            
            # 添加NTLBG约束损失
            ntlbg_loss = self.ntlbg_attention.ntlbg_core.compute_ntlbg_constraint_loss(
                representative_features,
                ntlbg_results['mu_q'],
                ntlbg_results['sigma_q']
            )
            
            # 总损失
            total_loss = loss + 0.1 * ntlbg_loss  # 调整权重
            
            outputs.update({
                'loss': total_loss,
                'task_loss': loss,
                'ntlbg_loss': ntlbg_loss
            })
        
        return outputs


def create_fixed_ntlbg_llm(config):
    """创建修复版NTLBG-LLM"""
    model = FixedNTLBGLLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 修复版NTLBG-LLM:")
    logger.info(f"   总参数: {total_params:,}")
    logger.info(f"   可训练参数: {trainable_params:,}")
    logger.info(f"   训练效率: {trainable_params/total_params:.1%}")
    
    return model
