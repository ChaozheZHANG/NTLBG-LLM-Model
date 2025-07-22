"""
真正的NTLBG-LLM实现，基于大模型架构
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, 
    CLIPVisionModel, CLIPImageProcessor,
    AutoModelForCausalLM
)
from .ntlbg_core import NTLBGAttention
import logging

logger = logging.getLogger(__name__)

class RealNTLBGLLM(nn.Module):
    """真正的NTLBG-LLM：集成统计理论的长视频理解模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 基础语言模型选择
        self.base_model_name = config.get('base_model_name', 'microsoft/DialoGPT-medium')
        
        # 视觉编码器
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            'openai/clip-vit-large-patch14'
        )
        self.vision_processor = CLIPImageProcessor.from_pretrained(
            'openai/clip-vit-large-patch14'
        )
        
        # 语言模型
        try:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.warning(f"加载{self.base_model_name}失败，使用备选模型: {e}")
            # 使用更简单的模型作为备选
            self.language_model = AutoModel.from_pretrained('microsoft/DialoGPT-medium')
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 获取模型维度
        vision_dim = self.vision_encoder.config.hidden_size  # 1024
        try:
            lang_dim = self.language_model.config.hidden_size
        except:
            lang_dim = self.language_model.config.n_embd if hasattr(self.language_model.config, 'n_embd') else 768
        
        # **NTLBG核心模块** - 这是关键！
        self.ntlbg_attention = NTLBGAttention(
            d_model=vision_dim,
            d_query=lang_dim,
            num_representatives=config.get('num_representatives', 6)
        )
        
        # 模态对齐层
        self.vision_projection = nn.Linear(vision_dim, lang_dim)
        self.temporal_encoding = nn.Parameter(torch.randn(1000, lang_dim))  # 支持1000帧
        
        # 多模态融合
        self.multimodal_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=lang_dim,
                nhead=8,
                dim_feedforward=lang_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 任务头
        try:
            vocab_size = self.language_model.config.vocab_size
        except:
            vocab_size = len(self.tokenizer)
            
        self.task_head = nn.Sequential(
            nn.Linear(lang_dim, lang_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lang_dim, vocab_size)
        )
        
        # 冻结部分参数以提高训练效率
        self._freeze_base_models()
        
        logger.info(f"✅ 真正的NTLBG-LLM初始化完成")
        logger.info(f"   视觉编码器: {vision_dim}D")
        logger.info(f"   语言模型: {lang_dim}D")
        logger.info(f"   NTLBG代表点: {config.get('num_representatives', 6)}个")
    
    def _freeze_base_models(self):
        """冻结基础模型的部分参数"""
        # 冻结视觉编码器（除了最后一层）
        for name, param in self.vision_encoder.named_parameters():
            if 'layer.23' not in name:  # 只训练最后一层
                param.requires_grad = False
        
        # 冻结语言模型的大部分参数
        for name, param in self.language_model.named_parameters():
            # 只训练最后几层
            if not any(layer in name for layer in ['layer.23', 'layer.24', 'layer.25', 'lm_head', 'h.23', 'h.24']):
                param.requires_grad = False
        
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(f"🧊 冻结参数比例: {frozen_params/total_params:.1%}")
    
    def encode_video_frames(self, video_frames):
        """编码视频帧为特征序列"""
        if not video_frames or len(video_frames) == 0:
            # 返回空特征
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.vision_encoder.config.hidden_size, device=device)
        
        try:
            # 预处理图像
            if hasattr(video_frames[0], 'size'):  # PIL Images
                inputs = self.vision_processor(video_frames, return_tensors="pt")
            else:  # 已经是tensor
                inputs = {'pixel_values': torch.stack(video_frames)}
            
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 编码视频帧
            with torch.no_grad():
                vision_outputs = self.vision_encoder(**inputs)
            
            # 获取特征并添加时序编码
            frame_features = vision_outputs.last_hidden_state  # [T, seq_len, hidden_size]
            
            # 简化：取CLS token或平均池化
            if frame_features.dim() == 3:
                frame_features = frame_features.mean(dim=1)  # [T, hidden_size]
            
            # 添加时序位置编码
            T = frame_features.shape[0]
            temporal_pos = self.temporal_encoding[:T].to(device)  # [T, hidden_size]
            frame_features = frame_features + temporal_pos
            
            return frame_features.unsqueeze(0)  # [1, T, hidden_size]
            
        except Exception as e:
            logger.warning(f"视频编码失败: {e}")
            device = next(self.parameters()).device
            return torch.zeros(1, 8, self.vision_encoder.config.hidden_size, device=device)
    
    def encode_text(self, text_input):
        """编码文本输入"""
        if not text_input:
            device = next(self.parameters()).device
            try:
                hidden_size = self.language_model.config.hidden_size
            except:
                hidden_size = self.language_model.config.n_embd if hasattr(self.language_model.config, 'n_embd') else 768
            return torch.zeros(1, hidden_size, device=device)
        
        try:
            # 标记化文本
            tokens = self.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            device = next(self.parameters()).device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # 编码文本
            with torch.no_grad():
                if hasattr(self.language_model, 'forward'):
                    outputs = self.language_model(**tokens, output_hidden_states=True)
                    if hasattr(outputs, 'hidden_states'):
                        text_features = outputs.hidden_states[-1].mean(dim=1)  # [1, hidden_size]
                    else:
                        text_features = outputs.last_hidden_state.mean(dim=1)
                else:
                    # 简化处理
                    embeddings = self.language_model.get_input_embeddings()
                    text_features = embeddings(tokens['input_ids']).mean(dim=1)
            
            return text_features  # [1, hidden_size]
            
        except Exception as e:
            logger.warning(f"文本编码失败: {e}")
            device = next(self.parameters()).device
            try:
                hidden_size = self.language_model.config.hidden_size
            except:
                hidden_size = 768
            return torch.zeros(1, hidden_size, device=device)
    
    def forward(self, video_frames=None, text_input=None, labels=None):
        """NTLBG-LLM前向传播"""
        device = next(self.parameters()).device
        
        # 1. 编码视频和文本
        video_features = self.encode_video_frames(video_frames)  # [1, T, vision_dim]
        text_features = self.encode_text(text_input)  # [1, lang_dim]
        
        # 2. 投影视觉特征到语言空间
        video_features_proj = self.vision_projection(video_features)  # [1, T, lang_dim]
        
        # 3. **NTLBG核心处理** - 这是我们的创新！
        ntlbg_results = self.ntlbg_attention(
            video_features=video_features_proj,
            query_embedding=text_features
        )
        
        # 4. 获取NTLBG选择的代表点特征
        representative_features = ntlbg_results['representative_features']  # [1, K, lang_dim]
        attended_features = ntlbg_results['attended_features']  # [1, 1, lang_dim]
        
        # 5. 多模态融合
        # 合并文本、代表点和注意力特征
        fused_input = torch.cat([
            attended_features,  # 查询-视频注意力特征
            representative_features,  # NTLBG代表点特征
            text_features.unsqueeze(1)  # 原始文本特征
        ], dim=1)  # [1, 1+K+1, lang_dim]
        
        fused_features = self.multimodal_fusion(fused_input)  # [1, 1+K+1, lang_dim]
        
        # 6. 任务预测
        # 使用融合特征的均值进行预测
        final_features = fused_features.mean(dim=1)  # [1, lang_dim]
        logits = self.task_head(final_features)  # [1, vocab_size]
        
        # 7. 构建输出
        outputs = {
            'logits': logits,
            'representative_features': representative_features,
            'representative_indices': ntlbg_results['representative_indices'],
            'mahalanobis_distances': ntlbg_results['mahalanobis_distances'],
            'attention_weights': ntlbg_results.get('attention_weights'),
            'ntlbg_constraint_loss': None
        }
        
        # 8. 计算损失
        if labels is not None:
            # 主任务损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            if labels.dim() == 1:
                # 单标签分类
                task_loss = loss_fct(logits, labels)
            else:
                # 序列生成
                task_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # NTLBG约束损失
            ntlbg_constraint_loss = self.ntlbg_attention.ntlbg_core.compute_ntlbg_constraint_loss(
                representative_features,
                ntlbg_results['mu_q'],
                ntlbg_results['sigma_q']
            )
            
            # 总损失
            total_loss = task_loss + 0.5 * ntlbg_constraint_loss
            
            outputs.update({
                'loss': total_loss,
                'task_loss': task_loss,
                'ntlbg_constraint_loss': ntlbg_constraint_loss
            })
        
        return outputs


def create_real_ntlbg_llm(config):
    """创建真正的NTLBG-LLM模型"""
    try:
        model = RealNTLBGLLM(config)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 NTLBG-LLM统计:")
        logger.info(f"   总参数: {total_params:,}")
        logger.info(f"   可训练参数: {trainable_params:,}")
        logger.info(f"   训练效率: {trainable_params/total_params:.1%}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ NTLBG-LLM创建失败: {e}")
        raise
