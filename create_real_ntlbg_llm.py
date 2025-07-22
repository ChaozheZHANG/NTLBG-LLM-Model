"""
基于LLaVA创建真正的NTLBG-LLM
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
import os

class RealNTLBGLLM(nn.Module):
    """真正的NTLBG-LLM，基于LLaVA架构"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 视觉编码器 (CLIP)
        self.vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # 语言模型 (选择一个开源模型)
        model_choices = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large", 
            "facebook/opt-1.3b",
            "EleutherAI/gpt-neo-1.3B"
        ]
        
        # 尝试加载可用的模型
        self.language_model = None
        self.tokenizer = None
        
        for model_name in model_choices:
            try:
                print(f"🔄 尝试加载: {model_name}")
                self.language_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # 添加pad_token如果不存在
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                print(f"✅ 成功加载: {model_name}")
                break
            except Exception as e:
                print(f"❌ 加载失败 {model_name}: {e}")
                continue
        
        if self.language_model is None:
            raise RuntimeError("❌ 无法加载任何语言模型")
        
        # NTLBG核心组件
        vision_dim = self.vision_tower.config.hidden_size
        lang_dim = self.language_model.config.hidden_size
        
        # 代表点数量
        self.num_representatives = config.get('num_representatives', 6)
        
        # 视觉到语言的投影
        self.vision_proj = nn.Linear(vision_dim, lang_dim)
        
        # NTLBG代表点学习
        self.representative_tokens = nn.Parameter(
            torch.randn(self.num_representatives, lang_dim)
        )
        
        # 注意力机制用于选择代表点
        self.attention = nn.MultiheadAttention(lang_dim, num_heads=8, batch_first=True)
        
        # 融合层
        self.fusion_layer = nn.TransformerEncoderLayer(
            d_model=lang_dim, 
            nhead=8, 
            dim_feedforward=lang_dim*4,
            batch_first=True
        )
        
        # 输出层
        self.output_proj = nn.Linear(lang_dim, self.language_model.config.vocab_size)
        
        print(f"✅ NTLBG-LLM 初始化完成")
        print(f"   视觉编码器: {vision_dim}D")
        print(f"   语言模型: {lang_dim}D")
        print(f"   代表点数量: {self.num_representatives}")
    
    def encode_video_frames(self, frames):
        """编码视频帧"""
        if len(frames) == 0:
            return torch.zeros(1, 768).to(next(self.parameters()).device)
        
        # 处理图像
        try:
            inputs = self.vision_processor(frames, return_tensors="pt")
            inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
            
            # 获取视觉特征
            vision_outputs = self.vision_tower(**inputs)
            vision_features = vision_outputs.last_hidden_state.mean(dim=1)  # [batch, dim]
            
            return vision_features
        except Exception as e:
            print(f"❌ 视频编码失败: {e}")
            return torch.zeros(1, 768).to(next(self.parameters()).device)
    
    def ntlbg_representation_learning(self, features):
        """NTLBG代表点学习"""
        batch_size = features.shape[0]
        
        # 扩展代表点到batch
        repr_tokens = self.representative_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_repr, dim]
        
        # 使用注意力选择最重要的代表点
        attended_repr, attention_weights = self.attention(
            repr_tokens, features.unsqueeze(1), features.unsqueeze(1)
        )
        
        # 融合特征
        fused_features = self.fusion_layer(
            torch.cat([attended_repr, features.unsqueeze(1)], dim=1)
        )
        
        # 取平均作为最终表示
        final_repr = fused_features.mean(dim=1)  # [batch, dim]
        
        return final_repr, attention_weights
    
    def forward(self, video_frames, text_input, questions=None):
        """前向传播"""
        device = next(self.parameters()).device
        
        # 编码视频
        if video_frames:
            vision_features = self.encode_video_frames(video_frames)
            vision_features = self.vision_proj(vision_features)
        else:
            vision_features = torch.zeros(1, self.language_model.config.hidden_size).to(device)
        
        # NTLBG表示学习
        ntlbg_features, attention_weights = self.ntlbg_representation_learning(vision_features)
        
        # 编码文本
        if text_input:
            text_tokens = self.tokenizer(
                text_input, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            
            # 获取文本特征
            lang_outputs = self.language_model(**text_tokens)
            text_features = lang_outputs.last_hidden_state.mean(dim=1)
        else:
            text_features = torch.zeros(1, self.language_model.config.hidden_size).to(device)
        
        # 融合视觉和文本特征
        combined_features = ntlbg_features + text_features
        
        # 生成输出
        output_logits = self.output_proj(combined_features)
        
        return {
            "logits": output_logits,
            "attention_weights": attention_weights,
            "ntlbg_features": ntlbg_features
        }

def create_model():
    """创建NTLBG-LLM模型"""
    config = {
        "num_representatives": 6,
        "vision_encoder": "openai/clip-vit-large-patch14"
    }
    
    try:
        model = RealNTLBGLLM(config)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

if __name__ == "__main__":
    print("🚀 创建真正的NTLBG-LLM")
    model = create_model()
    
    if model:
        # 保存模型架构
        torch.save(model.state_dict(), "/workspace/NTLBG-LLM/models/ntlbg_llm_base.pth")
        print("✅ 模型架构已保存")
    
    print("🎯 下一步: 使用这个真正的模型进行训练")
