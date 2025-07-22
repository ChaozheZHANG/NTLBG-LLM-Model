"""
修复版NTLBG训练脚本
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
import sys
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.append('/workspace/NTLBG-LLM')
sys.path.append('/workspace/NTLBG-LLM/src')

# 导入修复版模型
from src.models.ntlbg_llm_fixed import create_fixed_ntlbg_llm

# 导入官方数据加载器
try:
    from longvideobench import LongVideoBenchDataset
    HAS_OFFICIAL_LOADER = True
    logger.info("✅ 成功导入官方LongVideoBench数据加载器")
except ImportError:
    HAS_OFFICIAL_LOADER = False
    logger.warning("⚠️ 未安装官方LongVideoBench包，使用简化数据加载器")

class FixedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ 使用设备: {self.device}")
        
        # 创建修复版模型
        logger.info("🔨 创建修复版NTLBG-LLM...")
        model_config = {
            'base_model_name': 'microsoft/DialoGPT-medium',
            'num_representatives': config.get('num_representatives', 6)
        }
        
        self.model = create_fixed_ntlbg_llm(model_config).to(self.device)
        
        # 创建数据集
        self.train_dataset, self.val_dataset = self._create_datasets()
        
        # 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.get('learning_rate', 5e-5),  # 稍高的学习率
            weight_decay=config.get('weight_decay', 0.01),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.get('num_epochs', 5),
            eta_min=1e-7
        )
        
        logger.info(f"✅ 修复版训练器初始化完成")
        logger.info(f"   训练样本: {len(self.train_dataset) if self.train_dataset else 0}")
        logger.info(f"   验证样本: {len(self.val_dataset) if self.val_dataset else 0}")
        logger.info(f"   可训练参数: {sum(p.numel() for p in trainable_params):,}")
    
    def _create_datasets(self):
        """创建数据集"""
        if HAS_OFFICIAL_LOADER:
            return self._create_official_datasets()
        else:
            return self._create_simple_datasets()
    
    def _create_official_datasets(self):
        """使用官方数据加载器"""
        try:
            data_path = "/workspace/NTLBG-LLM/data/longvideobench"
# 修复NTLBG核心模块




tree
cat > src/models/ntlbg_llm_fixed.py << 'EOF'
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
   def _create_official_datasets(self):
       """使用官方数据加载器"""
       try:
           data_path = "/workspace/NTLBG-LLM/data/longvideobench"
           
           # 加载验证集数据
           val_dataset = LongVideoBenchDataset(
               data_path, 
               "lvb_val.json", 
               max_num_frames=32
           )
           
           # 将验证集分为训练和验证
           total_size = len(val_dataset)
           train_size = int(0.8 * total_size)
           val_size = total_size - train_size
           
           # 随机分割
           indices = torch.randperm(total_size).tolist()
           train_indices = indices[:train_size]
           val_indices = indices[train_size:]
           
           train_dataset = Subset(val_dataset, train_indices)
           val_dataset_subset = Subset(val_dataset, val_indices)
           
           logger.info(f"✅ 使用官方LongVideoBench数据")
           logger.info(f"   原始数据: {total_size} 样本")
           logger.info(f"   训练: {len(train_dataset)} 样本")
           logger.info(f"   验证: {len(val_dataset_subset)} 样本")
           
           return train_dataset, val_dataset_subset
           
       except Exception as e:
           logger.error(f"❌ 官方数据加载失败: {e}")
           return self._create_simple_datasets()
   
   def _create_simple_datasets(self):
       """简化数据集（备用方案）"""
       from src.data.fixed_dataset import FixedNTLBGDataset
       
       train_dataset = FixedNTLBGDataset(
           "/workspace/NTLBG-LLM/data",
           split="train",
           max_frames=32
       )
       
       val_dataset = FixedNTLBGDataset(
           "/workspace/NTLBG-LLM/data",
           split="val", 
           max_frames=32
       )
       
       return train_dataset, val_dataset
   
   def collate_fn(self, batch):
       """处理批次数据"""
       if HAS_OFFICIAL_LOADER and hasattr(batch[0], 'get'):
           # 官方数据格式
           return self._collate_official(batch)
       else:
           # 简化数据格式
           return self._collate_simple(batch)
   
   def _collate_official(self, batch):
       """处理官方LongVideoBench数据"""
       processed_batch = []
       
       for sample in batch:
           try:
               inputs = sample.get("inputs", [])
               
               # 分离视频帧和文本
               video_frames = []
               text_parts = []
               
               for item in inputs:
                   if hasattr(item, 'size'):  # PIL Image
                       video_frames.append(item)
                   elif isinstance(item, str):
                       text_parts.append(item)
               
               # 构造文本
               combined_text = " ".join(text_parts)
               question = sample.get('question', '')
               if question:
                   combined_text += f" Question: {question}"
               
               # 处理答案
               answer = sample.get('answer', 0)
               if isinstance(answer, (list, tuple)):
                   answer = answer[0] if len(answer) > 0 else 0
               
               processed_batch.append({
                   'video_frames': video_frames,
                   'text': combined_text,
                   'answer': int(answer),
                   'question': question
               })
               
           except Exception as e:
               logger.warning(f"⚠️ 处理样本失败: {e}")
               # 添加空样本
               processed_batch.append({
                   'video_frames': [],
                   'text': "empty sample",
                   'answer': 0,
                   'question': ""
               })
       
       return processed_batch
   
   def _collate_simple(self, batch):
       """处理简化数据"""
       processed_batch = []
       for sample in batch:
           processed_batch.append({
               'video_frames': sample.get('video_frames', []),
               'text': f"{sample.get('text', '')} {sample.get('question', '')}",
               'answer': sample.get('answer', 0),
               'question': sample.get('question', '')
           })
       return processed_batch
   
   def train_epoch(self):
       """训练一个epoch"""
       self.model.train()
       total_loss = 0
       total_task_loss = 0
       total_ntlbg_loss = 0
       num_batches = 0
       
       # 创建数据加载器
       train_loader = DataLoader(
           self.train_dataset,
           batch_size=self.config.get('batch_size', 2),
           shuffle=True,
           num_workers=0,
           collate_fn=self.collate_fn
       )
       
       progress_bar = tqdm(train_loader, desc="训练中")
       
       for batch in progress_bar:
           self.optimizer.zero_grad()
           
           batch_loss = 0
           batch_task_loss = 0
           batch_ntlbg_loss = 0
           valid_samples = 0
           
           for sample in batch:
               try:
                   # 准备标签
                   answer = sample['answer']
                   labels = torch.tensor([answer], device=self.device, dtype=torch.long)
                   
                   # 前向传播
                   outputs = self.model(
                       video_frames=sample['video_frames'],
                       text_input=sample['text'],
                       labels=labels,
                       return_loss=True
                   )
                   
                   if 'loss' in outputs:
                       loss = outputs['loss']
                       task_loss = outputs.get('task_loss', loss)
                       ntlbg_loss = outputs.get('ntlbg_loss', torch.tensor(0.0))
                       
                       batch_loss += loss
                       batch_task_loss += task_loss
                       batch_ntlbg_loss += ntlbg_loss
                       valid_samples += 1
                   
               except Exception as e:
                   logger.warning(f"⚠️ 训练样本失败: {e}")
                   continue
           
           # 只有当有有效样本时才更新
           if valid_samples > 0:
               avg_loss = batch_loss / valid_samples
               avg_task_loss = batch_task_loss / valid_samples
               avg_ntlbg_loss = batch_ntlbg_loss / valid_samples
               
               # 反向传播
               avg_loss.backward()
               
               # 梯度裁剪
               torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
               
               # 优化步骤
               self.optimizer.step()
               
               total_loss += avg_loss.item()
               total_task_loss += avg_task_loss.item()
               total_ntlbg_loss += avg_ntlbg_loss.item()
               num_batches += 1
               
               progress_bar.set_postfix({
                   'loss': f'{avg_loss.item():.4f}',
                   'task': f'{avg_task_loss.item():.4f}',
                   'ntlbg': f'{avg_ntlbg_loss.item():.4f}'
               })
       
       return {
           'total_loss': total_loss / max(num_batches, 1),
           'task_loss': total_task_loss / max(num_batches, 1),
           'ntlbg_loss': total_ntlbg_loss / max(num_batches, 1)
       }
   
   def evaluate(self):
       """评估模型"""
       self.model.eval()
       correct_predictions = 0
       total_predictions = 0
       
       # 创建验证数据加载器
       val_loader = DataLoader(
           self.val_dataset,
           batch_size=self.config.get('batch_size', 2),
           shuffle=False,
           num_workers=0,
           collate_fn=self.collate_fn
       )
       
       with torch.no_grad():
           for batch in tqdm(val_loader, desc="评估中"):
               for sample in batch:
                   try:
                       # 前向传播
                       outputs = self.model(
                           video_frames=sample['video_frames'],
                           text_input=sample['text'],
                           return_loss=False
                       )
                       
                       # 预测答案
                       if 'classification_logits' in outputs:
                           pred = torch.argmax(outputs['classification_logits'], dim=-1).cpu().item()
                       else:
                           pred = torch.argmax(outputs['logits'], dim=-1).cpu().item()
                       
                       # 评估正确性
                       target = sample['answer']
                       if pred == target:
                           correct_predictions += 1
                       
                       total_predictions += 1
                       
                   except Exception as e:
                       logger.warning(f"⚠️ 评估样本失败: {e}")
                       total_predictions += 1
       
       accuracy = correct_predictions / max(total_predictions, 1)
       return accuracy
   
   def train(self):
       """完整训练流程"""
       logger.info("🚀 开始修复版NTLBG-LLM训练")
       logger.info("=" * 60)
       
       num_epochs = self.config.get('num_epochs', 5)
       best_accuracy = 0
       patience = 3
       patience_counter = 0
       
       results = {
           'train_losses': [],
           'task_losses': [],
           'ntlbg_losses': [],
           'val_accuracies': [],
           'best_accuracy': 0,
           'training_time': datetime.now().isoformat(),
           'config': self.config
       }
       
       for epoch in range(num_epochs):
           logger.info(f"\n📚 Epoch {epoch+1}/{num_epochs}")
           logger.info(f"   学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
           logger.info("-" * 40)
           
           # 训练
           train_metrics = self.train_epoch()
           logger.info(f"   ✅ 训练损失: {train_metrics['total_loss']:.4f}")
           logger.info(f"      任务损失: {train_metrics['task_loss']:.4f}")
           logger.info(f"      NTLBG损失: {train_metrics['ntlbg_loss']:.4f}")
           
           # 评估
           val_accuracy = self.evaluate()
           logger.info(f"   ✅ 验证准确率: {val_accuracy:.4f}")
           
           # 学习率调整
           self.scheduler.step()
           
           # 记录结果
           results['train_losses'].append(train_metrics['total_loss'])
           results['task_losses'].append(train_metrics['task_loss'])
           results['ntlbg_losses'].append(train_metrics['ntlbg_loss'])
           results['val_accuracies'].append(val_accuracy)
           
           # 早停和模型保存
           if val_accuracy > best_accuracy:
               best_accuracy = val_accuracy
               results['best_accuracy'] = best_accuracy
               patience_counter = 0
               
               # 保存最佳模型
               os.makedirs("outputs/models", exist_ok=True)
               torch.save(self.model.state_dict(), "outputs/models/best_fixed_ntlbg_llm.pth")
               logger.info(f"   🎯 保存最佳模型 (准确率: {best_accuracy:.4f})")
           else:
               patience_counter += 1
               if patience_counter >= patience:
                   logger.info(f"   🛑 早停：{patience}个epoch没有改进")
                   break
       
       # 保存训练结果
       os.makedirs("outputs", exist_ok=True)
       with open("outputs/fixed_training_results.json", "w") as f:
           json.dump(results, f, indent=2)
       
       logger.info(f"\n🎉 修复版训练完成!")
       logger.info(f"   🏆 最佳准确率: {best_accuracy:.4f}")
       logger.info(f"   📁 模型保存: outputs/models/best_fixed_ntlbg_llm.pth")
       
       return results


def main():
   """主函数"""
   logger.info("🎯 修复版NTLBG-LLM训练开始")
   logger.info("=" * 80)
   
   config = {
       'batch_size': 2,  # 适中的批次大小
       'learning_rate': 5e-5,  # 合适的学习率
       'num_epochs': 8,
       'num_representatives': 6,
       'weight_decay': 0.01
   }
   
   logger.info("⚙️ 训练配置:")
   for key, value in config.items():
       logger.info(f"   {key}: {value}")
   
   try:
       trainer = FixedTrainer(config)
       results = trainer.train()
       
       logger.info("\n🎊 最终结果:")
       logger.info(f"   🎯 最佳准确率: {results['best_accuracy']:.4f}")
       logger.info(f"   📈 训练损失轨迹: {[f'{loss:.3f}' for loss in results['train_losses'][-3:]]}")
       logger.info(f"   📊 验证准确率轨迹: {[f'{acc:.3f}' for acc in results['val_accuracies'][-3:]]}")
       
       # 检查是否有改进
       if results['best_accuracy'] > 0.3:
           logger.info(f"   ✅ 性能良好！超过随机猜测基线")
       elif results['best_accuracy'] > 0.1:
           logger.info(f"   📈 有所改进，但仍需优化")
       else:
           logger.info(f"   ⚠️ 性能较低，需要调试模型或数据")
       
       return results
       
   except Exception as e:
       logger.error(f"❌ 训练失败: {e}")
       import traceback
       traceback.print_exc()
       return None


if __name__ == "__main__":
   results = main()
   if results and results['best_accuracy'] > 0.2:
       print("\n🎉 训练成功！可以进行评估了")
   else:
       print("\n⚠️ 训练效果不佳，建议检查配置")
   def _create_official_datasets(self):
       """使用官方数据加载器"""
       try:
           data_path = "/workspace/NTLBG-LLM/data/longvideobench"
           
           # 加载验证集数据
           val_dataset = LongVideoBenchDataset(
               data_path, 
               "lvb_val.json", 
               max_num_frames=32
           )
           
           # 将验证集分为训练和验证
           total_size = len(val_dataset)
           train_size = int(0.8 * total_size)
           val_size = total_size - train_size
           
           # 随机分割
           indices = torch.randperm(total_size).tolist()
           train_indices = indices[:train_size]
           val_indices = indices[train_size:]
           
           train_dataset = Subset(val_dataset, train_indices)
           val_dataset_subset = Subset(val_dataset, val_indices)
           
           logger.info(f"✅ 使用官方LongVideoBench数据")
           logger.info(f"   原始数据: {total_size} 样本")
           logger.info(f"   训练: {len(train_dataset)} 样本")
           logger.info(f"   验证: {len(val_dataset_subset)} 样本")
           
           return train_dataset, val_dataset_subset
           
       except Exception as e:
           logger.error(f"❌ 官方数据加载失败: {e}")
           return self._create_simple_datasets()
   
   def _create_simple_datasets(self):
       """简化数据集（备用方案）"""
       from src.data.fixed_dataset import FixedNTLBGDataset
       
       train_dataset = FixedNTLBGDataset(
           "/workspace/NTLBG-LLM/data",
           split="train",
           max_frames=32
       )
       
       val_dataset = FixedNTLBGDataset(
           "/workspace/NTLBG-LLM/data",
           split="val", 
           max_frames=32
       )
       
       return train_dataset, val_dataset
   
   def collate_fn(self, batch):
       """处理批次数据"""
       if HAS_OFFICIAL_LOADER and hasattr(batch[0], 'get'):
           # 官方数据格式
           return self._collate_official(batch)
       else:
           # 简化数据格式
           return self._collate_simple(batch)
   
   def _collate_official(self, batch):
       """处理官方LongVideoBench数据"""
       processed_batch = []
       
       for sample in batch:
           try:
               inputs = sample.get("inputs", [])
               
               # 分离视频帧和文本
               video_frames = []
               text_parts = []
               
               for item in inputs:
                   if hasattr(item, 'size'):  # PIL Image
                       video_frames.append(item)
                   elif isinstance(item, str):
                       text_parts.append(item)
               
               # 构造文本
               combined_text = " ".join(text_parts)
               question = sample.get('question', '')
               if question:
                   combined_text += f" Question: {question}"
               
               # 处理答案
               answer = sample.get('answer', 0)
               if isinstance(answer, (list, tuple)):
                   answer = answer[0] if len(answer) > 0 else 0
               
               processed_batch.append({
                   'video_frames': video_frames,
                   'text': combined_text,
                   'answer': int(answer),
                   'question': question
               })
               
           except Exception as e:
               logger.warning(f"⚠️ 处理样本失败: {e}")
               # 添加空样本
               processed_batch.append({
                   'video_frames': [],
                   'text': "empty sample",
                   'answer': 0,
                   'question': ""
               })
       
       return processed_batch
   
   def _collate_simple(self, batch):
       """处理简化数据"""
       processed_batch = []
       for sample in batch:
           processed_batch.append({
               'video_frames': sample.get('video_frames', []),
               'text': f"{sample.get('text', '')} {sample.get('question', '')}",
               'answer': sample.get('answer', 0),
               'question': sample.get('question', '')
           })
       return processed_batch
   
   def train_epoch(self):
       """训练一个epoch"""
       self.model.train()
       total_loss = 0
       total_task_loss = 0
       total_ntlbg_loss = 0
       num_batches = 0
       
       # 创建数据加载器
       train_loader = DataLoader(
           self.train_dataset,
           batch_size=self.config.get('batch_size', 2),
           shuffle=True,
           num_workers=0,
           collate_fn=self.collate_fn
       )
       
       progress_bar = tqdm(train_loader, desc="训练中")
       
       for batch in progress_bar:
           self.optimizer.zero_grad()
           
           batch_loss = 0
           batch_task_loss = 0
           batch_ntlbg_loss = 0
           valid_samples = 0
           
           for sample in batch:
               try:
                   # 准备标签
                   answer = sample['answer']
                   labels = torch.tensor([answer], device=self.device, dtype=torch.long)
                   
                   # 前向传播
                   outputs = self.model(
                       video_frames=sample['video_frames'],
                       text_input=sample['text'],
                       labels=labels,
                       return_loss=True
                   )
                   
                   if 'loss' in outputs:
                       loss = outputs['loss']
                       task_loss = outputs.get('task_loss', loss)
                       ntlbg_loss = outputs.get('ntlbg_loss', torch.tensor(0.0))
                       
                       batch_loss += loss
                       batch_task_loss += task_loss
                       batch_ntlbg_loss += ntlbg_loss
                       valid_samples += 1
                   
               except Exception as e:
                   logger.warning(f"⚠️ 训练样本失败: {e}")
                   continue
           
           # 只有当有有效样本时才更新
           if valid_samples > 0:
               avg_loss = batch_loss / valid_samples
               avg_task_loss = batch_task_loss / valid_samples
               avg_ntlbg_loss = batch_ntlbg_loss / valid_samples
               
               # 反向传播
               avg_loss.backward()
               
               # 梯度裁剪
               torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
               
               # 优化步骤
               self.optimizer.step()
               
               total_loss += avg_loss.item()
               total_task_loss += avg_task_loss.item()
               total_ntlbg_loss += avg_ntlbg_loss.item()
               num_batches += 1
               
               progress_bar.set_postfix({
                   'loss': f'{avg_loss.item():.4f}',
                   'task': f'{avg_task_loss.item():.4f}',
                   'ntlbg': f'{avg_ntlbg_loss.item():.4f}'
               })
       
       return {
           'total_loss': total_loss / max(num_batches, 1),
           'task_loss': total_task_loss / max(num_batches, 1),
           'ntlbg_loss': total_ntlbg_loss / max(num_batches, 1)
       }
   
   def evaluate(self):
       """评估模型"""
       self.model.eval()
       correct_predictions = 0
       total_predictions = 0
       
       # 创建验证数据加载器
       val_loader = DataLoader(
           self.val_dataset,
           batch_size=self.config.get('batch_size', 2),
           shuffle=False,
           num_workers=0,
           collate_fn=self.collate_fn
       )
       
       with torch.no_grad():
           for batch in tqdm(val_loader, desc="评估中"):
               for sample in batch:
                   try:
                       # 前向传播
                       outputs = self.model(
                           video_frames=sample['video_frames'],
                           text_input=sample['text'],
                           return_loss=False
                       )
                       
                       # 预测答案
                       if 'classification_logits' in outputs:
                           pred = torch.argmax(outputs['classification_logits'], dim=-1).cpu().item()
                       else:
                           pred = torch.argmax(outputs['logits'], dim=-1).cpu().item()
                       
                       # 评估正确性
                       target = sample['answer']
                       if pred == target:
                           correct_predictions += 1
                       
                       total_predictions += 1
                       
                   except Exception as e:
                       logger.warning(f"⚠️ 评估样本失败: {e}")
                       total_predictions += 1
       
       accuracy = correct_predictions / max(total_predictions, 1)
       return accuracy
   
   def train(self):
       """完整训练流程"""
       logger.info("🚀 开始修复版NTLBG-LLM训练")
       logger.info("=" * 60)
       
       num_epochs = self.config.get('num_epochs', 5)
       best_accuracy = 0
       patience = 3
       patience_counter = 0
       
       results = {
           'train_losses': [],
           'task_losses': [],
           'ntlbg_losses': [],
           'val_accuracies': [],
           'best_accuracy': 0,
           'training_time': datetime.now().isoformat(),
           'config': self.config
       }
       
       for epoch in range(num_epochs):
           logger.info(f"\n📚 Epoch {epoch+1}/{num_epochs}")
           logger.info(f"   学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
           logger.info("-" * 40)
           
           # 训练
           train_metrics = self.train_epoch()
           logger.info(f"   ✅ 训练损失: {train_metrics['total_loss']:.4f}")
           logger.info(f"      任务损失: {train_metrics['task_loss']:.4f}")
           logger.info(f"      NTLBG损失: {train_metrics['ntlbg_loss']:.4f}")
           
           # 评估
           val_accuracy = self.evaluate()
           logger.info(f"   ✅ 验证准确率: {val_accuracy:.4f}")
           
           # 学习率调整
           self.scheduler.step()
           
           # 记录结果
           results['train_losses'].append(train_metrics['total_loss'])
           results['task_losses'].append(train_metrics['task_loss'])
           results['ntlbg_losses'].append(train_metrics['ntlbg_loss'])
           results['val_accuracies'].append(val_accuracy)
           
           # 早停和模型保存
           if val_accuracy > best_accuracy:
               best_accuracy = val_accuracy
               results['best_accuracy'] = best_accuracy
               patience_counter = 0
               
               # 保存最佳模型
               os.makedirs("outputs/models", exist_ok=True)
               torch.save(self.model.state_dict(), "outputs/models/best_fixed_ntlbg_llm.pth")
               logger.info(f"   🎯 保存最佳模型 (准确率: {best_accuracy:.4f})")
           else:
               patience_counter += 1
               if patience_counter >= patience:
                   logger.info(f"   🛑 早停：{patience}个epoch没有改进")
                   break
       
       # 保存训练结果
       os.makedirs("outputs", exist_ok=True)
       with open("outputs/fixed_training_results.json", "w") as f:
           json.dump(results, f, indent=2)
       
       logger.info(f"\n🎉 修复版训练完成!")
       logger.info(f"   🏆 最佳准确率: {best_accuracy:.4f}")
       logger.info(f"   📁 模型保存: outputs/models/best_fixed_ntlbg_llm.pth")
       
       return results


def main():
   """主函数"""
   logger.info("🎯 修复版NTLBG-LLM训练开始")
   logger.info("=" * 80)
   
   config = {
       'batch_size': 2,  # 适中的批次大小
       'learning_rate': 5e-5,  # 合适的学习率
       'num_epochs': 8,
       'num_representatives': 6,
       'weight_decay': 0.01
   }
   
   logger.info("⚙️ 训练配置:")
   for key, value in config.items():
       logger.info(f"   {key}: {value}")
   
   try:
       trainer = FixedTrainer(config)
       results = trainer.train()
       
       logger.info("\n🎊 最终结果:")
       logger.info(f"   🎯 最佳准确率: {results['best_accuracy']:.4f}")
       logger.info(f"   📈 训练损失轨迹: {[f'{loss:.3f}' for loss in results['train_losses'][-3:]]}")
       logger.info(f"   📊 验证准确率轨迹: {[f'{acc:.3f}' for acc in results['val_accuracies'][-3:]]}")
       
       # 检查是否有改进
       if results['best_accuracy'] > 0.3:
           logger.info(f"   ✅ 性能良好！超过随机猜测基线")
       elif results['best_accuracy'] > 0.1:
           logger.info(f"   📈 有所改进，但仍需优化")
       else:
           logger.info(f"   ⚠️ 性能较低，需要调试模型或数据")
       
       return results
       
   except Exception as e:
       logger.error(f"❌ 训练失败: {e}")
       import traceback
       traceback.print_exc()
       return None


if __name__ == "__main__":
   results = main()
   if results and results['best_accuracy'] > 0.2:
       print("\n🎉 训练成功！可以进行评估了")
   else:
       print("\n⚠️ 训练效果不佳，建议检查配置")
