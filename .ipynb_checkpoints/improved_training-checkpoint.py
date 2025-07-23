"""
改进的NTLBG-LLM训练脚本 - 避免过拟合
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
import random

# 添加路径
sys.path.append('/workspace/NTLBG-LLM')
sys.path.append('/workspace/NTLBG-LLM/LongVideoBench_official')

# 导入官方数据加载器
from longvideobench import LongVideoBenchDataset
from create_real_ntlbg_llm import RealNTLBGLLM

class ImprovedNTLBGTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 使用设备: {self.device}")
        
        # 创建模型
        print("🔨 创建NTLBG-LLM模型...")
        self.model = RealNTLBGLLM(config).to(self.device)
        
        # 创建真实数据集
        self.create_real_datasets()
        
        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-5),  # 更小的学习率
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.get('num_epochs', 5)
        )
        
        print(f"✅ 改进训练器初始化完成")
    
    def create_real_datasets(self):
        """创建真实的数据集"""
        data_path = "/workspace/NTLBG-LLM/data/longvideobench"
        
        try:
            # 加载完整验证集
            full_val_dataset = LongVideoBenchDataset(
                data_path, 
                "lvb_val.json", 
                max_num_frames=16
            )
            
            # 将验证集分为训练和验证
            total_val = len(full_val_dataset)
            train_size = int(0.8 * total_val)  # 80%用于训练
            val_size = total_val - train_size
            
            # 随机分割
            indices = list(range(total_val))
            random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            self.train_dataset = Subset(full_val_dataset, train_indices)
            self.val_dataset = Subset(full_val_dataset, val_indices)
            
            # 创建数据加载器
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.get('batch_size', 2),
                shuffle=True,
                num_workers=0,
                collate_fn=self.collate_fn
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.get('batch_size', 2),
                shuffle=False,
                num_workers=0,
                collate_fn=self.collate_fn
            )
            
            print(f"✅ 数据集创建完成:")
            print(f"   训练样本: {len(self.train_dataset)}")
            print(f"   验证样本: {len(self.val_dataset)}")
            
        except Exception as e:
            print(f"❌ 真实数据集创建失败: {e}")
            # 使用简单数据集作为备选
            self.create_simple_datasets()
    
    def create_simple_datasets(self):
        """创建简单数据集作为备选"""
        from fixed_dataset import FixedNTLBGDataset
        
        self.train_dataset = FixedNTLBGDataset(
            "/workspace/NTLBG-LLM/data",
            split="train"
        )
        self.val_dataset = FixedNTLBGDataset(
            "/workspace/NTLBG-LLM/data", 
            split="val"
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 2),
            shuffle=True,
            collate_fn=self.simple_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 2),
            shuffle=False,
            collate_fn=self.simple_collate_fn
        )
    
    def collate_fn(self, batch):
        """处理LongVideoBench数据"""
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
                
                combined_text = " ".join(text_parts)
                answer = sample.get('answer', 0)
                if isinstance(answer, (list, tuple)):
                    answer = answer[0] if len(answer) > 0 else 0
                
                processed_batch.append({
                    'video_frames': video_frames,
                    'text': combined_text,
                    'answer': int(answer)
                })
                
            except Exception as e:
                print(f"❌ 批处理样本失败: {e}")
                # 添加空样本
                processed_batch.append({
                    'video_frames': [],
                    'text': "empty sample",
                    'answer': 0
                })
        
        return processed_batch
    
    def simple_collate_fn(self, batch):
        """处理简单数据"""
        processed_batch = []
        for sample in batch:
            processed_batch.append({
                'video_frames': sample.get('video_frames', []),
                'text': sample.get('text', '') + " " + sample.get('question', ''),
                'answer': sample.get('answer', 0)
            })
        return processed_batch
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            batch_loss = 0
            valid_samples = 0
            
            for sample in batch:
                try:
                    # 前向传播
                    outputs = self.model(
                        video_frames=sample['video_frames'],
                        text_input=sample['text']
                    )
                    
                    # 计算损失
                    logits = outputs['logits']
                    target = torch.tensor(sample['answer'], device=self.device)
                    
                    # 确保维度正确
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    
                    # 多选择题损失
                    if logits.shape[-1] >= 4:
                        choice_logits = logits[:, :4]
                        loss = F.cross_entropy(choice_logits, target.unsqueeze(0))
                    else:
                        # MSE备选
                        loss = F.mse_loss(logits.float(), target.float().unsqueeze(0).unsqueeze(0))
                    
                    batch_loss += loss
                    valid_samples += 1
                    
                except Exception as e:
                    # 跳过有问题的样本
                    continue
            
            # 只有当有有效样本时才更新
            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                
                # 反向传播
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                for sample in batch:
                    try:
                        outputs = self.model(
                            video_frames=sample['video_frames'],
                            text_input=sample['text']
                        )
                        
                        logits = outputs['logits']
                        target = sample['answer']
                        
                        if logits.shape[-1] >= 4:
                            pred = torch.argmax(logits[:, :4], dim=-1).cpu().item()
                            correct += (pred == target)
                        
                        total += 1
                        
                    except Exception as e:
                        total += 1  # 计数但不加分
        
        return correct / max(total, 1)
    
    def train(self):
        """完整训练流程"""
        print("🚀 开始改进的NTLBG-LLM训练")
        print("=" * 60)
        
        num_epochs = self.config.get('num_epochs', 10)
        best_accuracy = 0
        patience = 3
        patience_counter = 0
        
        results = {
            'train_losses': [],
            'val_accuracies': [],
            'best_accuracy': 0,
            'training_time': datetime.now().isoformat(),
            'config': self.config
        }
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
            print(f"   学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 40)
            
            # 训练
            train_loss = self.train_epoch()
            print(f"   ✅ 训练损失: {train_loss:.4f}")
            
            # 评估
            val_accuracy = self.evaluate()
            print(f"   ✅ 验证准确率: {val_accuracy:.4f}")
            
            # 学习率调整
            self.scheduler.step()
            
            # 记录结果
            results['train_losses'].append(train_loss)
            results['val_accuracies'].append(val_accuracy)
            
            # 早停检查
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                results['best_accuracy'] = best_accuracy
                patience_counter = 0
                
                # 保存最佳模型
                os.makedirs("outputs/models", exist_ok=True)
                torch.save(self.model.state_dict(), "outputs/models/improved_ntlbg_llm.pth")
                print(f"   🎯 保存最佳模型 (准确率: {best_accuracy:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   🛑 早停：{patience}个epoch没有改进")
                    break
        
        # 保存结果
        with open("outputs/improved_training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 改进训练完成!")
        print(f"   🏆 最佳准确率: {best_accuracy:.4f}")
        
        return results

def main():
    config = {
        'batch_size': 2,
        'learning_rate': 1e-5,  # 更小的学习率
        'num_epochs': 10,
        'max_frames': 16,
        'num_representatives': 6,
        'weight_decay': 0.01
    }
    
    trainer = ImprovedNTLBGTrainer(config)
    results = trainer.train()
    
    return results

if __name__ == "__main__":
    main()
