import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import yaml
import json
from tqdm import tqdm

class SimpleVideoQADataset(Dataset):
    def __init__(self, data_dirs, max_samples=1000):
        self.samples = []
        
        # 扫描数据集
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                # 查找JSON文件
                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        if file.endswith('.json') and len(self.samples) < max_samples:
                            try:
                                with open(os.path.join(root, file), 'r') as f:
                                    data = json.load(f)
                                    if isinstance(data, list):
                                        self.samples.extend(data[:10])  # 每个文件取10个样本
                                    elif isinstance(data, dict):
                                        self.samples.append(data)
                            except:
                                continue
        
        print(f"✅ 加载了 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 模拟数据
        return {
            'video_features': torch.randn(64, 768),  # 64帧，768维特征
            'input_ids': torch.randint(1, 1000, (128,)),  # 文本token
            'attention_mask': torch.ones(128),
            'labels': torch.randint(1, 1000, (128,))
        }

class SimpleNTLBGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_encoder = nn.Linear(768, 512)
        self.frame_selector = nn.Linear(512, 1)  # NTLBG选择器
        self.text_encoder = nn.Embedding(1000, 512)
        self.fusion = nn.MultiheadAttention(512, 8, batch_first=True)
        self.classifier = nn.Linear(512, 1000)
        
    def forward(self, video_features, input_ids, attention_mask, labels=None):
        # 视频编码
        video_encoded = torch.relu(self.video_encoder(video_features))  # [B, T, 512]
        
        # NTLBG代表点选择（简化版）
        frame_scores = self.frame_selector(video_encoded).squeeze(-1)  # [B, T]
        _, top_indices = torch.topk(frame_scores, k=6, dim=1)  # 选择6个代表点
        
        # 收集代表点
        batch_size = video_features.size(0)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, 6)
        representatives = video_encoded[batch_indices, top_indices]  # [B, 6, 512]
        
        # 文本编码
        text_encoded = self.text_encoder(input_ids)  # [B, L, 512]
        
        # 多模态融合
        fused, _ = self.fusion(text_encoded, representatives, representatives)
        
        # 分类
        logits = self.classifier(fused)  # [B, L, 1000]
        
        outputs = {'logits': logits, 'representatives': representatives}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, 1000), labels.view(-1))
            outputs['loss'] = loss
            
        return outputs

def train():
    print("🚀 开始NTLBG-LLM训练")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 数据集
    data_dirs = ['data/longvideobench', 'data/video_mme', 'data/mlvu']
    dataset = SimpleVideoQADataset(data_dirs)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 模型
    model = SimpleNTLBGModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    print(f"🤖 模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 训练循环
    model.train()
    for epoch in range(3):  # 3个epoch的演示
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")
        
        for batch in progress_bar:
            # 移动到设备
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 记录
            epoch_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'reps': outputs['representatives'].shape[1]
            })
        
        avg_loss = epoch_loss / num_batches
        print(f"✅ Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")
        
        # 保存检查点
        os.makedirs('outputs/checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, f'outputs/checkpoints/checkpoint_epoch_{epoch+1}.pt')
    
    print("🎉 训练完成！")
    print("📊 检查点已保存到 outputs/checkpoints/")

if __name__ == "__main__":
    train()
