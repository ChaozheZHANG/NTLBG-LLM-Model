#!/usr/bin/env python3
"""
NTLBG-LLM训练脚本
用法: python scripts/train_ntlbg.py --config configs/ntlbg_base_config.json
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from collections import defaultdict
import random
import warnings

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入我们的模块
from src.models.ntlbg_llm import create_ntlbg_llm
from src.data.datasets import create_dataloaders


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def get_device_info():
    """获取设备信息"""
    print("=== 环境信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


class LossWeightScheduler:
    """损失权重动态调度器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.initial_weights = config['loss_weights']
        self.schedule_type = config.get('weight_schedule', 'static')
        self.warmup_steps = config.get('weight_warmup_steps', 1000)
    
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
            weights['temporal'] = self.initial_weights['temporal'] * progress
            weights['info'] = self.initial_weights['info'] * progress
            
            return weights
        
        else:
            return self.initial_weights


class NTLBGTrainer:
    """NTLBG-LLM训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 config: Dict,
                 device: torch.device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # 将模型移到设备
        self.model.to(device)
        
        # 设置优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()
        
        # 损失权重调度器
        self.loss_weight_scheduler = LossWeightScheduler(config.get('loss_config', {}))
        
        # 设置日志
        self.logger = setup_logging(config['output_dir'])
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # 训练统计
        self.train_stats = {
            'total_steps': 0,
            'epoch': 0,
            'losses': [],
            'learning_rates': []
        }
    
    def _setup_optimizer(self):
        """设置优化器"""
        training_config = self.config['training_config']
        
        # 分组参数：不同的学习率
        base_model_params = []
        ntlbg_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'base_llm' in name:
                base_model_params.append(param)
            elif any(module in name for module in ['ntlbg_attention', 'rich_constructor', 'temporal_aligner']):
                ntlbg_params.append(param)
            else:
                other_params.append(param)
        
        # 使用不同的学习率
        param_groups = [
            {'params': base_model_params, 'lr': training_config['base_lr'] * 0.1},
            {'params': ntlbg_params, 'lr': training_config['base_lr']},
            {'params': other_params, 'lr': training_config['base_lr'] * 0.5}
        ]
        
        if training_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=training_config['weight_decay'],
                eps=1e-8
            )
        elif training_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        training_config = self.config['training_config']
        
        total_steps = len(self.train_dataloader) * training_config['num_epochs']
        warmup_steps = int(total_steps * training_config['warmup_ratio'])
        
        if training_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps,
                eta_min=training_config['base_lr'] * 0.01
            )
        elif training_config['scheduler'] == 'linear':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return max(0.01, (total_steps - step) / (total_steps - warmup_steps))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到GPU
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(
                video_frames=batch['video_features'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # 计算损失
            current_weights = self.loss_weight_scheduler.get_weights(
                self.train_stats['total_steps'], epoch
            )
            
            total_loss = self._compute_loss(outputs, current_weights)
            
            # 反向传播
            total_loss['total'].backward()
            
            # 梯度裁剪
            if self.config['training_config'].get('max_grad_norm', 0) > 0:
                clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training_config']['max_grad_norm']
                )
            
            # 优化器步进
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录损失
            for key, value in total_loss.items():
                epoch_losses[key].append(value.item())
            
            # 更新统计信息
            self.train_stats['total_steps'] += 1
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_stats['learning_rates'].append(current_lr)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{total_loss['total'].item():.4f}",
                'lr': f"{current_lr:.2e}",
                'ntlbg': f"{total_loss.get('ntlbg', 0):.4f}" if 'ntlbg' in total_loss else 0
            })
            
            # 日志记录
            if batch_idx % self.config['logging_config']['log_interval'] == 0:
                self.logger.info(
                    f"Epoch {epoch}, Step {self.train_stats['total_steps']}, "
                    f"Loss: {total_loss['total'].item():.4f}, "
                    f"LR: {current_lr:.2e}"
                )
        
        # 计算平均损失
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def _compute_loss(self, outputs: Dict, weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """计算多任务损失"""
        losses = {}
        
        # 从模型输出中提取损失
        loss_components = outputs.get('loss_components', {})
        
        # 任务损失
        task_loss = loss_components.get('task_loss', torch.tensor(0.0, device=self.device))
        losses['task'] = task_loss
        
        # NTLBG约束损失
        ntlbg_loss = loss_components.get('ntlbg_loss', torch.tensor(0.0, device=self.device))
        losses['ntlbg'] = ntlbg_loss
        
        # 特征对齐损失
        alignment_loss = loss_components.get('alignment_loss', torch.tensor(0.0, device=self.device))
        losses['alignment'] = alignment_loss
        
        # 时序连贯性损失
        temporal_loss = loss_components.get('temporal_loss', torch.tensor(0.0, device=self.device))
        losses['temporal'] = temporal_loss
        
        # 信息保持损失
        info_loss = loss_components.get('info_loss', torch.tensor(0.0, device=self.device))
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
        
        return losses
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                outputs = self.model(
                    video_frames=batch['video_features'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # 使用当前权重计算损失
                current_weights = self.loss_weight_scheduler.get_weights(
                    self.train_stats['total_steps'], epoch
                )
                total_loss = self._compute_loss(outputs, current_weights)
                
                for key, value in total_loss.items():
                    val_losses[key].append(value.item())
        
        # 计算平均损失
        avg_val_losses = {key: np.mean(values) for key, values in val_losses.items()}
        
        return avg_val_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'train_stats': self.train_stats,
            'config': self.config
        }
        
        # 保存当前检查点
        checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['output_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config['training_config']['num_epochs']}")
        self.logger.info(f"Total training steps: {len(self.train_dataloader) * self.config['training_config']['num_epochs']}")
        
        for epoch in range(self.config['training_config']['num_epochs']):
            self.train_stats['epoch'] = epoch
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_losses = self.validate(epoch)
            
            # 检查是否是最佳模型
            current_val_loss = val_losses['total']
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss
            
            # 保存检查点
            if epoch % self.config['logging_config']['save_interval'] == 0:
                self.save_checkpoint(epoch, current_val_loss, is_best)
            
            # 记录日志
            self.logger.info(f"Epoch {epoch}")
            self.logger.info(f"Train - Total: {train_losses['total']:.4f}, Task: {train_losses['task']:.4f}, "
                           f"NTLBG: {train_losses.get('ntlbg', 0):.4f}")
            self.logger.info(f"Val - Total: {val_losses['total']:.4f}, Task: {val_losses['task']:.4f}, "
                           f"NTLBG: {val_losses.get('ntlbg', 0):.4f}")
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练NTLBG-LLM模型")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="配置文件路径"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="调试模式：使用小数据集快速测试"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="干运行：只验证配置和数据加载"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 调试模式设置
    if args.debug:
        print("🐛 调试模式启用")
        config['training_config']['num_epochs'] = 2
        config['training_config']['batch_size'] = 2
        config['data_config']['max_video_frames'] = 20
        config['logging_config']['use_wandb'] = False
    
    # 设置随机种子
    set_seed(config['training_config']['seed'])
    
    # 获取设备信息
    device = get_device_info()
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 保存配置文件副本
    config_save_path = os.path.join(config['output_dir'], 'config.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    try:
        # 创建数据加载器
        print("📁 创建数据加载器...")
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config)
        print(f"训练样本数: {len(train_dataloader.dataset)}")
        print(f"验证样本数: {len(val_dataloader.dataset)}")
        print(f"测试样本数: {len(test_dataloader.dataset)}")
        
        if args.dry_run:
            print("✅ 干运行完成：数据加载正常")
            return
        
        # 创建模型
        print("🤖 创建NTLBG-LLM模型...")
        model = create_ntlbg_llm(config['model_config'])
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
        
        # 创建训练器
        print("🏃 创建训练器...")
        trainer = NTLBGTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            device=device
        )
        
        # 开始训练
        print("🚀 开始训练...")
        trainer.train()
        
        print("✅ 训练完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 