"""
NTLBG-LLM主训练器模块
包含完整的训练循环和检查点管理
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from collections import defaultdict
import random
import warnings
import wandb
from contextlib import nullcontext

from .losses import NTLBGLossComputer
from .scheduler import TrainingScheduler


class NTLBGTrainer:
    """NTLBG-LLM训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 config: Dict,
                 device: torch.device,
                 logger: Optional[logging.Logger] = None):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.logger = logger or self._setup_logger()
        
        # 将模型移动到设备
        self.model.to(device)
        
        # 设置训练配置
        self.training_config = config['training_config']
        self.logging_config = config.get('logging_config', {})
        
        # 创建损失计算器
        self.loss_computer = NTLBGLossComputer(config, device)
        
        # 创建训练调度器（优化器、学习率调度器、梯度裁剪）
        self.training_scheduler = TrainingScheduler(model, config)
        
        # 训练统计
        self.train_stats = {
            'epoch': 0,
            'total_steps': 0,
            'best_val_loss': float('inf'),
            'best_model_path': None,
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'grad_norms': []
        }
        
        # 设置混合精度训练
        self.use_amp = config.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 设置分布式训练
        self.is_distributed = config.get('distributed', False)
        self.local_rank = config.get('local_rank', 0)
        
        # 设置wandb
        self.use_wandb = self.logging_config.get('use_wandb', False)
        if self.use_wandb and self.local_rank == 0:
            self._setup_wandb()
        
        # 创建输出目录
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置检查点保存
        self.save_interval = self.logging_config.get('save_interval', 1)
        self.save_total_limit = self.logging_config.get('save_total_limit', 5)
        
        # 早停
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        
        self.logger.info(f"Trainer initialized with device: {device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_wandb(self):
        """设置wandb"""
        try:
            wandb.init(
                project=self.logging_config.get('wandb_project', 'ntlbg-llm'),
                name=self.logging_config.get('wandb_name', f'ntlbg-{datetime.now().strftime("%Y%m%d-%H%M%S")}'),
                config=self.config,
                dir=self.output_dir
            )
            self.logger.info("Wandb initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        # 设置进度条
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch}/{self.training_config['num_epochs']}",
            disable=self.local_rank != 0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            batch = self._move_to_device(batch)
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    video_frames=batch['video_features'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # 计算损失
                losses = self.loss_computer.compute_loss(
                    outputs, 
                    batch['labels'], 
                    self.train_stats['total_steps'], 
                    epoch
                )
                
                total_loss = losses['total']
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.training_scheduler.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.training_scheduler.optimizer.step()
            
            # 更新学习率
            self.training_scheduler.lr_scheduler.step()
            
            # 梯度裁剪
            if self.training_config.get('max_grad_norm', 0) > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.training_scheduler.optimizer)
                
                grad_norm = clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['max_grad_norm']
                ).item()
                
                self.train_stats['grad_norms'].append(grad_norm)
            
            # 清零梯度
            self.training_scheduler.optimizer.zero_grad()
            
            # 记录损失
            for key, value in losses.items():
                if key != 'weights' and torch.is_tensor(value):
                    epoch_losses[key].append(value.item())
            
            # 更新统计信息
            self.train_stats['total_steps'] += 1
            current_lr = self.training_scheduler.lr_scheduler.get_current_lr()
            self.train_stats['learning_rates'].append(current_lr)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{current_lr:.2e}",
                'ntlbg': f"{losses.get('ntlbg', torch.tensor(0.0)).item():.4f}"
            })
            
            # 日志记录
            if batch_idx % self.logging_config.get('log_interval', 100) == 0:
                self._log_training_step(epoch, batch_idx, losses, current_lr)
            
            # wandb记录
            if self.use_wandb and self.train_stats['total_steps'] % self.logging_config.get('wandb_log_interval', 50) == 0:
                self._log_to_wandb(losses, current_lr, 'train')
        
        # 计算平均损失
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_dataloader, 
                desc=f"Validation {epoch}",
                disable=self.local_rank != 0
            )
            
            for batch in progress_bar:
                batch = self._move_to_device(batch)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(
                        video_frames=batch['video_features'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    
                    # 计算损失
                    losses = self.loss_computer.compute_loss(
                        outputs, 
                        batch['labels'], 
                        self.train_stats['total_steps'], 
                        epoch
                    )
                
                # 记录损失
                for key, value in losses.items():
                    if key != 'weights' and torch.is_tensor(value):
                        val_losses[key].append(value.item())
                
                # 更新进度条
                progress_bar.set_postfix({
                    'val_loss': f"{losses['total'].item():.4f}",
                    'val_ntlbg': f"{losses.get('ntlbg', torch.tensor(0.0)).item():.4f}"
                })
        
        # 计算平均损失
        avg_val_losses = {key: np.mean(values) for key, values in val_losses.items()}
        
        # wandb记录
        if self.use_wandb:
            self._log_to_wandb(avg_val_losses, None, 'val')
        
        return avg_val_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        if self.local_rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'training_scheduler_state_dict': self.training_scheduler.get_state_dict(),
            'val_loss': val_loss,
            'train_stats': self.train_stats,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        # 保存当前检查点
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.train_stats['best_model_path'] = best_path
            self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # 清理旧的检查点
        self._cleanup_checkpoints()
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载训练调度器状态
        self.training_scheduler.load_state_dict(checkpoint['training_scheduler_state_dict'])
        
        # 加载训练统计
        self.train_stats = checkpoint['train_stats']
        
        # 加载scaler状态
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded successfully. Resuming from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']
    
    def train(self) -> Dict[str, Any]:
        """主训练循环"""
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.training_config['num_epochs']}")
        self.logger.info(f"Total training steps: {len(self.train_dataloader) * self.training_config['num_epochs']}")
        
        start_epoch = 0
        
        # 如果有检查点，加载它
        resume_from = self.config.get('resume_from_checkpoint')
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        
        # 训练循环
        for epoch in range(start_epoch, self.training_config['num_epochs']):
            self.train_stats['epoch'] = epoch
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_losses = self.validate(epoch)
            
            # 记录损失
            self.train_stats['train_losses'].append(train_losses)
            self.train_stats['val_losses'].append(val_losses)
            
            # 检查是否是最佳模型
            current_val_loss = val_losses['total']
            is_best = current_val_loss < self.train_stats['best_val_loss']
            if is_best:
                self.train_stats['best_val_loss'] = current_val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # 保存检查点
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, current_val_loss, is_best)
            
            # 日志记录
            self._log_epoch_end(epoch, train_losses, val_losses)
            
            # 早停检查
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # 训练完成
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.train_stats['best_val_loss']:.4f}")
        
        # 保存最终模型
        self.save_checkpoint(epoch, current_val_loss, False)
        
        # 关闭wandb
        if self.use_wandb:
            wandb.finish()
        
        return {
            'best_val_loss': self.train_stats['best_val_loss'],
            'total_epochs': epoch + 1,
            'total_steps': self.train_stats['total_steps'],
            'best_model_path': self.train_stats['best_model_path']
        }
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """将batch数据移动到设备"""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v 
            for k, v in batch.items()
        }
    
    def _log_training_step(self, epoch: int, batch_idx: int, losses: Dict, lr: float):
        """记录训练步骤"""
        self.logger.info(
            f"Epoch {epoch}, Step {self.train_stats['total_steps']}, "
            f"Batch {batch_idx}, Loss: {losses['total'].item():.4f}, "
            f"NTLBG: {losses.get('ntlbg', torch.tensor(0.0)).item():.4f}, "
            f"LR: {lr:.2e}"
        )
    
    def _log_epoch_end(self, epoch: int, train_losses: Dict, val_losses: Dict):
        """记录epoch结束"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Epoch {epoch} Summary:")
        self.logger.info(f"Train - Total: {train_losses['total']:.4f}, "
                        f"Task: {train_losses['task']:.4f}, "
                        f"NTLBG: {train_losses.get('ntlbg', 0):.4f}")
        self.logger.info(f"Val - Total: {val_losses['total']:.4f}, "
                        f"Task: {val_losses['task']:.4f}, "
                        f"NTLBG: {val_losses.get('ntlbg', 0):.4f}")
        self.logger.info(f"Best Val Loss: {self.train_stats['best_val_loss']:.4f}")
        self.logger.info(f"Early Stopping Counter: {self.early_stopping_counter}")
        self.logger.info(f"{'='*50}\n")
    
    def _log_to_wandb(self, losses: Dict, lr: Optional[float], phase: str):
        """记录到wandb"""
        if not self.use_wandb:
            return
        
        log_dict = {}
        for key, value in losses.items():
            if key != 'weights' and torch.is_tensor(value):
                log_dict[f'{phase}/{key}_loss'] = value.item()
            elif key != 'weights' and isinstance(value, (int, float)):
                log_dict[f'{phase}/{key}_loss'] = value
        
        if lr is not None:
            log_dict['learning_rate'] = lr
        
        log_dict['epoch'] = self.train_stats['epoch']
        log_dict['step'] = self.train_stats['total_steps']
        
        wandb.log(log_dict)
    
    def _cleanup_checkpoints(self):
        """清理旧的检查点"""
        if self.save_total_limit <= 0:
            return
        
        # 获取所有检查点文件
        checkpoint_files = []
        for file in os.listdir(self.output_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
                epoch_num = int(file.split('_')[2].split('.')[0])
                checkpoint_files.append((epoch_num, file))
        
        # 按epoch排序
        checkpoint_files.sort(key=lambda x: x[0])
        
        # 删除超出限制的旧文件
        while len(checkpoint_files) > self.save_total_limit:
            epoch_num, filename = checkpoint_files.pop(0)
            file_path = os.path.join(self.output_dir, filename)
            try:
                os.remove(file_path)
                self.logger.info(f"Removed old checkpoint: {filename}")
            except OSError as e:
                self.logger.warning(f"Failed to remove {filename}: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'total_epochs': self.train_stats['epoch'],
            'total_steps': self.train_stats['total_steps'],
            'best_val_loss': self.train_stats['best_val_loss'],
            'best_model_path': self.train_stats['best_model_path'],
            'final_lr': self.training_scheduler.lr_scheduler.get_current_lr(),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'config': self.config
        } 