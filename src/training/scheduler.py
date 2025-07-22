"""
NTLBG-LLM学习率调度器模块
支持多种学习率调度策略
"""

import torch
import torch.optim as optim
from typing import Dict, Optional, Union
import numpy as np
import math


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer: optim.Optimizer, config: Dict):
        self.optimizer = optimizer
        self.config = config
        self.scheduler_type = config.get('scheduler', 'cosine')
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.max_steps = config.get('max_steps', 10000)
        self.initial_lr = config.get('learning_rate', 1e-4)
        self.min_lr = config.get('min_lr', 1e-6)
        
        # 创建调度器
        self.scheduler = self._create_scheduler()
        
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if self.scheduler_type == 'cosine':
            return self._create_cosine_scheduler()
        elif self.scheduler_type == 'linear':
            return self._create_linear_scheduler()
        elif self.scheduler_type == 'exponential':
            return self._create_exponential_scheduler()
        elif self.scheduler_type == 'step':
            return self._create_step_scheduler()
        elif self.scheduler_type == 'plateau':
            return self._create_plateau_scheduler()
        elif self.scheduler_type == 'polynomial':
            return self._create_polynomial_scheduler()
        else:
            return None
    
    def _create_cosine_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建余弦退火调度器"""
        def lr_lambda(step):
            if step < self.warmup_steps:
                # 预热阶段
                return step / self.warmup_steps
            else:
                # 余弦退火阶段
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                progress = min(1.0, progress)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = (1 - self.min_lr / self.initial_lr) * cosine_decay + self.min_lr / self.initial_lr
                return decayed
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_linear_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建线性衰减调度器"""
        def lr_lambda(step):
            if step < self.warmup_steps:
                # 预热阶段
                return step / self.warmup_steps
            else:
                # 线性衰减阶段
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                progress = min(1.0, progress)
                return max(self.min_lr / self.initial_lr, 1.0 - progress)
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_exponential_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建指数衰减调度器"""
        gamma = self.config.get('gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
    
    def _create_step_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建阶梯衰减调度器"""
        step_size = self.config.get('step_size', 1000)
        gamma = self.config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    
    def _create_plateau_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建自适应调度器"""
        patience = self.config.get('patience', 10)
        factor = self.config.get('factor', 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=factor, patience=patience, verbose=True
        )
    
    def _create_polynomial_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建多项式衰减调度器"""
        power = self.config.get('power', 0.9)
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # 预热阶段
                return step / self.warmup_steps
            else:
                # 多项式衰减阶段
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                progress = min(1.0, progress)
                return max(self.min_lr / self.initial_lr, (1 - progress) ** power)
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """更新学习率"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is not None and 'val_loss' in metrics:
                    self.scheduler.step(metrics['val_loss'])
            else:
                self.scheduler.step()
    
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr_schedule_info(self) -> Dict[str, Union[str, float, int]]:
        """获取学习率调度信息"""
        return {
            'scheduler_type': self.scheduler_type,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr,
            'current_lr': self.get_current_lr()
        }


class OptimizerFactory:
    """优化器工厂"""
    
    @staticmethod
    def create_optimizer(model: torch.nn.Module, config: Dict) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = config.get('optimizer', 'adamw')
        learning_rate = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 0.01)
        
        # 参数分组
        param_groups = OptimizerFactory._get_param_groups(model, config)
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                param_groups,
                lr=learning_rate,
                betas=config.get('betas', (0.9, 0.999)),
                eps=config.get('eps', 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(
                param_groups,
                lr=learning_rate,
                betas=config.get('betas', (0.9, 0.999)),
                eps=config.get('eps', 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                param_groups,
                lr=learning_rate,
                momentum=config.get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=config.get('nesterov', False)
            )
        elif optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(
                param_groups,
                lr=learning_rate,
                alpha=config.get('alpha', 0.99),
                eps=config.get('eps', 1e-8),
                weight_decay=weight_decay,
                momentum=config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    @staticmethod
    def _get_param_groups(model: torch.nn.Module, config: Dict) -> list:
        """获取参数分组"""
        # 默认参数组
        param_groups = [{'params': [], 'weight_decay': config.get('weight_decay', 0.01)}]
        
        # 无权重衰减的参数组（如bias和layernorm）
        no_decay_params = []
        
        # 不同学习率的参数组
        special_lr_params = {}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 检查是否需要特殊学习率
            special_lr = OptimizerFactory._get_special_lr(name, config)
            if special_lr is not None:
                if special_lr not in special_lr_params:
                    special_lr_params[special_lr] = []
                special_lr_params[special_lr].append(param)
                continue
            
            # 检查是否需要权重衰减
            if OptimizerFactory._no_weight_decay(name):
                no_decay_params.append(param)
            else:
                param_groups[0]['params'].append(param)
        
        # 添加无权重衰减的参数组
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params,
                'weight_decay': 0.0
            })
        
        # 添加特殊学习率的参数组
        for lr, params in special_lr_params.items():
            param_groups.append({
                'params': params,
                'lr': lr,
                'weight_decay': config.get('weight_decay', 0.01)
            })
        
        return param_groups
    
    @staticmethod
    def _no_weight_decay(param_name: str) -> bool:
        """判断参数是否需要权重衰减"""
        no_decay_keywords = ['bias', 'LayerNorm', 'layernorm', 'norm', 'embedding']
        return any(keyword in param_name for keyword in no_decay_keywords)
    
    @staticmethod
    def _get_special_lr(param_name: str, config: Dict) -> Optional[float]:
        """获取特殊学习率"""
        special_lr_config = config.get('special_lr', {})
        
        for pattern, lr in special_lr_config.items():
            if pattern in param_name:
                return lr
        
        return None


class GradientClipping:
    """梯度裁剪"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.clip_type = config.get('clip_type', 'norm')
        
    def clip_gradients(self, model: torch.nn.Module) -> float:
        """裁剪梯度"""
        if self.clip_type == 'norm':
            return torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.max_grad_norm
            ).item()
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                self.max_grad_norm
            )
            return self.max_grad_norm
        else:
            return 0.0
    
    def get_grad_norm(self, model: torch.nn.Module) -> float:
        """获取梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** 0.5


class TrainingScheduler:
    """训练调度器 - 整合学习率调度和梯度裁剪"""
    
    def __init__(self, model: torch.nn.Module, config: Dict):
        self.model = model
        self.config = config
        
        # 创建优化器
        self.optimizer = OptimizerFactory.create_optimizer(model, config['optimizer_config'])
        
        # 创建学习率调度器
        self.lr_scheduler = LearningRateScheduler(self.optimizer, config['scheduler_config'])
        
        # 创建梯度裁剪器
        self.gradient_clipper = GradientClipping(config['gradient_config'])
        
        # 训练统计
        self.step_count = 0
        self.epoch_count = 0
        
    def step(self, loss: torch.Tensor, metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """执行一步训练"""
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = self.gradient_clipper.clip_gradients(self.model)
        
        # 优化器步进
        self.optimizer.step()
        
        # 学习率调度
        self.lr_scheduler.step(metrics)
        
        # 更新统计
        self.step_count += 1
        
        # 返回训练统计
        return {
            'learning_rate': self.lr_scheduler.get_current_lr(),
            'grad_norm': grad_norm,
            'step_count': self.step_count
        }
    
    def epoch_end(self):
        """epoch结束时调用"""
        self.epoch_count += 1
    
    def get_state_dict(self) -> Dict:
        """获取状态字典"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.scheduler.state_dict() if self.lr_scheduler.scheduler else None,
            'step_count': self.step_count,
            'epoch_count': self.epoch_count
        }
    
    def load_state_dict(self, state_dict: Dict):
        """加载状态字典"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.lr_scheduler.scheduler and state_dict['lr_scheduler']:
            self.lr_scheduler.scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.step_count = state_dict['step_count']
        self.epoch_count = state_dict['epoch_count']
    
    def get_training_info(self) -> Dict:
        """获取训练信息"""
        return {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'current_lr': self.lr_scheduler.get_current_lr(),
            'lr_schedule_info': self.lr_scheduler.get_lr_schedule_info(),
            'optimizer_info': {
                'type': self.config['optimizer_config'].get('optimizer', 'adamw'),
                'weight_decay': self.config['optimizer_config'].get('weight_decay', 0.01),
                'param_groups': len(self.optimizer.param_groups)
            }
        } 