"""
NTLBG-LLM训练模块

包含训练循环、损失计算、学习率调度等功能
"""

from .trainer import NTLBGTrainer
from .losses import NTLBGLossComputer, LossWeightScheduler
from .scheduler import (
    LearningRateScheduler,
    OptimizerFactory,
    GradientClipping,
    TrainingScheduler
)

__all__ = [
    # 主训练器
    'NTLBGTrainer',
    
    # 损失计算
    'NTLBGLossComputer',
    'LossWeightScheduler',
    
    # 学习率调度
    'LearningRateScheduler',
    'OptimizerFactory',
    'GradientClipping',
    'TrainingScheduler'
] 