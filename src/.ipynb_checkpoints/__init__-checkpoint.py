"""
NTLBG-LLM - 基于统计代表点理论的长视频理解模型

"Statistical Representative Point Sampling for Efficient Long Video Understanding"

本项目实现了一个基于统计学代表点理论的长视频理解模型，专门用于解决长视频中的关键帧选择和视频问答任务。
该模型将多元正态分布的代表点理论应用到视频理解中，通过NTLBG（Novel Temporal Long-form Best-view Grounding）
约束来选择最具代表性的视频帧。

主要模块：
- models: 核心模型架构（NTLBG注意力、富代表点构造器、主模型）
- data: 数据处理模块（视频加载、数据集封装）
- training: 训练模块（训练器、损失函数、调度器）
- evaluation: 评估模块（评估指标、结果可视化）
"""

# 模型相关
from .models.ntlbg_attention import NTLBGAttention
from .models.rich_points import RichRepresentativePointConstructor
from .models.ntlbg_llm import NTLBGLLM, create_ntlbg_llm

# 数据相关
from .data.video_loader import VideoLoader
from .data.datasets import VideoQADataset, create_dataloaders

# 训练相关
from .training.trainer import NTLBGTrainer
from .training.losses import NTLBGLossComputer, LossWeightScheduler
from .training.scheduler import (
    LearningRateScheduler, 
    OptimizerFactory, 
    GradientClipping,
    TrainingScheduler
)

# 评估相关
from .evaluation.metrics import (
    VideoQAMetrics, 
    NTLBGSpecificMetrics, 
    EvaluationRunner
)
from .evaluation.visualizer import NTLBGVisualizer

__version__ = "1.0.0"
__author__ = "NTLBG-LLM Team"
__email__ = "contact@ntlbg-llm.org"
__description__ = "Statistical Representative Point Sampling for Efficient Long Video Understanding"

__all__ = [
    # 模型
    'NTLBGAttention',
    'RichRepresentativePointConstructor', 
    'NTLBGLLM',
    'create_ntlbg_llm',
    
    # 数据
    'VideoLoader',
    'VideoQADataset',
    'create_dataloaders',
    
    # 训练
    'NTLBGTrainer',
    'NTLBGLossComputer',
    'LossWeightScheduler',
    'LearningRateScheduler',
    'OptimizerFactory',
    'GradientClipping',
    'TrainingScheduler',
    
    # 评估
    'VideoQAMetrics',
    'NTLBGSpecificMetrics',
    'EvaluationRunner',
    'NTLBGVisualizer',
    
    # 版本信息
    '__version__',
    '__author__',
    '__email__',
    '__description__'
]


def get_version():
    """获取版本信息"""
    return __version__


def get_model_info():
    """获取模型信息"""
    return {
        'name': 'NTLBG-LLM',
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'email': __email__,
        'components': {
            'ntlbg_attention': 'NTLBG注意力模块',
            'rich_points': '富代表点构造器',
            'ntlbg_llm': '主模型架构',
            'training': '训练模块',
            'evaluation': '评估模块'
        }
    }


def print_model_info():
    """打印模型信息"""
    info = get_model_info()
    print(f"=== {info['name']} v{info['version']} ===")
    print(f"描述: {info['description']}")
    print(f"作者: {info['author']}")
    print(f"邮箱: {info['email']}")
    print("\n主要组件:")
    for component, description in info['components'].items():
        print(f"  - {component}: {description}")
    print("=" * 50) 