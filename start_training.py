#!/usr/bin/env python3
"""
NTLBG-LLM H200训练启动脚本
"""
import os
import sys
import yaml
import torch
import logging
from datetime import datetime

# 导入您的训练模块（需要根据实际代码调整）
# from src.training.trainer import NTLBGTrainer
# from src.models.ntlbg_llm import NTLBGLLM

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_environment():
    print("🔍 环境检查...")
    
    # GPU检查
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB")
    else:
        print("❌ CUDA不可用")
        return False
    
    # 数据集检查
    datasets = ["longvideobench", "video_mme", "mlvu"]
    valid_datasets = []
    
    for dataset in datasets:
        path = f"data/{dataset}"
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            size = os.popen(f"du -sh {path} 2>/dev/null").read().split()[0]
            print(f"✅ {dataset}: {size}")
            valid_datasets.append(dataset)
        else:
            print(f"❌ {dataset}: 不可用")
    
    print(f"📊 可用数据集: {len(valid_datasets)}/3")
    return len(valid_datasets) >= 2

def load_config():
    with open("configs/train_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    print("🚀 启动NTLBG-LLM训练")
    print("=" * 50)
    
    # 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请修复后重试")
        return
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logging(config['output_dir'])
    logger.info("🎯 开始NTLBG-LLM训练")
    
    # 这里会调用您的实际训练代码
    # model = NTLBGLLM(config)
    # trainer = NTLBGTrainer(model, config)
    # trainer.train()
    
    print("✅ 训练脚本准备就绪")
    print("📝 请根据您的实际代码结构调整导入和训练逻辑")
    print("🔄 现在可以开始实际训练了!")

if __name__ == "__main__":
    main()
