#!/usr/bin/env python3
"""
NTLBG-LLM评估实验脚本
用于评估训练好的NTLBG-LLM模型
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ntlbg_llm import create_ntlbg_llm
from src.data.datasets import create_dataloaders
from src.evaluation.metrics import EvaluationRunner
from src.evaluation.visualizer import NTLBGVisualizer
import torch


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    """设置日志记录"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_evaluation(config: Dict[str, Any], 
                  model_path: str, 
                  output_dir: str,
                  logger: logging.Logger) -> Dict[str, Any]:
    """运行评估"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # 创建数据加载器
        logger.info("Creating data loaders...")
        _, _, test_dataloader = create_dataloaders(config)
        logger.info(f"Test samples: {len(test_dataloader.dataset)}")
        
        # 创建模型
        logger.info("Creating model...")
        model = create_ntlbg_llm(config['model_config'])
        
        # 加载模型权重
        logger.info(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # 创建评估器
        evaluator = EvaluationRunner(config)
        
        # 运行评估
        logger.info("Running evaluation...")
        total_batches = len(test_dataloader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                # 将数据移到设备
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # 前向传播
                outputs = model(
                    video_frames=batch['video_features'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # 模拟预测和参考答案（实际应用中需要实现文本生成）
                batch_size = batch['video_features'].size(0)
                predictions = [f"Generated answer {i}" for i in range(batch_size)]
                references = [f"Reference answer {i}" for i in range(batch_size)]
                
                # 添加到评估器
                evaluator.evaluate_batch(
                    model_outputs=outputs,
                    batch=batch,
                    predictions=predictions,
                    references=references,
                    video_ids=[f"video_{batch_idx}_{i}" for i in range(batch_size)]
                )
                
                if batch_idx % 10 == 0:
                    logger.info(f"Evaluated {batch_idx + 1}/{total_batches} batches ({100*(batch_idx+1)/total_batches:.1f}%)")
        
        # 计算指标
        logger.info("Computing metrics...")
        metrics = evaluator.compute_all_metrics()
        
        # 打印摘要
        evaluator.print_summary()
        
        logger.info("Evaluation completed successfully!")
        
        return {
            'metrics': metrics,
            'config': config,
            'model_path': model_path,
            'total_samples': len(test_dataloader.dataset)
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NTLBG-LLM评估实验")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--output-dir", type=str, default="results/evaluation", help="输出目录")
    parser.add_argument("--experiment-name", type=str, default="evaluation", help="实验名称")
    
    args = parser.parse_args()
    
    # 创建输出目录
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(str(experiment_dir), args.experiment_name)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 运行评估
        results = run_evaluation(config, args.model_path, str(experiment_dir), logger)
        
        # 保存结果
        results_path = experiment_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # 创建可视化
        logger.info("Creating visualizations...")
        visualizer = NTLBGVisualizer(
            config=config,
            output_dir=str(experiment_dir / 'visualizations')
        )
        
        # 创建评估指标可视化
        visualizer.visualize_evaluation_metrics(
            metrics=results['metrics'],
            save_name=f"{args.experiment_name}_evaluation_metrics"
        )
        
        # 保存比较表格
        visualizer.save_comparison_table(
            results=results,
            save_name=f"{args.experiment_name}_comparison_table"
        )
        
        logger.info("Visualizations created successfully!")
        logger.info("Evaluation experiment completed!")
        
    except Exception as e:
        logger.error(f"Evaluation experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 