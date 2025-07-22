#!/usr/bin/env python3
"""
NTLBG-LLM训练实验脚本
用于执行NTLBG-LLM的训练实验
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
from src.training.trainer import NTLBGTrainer
from src.evaluation.metrics import EvaluationRunner
from src.evaluation.visualizer import NTLBGVisualizer


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


def create_experiment_config(base_config: Dict[str, Any], experiment_params: Dict[str, Any]) -> Dict[str, Any]:
    """创建实验配置"""
    config = base_config.copy()
    
    # 更新实验特定参数
    for key, value in experiment_params.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        else:
            config[key] = value
    
    return config


def run_training_experiment(config: Dict[str, Any], 
                          experiment_name: str,
                          output_dir: str,
                          logger: logging.Logger) -> Dict[str, Any]:
    """运行训练实验"""
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # 设置设备
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # 创建数据加载器
        logger.info("Creating data loaders...")
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config)
        logger.info(f"Train samples: {len(train_dataloader.dataset)}")
        logger.info(f"Val samples: {len(val_dataloader.dataset)}")
        logger.info(f"Test samples: {len(test_dataloader.dataset)}")
        
        # 创建模型
        logger.info("Creating NTLBG-LLM model...")
        model = create_ntlbg_llm(config['model_config'])
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # 创建训练器
        logger.info("Creating trainer...")
        trainer = NTLBGTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            device=device,
            logger=logger
        )
        
        # 开始训练
        logger.info("Starting training...")
        training_results = trainer.train()
        
        # 获取训练摘要
        training_summary = trainer.get_training_summary()
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"Total epochs: {training_results['total_epochs']}")
        
        return {
            'training_results': training_results,
            'training_summary': training_summary,
            'config': config,
            'experiment_name': experiment_name
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def run_evaluation_experiment(config: Dict[str, Any],
                            model_path: str,
                            experiment_name: str,
                            output_dir: str,
                            logger: logging.Logger) -> Dict[str, Any]:
    """运行评估实验"""
    
    logger.info(f"Starting evaluation experiment: {experiment_name}")
    
    # 设置设备
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
                
                # 生成预测
                predictions = []  # 这里需要实现实际的文本生成
                references = []   # 从batch中提取参考答案
                
                # 添加到评估器
                evaluator.evaluate_batch(
                    model_outputs=outputs,
                    batch=batch,
                    predictions=predictions,
                    references=references
                )
                
                if batch_idx % 10 == 0:
                    logger.info(f"Evaluated {batch_idx + 1}/{len(test_dataloader)} batches")
        
        # 计算指标
        logger.info("Computing metrics...")
        metrics = evaluator.compute_all_metrics()
        
        # 打印摘要
        evaluator.print_summary()
        
        # 保存结果
        results_path = Path(output_dir) / f"{experiment_name}_evaluation_results.json"
        evaluator.save_results(str(results_path))
        
        logger.info(f"Evaluation completed. Results saved to: {results_path}")
        
        return {
            'metrics': metrics,
            'config': config,
            'experiment_name': experiment_name,
            'model_path': model_path
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def create_visualizations(results: Dict[str, Any], output_dir: str, logger: logging.Logger):
    """创建可视化"""
    
    logger.info("Creating visualizations...")
    
    try:
        # 创建可视化器
        visualizer = NTLBGVisualizer(
            config=results['config'],
            output_dir=os.path.join(output_dir, 'visualizations')
        )
        
        # 获取训练数据
        training_summary = results.get('training_summary', {})
        train_losses = training_summary.get('train_losses', [])
        val_losses = training_summary.get('val_losses', [])
        
        # 创建训练进度可视化
        if train_losses and val_losses:
            visualizer.visualize_training_progress(
                train_losses=train_losses,
                val_losses=val_losses,
                learning_rates=training_summary.get('learning_rates', []),
                save_name=f"{results['experiment_name']}_training_progress"
            )
        
        # 创建评估指标可视化
        if 'metrics' in results:
            visualizer.visualize_evaluation_metrics(
                metrics=results['metrics'],
                save_name=f"{results['experiment_name']}_evaluation_metrics"
            )
            
            # 创建交互式仪表板
            visualizer.create_interactive_dashboard(
                train_losses=train_losses,
                val_losses=val_losses,
                metrics=results['metrics'],
                save_name=f"{results['experiment_name']}_dashboard"
            )
        
        # 创建综合报告
        if train_losses and val_losses and 'metrics' in results:
            report = visualizer.create_comprehensive_report(
                train_losses=train_losses,
                val_losses=val_losses,
                metrics=results['metrics'],
                config=results['config'],
                save_name=f"{results['experiment_name']}_comprehensive_report"
            )
            
            logger.info("Comprehensive report created successfully!")
        
        logger.info("Visualizations created successfully!")
        
    except Exception as e:
        logger.error(f"Failed to create visualizations: {str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NTLBG-LLM训练实验")
    parser.add_argument("--config", type=str, required=True, help="基础配置文件路径")
    parser.add_argument("--experiment-name", type=str, required=True, help="实验名称")
    parser.add_argument("--output-dir", type=str, default="results/experiments", help="输出目录")
    parser.add_argument("--mode", type=str, choices=['train', 'eval', 'both'], default='both', help="运行模式")
    parser.add_argument("--model-path", type=str, help="评估时使用的模型路径")
    
    # 实验参数
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch-size", type=int, help="批次大小")
    parser.add_argument("--learning-rate", type=float, help="学习率")
    parser.add_argument("--num-representatives", type=int, help="代表点数量")
    parser.add_argument("--ntlbg-weight", type=float, help="NTLBG损失权重")
    
    args = parser.parse_args()
    
    # 创建输出目录
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(str(experiment_dir), args.experiment_name)
    
    try:
        # 加载基础配置
        config = load_config(args.config)
        
        # 创建实验配置
        experiment_params = {}
        if args.epochs:
            experiment_params['training_config'] = {'num_epochs': args.epochs}
        if args.batch_size:
            experiment_params['training_config'] = experiment_params.get('training_config', {})
            experiment_params['training_config']['batch_size'] = args.batch_size
        if args.learning_rate:
            experiment_params['training_config'] = experiment_params.get('training_config', {})
            experiment_params['training_config']['learning_rate'] = args.learning_rate
        if args.num_representatives:
            experiment_params['model_config'] = {'num_representative_points': args.num_representatives}
        if args.ntlbg_weight:
            experiment_params['loss_weights'] = {'ntlbg': args.ntlbg_weight}
        
        # 更新输出目录
        experiment_params['output_dir'] = str(experiment_dir)
        
        # 创建实验配置
        experiment_config = create_experiment_config(config, experiment_params)
        
        # 保存实验配置
        config_path = experiment_dir / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Experiment configuration saved to: {config_path}")
        
        results = {}
        
        # 运行训练
        if args.mode in ['train', 'both']:
            training_results = run_training_experiment(
                config=experiment_config,
                experiment_name=args.experiment_name,
                output_dir=str(experiment_dir),
                logger=logger
            )
            results.update(training_results)
            
            # 保存训练结果
            training_results_path = experiment_dir / "training_results.json"
            with open(training_results_path, 'w', encoding='utf-8') as f:
                json.dump(training_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Training results saved to: {training_results_path}")
        
        # 运行评估
        if args.mode in ['eval', 'both']:
            model_path = args.model_path
            if not model_path and args.mode == 'both':
                model_path = results.get('training_results', {}).get('best_model_path')
            
            if model_path and os.path.exists(model_path):
                evaluation_results = run_evaluation_experiment(
                    config=experiment_config,
                    model_path=model_path,
                    experiment_name=args.experiment_name,
                    output_dir=str(experiment_dir),
                    logger=logger
                )
                results.update(evaluation_results)
            else:
                logger.warning(f"Model path not found: {model_path}")
        
        # 创建可视化
        if results:
            create_visualizations(results, str(experiment_dir), logger)
        
        # 保存最终结果
        final_results_path = experiment_dir / "final_results.json"
        with open(final_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Final results saved to: {final_results_path}")
        logger.info(f"Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 