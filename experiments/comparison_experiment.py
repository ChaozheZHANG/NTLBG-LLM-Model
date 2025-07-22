#!/usr/bin/env python3
"""
NTLBG-LLM对比实验脚本
用于比较不同NTLBG配置的性能
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train_experiment import run_training_experiment, run_evaluation_experiment, create_visualizations
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


def create_experiment_configurations(base_config: Dict[str, Any], 
                                   experiment_params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """创建实验配置列表"""
    configurations = []
    
    for i, params in enumerate(experiment_params):
        config = base_config.copy()
        
        # 更新实验特定参数
        for key, value in params.items():
            if key in config:
                if isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
            else:
                config[key] = value
        
        # 添加实验标识
        config['experiment_id'] = i
        config['experiment_params'] = params
        
        configurations.append(config)
    
    return configurations


def run_single_experiment(config: Dict[str, Any], 
                         experiment_name: str, 
                         output_dir: str) -> Dict[str, Any]:
    """运行单个实验"""
    
    experiment_id = config.get('experiment_id', 0)
    exp_name = f"{experiment_name}_exp_{experiment_id}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(str(exp_dir), exp_name)
    
    try:
        logger.info(f"Starting experiment {experiment_id}: {exp_name}")
        logger.info(f"Experiment parameters: {config.get('experiment_params', {})}")
        
        # 更新输出目录
        config['output_dir'] = str(exp_dir)
        
        # 保存配置
        config_path = exp_dir / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 运行训练
        training_results = run_training_experiment(
            config=config,
            experiment_name=exp_name,
            output_dir=str(exp_dir),
            logger=logger
        )
        
        # 运行评估
        model_path = training_results['training_results']['best_model_path']
        evaluation_results = run_evaluation_experiment(
            config=config,
            model_path=model_path,
            experiment_name=exp_name,
            output_dir=str(exp_dir),
            logger=logger
        )
        
        # 合并结果
        results = {
            **training_results,
            **evaluation_results,
            'experiment_id': experiment_id,
            'experiment_name': exp_name
        }
        
        # 创建可视化
        create_visualizations(results, str(exp_dir), logger)
        
        # 保存结果
        results_path = exp_dir / "final_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Experiment {experiment_id} completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {str(e)}")
        return {
            'experiment_id': experiment_id,
            'experiment_name': exp_name,
            'error': str(e),
            'status': 'failed'
        }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """聚合实验结果"""
    
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    if not successful_results:
        return {
            'summary': {
                'total_experiments': len(results),
                'successful_experiments': 0,
                'failed_experiments': len(failed_results)
            },
            'failed_experiments': failed_results
        }
    
    # 提取关键指标
    comparison_data = []
    
    for result in successful_results:
        experiment_params = result.get('config', {}).get('experiment_params', {})
        metrics = result.get('metrics', {})
        training_results = result.get('training_results', {})
        
        # 创建比较记录
        record = {
            'experiment_id': result['experiment_id'],
            'experiment_name': result['experiment_name'],
            'experiment_params': experiment_params,
            
            # 训练指标
            'best_val_loss': training_results.get('best_val_loss', float('inf')),
            'total_epochs': training_results.get('total_epochs', 0),
            'total_steps': training_results.get('total_steps', 0),
            
            # 评估指标
            'exact_match_accuracy': metrics.get('video_qa', {}).get('exact_match_accuracy', 0),
            'rouge_l': metrics.get('video_qa', {}).get('rougeL', 0),
            'bleu': metrics.get('video_qa', {}).get('bleu', 0),
            'f1': metrics.get('video_qa', {}).get('f1', 0),
            
            # NTLBG指标
            'coverage_score': metrics.get('ntlbg', {}).get('coverage_score', {}).get('mean', 0),
            'diversity_score': metrics.get('ntlbg', {}).get('diversity_score', {}).get('mean', 0),
            'query_relevance': metrics.get('ntlbg', {}).get('query_relevance', {}).get('mean', 0),
            'attention_efficiency': metrics.get('ntlbg', {}).get('attention_efficiency', {}).get('mean', 0),
            
            # 综合指标
            'overall_performance': metrics.get('summary', {}).get('overall_performance', 0),
            'generation_quality': metrics.get('summary', {}).get('generation_quality', 0),
            'ntlbg_effectiveness': metrics.get('summary', {}).get('ntlbg_effectiveness', 0),
            'efficiency': metrics.get('summary', {}).get('efficiency', 0)
        }
        
        comparison_data.append(record)
    
    # 创建排名
    df = pd.DataFrame(comparison_data)
    
    # 按不同指标排名
    rankings = {}
    key_metrics = ['exact_match_accuracy', 'rouge_l', 'bleu', 'f1', 'overall_performance']
    
    for metric in key_metrics:
        if metric in df.columns:
            df_sorted = df.sort_values(metric, ascending=False)
            rankings[metric] = df_sorted[['experiment_id', 'experiment_name', metric]].to_dict('records')
    
    # 统计信息
    summary = {
        'total_experiments': len(results),
        'successful_experiments': len(successful_results),
        'failed_experiments': len(failed_results),
        'best_experiment': {
            'by_exact_match': rankings.get('exact_match_accuracy', [{}])[0],
            'by_rouge_l': rankings.get('rouge_l', [{}])[0],
            'by_overall_performance': rankings.get('overall_performance', [{}])[0]
        },
        'average_metrics': {
            metric: df[metric].mean() if metric in df.columns else 0
            for metric in key_metrics
        },
        'std_metrics': {
            metric: df[metric].std() if metric in df.columns else 0
            for metric in key_metrics
        }
    }
    
    return {
        'summary': summary,
        'comparison_data': comparison_data,
        'rankings': rankings,
        'failed_experiments': failed_results,
        'detailed_results': successful_results
    }


def create_comparison_visualizations(aggregated_results: Dict[str, Any], 
                                   output_dir: str, 
                                   experiment_name: str):
    """创建比较可视化"""
    
    comparison_data = aggregated_results['comparison_data']
    if not comparison_data:
        return
    
    # 创建可视化器
    visualizer = NTLBGVisualizer(
        config={},
        output_dir=os.path.join(output_dir, 'comparison_visualizations')
    )
    
    df = pd.DataFrame(comparison_data)
    
    # 1. 性能对比图
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{experiment_name} - Performance Comparison', fontsize=16, fontweight='bold')
    
    # 主要指标对比
    main_metrics = ['exact_match_accuracy', 'rouge_l', 'bleu', 'f1']
    
    for i, metric in enumerate(main_metrics):
        ax = axes[i//2, i%2]
        if metric in df.columns:
            bars = ax.bar(df['experiment_id'], df[metric], alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Experiment ID')
            ax.set_ylabel('Score')
            
            # 添加数值标签
            for j, (bar, value) in enumerate(zip(bars, df[metric])):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'comparison_visualizations' / f'{experiment_name}_performance_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. NTLBG指标对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{experiment_name} - NTLBG Metrics Comparison', fontsize=16, fontweight='bold')
    
    ntlbg_metrics = ['coverage_score', 'diversity_score', 'query_relevance', 'attention_efficiency']
    
    # 雷达图
    ax1 = axes[0]
    angles = np.linspace(0, 2*np.pi, len(ntlbg_metrics), endpoint=False)
    
    for i, row in df.iterrows():
        values = [row.get(metric, 0) for metric in ntlbg_metrics]
        values_plot = values + [values[0]]
        angles_plot = np.concatenate([angles, [angles[0]]])
        
        ax1.plot(angles_plot, values_plot, 'o-', linewidth=2, alpha=0.7, 
                label=f'Exp {row["experiment_id"]}')
    
    ax1.set_xticks(angles)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in ntlbg_metrics])
    ax1.set_ylim(0, 1)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_title('NTLBG Metrics Radar', fontweight='bold')
    ax1.grid(True)
    
    # 条形图
    ax2 = axes[1]
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(ntlbg_metrics):
        if metric in df.columns:
            ax2.bar(x + i * width, df[metric], width, alpha=0.7, label=metric.replace('_', ' ').title())
    
    ax2.set_xlabel('Experiment ID')
    ax2.set_ylabel('Score')
    ax2.set_title('NTLBG Metrics Bar Chart', fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(df['experiment_id'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'comparison_visualizations' / f'{experiment_name}_ntlbg_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 保存比较表格
    comparison_table_path = Path(output_dir) / 'comparison_visualizations' / f'{experiment_name}_comparison_table.csv'
    df.to_csv(comparison_table_path, index=False)
    
    # 4. 创建排名表格
    rankings_path = Path(output_dir) / 'comparison_visualizations' / f'{experiment_name}_rankings.json'
    with open(rankings_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_results['rankings'], f, indent=2, ensure_ascii=False)
    
    print(f"Comparison visualizations saved to: {Path(output_dir) / 'comparison_visualizations'}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NTLBG-LLM对比实验")
    parser.add_argument("--config", type=str, required=True, help="基础配置文件路径")
    parser.add_argument("--experiment-name", type=str, required=True, help="实验名称")
    parser.add_argument("--output-dir", type=str, default="results/comparison_experiments", help="输出目录")
    parser.add_argument("--parallel", type=int, default=1, help="并行实验数量")
    
    # 实验参数范围
    parser.add_argument("--num-representatives", type=str, default="3,6,9", help="代表点数量（逗号分隔）")
    parser.add_argument("--ntlbg-weights", type=str, default="0.1,0.5,1.0", help="NTLBG损失权重（逗号分隔）")
    parser.add_argument("--learning-rates", type=str, default="1e-4,5e-4,1e-3", help="学习率（逗号分隔）")
    parser.add_argument("--batch-sizes", type=str, default="4,8", help="批次大小（逗号分隔）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(str(experiment_dir), args.experiment_name)
    
    try:
        # 加载基础配置
        base_config = load_config(args.config)
        
        # 解析实验参数
        num_representatives = [int(x) for x in args.num_representatives.split(',')]
        ntlbg_weights = [float(x) for x in args.ntlbg_weights.split(',')]
        learning_rates = [float(x) for x in args.learning_rates.split(',')]
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
        
        # 创建实验参数组合
        experiment_params = []
        for num_rep in num_representatives:
            for ntlbg_weight in ntlbg_weights:
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        params = {
                            'model_config': {'num_representative_points': num_rep},
                            'loss_weights': {'ntlbg': ntlbg_weight},
                            'training_config': {
                                'learning_rate': lr,
                                'batch_size': batch_size
                            }
                        }
                        experiment_params.append(params)
        
        logger.info(f"Created {len(experiment_params)} experiment configurations")
        
        # 创建实验配置
        configurations = create_experiment_configurations(base_config, experiment_params)
        
        # 保存实验配置
        configs_path = experiment_dir / "experiment_configurations.json"
        with open(configs_path, 'w', encoding='utf-8') as f:
            json.dump(configurations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Experiment configurations saved to: {configs_path}")
        
        # 运行实验
        results = []
        
        if args.parallel > 1:
            # 并行运行
            logger.info(f"Running {len(configurations)} experiments in parallel (max_workers={args.parallel})")
            
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                future_to_config = {
                    executor.submit(run_single_experiment, config, args.experiment_name, str(experiment_dir)): config
                    for config in configurations
                }
                
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed experiment {result.get('experiment_id', 'unknown')}")
                    except Exception as e:
                        logger.error(f"Experiment failed: {str(e)}")
                        results.append({
                            'experiment_id': config.get('experiment_id', 'unknown'),
                            'error': str(e),
                            'status': 'failed'
                        })
        else:
            # 顺序运行
            logger.info(f"Running {len(configurations)} experiments sequentially")
            
            for config in configurations:
                try:
                    result = run_single_experiment(config, args.experiment_name, str(experiment_dir))
                    results.append(result)
                    logger.info(f"Completed experiment {result.get('experiment_id', 'unknown')}")
                except Exception as e:
                    logger.error(f"Experiment {config.get('experiment_id', 'unknown')} failed: {str(e)}")
                    results.append({
                        'experiment_id': config.get('experiment_id', 'unknown'),
                        'error': str(e),
                        'status': 'failed'
                    })
        
        # 聚合结果
        logger.info("Aggregating results...")
        aggregated_results = aggregate_results(results)
        
        # 保存聚合结果
        aggregated_path = experiment_dir / "aggregated_results.json"
        with open(aggregated_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Aggregated results saved to: {aggregated_path}")
        
        # 创建比较可视化
        logger.info("Creating comparison visualizations...")
        create_comparison_visualizations(aggregated_results, str(experiment_dir), args.experiment_name)
        
        # 打印摘要
        summary = aggregated_results['summary']
        logger.info(f"\n{'='*50}")
        logger.info(f"COMPARISON EXPERIMENT SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total experiments: {summary['total_experiments']}")
        logger.info(f"Successful experiments: {summary['successful_experiments']}")
        logger.info(f"Failed experiments: {summary['failed_experiments']}")
        
        if summary['successful_experiments'] > 0:
            logger.info(f"\nBest performing experiments:")
            for metric, best_exp in summary['best_experiment'].items():
                if best_exp:
                    logger.info(f"  {metric}: Experiment {best_exp.get('experiment_id', 'unknown')} "
                               f"(Score: {best_exp.get(metric.replace('by_', ''), 0):.4f})")
            
            logger.info(f"\nAverage metrics:")
            for metric, avg_value in summary['average_metrics'].items():
                logger.info(f"  {metric}: {avg_value:.4f} ± {summary['std_metrics'][metric]:.4f}")
        
        logger.info(f"{'='*50}")
        logger.info(f"Comparison experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Comparison experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 