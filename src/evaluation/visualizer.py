"""
NTLBG-LLM结果可视化模块
用于可视化NTLBG模型的训练和评估结果
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime


class NTLBGVisualizer:
    """NTLBG可视化器"""
    
    def __init__(self, config: Dict, output_dir: str = "results/visualizations"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 颜色配置
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#7209B7',
            'light': '#F8F9FA',
            'dark': '#212529'
        }
    
    def visualize_training_progress(self, 
                                  train_losses: List[Dict[str, float]], 
                                  val_losses: List[Dict[str, float]],
                                  learning_rates: List[float],
                                  save_name: str = "training_progress"):
        """可视化训练进度"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NTLBG-LLM Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. 总损失
        ax1 = axes[0, 0]
        train_total = [loss.get('total', 0) for loss in train_losses]
        val_total = [loss.get('total', 0) for loss in val_losses]
        
        ax1.plot(epochs, train_total, label='Train', color=self.colors['primary'], linewidth=2)
        ax1.plot(epochs, val_total, label='Validation', color=self.colors['secondary'], linewidth=2)
        ax1.set_title('Total Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 分解损失
        ax2 = axes[0, 1]
        loss_components = ['task', 'ntlbg', 'alignment', 'temporal', 'info']
        for i, component in enumerate(loss_components):
            train_component = [loss.get(component, 0) for loss in train_losses]
            if any(x > 0 for x in train_component):  # 只显示非零损失
                ax2.plot(epochs, train_component, label=f'{component.capitalize()}', linewidth=2)
        
        ax2.set_title('Loss Components (Training)', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 学习率
        ax3 = axes[1, 0]
        ax3.plot(learning_rates, color=self.colors['accent'], linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. NTLBG损失详细
        ax4 = axes[1, 1]
        train_ntlbg = [loss.get('ntlbg', 0) for loss in train_losses]
        val_ntlbg = [loss.get('ntlbg', 0) for loss in val_losses]
        
        ax4.plot(epochs, train_ntlbg, label='Train NTLBG', color=self.colors['info'], linewidth=2)
        ax4.plot(epochs, val_ntlbg, label='Val NTLBG', color=self.colors['success'], linewidth=2)
        ax4.set_title('NTLBG Constraint Loss', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training progress visualization saved to: {save_path}")
    
    def visualize_representative_points(self, 
                                      video_features: torch.Tensor,
                                      representative_points: torch.Tensor,
                                      attention_weights: torch.Tensor,
                                      query_text: str = "",
                                      save_name: str = "representative_points"):
        """可视化代表点选择"""
        
        # 降维到2D用于可视化
        features_2d = self._reduce_dimensions(video_features, method='pca')
        rep_points_2d = self._reduce_dimensions(representative_points, method='pca')
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Representative Points Selection\nQuery: {query_text}', fontsize=14, fontweight='bold')
        
        # 1. 特征空间中的代表点
        ax1 = axes[0]
        
        # 绘制所有视频帧
        scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=attention_weights.cpu().numpy(),
                            cmap='viridis', alpha=0.6, s=50)
        
        # 绘制代表点
        ax1.scatter(rep_points_2d[:, 0], rep_points_2d[:, 1], 
                   c='red', s=200, marker='*', 
                   edgecolors='black', linewidth=2, 
                   label='Representative Points')
        
        ax1.set_title('Feature Space Distribution', fontweight='bold')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend()
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax1, label='Attention Weight')
        
        # 2. 注意力权重分布
        ax2 = axes[1]
        
        frame_indices = range(len(attention_weights))
        bars = ax2.bar(frame_indices, attention_weights.cpu().numpy(), 
                      color=self.colors['primary'], alpha=0.7)
        
        # 高亮代表点
        rep_indices = torch.topk(attention_weights, k=len(representative_points))[1]
        for idx in rep_indices:
            bars[idx].set_color(self.colors['accent'])
            bars[idx].set_alpha(1.0)
        
        ax2.set_title('Attention Weights Distribution', fontweight='bold')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Attention Weight')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Representative points visualization saved to: {save_path}")
    
    def visualize_distribution_ellipsoids(self, 
                                        representative_points: torch.Tensor,
                                        distribution_mean: torch.Tensor,
                                        distribution_cov: torch.Tensor,
                                        save_name: str = "distribution_ellipsoids"):
        """可视化分布椭球"""
        
        # 降维到2D
        points_2d = self._reduce_dimensions(representative_points, method='pca')
        mean_2d = self._reduce_dimensions(distribution_mean.unsqueeze(0), method='pca').squeeze(0)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制代表点
        ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                  c=self.colors['primary'], s=100, alpha=0.8, 
                  label='Representative Points')
        
        # 绘制均值点
        ax.scatter(mean_2d[0], mean_2d[1], 
                  c=self.colors['accent'], s=200, marker='x', 
                  linewidth=3, label='Distribution Mean')
        
        # 绘制等高椭球（近似）
        # 由于PCA降维，这里是近似的椭球投影
        try:
            # 计算2D协方差矩阵（近似）
            cov_2d = np.cov(points_2d.T)
            
            # 计算椭球参数
            eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
            
            # 不同置信度的椭球
            confidence_levels = [0.5, 0.8, 0.95]
            colors = [self.colors['info'], self.colors['secondary'], self.colors['success']]
            
            for i, (conf, color) in enumerate(zip(confidence_levels, colors)):
                # 卡方分布临界值
                chi2_val = 2.0 * np.log(1.0 / (1.0 - conf))
                
                # 椭球半轴长度
                a = np.sqrt(chi2_val * eigenvals[0])
                b = np.sqrt(chi2_val * eigenvals[1])
                
                # 椭球角度
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # 创建椭球
                ellipse = Ellipse(xy=mean_2d, width=2*a, height=2*b, 
                                angle=angle, facecolor=color, alpha=0.2, 
                                edgecolor=color, linewidth=2)
                ax.add_patch(ellipse)
                
                if i == 0:  # 只在第一个椭球上添加标签
                    ellipse.set_label('Confidence Ellipsoids')
        
        except Exception as e:
            print(f"Warning: Could not draw ellipsoids: {e}")
        
        ax.set_title('Distribution Ellipsoids of Representative Points', fontweight='bold', fontsize=14)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution ellipsoids visualization saved to: {save_path}")
    
    def visualize_evaluation_metrics(self, 
                                   metrics: Dict[str, Any],
                                   save_name: str = "evaluation_metrics"):
        """可视化评估指标"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NTLBG-LLM Evaluation Metrics', fontsize=16, fontweight='bold')
        
        # 1. 主要指标雷达图
        ax1 = axes[0, 0]
        
        main_metrics = ['exact_match_accuracy', 'rougeL', 'bleu', 'f1']
        values = [metrics.get(metric, 0) for metric in main_metrics]
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(main_metrics), endpoint=False)
        values_plot = values + [values[0]]  # 闭合图形
        angles_plot = np.concatenate([angles, [angles[0]]])
        
        ax1.plot(angles_plot, values_plot, 'o-', linewidth=2, color=self.colors['primary'])
        ax1.fill(angles_plot, values_plot, alpha=0.25, color=self.colors['primary'])
        ax1.set_xticks(angles)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in main_metrics])
        ax1.set_ylim(0, 1)
        ax1.set_title('Main Metrics', fontweight='bold')
        ax1.grid(True)
        
        # 2. 问题类型准确率
        ax2 = axes[0, 1]
        
        if 'by_question_type' in metrics:
            type_data = metrics['by_question_type']
            types = list(type_data.keys())
            accuracies = [type_data[t].get('exact_match', {}).get('mean', 0) for t in types]
            
            bars = ax2.bar(types, accuracies, color=self.colors['secondary'], alpha=0.7)
            ax2.set_title('Accuracy by Question Type', fontweight='bold')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # 3. NTLBG特定指标
        ax3 = axes[1, 0]
        
        if 'ntlbg' in metrics:
            ntlbg_metrics = metrics['ntlbg']
            ntlbg_names = list(ntlbg_metrics.keys())
            ntlbg_values = [ntlbg_metrics[name].get('mean', 0) for name in ntlbg_names]
            
            bars = ax3.bar(ntlbg_names, ntlbg_values, color=self.colors['info'], alpha=0.7)
            ax3.set_title('NTLBG Specific Metrics', fontweight='bold')
            ax3.set_ylabel('Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, val in zip(bars, ntlbg_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # 4. 综合性能指标
        ax4 = axes[1, 1]
        
        if 'summary' in metrics:
            summary = metrics['summary']
            summary_names = list(summary.keys())
            summary_values = list(summary.values())
            
            bars = ax4.bar(summary_names, summary_values, color=self.colors['accent'], alpha=0.7)
            ax4.set_title('Summary Metrics', fontweight='bold')
            ax4.set_ylabel('Score')
            ax4.tick_params(axis='x', rotation=45)
            ax4.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, val in zip(bars, summary_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation metrics visualization saved to: {save_path}")
    
    def create_interactive_dashboard(self, 
                                   train_losses: List[Dict[str, float]],
                                   val_losses: List[Dict[str, float]],
                                   metrics: Dict[str, Any],
                                   save_name: str = "interactive_dashboard"):
        """创建交互式仪表板"""
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Loss', 'Metrics by Type', 'NTLBG Metrics'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}], 
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        epochs = list(range(1, len(train_losses) + 1))
        
        # 1. 训练损失
        train_total = [loss.get('total', 0) for loss in train_losses]
        train_ntlbg = [loss.get('ntlbg', 0) for loss in train_losses]
        
        fig.add_trace(
            go.Scatter(x=epochs, y=train_total, name='Total Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=train_ntlbg, name='NTLBG Loss', line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        # 2. 验证损失
        val_total = [loss.get('total', 0) for loss in val_losses]
        val_ntlbg = [loss.get('ntlbg', 0) for loss in val_losses]
        
        fig.add_trace(
            go.Scatter(x=epochs, y=val_total, name='Val Total', line=dict(color='lightblue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_ntlbg, name='Val NTLBG', line=dict(color='lightcoral')),
            row=1, col=2, secondary_y=True
        )
        
        # 3. 分类型指标
        if 'by_question_type' in metrics:
            type_data = metrics['by_question_type']
            types = list(type_data.keys())
            accuracies = [type_data[t].get('exact_match', {}).get('mean', 0) for t in types]
            
            fig.add_trace(
                go.Bar(x=types, y=accuracies, name='Accuracy by Type'),
                row=2, col=1
            )
        
        # 4. NTLBG指标
        if 'ntlbg' in metrics:
            ntlbg_metrics = metrics['ntlbg']
            ntlbg_names = list(ntlbg_metrics.keys())
            ntlbg_values = [ntlbg_metrics[name].get('mean', 0) for name in ntlbg_names]
            
            fig.add_trace(
                go.Bar(x=ntlbg_names, y=ntlbg_values, name='NTLBG Metrics'),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="NTLBG-LLM Interactive Dashboard",
            showlegend=True,
            height=700
        )
        
        # 保存HTML文件
        save_path = self.output_dir / f"{save_name}.html"
        fig.write_html(str(save_path))
        
        print(f"Interactive dashboard saved to: {save_path}")
    
    def visualize_attention_heatmap(self, 
                                  attention_weights: torch.Tensor,
                                  video_frames: Optional[List[np.ndarray]] = None,
                                  save_name: str = "attention_heatmap"):
        """可视化注意力热力图"""
        
        # 转换为numpy数组
        attention = attention_weights.cpu().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 创建热力图
        im = ax.imshow(attention.reshape(1, -1), cmap='hot', aspect='auto')
        
        # 设置标签
        ax.set_title('Attention Weights Heatmap', fontweight='bold', fontsize=14)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Attention Weight')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        # 如果有视频帧，可以添加关键帧预览
        if video_frames is not None:
            # 选择注意力最高的几帧
            top_k = min(5, len(video_frames))
            top_indices = np.argsort(attention)[-top_k:]
            
            # 在下方添加关键帧预览
            fig.set_size_inches(12, 10)
            
            for i, idx in enumerate(top_indices):
                ax_frame = fig.add_subplot(3, top_k, 2*top_k + i + 1)
                ax_frame.imshow(video_frames[idx])
                ax_frame.set_title(f'Frame {idx}\nWeight: {attention[idx]:.3f}')
                ax_frame.axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention heatmap visualization saved to: {save_path}")
    
    def _reduce_dimensions(self, 
                          features: torch.Tensor, 
                          method: str = 'pca',
                          n_components: int = 2) -> np.ndarray:
        """降维到2D用于可视化"""
        
        # 转换为numpy数组
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features
        
        # 展平为2D
        if features_np.ndim > 2:
            features_np = features_np.reshape(features_np.shape[0], -1)
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        reduced = reducer.fit_transform(features_np)
        return reduced
    
    def save_comparison_table(self, 
                            results: Dict[str, Any],
                            save_name: str = "comparison_table"):
        """保存比较表格"""
        
        # 创建比较数据
        comparison_data = []
        
        if 'video_qa' in results:
            qa_metrics = results['video_qa']
            comparison_data.append({
                'Metric': 'Exact Match Accuracy',
                'Value': f"{qa_metrics.get('exact_match_accuracy', 0):.4f}",
                'Category': 'Video QA'
            })
            comparison_data.append({
                'Metric': 'ROUGE-L',
                'Value': f"{qa_metrics.get('rougeL', 0):.4f}",
                'Category': 'Video QA'
            })
            comparison_data.append({
                'Metric': 'BLEU',
                'Value': f"{qa_metrics.get('bleu', 0):.4f}",
                'Category': 'Video QA'
            })
            comparison_data.append({
                'Metric': 'F1 Score',
                'Value': f"{qa_metrics.get('f1', 0):.4f}",
                'Category': 'Video QA'
            })
        
        if 'ntlbg' in results:
            ntlbg_metrics = results['ntlbg']
            for metric_name, metric_data in ntlbg_metrics.items():
                comparison_data.append({
                    'Metric': metric_name.replace('_', ' ').title(),
                    'Value': f"{metric_data.get('mean', 0):.4f} ± {metric_data.get('std', 0):.4f}",
                    'Category': 'NTLBG'
                })
        
        # 创建DataFrame
        df = pd.DataFrame(comparison_data)
        
        # 保存为CSV
        csv_path = self.output_dir / f"{save_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # 保存为HTML表格
        html_path = self.output_dir / f"{save_name}.html"
        df.to_html(html_path, index=False, table_id='comparison_table')
        
        print(f"Comparison table saved to: {csv_path} and {html_path}")
    
    def create_comprehensive_report(self, 
                                  train_losses: List[Dict[str, float]],
                                  val_losses: List[Dict[str, float]],
                                  metrics: Dict[str, Any],
                                  config: Dict[str, Any],
                                  save_name: str = "comprehensive_report"):
        """创建综合报告"""
        
        # 生成所有可视化
        self.visualize_training_progress(train_losses, val_losses, [], 
                                       f"{save_name}_training")
        self.visualize_evaluation_metrics(metrics, f"{save_name}_metrics")
        self.create_interactive_dashboard(train_losses, val_losses, metrics, 
                                        f"{save_name}_dashboard")
        self.save_comparison_table(metrics, f"{save_name}_comparison")
        
        # 创建汇总报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'training_summary': {
                'total_epochs': len(train_losses),
                'final_train_loss': train_losses[-1]['total'] if train_losses else 0,
                'final_val_loss': val_losses[-1]['total'] if val_losses else 0,
                'best_val_loss': min([loss['total'] for loss in val_losses]) if val_losses else 0
            },
            'evaluation_summary': metrics,
            'files_generated': [
                f"{save_name}_training.png",
                f"{save_name}_metrics.png", 
                f"{save_name}_dashboard.html",
                f"{save_name}_comparison.csv"
            ]
        }
        
        # 保存报告
        report_path = self.output_dir / f"{save_name}_summary.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive report saved to: {report_path}")
        print(f"All visualizations saved in: {self.output_dir}")
        
        return report 