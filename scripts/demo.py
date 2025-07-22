#!/usr/bin/env python3
"""
NTLBG-LLM演示脚本
展示模型的核心功能和代表点选择效果
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ntlbg_llm import create_ntlbg_llm
from src.models.ntlbg_attention import NTLBGAttention
from src.models.rich_points import RichRepresentativePointConstructor
from src.data.video_loader import VideoLoader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="NTLBG-LLM模型演示")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/ntlbg_base_config.json",
        help="配置文件路径"
    )
    parser.add_argument(
        "--save_plots", 
        action="store_true",
        help="保存可视化图片"
    )
    return parser.parse_args()


def demo_ntlbg_attention():
    """演示NTLBG注意力机制"""
    print("🎯 演示NTLBG注意力机制")
    print("=" * 50)
    
    # 创建模拟数据
    batch_size, T, d_model = 2, 50, 768
    video_features = torch.randn(batch_size, T, d_model)
    query_embedding = torch.randn(batch_size, d_model)
    
    # 初始化NTLBG注意力
    ntlbg_attention = NTLBGAttention(
        d_model=d_model,
        d_query=d_model,
        num_representatives=6
    )
    
    # 前向传播
    results = ntlbg_attention(video_features, query_embedding, return_stats=True)
    
    print(f"📊 输入视频特征形状: {video_features.shape}")
    print(f"📊 查询嵌入形状: {query_embedding.shape}")
    print(f"📊 代表点索引: {results['representative_indices'][0]}")
    print(f"📊 代表点权重: {results['weights'][0]}")
    print(f"📊 选择的帧: {results['representative_indices'][0].tolist()}")
    
    # 计算NTLBG约束损失
    constraint_loss = ntlbg_attention.compute_ntlbg_constraint_loss(
        results['representative_features'],
        results['mu_q'],
        results['sigma_q']
    )
    print(f"📊 NTLBG约束损失: {constraint_loss.item():.6f}")
    
    return results


def demo_rich_points(ntlbg_results):
    """演示富代表点构造"""
    print("\n🌟 演示富代表点构造")
    print("=" * 50)
    
    # 创建模拟数据
    batch_size, T, d_visual = 2, 50, 768
    video_features = torch.randn(batch_size, T, d_visual)
    representative_indices = ntlbg_results['representative_indices']
    
    # 初始化富代表点构造器
    rich_constructor = RichRepresentativePointConstructor(
        d_visual=d_visual,
        d_context=256,
        d_temporal=64
    )
    
    # 构造富代表点
    rich_results = rich_constructor(
        video_features=video_features,
        representative_indices=representative_indices
    )
    
    print(f"📊 富代表点特征形状: {rich_results['rich_features'].shape}")
    print(f"📊 上下文特征形状: {rich_results['context_features'].shape}")
    print(f"📊 代表性权重: {rich_results['representativeness_weights'][0]}")
    print(f"📊 覆盖范围: {rich_results['coverage_ranges'][0]}")
    
    # 计算损失
    info_loss = rich_constructor.compute_information_preservation_loss(
        video_features, rich_results['rich_features'], representative_indices
    )
    temporal_loss = rich_constructor.compute_temporal_coherence_loss(
        rich_results['rich_features'], representative_indices
    )
    
    print(f"📊 信息保持损失: {info_loss.item():.6f}")
    print(f"📊 时序连贯性损失: {temporal_loss.item():.6f}")
    
    return rich_results


def demo_full_model(config):
    """演示完整的NTLBG-LLM模型"""
    print("\n🚀 演示完整NTLBG-LLM模型")
    print("=" * 50)
    
    # 创建模拟数据
    batch_size = 2
    T = 30  # 视频帧数
    seq_len = 20  # 文本长度
    vocab_size = 32000
    
    # 模拟视频帧
    video_frames = torch.randn(batch_size, T, 3, 224, 224)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 创建模型
    model = create_ntlbg_llm(config['model_config'])
    
    print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"📊 可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(
            video_frames=video_frames,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_ntlbg_stats=True
        )
    
    print(f"📊 输出logits形状: {outputs['logits'].shape}")
    print(f"📊 总损失: {outputs['loss'].item():.6f}")
    print(f"📊 损失组件:")
    for name, loss in outputs['loss_components'].items():
        print(f"   {name}: {loss.item():.6f}")
    
    print(f"📊 代表点索引: {outputs['representative_indices'][0]}")
    print(f"📊 代表点权重: {outputs['representative_weights'][0]}")
    
    # 生成测试
    print("\n🎬 测试生成功能...")
    generation_outputs = model.generate(
        video_frames=video_frames,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False
    )
    
    print(f"📊 生成的token形状: {generation_outputs['generated_ids'].shape}")
    print(f"📊 生成的tokens: {generation_outputs['generated_ids'][0]}")
    
    return outputs


def visualize_representative_selection(ntlbg_results, save_plots=False):
    """可视化代表点选择结果"""
    print("\n📊 可视化代表点选择")
    print("=" * 50)
    
    # 提取第一个batch的结果
    indices = ntlbg_results['representative_indices'][0].cpu().numpy()
    weights = ntlbg_results['weights'][0].cpu().numpy()
    
    # 创建时序图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制代表点分布
    video_length = 50  # 假设视频长度
    frames = np.arange(video_length)
    
    # 所有帧
    ax1.scatter(frames, np.ones_like(frames), alpha=0.3, s=20, label='All frames')
    
    # 代表点
    ax1.scatter(indices, np.ones_like(indices), c='red', s=100, marker='*', label='Representative points')
    
    # 添加权重信息
    for i, (idx, weight) in enumerate(zip(indices, weights)):
        ax1.annotate(f'{idx}\n({weight:.3f})', 
                    (idx, 1), 
                    xytext=(0, 20), 
                    textcoords='offset points', 
                    ha='center', 
                    fontsize=8)
    
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Selection')
    ax1.set_title('NTLBG Representative Point Selection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制权重分布
    ax2.bar(range(len(weights)), weights, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Representative Point Index')
    ax2.set_ylabel('Weight')
    ax2.set_title('Representative Point Weights')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('representative_selection.png', dpi=300, bbox_inches='tight')
        print("📊 可视化结果已保存到 representative_selection.png")
    else:
        plt.show()
    
    plt.close()


def performance_benchmark():
    """性能基准测试"""
    print("\n⚡ 性能基准测试")
    print("=" * 50)
    
    import time
    
    # 测试不同视频长度的性能
    video_lengths = [20, 50, 100, 200]
    batch_size = 2
    d_model = 768
    
    ntlbg_attention = NTLBGAttention(
        d_model=d_model,
        d_query=d_model,
        num_representatives=6
    )
    
    for T in video_lengths:
        video_features = torch.randn(batch_size, T, d_model)
        query_embedding = torch.randn(batch_size, d_model)
        
        # 预热
        for _ in range(5):
            _ = ntlbg_attention(video_features, query_embedding)
        
        # 计时
        start_time = time.time()
        for _ in range(10):
            _ = ntlbg_attention(video_features, query_embedding)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"📊 视频长度 {T}: {avg_time:.4f}s 每次前向传播")
    
    print("📊 性能测试完成!")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("🎬 NTLBG-LLM模型演示")
    print("=" * 60)
    print(f"📋 配置文件: {args.config}")
    print(f"📋 PyTorch版本: {torch.__version__}")
    print(f"📋 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 1. 演示NTLBG注意力机制
    ntlbg_results = demo_ntlbg_attention()
    
    # 2. 演示富代表点构造
    rich_results = demo_rich_points(ntlbg_results)
    
    # 3. 演示完整模型
    model_outputs = demo_full_model(config)
    
    # 4. 可视化结果
    visualize_representative_selection(ntlbg_results, save_plots=args.save_plots)
    
    # 5. 性能基准测试
    performance_benchmark()
    
    print("\n✅ 演示完成！")
    print("🎯 NTLBG-LLM模型核心功能演示完毕")
    print("📊 关键特性:")
    print("   - 基于统计学理论的代表点选择")
    print("   - 富代表点的时空上下文信息")
    print("   - 多任务损失函数优化")
    print("   - 端到端的视频问答能力")


if __name__ == "__main__":
    main() 