#!/bin/bash

echo "🎯 启动AAAI 2026论文实验"
echo "================================"

# 确保环境激活
conda activate ntlbg-llm

# 检查GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# 检查数据集
echo "📊 检查数据集..."
for dataset in longvideobench video_mme mlvu; do
    if [ -d "data/$dataset" ]; then
        size=$(du -sh "data/$dataset" 2>/dev/null | cut -f1)
        echo "✅ $dataset: $size"
    else
        echo "❌ $dataset: 不存在"
    fi
done

# 运行论文实验
echo "🚀 开始论文实验..."
python run_paper_experiments.py

echo "✅ 论文实验完成！"
echo "📁 结果保存在: paper_results/"
