# NTLBG-LLM项目管理Makefile
# 用法: make <target>

.PHONY: setup install train evaluate test clean help

# 默认目标
help:
	@echo "NTLBG-LLM项目管理命令："
	@echo "  setup       - 设置项目环境"
	@echo "  install     - 安装依赖包"
	@echo "  train       - 训练模型"
	@echo "  train-debug - 调试模式训练"
	@echo "  evaluate    - 评估模型"
	@echo "  test        - 运行测试"
	@echo "  clean       - 清理输出文件"
	@echo "  demo        - 运行模型演示"
	@echo "  format      - 格式化代码"

# 设置环境
setup:
	@echo "🚀 设置NTLBG-LLM环境..."
	conda create -n ntlbg-llm python=3.9 -y || echo "环境可能已存在"
	conda activate ntlbg-llm
	@echo "环境设置完成！"

# 安装依赖
install:
	@echo "📦 安装依赖包..."
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install transformers>=4.30.0
	pip install opencv-python
	pip install matplotlib seaborn
	pip install tqdm
	pip install scikit-learn
	pip install numpy
	pip install Pillow
	pip install datasets
	@echo "依赖包安装完成！"

# 创建必要目录
create-dirs:
	@echo "📁 创建项目目录..."
	mkdir -p data/videos
	mkdir -p outputs
	mkdir -p results
	mkdir -p logs
	@echo "目录创建完成！"

# 训练模型
train: create-dirs
	@echo "🏋️ 开始训练NTLBG-LLM模型..."
	python scripts/train_ntlbg.py --config configs/ntlbg_base_config.json

# 调试模式训练
train-debug: create-dirs
	@echo "🐛 调试模式训练..."
	python scripts/train_ntlbg.py --config configs/ntlbg_base_config.json --debug

# 干运行（验证配置）
dry-run: create-dirs
	@echo "🔍 验证配置和数据..."
	python scripts/train_ntlbg.py --config configs/ntlbg_base_config.json --dry_run

# 评估模型
evaluate:
	@echo "📊 评估模型性能..."
	python scripts/evaluate_ntlbg.py \
		--model_path outputs/ntlbg_baseline/best_model.pt \
		--data_path data/test.jsonl \
		--output_dir results/evaluation

# 测试模块
test:
	@echo "🧪 运行模块测试..."
	python -m pytest tests/ -v || echo "测试需要pytest：pip install pytest"

# 运行演示
demo:
	@echo "🎬 运行NTLBG-LLM演示..."
	python scripts/demo.py --config configs/ntlbg_base_config.json

# 格式化代码
format:
	@echo "✨ 格式化代码..."
	black src/ scripts/ --line-length 88 || echo "需要安装black：pip install black"
	isort src/ scripts/ || echo "需要安装isort：pip install isort"

# 清理输出文件
clean:
	@echo "🧹 清理输出文件..."
	rm -rf outputs/*
	rm -rf results/*
	rm -rf logs/*
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "清理完成！"

# 清理所有（包括模型）
clean-all: clean
	@echo "🧹 完全清理..."
	rm -rf data/videos/*
	@echo "完全清理完成！"

# 下载示例数据
download-data:
	@echo "⬇️ 下载示例数据..."
	@echo "创建示例视频文件（占位符）..."
	for i in {1..10}; do \
		touch data/videos/video_00$$i.mp4; \
	done
	for i in {1..5}; do \
		touch data/videos/video_val_00$$i.mp4; \
	done
	for i in {1..5}; do \
		touch data/videos/video_test_00$$i.mp4; \
	done
	@echo "示例数据创建完成！"

# 检查环境
check-env:
	@echo "🔍 检查环境..."
	@echo "Python版本:"
	python --version
	@echo "PyTorch版本:"
	python -c "import torch; print(torch.__version__)" || echo "❌ PyTorch未安装"
	@echo "CUDA可用性:"
	python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" || echo "❌ 无法检查CUDA"
	@echo "GPU信息:"
	python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')" || echo "❌ 无法检查GPU"

# 生成项目报告
report:
	@echo "📋 生成项目报告..."
	@echo "项目结构:" > project_report.txt
	find . -name "*.py" -o -name "*.json" -o -name "*.md" | grep -v __pycache__ | sort >> project_report.txt
	@echo "" >> project_report.txt
	@echo "代码统计:" >> project_report.txt
	find . -name "*.py" -exec wc -l {} + | tail -1 >> project_report.txt
	@echo "报告生成完成: project_report.txt"

# 启动tensorboard
tensorboard:
	@echo "📊 启动TensorBoard..."
	tensorboard --logdir=logs --port=6006 || echo "需要安装tensorboard：pip install tensorboard"

# 完整的设置流程
setup-all: setup install create-dirs download-data
	@echo "✅ 完整环境设置完成！"
	@echo "现在可以运行："
	@echo "  make train-debug  # 调试模式训练"
	@echo "  make train        # 正式训练"

# 快速测试
quick-test:
	@echo "⚡ 快速测试..."
	python -c "from src.models.ntlbg_attention import NTLBGAttention; print('✅ NTLBG模块导入成功')"
	python -c "from src.models.rich_points import RichRepresentativePointConstructor; print('✅ 富代表点模块导入成功')"
	python -c "from src.models.ntlbg_llm import create_ntlbg_llm; print('✅ 主模型导入成功')"
	python -c "from src.data.video_loader import VideoLoader; print('✅ 视频加载器导入成功')"
	python -c "from src.data.datasets import VideoQADataset; print('✅ 数据集导入成功')"
	@echo "✅ 所有模块测试通过！"

# 开发者工具
dev-setup:
	@echo "👨‍💻 设置开发环境..."
	pip install black isort flake8 pytest
	pip install jupyter notebook
	@echo "开发工具安装完成！"

# 启动Jupyter
jupyter:
	@echo "📔 启动Jupyter Notebook..."
	jupyter notebook --port=8888 --no-browser

# 打包项目
package:
	@echo "📦 打包项目..."
	tar -czf ntlbg-llm-$(shell date +%Y%m%d).tar.gz \
		--exclude='outputs' \
		--exclude='results' \
		--exclude='logs' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.git' \
		.
	@echo "项目打包完成！" 