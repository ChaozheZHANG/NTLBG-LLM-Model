# NTLBG-LLM Long Video Configuration System

A comprehensive configuration management system for NTLBG-LLM long video understanding tasks.

## Overview

This configuration system provides optimized settings for long video understanding using Novel Temporal Long-form Best-view Grounding (NTLBG). The system automatically adapts to different video lengths and hardware configurations.

## Key Features

- **Adaptive Representative Points**: Automatically calculates optimal number of representative points based on video length
- **GPU-Optimized Configurations**: Supports V100, A100, and RTX series GPUs
- **Multi-Dataset Support**: Configured for LongVideoBench, Video-MME, and MLVU datasets
- **Memory-Efficient**: Optimized for large-scale long video processing
- **Comprehensive Experiments**: Includes ablation studies, SOTA comparisons, and efficiency analysis

## File Structure

```
configs/
├── longvideo_config.json           # Main configuration file
├── experiments/
│   ├── ablation_studies.json       # Ablation experiment configurations
│   ├── sota_comparison.json        # SOTA baseline comparisons
│   ├── efficiency_analysis.json    # Performance analysis experiments
│   └── theoretical_validation.json # Theoretical validation experiments
└── README.md                       # This file
```

## Main Configuration (`longvideo_config.json`)

### Video Configuration
- **max_frames**: 4096 (maximum frames per video)
- **num_representatives**: 512 (representative points extracted)
- **coverage_ratio**: 0.125 (12.5% of frames become representatives)
- **frame_resolution**: [224, 224] (input frame size)

### Model Configuration
- **base_model**: Qwen/Qwen2-VL-7B-Instruct
- **ntlbg_hidden_size**: 4096
- **flash_attention**: Enabled for efficiency
- **gradient_checkpointing**: Enabled for memory optimization

### Training Configuration
- **batch_size**: 1 (with gradient accumulation)
- **gradient_accumulation_steps**: 32
- **learning_rate**: 5e-5
- **fp16**: Enabled for A100/V100 GPUs
- **max_steps**: 10000

## Long Video Categories

### 1. Moderate Long (`moderate_long`)
- **Duration**: 5-10 minutes
- **Max Frames**: 2048
- **Representatives**: 256
- **GPU Requirement**: V100 32GB
- **Batch Size**: 2

### 2. Long (`long`)
- **Duration**: 10-20 minutes
- **Max Frames**: 4096
- **Representatives**: 512
- **GPU Requirement**: A100 40GB
- **Batch Size**: 1

### 3. Very Long (`very_long`)
- **Duration**: 20+ minutes
- **Max Frames**: 8192
- **Representatives**: 1024
- **GPU Requirement**: A100 80GB
- **Batch Size**: 1

## Hardware Configurations

### V100 32GB
- **Memory**: 32GB
- **Recommended**: moderate_long configuration
- **Max Batch Size**: 2
- **Precision**: FP16

### A100 40GB
- **Memory**: 40GB
- **Recommended**: long configuration
- **Max Batch Size**: 1
- **Precision**: BF16/FP16

### A100 80GB
- **Memory**: 80GB
- **Recommended**: very_long configuration
- **Max Batch Size**: 1
- **Precision**: BF16

## Dataset Configuration

### Supported Datasets
1. **LongVideoBench** (40% weight)
   - Focus on long-form video understanding
   - Documentary and lecture videos
   
2. **Video-MME** (30% weight)
   - Multi-modal evaluation
   - Long and medium-form videos
   
3. **MLVU** (30% weight)
   - Multi-modal long video understanding
   - Comprehensive evaluation metrics

## Experiment Configurations

### Ablation Studies (`ablation_studies.json`)
- **NTLBG Constraint Analysis**: Effect of statistical constraints
- **Representative Count Studies**: 256, 512, 1024 representatives
- **Loss Weight Analysis**: Optimal loss component weights
- **Feature Alignment**: Different alignment strategies
- **Rich Points Analysis**: Spatial-temporal feature enhancement
- **Temporal Modeling**: Long-range dependency analysis

### SOTA Comparison (`sota_comparison.json`)
- **Uniform Sampling**: Traditional uniform frame sampling
- **CLIP-based Sampling**: CLIP-guided frame selection
- **Qwen2-VL**: Direct Qwen2-VL baseline
- **LLaVA-Video**: Video-adapted LLaVA
- **Video-ChatGPT**: Conversational video model
- **VideoLLaMA**: Multi-modal video understanding
- **Valley**: Video-language model
- **Video-LLaMA2**: Enhanced video understanding
- **PLLaVA**: Pooling-based video LLaVA

### Efficiency Analysis (`efficiency_analysis.json`)
- **Video Length Scaling**: 5min, 10min, 20min, 30min
- **Representative Scaling**: 128, 256, 512, 1024 points
- **Batch Size Scaling**: Memory vs. throughput analysis
- **Hardware Comparison**: V100, A100-40GB, A100-80GB

### Theoretical Validation (`theoretical_validation.json`)
- **Equicontour Properties**: Statistical distribution validation
- **Information Optimality**: Information preservation analysis
- **Convergence Analysis**: Training stability validation
- **Robustness Testing**: Noise and perturbation analysis
- **Statistical Properties**: Representative point quality

## Usage

### Using the Configuration Manager

```bash
# Generate configuration for long videos
python scripts/longvideo_config_manager.py --action generate --category long --gpu A100_40GB

# Optimize existing configuration for specific hardware
python scripts/longvideo_config_manager.py --action optimize --config longvideo_config.json --gpu V100_32GB

# Validate configuration
python scripts/longvideo_config_manager.py --action validate --config longvideo_config.json

# Analyze memory requirements
python scripts/longvideo_config_manager.py --action analyze
```

### Generate Experiment Configurations

```bash
# Generate ablation study configs
python scripts/longvideo_config_manager.py --action generate --experiment ablation_studies

# Generate SOTA comparison configs
python scripts/longvideo_config_manager.py --action generate --experiment sota_comparison
```

### Training Examples

```bash
# Train with long video configuration
python scripts/train_ntlbg.py --config configs/longvideo_config.json

# Run ablation experiments
python scripts/train_ntlbg.py --config configs/experiments/ablation_studies.json --experiment ablation_ntlbg_constraint

# Run SOTA comparison
python scripts/train_ntlbg.py --config configs/experiments/sota_comparison.json --experiment sota_uniform_sampling
```

## Memory Optimization

### Gradient Checkpointing
- Enabled by default for large models
- Reduces memory usage by ~40%
- Slight computational overhead

### Mixed Precision Training
- FP16 for V100 GPUs
- BF16 for A100 GPUs
- Automatic loss scaling

### Distributed Training
- Multi-GPU support with DDP
- Gradient accumulation for large effective batch sizes
- ZeRO optimization for memory efficiency

## Performance Targets

### Accuracy Targets
- **LongVideoBench**: ≥70%
- **Video-MME**: ≥65%
- **MLVU**: ≥68%

### Efficiency Targets
- **Speedup**: 2.5x over uniform sampling
- **Memory Reduction**: 40% vs. full-frame processing
- **Training Time**: <100 GPU hours

## Representative Point Calculation

The system automatically calculates optimal representative points:

```python
representatives = max(16, int(max_frames * 0.125))
representatives = min(representatives, int(max_frames * 0.5))
representatives = (representatives // 8) * 8  # Align to 8
```

### Constraints
- **Minimum**: 16 representatives
- **Maximum**: 50% of total frames
- **Alignment**: Multiple of 8 for efficient computation
- **Coverage**: 12.5% of frames by default

## Advanced Features

### Rich Points
- Multi-scale spatial features
- Temporal context integration
- Enhanced representative quality

### Temporal Modeling
- Long-range dependencies
- Hierarchical attention
- Segment-based processing

### Statistical Constraints
- Equicontour properties
- Information preservation
- Diversity regularization

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch_size to 1
   - Increase gradient_accumulation_steps
   - Enable cpu_offload in memory_optimization

2. **Training Instability**
   - Reduce learning_rate
   - Increase warmup_steps
   - Check gradient_accumulation_steps

3. **Slow Training**
   - Enable flash_attention
   - Use bf16 on A100 GPUs
   - Increase dataloader_num_workers

### Performance Optimization

1. **Memory Efficiency**
   - Use gradient_checkpointing
   - Enable mixed precision
   - Optimize dataloader settings

2. **Training Speed**
   - Use multiple GPUs
   - Optimize batch size
   - Enable compilation optimizations

## Configuration Validation

The system includes automatic validation:
- Memory requirement checks
- Hardware compatibility
- Parameter consistency
- Dataset availability

## Future Enhancements

- [ ] Support for longer video sequences (>30 minutes)
- [ ] Advanced sampling strategies
- [ ] Model compression techniques
- [ ] Real-time inference optimization
- [ ] Multi-language support 