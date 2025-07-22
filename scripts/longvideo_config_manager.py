#!/usr/bin/env python3
"""
Long Video Configuration Manager

A comprehensive configuration management system for NTLBG-LLM long video understanding.
Provides automatic configuration generation, optimization, and experiment management.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math

class LongVideoConfigManager:
    """Intelligent configuration manager for long video understanding experiments."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.base_config_path = self.config_dir / "longvideo_config.json"
        self.experiments_dir = self.config_dir / "experiments"
        
        # GPU hardware specifications
        self.gpu_specs = {
            "V100_32GB": {"memory": 32, "compute_capability": 7.0, "tensor_cores": True},
            "A100_40GB": {"memory": 40, "compute_capability": 8.0, "tensor_cores": True},
            "A100_80GB": {"memory": 80, "compute_capability": 8.0, "tensor_cores": True},
            "RTX_4090": {"memory": 24, "compute_capability": 8.9, "tensor_cores": True}
        }
        
        # Load base configuration
        self.load_base_config()
    
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base long video configuration."""
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {self.base_config_path}")
        
        with open(self.base_config_path, 'r') as f:
            self.base_config = json.load(f)
        
        return self.base_config
    
    def analyze_dataset_requirements(self, dataset_path: str) -> Dict[str, Any]:
        """Analyze dataset characteristics to determine optimal configuration."""
        # This would analyze video lengths, resolutions, etc.
        # For now, return default analysis
        return {
            "avg_video_length": 600,  # 10 minutes
            "max_video_length": 1800,  # 30 minutes
            "avg_fps": 30,
            "resolution": [1920, 1080],
            "total_videos": 10000,
            "video_types": ["long_form", "documentary", "lecture"]
        }
    
    def calculate_optimal_representatives(self, max_frames: int, 
                                      coverage_ratio: float = 0.125,
                                      min_reps: int = 16,
                                      max_ratio: float = 0.5,
                                      alignment: int = 8) -> int:
        """Calculate optimal number of representative points."""
        # Base calculation
        representatives = max(min_reps, int(max_frames * coverage_ratio))
        
        # Apply maximum ratio constraint
        representatives = min(representatives, int(max_frames * max_ratio))
        
        # Align to specified boundary
        representatives = (representatives // alignment) * alignment
        
        return representatives
    
    def estimate_memory_requirements(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory requirements for given configuration."""
        video_config = config["video_config"]
        training_config = config["training_config"]
        
        # Basic memory calculations (simplified)
        frame_memory = video_config["max_frames"] * 224 * 224 * 3 * 4  # FP32
        model_memory = 7 * 1024 * 1024 * 1024  # 7B parameters
        representative_memory = video_config["num_representatives"] * 4096 * 4  # Feature dim
        
        batch_memory = frame_memory * training_config["batch_size"]
        gradient_memory = model_memory * 2  # Gradients + optimizer states
        
        total_memory = (batch_memory + model_memory + representative_memory + gradient_memory) / (1024**3)
        
        return {
            "total_gb": total_memory,
            "model_gb": model_memory / (1024**3),
            "video_gb": batch_memory / (1024**3),
            "representatives_gb": representative_memory / (1024**3),
            "gradients_gb": gradient_memory / (1024**3)
        }
    
    def recommend_gpu_configuration(self, target_memory: float) -> Tuple[str, Dict[str, Any]]:
        """Recommend GPU configuration based on memory requirements."""
        suitable_gpus = []
        
        for gpu_name, specs in self.gpu_specs.items():
            if specs["memory"] >= target_memory * 1.2:  # 20% safety margin
                suitable_gpus.append((gpu_name, specs))
        
        if not suitable_gpus:
            return "A100_80GB", self.gpu_specs["A100_80GB"]
        
        # Sort by memory (ascending) to find most economical option
        suitable_gpus.sort(key=lambda x: x[1]["memory"])
        return suitable_gpus[0]
    
    def generate_longvideo_config(self, video_length_category: str = "long", 
                                gpu_type: Optional[str] = None) -> Dict[str, Any]:
        """Generate configuration for specific long video category."""
        config = self.base_config.copy()
        
        # Get variant configuration
        if video_length_category in config["longvideo_variants"]:
            variant = config["longvideo_variants"][video_length_category]
            
            # Update video configuration
            config["video_config"]["max_frames"] = variant["max_frames"]
            config["video_config"]["num_representatives"] = variant["num_representatives"]
            
            # Update training configuration
            config["training_config"]["batch_size"] = variant["batch_size"]
            config["training_config"]["gradient_accumulation_steps"] = variant["gradient_accumulation_steps"]
            
            # Update hardware configuration
            if gpu_type:
                config["hardware_config"]["gpu_type"] = gpu_type
            
        return config
    
    def create_experiment_suite(self) -> Dict[str, Dict[str, Any]]:
        """Create complete experiment suite with all configurations."""
        experiments = {}
        
        # Load experiment configurations
        experiment_files = [
            "ablation_studies.json",
            "sota_comparison.json", 
            "efficiency_analysis.json",
            "theoretical_validation.json"
        ]
        
        for exp_file in experiment_files:
            exp_path = self.experiments_dir / exp_file
            if exp_path.exists():
                with open(exp_path, 'r') as f:
                    exp_config = json.load(f)
                    experiments[exp_file.replace('.json', '')] = exp_config
        
        return experiments
    
    def generate_training_configs(self, experiment_type: str) -> List[Dict[str, Any]]:
        """Generate training configurations for specific experiment type."""
        configs = []
        base_config = self.base_config.copy()
        
        if experiment_type == "ablation_studies":
            # Generate configurations for ablation studies
            variants = [
                {"name": "baseline", "ntlbg_constraint": 0.0},
                {"name": "with_ntlbg", "ntlbg_constraint": 0.8},
                {"name": "representatives_256", "num_representatives": 256},
                {"name": "representatives_512", "num_representatives": 512},
                {"name": "representatives_1024", "num_representatives": 1024},
            ]
            
            for variant in variants:
                config = base_config.copy()
                config["experiment_name"] = f"ablation_{variant['name']}"
                
                # Apply variant settings
                for key, value in variant.items():
                    if key == "name":
                        continue
                    elif key == "num_representatives":
                        config["video_config"][key] = value
                    else:
                        config["training_config"][key] = value
                
                configs.append(config)
        
        elif experiment_type == "sota_comparison":
            # Generate configurations for SOTA comparison
            baselines = [
                {"name": "uniform_sampling", "sampling_strategy": "uniform"},
                {"name": "clip_based", "sampling_strategy": "clip_based"},
                {"name": "qwen2_vl", "base_model": "Qwen/Qwen2-VL-7B-Instruct"},
                {"name": "llava_video", "base_model": "llava-hf/llava-1.5-7b-hf"},
            ]
            
            for baseline in baselines:
                config = base_config.copy()
                config["experiment_name"] = f"sota_{baseline['name']}"
                
                # Apply baseline settings
                for key, value in baseline.items():
                    if key == "name":
                        continue
                    elif key == "sampling_strategy":
                        config["video_config"]["frame_sampling_strategy"] = value
                    elif key == "base_model":
                        config["model_config"]["base_model"] = value
                
                configs.append(config)
        
        return configs
    
    def optimize_for_hardware(self, config: Dict[str, Any], 
                            target_gpu: str) -> Dict[str, Any]:
        """Optimize configuration for specific hardware."""
        gpu_specs = self.gpu_specs.get(target_gpu, self.gpu_specs["A100_40GB"])
        optimized_config = config.copy()
        
        # Memory-based optimizations
        memory_estimate = self.estimate_memory_requirements(config)
        
        if memory_estimate["total_gb"] > gpu_specs["memory"] * 0.8:
            # Reduce batch size
            optimized_config["training_config"]["batch_size"] = 1
            
            # Increase gradient accumulation
            optimized_config["training_config"]["gradient_accumulation_steps"] *= 2
            
            # Enable memory optimizations
            optimized_config["hardware_config"]["memory_optimization"]["cpu_offload"] = True
            optimized_config["hardware_config"]["memory_optimization"]["zero_stage"] = 3
        
        # GPU-specific optimizations
        if "A100" in target_gpu:
            optimized_config["training_config"]["bf16"] = True
            optimized_config["training_config"]["fp16"] = False
        
        optimized_config["hardware_config"]["gpu_type"] = target_gpu
        
        return optimized_config
    
    def save_config(self, config: Dict[str, Any], filename: str) -> str:
        """Save configuration to file."""
        output_path = self.config_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(output_path)
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check video configuration
        video_config = config.get("video_config", {})
        if video_config.get("num_representatives", 0) > video_config.get("max_frames", 0):
            issues.append("num_representatives cannot exceed max_frames")
        
        # Check memory requirements
        memory_estimate = self.estimate_memory_requirements(config)
        gpu_type = config.get("hardware_config", {}).get("gpu_type", "A100_40GB")
        gpu_memory = self.gpu_specs.get(gpu_type, {}).get("memory", 40)
        
        if memory_estimate["total_gb"] > gpu_memory:
            issues.append(f"Estimated memory ({memory_estimate['total_gb']:.1f}GB) exceeds GPU memory ({gpu_memory}GB)")
        
        # Check training configuration
        training_config = config.get("training_config", {})
        if training_config.get("batch_size", 1) * training_config.get("gradient_accumulation_steps", 1) > 64:
            issues.append("Effective batch size is very large and may cause instability")
        
        return issues

def main():
    """Main CLI interface for configuration management."""
    parser = argparse.ArgumentParser(description="Long Video Configuration Manager")
    parser.add_argument("--action", choices=["generate", "optimize", "validate", "analyze"], 
                       required=True, help="Action to perform")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--gpu", type=str, choices=["V100_32GB", "A100_40GB", "A100_80GB", "RTX_4090"],
                       help="Target GPU type")
    parser.add_argument("--category", type=str, choices=["moderate_long", "long", "very_long"],
                       default="long", help="Video length category")
    parser.add_argument("--experiment", type=str, choices=["ablation_studies", "sota_comparison"],
                       help="Experiment type")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = LongVideoConfigManager()
    
    if args.action == "generate":
        if args.experiment:
            # Generate experiment configurations
            configs = manager.generate_training_configs(args.experiment)
            for i, config in enumerate(configs):
                output_path = f"{args.experiment}_{i}.json"
                if args.output:
                    output_path = f"{args.output}_{i}.json"
                
                saved_path = manager.save_config(config, output_path)
                print(f"Generated config: {saved_path}")
        else:
            # Generate single configuration
            config = manager.generate_longvideo_config(args.category, args.gpu)
            
            output_path = args.output or f"longvideo_{args.category}.json"
            saved_path = manager.save_config(config, output_path)
            print(f"Generated config: {saved_path}")
    
    elif args.action == "optimize":
        if not args.config:
            print("Error: --config required for optimization")
            return
        
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        gpu_type = args.gpu or "A100_40GB"
        optimized_config = manager.optimize_for_hardware(config, gpu_type)
        
        output_path = args.output or f"optimized_{Path(args.config).stem}.json"
        saved_path = manager.save_config(optimized_config, output_path)
        print(f"Optimized config saved: {saved_path}")
    
    elif args.action == "validate":
        if not args.config:
            print("Error: --config required for validation")
            return
        
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        issues = manager.validate_config(config)
        
        if issues:
            print("Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid!")
    
    elif args.action == "analyze":
        # Analyze current configuration
        memory_estimate = manager.estimate_memory_requirements(manager.base_config)
        recommended_gpu, specs = manager.recommend_gpu_configuration(memory_estimate["total_gb"])
        
        print("Configuration Analysis:")
        print(f"  Estimated Memory: {memory_estimate['total_gb']:.1f}GB")
        print(f"  Recommended GPU: {recommended_gpu}")
        print(f"  GPU Memory: {specs['memory']}GB")
        print(f"  Representatives: {manager.base_config['video_config']['num_representatives']}")
        print(f"  Max Frames: {manager.base_config['video_config']['max_frames']}")

if __name__ == "__main__":
    main() 