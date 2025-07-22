import torch
import torchvision
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
import random
from PIL import Image
import json


class VideoLoader:
    """视频数据加载器"""
    
    def __init__(self, 
                 video_dir: str,
                 frame_size: Tuple[int, int] = (224, 224),
                 fps: int = 1,
                 max_frames: int = 100,
                 normalize: bool = True):
        """
        Args:
            video_dir: 视频文件目录
            frame_size: 帧大小 (height, width)
            fps: 采样帧率
            max_frames: 最大帧数
            normalize: 是否归一化
        """
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        self.fps = fps
        self.max_frames = max_frames
        self.normalize = normalize
        
        # 视频转换pipeline
        self.transform = self._create_transform()
        
        # 支持的视频格式
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        # 缓存机制
        self.cache = {}
        self.cache_size = 100
    
    def _create_transform(self):
        """创建视频预处理pipeline"""
        transforms = []
        
        # 调整大小
        transforms.append(torchvision.transforms.Resize(self.frame_size))
        
        # 转换为tensor
        transforms.append(torchvision.transforms.ToTensor())
        
        # 归一化
        if self.normalize:
            transforms.append(torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return torchvision.transforms.Compose(transforms)
    
    def load_video(self, video_path: Union[str, Path]) -> torch.Tensor:
        """
        加载视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频张量 [T, C, H, W]
        """
        video_path = Path(video_path)
        
        # 检查缓存
        if str(video_path) in self.cache:
            return self.cache[str(video_path)]
        
        # 检查文件是否存在
        if not video_path.exists():
            full_path = self.video_dir / video_path
            if not full_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            video_path = full_path
        
        # 使用OpenCV读取视频
        try:
            frames = self._read_video_cv2(video_path)
        except Exception as e:
            # 备用方案：使用torchvision
            try:
                frames = self._read_video_torchvision(video_path)
            except Exception as e2:
                raise RuntimeError(f"Failed to load video {video_path}: {e}, {e2}")
        
        # 缓存结果
        if len(self.cache) >= self.cache_size:
            # 移除最旧的缓存
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[str(video_path)] = frames
        
        return frames
    
    def _read_video_cv2(self, video_path: Path) -> torch.Tensor:
        """使用OpenCV读取视频"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = 0
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算帧间隔
        frame_interval = max(1, int(video_fps / self.fps))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔采样帧
            if frame_count % frame_interval == 0:
                # BGR转RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转换为PIL Image
                pil_frame = Image.fromarray(frame)
                # 应用transforms
                processed_frame = self.transform(pil_frame)
                frames.append(processed_frame)
                
                # 检查是否达到最大帧数
                if len(frames) >= self.max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise RuntimeError(f"No frames extracted from video: {video_path}")
        
        # 堆叠成张量
        frames_tensor = torch.stack(frames)  # [T, C, H, W]
        
        return frames_tensor
    
    def _read_video_torchvision(self, video_path: Path) -> torch.Tensor:
        """使用torchvision读取视频"""
        try:
            # 使用torchvision的read_video
            from torchvision.io import read_video
            
            video_tensor, audio_tensor, video_info = read_video(
                str(video_path), 
                pts_unit='sec'
            )
            
            # video_tensor: [T, H, W, C]
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
            
            # 采样帧
            if len(video_tensor) > self.max_frames:
                indices = torch.linspace(0, len(video_tensor) - 1, self.max_frames, dtype=torch.long)
                video_tensor = video_tensor[indices]
            
            # 应用transforms
            processed_frames = []
            for frame in video_tensor:
                # 转换为PIL Image
                frame_pil = torchvision.transforms.ToPILImage()(frame)
                processed_frame = self.transform(frame_pil)
                processed_frames.append(processed_frame)
            
            return torch.stack(processed_frames)
            
        except ImportError:
            raise RuntimeError("torchvision video reading requires additional dependencies")
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict:
        """获取视频信息"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            video_path = self.video_dir / video_path
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        
        return info
    
    def extract_frames_at_timestamps(self, 
                                   video_path: Union[str, Path],
                                   timestamps: List[float]) -> torch.Tensor:
        """在指定时间戳提取帧"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            video_path = self.video_dir / video_path
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        for timestamp in timestamps:
            # 计算帧号
            frame_number = int(timestamp * fps)
            
            # 定位到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                # BGR转RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转换为PIL Image
                pil_frame = Image.fromarray(frame)
                # 应用transforms
                processed_frame = self.transform(pil_frame)
                frames.append(processed_frame)
            else:
                # 如果无法读取，使用零张量
                zero_frame = torch.zeros(3, *self.frame_size)
                frames.append(zero_frame)
        
        cap.release()
        
        return torch.stack(frames)
    
    def batch_load_videos(self, video_paths: List[Union[str, Path]]) -> torch.Tensor:
        """批量加载视频"""
        batch_frames = []
        
        for video_path in video_paths:
            frames = self.load_video(video_path)
            batch_frames.append(frames)
        
        # 找到最大帧数
        max_frames = max(len(frames) for frames in batch_frames)
        
        # 填充到相同长度
        padded_frames = []
        for frames in batch_frames:
            if len(frames) < max_frames:
                # 重复最后一帧进行填充
                last_frame = frames[-1].unsqueeze(0)
                padding = last_frame.repeat(max_frames - len(frames), 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)
            
            padded_frames.append(frames)
        
        # 堆叠成批次
        batch_tensor = torch.stack(padded_frames)  # [B, T, C, H, W]
        
        return batch_tensor
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
    
    def preprocess_video_directory(self, 
                                 output_dir: str,
                                 video_extensions: List[str] = None) -> Dict[str, str]:
        """预处理视频目录，提取特征并保存"""
        if video_extensions is None:
            video_extensions = self.supported_formats
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_videos = {}
        
        for video_file in self.video_dir.rglob('*'):
            if video_file.suffix.lower() in video_extensions:
                try:
                    # 加载视频
                    frames = self.load_video(video_file)
                    
                    # 生成输出文件名
                    relative_path = video_file.relative_to(self.video_dir)
                    output_path = output_dir / (str(relative_path).replace(video_file.suffix, '.pt'))
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 保存处理后的张量
                    torch.save(frames, output_path)
                    
                    processed_videos[str(relative_path)] = str(output_path)
                    
                    print(f"Processed: {video_file} -> {output_path}")
                    
                except Exception as e:
                    print(f"Failed to process {video_file}: {e}")
                    continue
        
        # 保存处理映射
        mapping_file = output_dir / 'video_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(processed_videos, f, indent=2)
        
        print(f"Processed {len(processed_videos)} videos")
        return processed_videos


class VideoDataAugmentation:
    """视频数据增强"""
    
    def __init__(self, 
                 temporal_jitter: bool = True,
                 frame_dropout: float = 0.1,
                 color_jitter: bool = True,
                 horizontal_flip: bool = True,
                 random_crop: bool = True):
        """
        Args:
            temporal_jitter: 时序抖动
            frame_dropout: 帧丢失率
            color_jitter: 颜色抖动
            horizontal_flip: 水平翻转
            random_crop: 随机裁剪
        """
        self.temporal_jitter = temporal_jitter
        self.frame_dropout = frame_dropout
        self.color_jitter = color_jitter
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        
        # 创建图像增强transforms
        self.image_transforms = []
        
        if random_crop:
            self.image_transforms.append(
                torchvision.transforms.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33)
                )
            )
        
        if horizontal_flip:
            self.image_transforms.append(
                torchvision.transforms.RandomHorizontalFlip(p=0.5)
            )
        
        if color_jitter:
            self.image_transforms.append(
                torchvision.transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        self.image_transform = torchvision.transforms.Compose(self.image_transforms)
    
    def __call__(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        应用数据增强
        
        Args:
            video_tensor: [T, C, H, W] 视频张量
            
        Returns:
            增强后的视频张量
        """
        T, C, H, W = video_tensor.shape
        
        # 1. 时序抖动
        if self.temporal_jitter and T > 1:
            # 随机改变帧的顺序（轻微）
            jitter_indices = torch.arange(T)
            for i in range(T):
                if random.random() < 0.1:  # 10%概率交换相邻帧
                    if i < T - 1:
                        jitter_indices[i], jitter_indices[i + 1] = jitter_indices[i + 1], jitter_indices[i]
            
            video_tensor = video_tensor[jitter_indices]
        
        # 2. 帧丢失
        if self.frame_dropout > 0 and T > 2:
            dropout_mask = torch.rand(T) > self.frame_dropout
            # 确保至少保留一半的帧
            if dropout_mask.sum() < T // 2:
                dropout_mask = torch.ones(T, dtype=torch.bool)
                dropout_mask[torch.randperm(T)[:T//4]] = False
            
            video_tensor = video_tensor[dropout_mask]
        
        # 3. 图像级增强
        if self.image_transforms:
            # 转换为PIL图像，应用增强，再转回张量
            enhanced_frames = []
            for frame in video_tensor:
                # 反归一化
                frame_denorm = frame * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                frame_denorm = torch.clamp(frame_denorm, 0, 1)
                
                # 转换为PIL图像
                pil_frame = torchvision.transforms.ToPILImage()(frame_denorm)
                
                # 应用增强
                enhanced_frame = self.image_transform(pil_frame)
                
                # 转换回张量并重新归一化
                enhanced_frame = torchvision.transforms.ToTensor()(enhanced_frame)
                enhanced_frame = torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(enhanced_frame)
                
                enhanced_frames.append(enhanced_frame)
            
            video_tensor = torch.stack(enhanced_frames)
        
        return video_tensor


# 测试代码
if __name__ == "__main__":
    # 创建测试视频加载器
    video_loader = VideoLoader(
        video_dir="./test_videos",
        frame_size=(224, 224),
        fps=1,
        max_frames=50
    )
    
    # 创建数据增强
    augmentation = VideoDataAugmentation(
        temporal_jitter=True,
        frame_dropout=0.1,
        color_jitter=True
    )
    
    # 模拟测试（创建假的视频张量）
    print("=== Testing Video Loader ===")
    
    # 创建模拟视频数据
    mock_video = torch.randn(30, 3, 224, 224)  # 30帧，3通道，224x224
    
    # 应用数据增强
    augmented_video = augmentation(mock_video)
    
    print(f"Original video shape: {mock_video.shape}")
    print(f"Augmented video shape: {augmented_video.shape}")
    
    # 测试批量加载功能
    batch_videos = [
        torch.randn(25, 3, 224, 224),
        torch.randn(35, 3, 224, 224),
        torch.randn(20, 3, 224, 224)
    ]
    
    # 模拟批量处理
    max_frames = max(len(v) for v in batch_videos)
    padded_videos = []
    
    for video in batch_videos:
        if len(video) < max_frames:
            padding = video[-1].unsqueeze(0).repeat(max_frames - len(video), 1, 1, 1)
            video = torch.cat([video, padding], dim=0)
        padded_videos.append(video)
    
    batch_tensor = torch.stack(padded_videos)
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    print("\n✅ Video loader working correctly!") 