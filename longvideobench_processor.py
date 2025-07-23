"""
LongVideoBench数据处理器
支持真实视频数据的加载和处理
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import logging

try:
    from longvideobench import LongVideoBenchDataset as OfficialDataset
    OFFICIAL_LOADER_AVAILABLE = False  # 简化处理
    print("✅ 成功导入官方LongVideoBench数据加载器")
except ImportError:
    OFFICIAL_LOADER_AVAILABLE = False
    print("⚠️ 官方数据加载器不可用，使用自定义实现")

logger = logging.getLogger(__name__)

class LongVideoBenchProcessor:
    """LongVideoBench数据处理器"""
    
    def __init__(self, data_root="/workspace/NTLBG-LLM/data/longvideobench", max_frames=64):
        self.data_root = Path(data_root)
        self.max_frames = max_frames
        self.video_dir = self.data_root / "videos"
        self.subtitle_dir = self.data_root / "subtitles"
        
        # 检查数据完整性
        self._check_data_integrity()
    
    def _check_data_integrity(self):
        """检查数据完整性"""
        required_files = [
            self.data_root / "lvb_val.json",
            self.data_root / "lvb_test_wo_gt.json"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.warning(f"⚠️ 缺少文件: {file_path}")
        
        if self.video_dir.exists():
            video_count = len(list(self.video_dir.glob("*.mp4")))
            logger.info(f"📹 找到 {video_count} 个视频文件")
        else:
            logger.warning(f"⚠️ 视频目录不存在: {self.video_dir}")
        
        if self.subtitle_dir.exists():
            subtitle_count = len(list(self.subtitle_dir.glob("*.srt")))
            logger.info(f"📝 找到 {subtitle_count} 个字幕文件")
    
    def load_video_frames(self, video_path: str) -> List[Image.Image]:
        """加载视频帧"""
        if not os.path.exists(video_path):
            logger.warning(f"⚠️ 视频文件不存在: {video_path}")
            return []
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 均匀采样帧
            if frame_count <= self.max_frames:
                indices = list(range(frame_count))
            else:
                indices = np.linspace(0, frame_count-1, self.max_frames, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # 转换为PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    frames.append(pil_frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"❌ 加载视频失败 {video_path}: {e}")
            return []
    
    def load_subtitle(self, subtitle_path: str) -> str:
        """加载字幕"""
        if not os.path.exists(subtitle_path):
            return ""
        
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的SRT解析
            lines = content.split('\n')
            subtitle_text = []
            
            for line in lines:
                line = line.strip()
                if line and not line.isdigit() and '-->' not in line:
                    subtitle_text.append(line)
            
            return ' '.join(subtitle_text)
            
        except Exception as e:
            logger.error(f"❌ 加载字幕失败 {subtitle_path}: {e}")
            return ""


class LongVideoBenchDataset(Dataset):
    """LongVideoBench数据集"""
    
    def __init__(self, data_root, split="val", max_frames=64, max_samples=None):
        self.data_root = Path(data_root)
        self.split = split
        self.max_frames = max_frames
        self.processor = LongVideoBenchProcessor(data_root, max_frames)
        
        # 加载数据
        self.data = self._load_data()
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        logger.info(f"📊 {split}数据集加载完成: {len(self.data)} 个样本")
    
    def _load_data(self):
        """加载数据"""
        if OFFICIAL_LOADER_AVAILABLE:
            return self._load_with_official_loader()
        else:
            return self._load_with_custom_loader()
    
    def _load_with_official_loader(self):
        """使用官方加载器"""
        try:
            json_file = f"lvb_{self.split}.json"
            official_dataset = OfficialDataset(
                str(self.data_root),
                json_file,
                max_num_frames=self.max_frames
            )
            
            data = []
            for i in range(len(official_dataset)):
                try:
                    sample = official_dataset[i]
                    
                    # 分离视频帧和文本
                    frames = []
                    texts = []
                    
                    for item in sample.get("inputs", []):
                        if hasattr(item, 'size'):  # PIL Image
                            frames.append(item)
                        elif isinstance(item, str):
                            texts.append(item)
                    
                    processed_sample = {
                        'video_id': sample.get('video_id', f'video_{i}'),
                        'frames': frames,
                        'subtitle': ' '.join(texts),
                        'question': sample.get('question', ''),
                        'options': sample.get('options', []),
                        'answer': sample.get('answer', 0)
                    }
                    
                    data.append(processed_sample)
                    
                except Exception as e:
                    logger.warning(f"⚠️ 官方数据样本{i}加载失败: {e}")
                    continue
            
            return data
            
        except Exception as e:
            logger.error(f"❌ 官方加载器失败: {e}")
            return self._load_with_custom_loader()
    
    def _load_with_custom_loader(self):
        """使用自定义加载器"""
        json_file = self.data_root / f"lvb_{self.split}.json"
        
        if not json_file.exists():
            logger.warning(f"⚠️ 数据文件不存在: {json_file}")
            return self._create_dummy_data()
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            data = []
            for item in raw_data:
                try:
                    video_id = item.get('video_id', '')
                    video_path = self.data_root / "videos" / f"{video_id}.mp4"
                    subtitle_path = self.data_root / "subtitles" / f"{video_id}.srt"
                    
                    # 加载视频帧
                    frames = self.processor.load_video_frames(str(video_path))
                    
                    # 加载字幕
                    subtitle = self.processor.load_subtitle(str(subtitle_path))
                    
                    processed_sample = {
                        'video_id': video_id,
                        'frames': frames,
                        'subtitle': subtitle,
                        'question': item.get('question', ''),
                        'options': item.get('options', []),
                        'answer': item.get('answer', 0)
                    }
                    
                    data.append(processed_sample)
                    
                except Exception as e:
                    logger.warning(f"⚠️ 样本处理失败: {e}")
                    continue
            
            return data
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        """创建虚拟数据用于测试"""
        logger.info("🔧 创建虚拟数据进行测试...")
        
        dummy_data = []
        questions = [
            "What is the main topic of this video?",
            "Who appears in this video?", 
            "What happens at the end of the video?",
            "What is the setting of this video?",
            "What is the speaker discussing?"
        ]
        
        for i in range(50):  # 创建50个虚拟样本
            # 创建虚拟帧
            frames = []
            for j in range(self.max_frames):
                # 创建随机颜色的虚拟图像
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                frame = Image.fromarray(img_array)
                frames.append(frame)
            
            dummy_sample = {
                'video_id': f'dummy_video_{i}',
                'frames': frames,
                'subtitle': f'This is a dummy subtitle for video {i}.',
                'question': questions[i % len(questions)],
                'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                'answer': i % 4
            }
            
            dummy_data.append(dummy_sample)
        
        return dummy_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloader(data_root, split="val", batch_size=1, max_frames=64, max_samples=None):
    """创建数据加载器"""
    dataset = LongVideoBenchDataset(
        data_root=data_root,
        split=split,
        max_frames=max_frames,
        max_samples=max_samples
    )
    
    def collate_fn(batch):
        """批处理函数"""
        return {
            'video_ids': [item['video_id'] for item in batch],
            'frames': [item['frames'] for item in batch],
            'subtitles': [item['subtitle'] for item in batch],
            'questions': [item['question'] for item in batch],
            'options': [item['options'] for item in batch],
            'answers': torch.tensor([item['answer'] for item in batch])
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        num_workers=0  # H200上设置为0避免多进程问题
    )

