import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import random
import warnings

from .video_loader import VideoLoader, VideoDataAugmentation


class VideoQADataset(Dataset):
    """视频问答数据集"""
    
    def __init__(self, 
                 data_path: str,
                 video_dir: str,
                 max_video_frames: int = 100,
                 max_text_length: int = 512,
                 tokenizer=None,
                 video_loader_config: Optional[Dict] = None,
                 augmentation: bool = False,
                 augmentation_config: Optional[Dict] = None):
        """
        Args:
            data_path: 数据文件路径（JSON或JSONL）
            video_dir: 视频文件目录
            max_video_frames: 最大视频帧数
            max_text_length: 最大文本长度
            tokenizer: 文本tokenizer
            video_loader_config: 视频加载器配置
            augmentation: 是否启用数据增强
            augmentation_config: 数据增强配置
        """
        self.data_path = data_path
        self.video_dir = video_dir
        self.max_video_frames = max_video_frames
        self.max_text_length = max_text_length
        self.tokenizer = tokenizer
        self.augmentation = augmentation
        
        # 初始化视频加载器
        video_loader_config = video_loader_config or {}
        self.video_loader = VideoLoader(
            video_dir=video_dir,
            max_frames=max_video_frames,
            **video_loader_config
        )
        
        # 初始化数据增强
        if augmentation:
            augmentation_config = augmentation_config or {}
            self.video_augmentation = VideoDataAugmentation(**augmentation_config)
        else:
            self.video_augmentation = None
        
        # 加载数据
        self.data = self._load_data()
        
        # 数据统计
        self.stats = self._compute_stats()
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Data statistics: {self.stats}")
    
    def _load_data(self) -> List[Dict]:
        """加载数据文件"""
        data = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            if self.data_path.endswith('.json'):
                data = json.load(f)
            elif self.data_path.endswith('.jsonl'):
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")
        
        # 数据验证和清理
        cleaned_data = []
        for sample in data:
            if self._validate_sample(sample):
                cleaned_data.append(sample)
        
        return cleaned_data
    
    def _validate_sample(self, sample: Dict) -> bool:
        """验证数据样本"""
        required_fields = ['video_id', 'question', 'answer']
        
        for field in required_fields:
            if field not in sample or not sample[field]:
                return False
        
        # 检查视频文件是否存在
        video_path = Path(self.video_dir) / sample['video_id']
        if not video_path.exists():
            # 尝试常见的视频扩展名
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            video_found = False
            for ext in video_extensions:
                if (video_path.parent / (video_path.stem + ext)).exists():
                    sample['video_id'] = video_path.stem + ext
                    video_found = True
                    break
            
            if not video_found:
                warnings.warn(f"Video file not found: {video_path}")
                return False
        
        return True
    
    def _compute_stats(self) -> Dict:
        """计算数据统计信息"""
        stats = {
            'total_samples': len(self.data),
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'unique_videos': 0,
            'answer_types': {}
        }
        
        if len(self.data) == 0:
            return stats
        
        question_lengths = []
        answer_lengths = []
        unique_videos = set()
        answer_types = {}
        
        for sample in self.data:
            question_lengths.append(len(sample['question'].split()))
            answer_lengths.append(len(sample['answer'].split()))
            unique_videos.add(sample['video_id'])
            
            # 统计答案类型
            answer_type = sample.get('answer_type', 'open')
            answer_types[answer_type] = answer_types.get(answer_type, 0) + 1
        
        stats['avg_question_length'] = np.mean(question_lengths)
        stats['avg_answer_length'] = np.mean(answer_lengths)
        stats['unique_videos'] = len(unique_videos)
        stats['answer_types'] = answer_types
        
        return stats
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. 加载视频特征
        video_features = self._load_video_features(sample['video_id'])
        
        # 2. 处理文本
        text_data = self._process_text(sample)
        
        # 3. 组合返回结果
        result = {
            'video_features': video_features,
            'video_id': sample['video_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'sample_id': sample.get('id', f'sample_{idx}'),
            **text_data
        }
        
        # 4. 添加额外信息
        if 'answer_type' in sample:
            result['answer_type'] = sample['answer_type']
        
        if 'timestamps' in sample:
            result['timestamps'] = sample['timestamps']
        
        if 'metadata' in sample:
            result['metadata'] = sample['metadata']
        
        return result
    
    def _load_video_features(self, video_id: str) -> torch.Tensor:
        """加载视频特征"""
        try:
            # 尝试加载预处理的特征
            if video_id.endswith('.pt'):
                features = torch.load(os.path.join(self.video_dir, video_id))
            else:
                # 实时加载视频
                features = self.video_loader.load_video(video_id)
            
            # 应用数据增强
            if self.video_augmentation is not None and self.training:
                features = self.video_augmentation(features)
            
            # 确保帧数不超过最大值
            if len(features) > self.max_video_frames:
                # 均匀采样
                indices = torch.linspace(0, len(features) - 1, self.max_video_frames, dtype=torch.long)
                features = features[indices]
            
            return features
            
        except Exception as e:
            # 出错时返回零张量
            warnings.warn(f"Failed to load video {video_id}: {e}")
            return torch.zeros(self.max_video_frames, 3, 224, 224)
    
    def _process_text(self, sample: Dict) -> Dict:
        """处理文本数据"""
        question = sample['question']
        answer = sample['answer']
        
        if self.tokenizer is not None:
            # 使用提供的tokenizer
            return self._tokenize_with_tokenizer(question, answer)
        else:
            # 使用简单的模拟tokenization
            return self._simple_tokenize(question, answer)
    
    def _tokenize_with_tokenizer(self, question: str, answer: str) -> Dict:
        """使用真实tokenizer进行tokenization"""
        # 组合问题和答案
        full_text = f"Question: {question} Answer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 创建标签（用于训练）
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 标签与input_ids相同，但问题部分设为-100（忽略）
        labels = input_ids.clone()
        
        # 找到答案的开始位置
        answer_start_token = "Answer:"
        answer_tokens = self.tokenizer.encode(answer_start_token, add_special_tokens=False)
        
        # 简化处理：假设答案在后半部分
        question_length = len(self.tokenizer.encode(f"Question: {question}", add_special_tokens=False))
        labels[:question_length] = -100  # 忽略问题部分
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _simple_tokenize(self, question: str, answer: str) -> Dict:
        """简单的模拟tokenization"""
        vocab_size = 32000
        
        # 模拟tokenization
        question_tokens = len(question.split())
        answer_tokens = len(answer.split())
        total_tokens = question_tokens + answer_tokens + 2  # +2 for special tokens
        
        # 截断到最大长度
        seq_len = min(total_tokens, self.max_text_length)
        
        # 创建模拟的token IDs
        input_ids = torch.randint(1, vocab_size, (seq_len,))
        attention_mask = torch.ones(seq_len)
        
        # 创建标签
        labels = input_ids.clone()
        # 前半部分（问题）设为-100
        labels[:seq_len//2] = -100
        
        # 填充到固定长度
        if seq_len < self.max_text_length:
            pad_length = self.max_text_length - seq_len
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length)])
            labels = torch.cat([labels, torch.full((pad_length,), -100, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def get_sample_by_video_id(self, video_id: str) -> List[Dict]:
        """根据视频ID获取样本"""
        samples = []
        for sample in self.data:
            if sample['video_id'] == video_id:
                samples.append(sample)
        return samples
    
    def get_random_sample(self) -> Dict:
        """获取随机样本"""
        idx = random.randint(0, len(self.data) - 1)
        return self.__getitem__(idx)
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False


class VideoQACollator:
    """视频问答数据集的collate函数"""
    
    def __init__(self, 
                 max_video_frames: int = 100,
                 max_text_length: int = 512,
                 pad_video: bool = True,
                 pad_text: bool = True):
        """
        Args:
            max_video_frames: 最大视频帧数
            max_text_length: 最大文本长度
            pad_video: 是否填充视频到固定长度
            pad_text: 是否填充文本到固定长度
        """
        self.max_video_frames = max_video_frames
        self.max_text_length = max_text_length
        self.pad_video = pad_video
        self.pad_text = pad_text
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        将batch中的样本组合成张量
        
        Args:
            batch: 样本列表
            
        Returns:
            批次张量字典
        """
        # 提取各个字段
        video_features = [sample['video_features'] for sample in batch]
        input_ids = [sample['input_ids'] for sample in batch]
        attention_mask = [sample['attention_mask'] for sample in batch]
        labels = [sample['labels'] for sample in batch]
        
        # 处理视频特征
        video_batch = self._collate_video_features(video_features)
        
        # 处理文本数据
        text_batch = self._collate_text_data(input_ids, attention_mask, labels)
        
        # 组合结果
        result = {
            'video_features': video_batch,
            **text_batch
        }
        
        # 添加元数据
        result['video_ids'] = [sample['video_id'] for sample in batch]
        result['questions'] = [sample['question'] for sample in batch]
        result['answers'] = [sample['answer'] for sample in batch]
        result['sample_ids'] = [sample['sample_id'] for sample in batch]
        
        return result
    
    def _collate_video_features(self, video_features: List[torch.Tensor]) -> torch.Tensor:
        """处理视频特征批次"""
        batch_size = len(video_features)
        
        if not self.pad_video:
            # 不填充：找到最大长度
            max_frames = max(len(features) for features in video_features)
        else:
            # 填充到固定长度
            max_frames = self.max_video_frames
        
        # 获取特征维度
        sample_feature = video_features[0]
        if len(sample_feature.shape) == 4:  # [T, C, H, W]
            C, H, W = sample_feature.shape[1:]
            batch_tensor = torch.zeros(batch_size, max_frames, C, H, W)
        else:  # [T, D] - 预提取特征
            D = sample_feature.shape[1]
            batch_tensor = torch.zeros(batch_size, max_frames, D)
        
        # 填充每个样本
        for i, features in enumerate(video_features):
            seq_len = min(len(features), max_frames)
            batch_tensor[i, :seq_len] = features[:seq_len]
            
            # 如果原始序列较短，重复最后一帧
            if len(features) < max_frames:
                last_frame = features[-1]
                for j in range(len(features), max_frames):
                    batch_tensor[i, j] = last_frame
        
        return batch_tensor
    
    def _collate_text_data(self, 
                          input_ids: List[torch.Tensor],
                          attention_mask: List[torch.Tensor],
                          labels: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """处理文本数据批次"""
        batch_size = len(input_ids)
        
        if not self.pad_text:
            # 不填充：找到最大长度
            max_length = max(len(ids) for ids in input_ids)
        else:
            # 使用固定长度
            max_length = self.max_text_length
        
        # 创建批次张量
        batch_input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        batch_attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
        batch_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        
        # 填充每个样本
        for i, (ids, mask, label) in enumerate(zip(input_ids, attention_mask, labels)):
            seq_len = min(len(ids), max_length)
            batch_input_ids[i, :seq_len] = ids[:seq_len]
            batch_attention_mask[i, :seq_len] = mask[:seq_len]
            batch_labels[i, :seq_len] = label[:seq_len]
        
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels
        }


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证和测试数据加载器"""
    
    # 数据集配置
    dataset_config = config.get('data_config', {})
    
    # 创建数据集
    train_dataset = VideoQADataset(
        data_path=dataset_config['train_data_path'],
        video_dir=dataset_config['video_dir'],
        max_video_frames=dataset_config['max_video_frames'],
        max_text_length=dataset_config['max_text_length'],
        augmentation=True,
        augmentation_config=dataset_config.get('augmentation_config', {})
    )
    
    val_dataset = VideoQADataset(
        data_path=dataset_config['val_data_path'],
        video_dir=dataset_config['video_dir'],
        max_video_frames=dataset_config['max_video_frames'],
        max_text_length=dataset_config['max_text_length'],
        augmentation=False
    )
    
    test_dataset = VideoQADataset(
        data_path=dataset_config['test_data_path'],
        video_dir=dataset_config['video_dir'],
        max_video_frames=dataset_config['max_video_frames'],
        max_text_length=dataset_config['max_text_length'],
        augmentation=False
    )
    
    # 创建collator
    collator = VideoQACollator(
        max_video_frames=dataset_config['max_video_frames'],
        max_text_length=dataset_config['max_text_length']
    )
    
    # 训练配置
    training_config = config.get('training_config', {})
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('system_config', {}).get('num_workers', 4),
        collate_fn=collator,
        pin_memory=config.get('system_config', {}).get('pin_memory', True),
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('system_config', {}).get('num_workers', 4),
        collate_fn=collator,
        pin_memory=config.get('system_config', {}).get('pin_memory', True)
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('system_config', {}).get('num_workers', 4),
        collate_fn=collator,
        pin_memory=config.get('system_config', {}).get('pin_memory', True)
    )
    
    return train_dataloader, val_dataloader, test_dataloader


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    test_data = [
        {
            "id": "sample_1",
            "video_id": "video_1.mp4",
            "question": "What is happening in this video?",
            "answer": "A person is walking down the street.",
            "answer_type": "open"
        },
        {
            "id": "sample_2", 
            "video_id": "video_2.mp4",
            "question": "What color is the car?",
            "answer": "Red",
            "answer_type": "factual"
        }
    ]
    
    # 保存测试数据
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        test_data_path = f.name
    
    # 创建临时视频目录
    temp_video_dir = tempfile.mkdtemp()
    
    try:
        # 创建数据集
        dataset = VideoQADataset(
            data_path=test_data_path,
            video_dir=temp_video_dir,
            max_video_frames=50,
            max_text_length=256
        )
        
        print("=== Testing VideoQA Dataset ===")
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset stats: {dataset.stats}")
        
        # 创建collator
        collator = VideoQACollator(
            max_video_frames=50,
            max_text_length=256
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collator
        )
        
        # 测试批次处理
        print("\n=== Testing Batch Processing ===")
        for batch in dataloader:
            print(f"Video features shape: {batch['video_features'].shape}")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Attention mask shape: {batch['attention_mask'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            print(f"Video IDs: {batch['video_ids']}")
            print(f"Questions: {batch['questions']}")
            break
        
        print("\n✅ VideoQA Dataset working correctly!")
        
    finally:
        # 清理临时文件
        os.unlink(test_data_path)
        os.rmdir(temp_video_dir) 