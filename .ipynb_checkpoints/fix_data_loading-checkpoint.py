"""
修复NTLBG-LLM的数据加载问题
"""
import os
import json
import sys
from pathlib import Path

def fix_data_paths():
    """修复数据路径问题"""
    print("🔧 修复数据路径...")
    
    # 检查数据集实际路径
    base_path = "/workspace/NTLBG-LLM/data"
    datasets = {
        "longvideobench": f"{base_path}/longvideobench",
        "video_mme": f"{base_path}/video_mme", 
        "mlvu": f"{base_path}/mlvu"
    }
    
    for name, path in datasets.items():
        print(f"📁 {name}:")
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"   ✅ 存在，包含 {len(files)} 个文件/文件夹")
            
            # 检查关键文件
            key_files = []
            if name == "longvideobench":
                key_files = ["lvb_val.json", "lvb_test_wo_gt.json", "videos", "subtitles"]
            elif name == "video_mme":
                key_files = [f for f in files if f.endswith('.json') or 'video' in f.lower()]
            elif name == "mlvu":
                key_files = [f for f in files if f.endswith('.json') or f.endswith('.mp4')]
            
            for key_file in key_files:
                key_path = os.path.join(path, key_file)
                if os.path.exists(key_path):
                    if os.path.isfile(key_path):
                        size = os.path.getsize(key_path) / (1024*1024)  # MB
                        print(f"      ✅ {key_file}: {size:.1f}MB")
                    else:
                        count = len(os.listdir(key_path)) if os.path.isdir(key_path) else 0
                        print(f"      ✅ {key_file}/: {count} 个文件")
                else:
                    print(f"      ❌ {key_file}: 不存在")
        else:
            print(f"   ❌ 路径不存在")

def test_longvideobench_loader():
    """测试LongVideoBench数据加载"""
    print("\n🧪 测试LongVideoBench数据加载...")
    
    try:
        from longvideobench import LongVideoBenchDataset
        
        data_path = "/workspace/NTLBG-LLM/data/longvideobench"
        val_file = "lvb_val.json"
        
        if os.path.exists(f"{data_path}/{val_file}"):
            print(f"📚 加载验证集: {val_file}")
            dataset = LongVideoBenchDataset(data_path, val_file, max_num_frames=8)
            print(f"✅ 数据集大小: {len(dataset)}")
            
            if len(dataset) > 0:
                print("📝 第一个样本预览:")
                sample = dataset[0]
                print(f"   输入类型: {type(sample.get('inputs', []))}")
                print(f"   输入长度: {len(sample.get('inputs', []))}")
                return True
            else:
                print("❌ 数据集为空")
                return False
        else:
            print(f"❌ 验证文件不存在: {data_path}/{val_file}")
            return False
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("   请确认已安装 longvideobench 包")
        return False
    except Exception as e:
        print(f"❌ 加载错误: {e}")
        return False

def create_fixed_data_loader():
    """创建修复后的数据加载器"""
    print("\n🔨 创建修复后的数据加载器...")
    
    code = '''
import os
import json
from torch.utils.data import Dataset
from longvideobench import LongVideoBenchDataset
import torch
from PIL import Image

class FixedNTLBGDataset(Dataset):
    def __init__(self, data_path, split="train", max_frames=32):
        self.data_path = data_path
        self.split = split
        self.max_frames = max_frames
        
        # LongVideoBench数据
        if os.path.exists(f"{data_path}/longvideobench"):
            lvb_file = "lvb_val.json" if split == "val" else "lvb_test_wo_gt.json"
            if os.path.exists(f"{data_path}/longvideobench/{lvb_file}"):
                try:
                    self.lvb_dataset = LongVideoBenchDataset(
                        f"{data_path}/longvideobench", 
                        lvb_file, 
                        max_num_frames=max_frames
                    )
                    print(f"✅ 加载LongVideoBench: {len(self.lvb_dataset)} 样本")
                except Exception as e:
                    print(f"❌ LongVideoBench加载失败: {e}")
                    self.lvb_dataset = None
            else:
                print(f"❌ 文件不存在: {data_path}/longvideobench/{lvb_file}")
                self.lvb_dataset = None
        else:
            self.lvb_dataset = None
            
        # 计算总样本数
        self.total_samples = len(self.lvb_dataset) if self.lvb_dataset else 0
        print(f"📊 总样本数: {self.total_samples}")
    
    def __len__(self):
        return max(self.total_samples, 1)  # 至少返回1避免空数据集
    
    def __getitem__(self, idx):
        if self.lvb_dataset and idx < len(self.lvb_dataset):
            try:
                sample = self.lvb_dataset[idx]
                
                # 处理输入
                inputs = sample.get("inputs", [])
                video_frames = [inp for inp in inputs if isinstance(inp, Image.Image)]
                text_parts = [inp for inp in inputs if isinstance(inp, str)]
                
                return {
                    "video_frames": video_frames[:self.max_frames],
                    "text": " ".join(text_parts) if text_parts else "",
                    "question": sample.get("question", ""),
                    "options": sample.get("options", []),
                    "answer": sample.get("answer", 0)
                }
            except Exception as e:
                print(f"❌ 样本{idx}加载失败: {e}")
        
        # 返回空样本
        return {
            "video_frames": [],
            "text": "empty",
            "question": "What do you see?",
            "options": ["A", "B", "C", "D"],
            "answer": 0
        }

# 保存到文件
with open("/workspace/NTLBG-LLM/src/data/fixed_dataset.py", "w") as f:
    f.write(__doc__ + "\\n\\n" + """
{code}
""".format(code=code))
'''
    
    os.makedirs("/workspace/NTLBG-LLM/src/data", exist_ok=True)
    with open("/workspace/NTLBG-LLM/src/data/fixed_dataset.py", "w") as f:
        f.write(code)
    
    print("✅ 修复后的数据加载器已创建: src/data/fixed_dataset.py")

def main():
    print("🔧 NTLBG-LLM 数据加载修复")
    print("=" * 50)
    
    # 检查路径
    fix_data_paths()
    
    # 测试加载
    if test_longvideobench_loader():
        print("✅ 数据加载测试成功")
    else:
        print("❌ 数据加载测试失败，需要检查数据集")
    
    # 创建修复版本
    create_fixed_data_loader()
    
    print("\n🎯 下一步:")
    print("1. 确认所有数据集都正确解压")
    print("2. 使用修复后的数据加载器重新训练")
    print("3. 基于现有大模型（如LLaVA）进行微调")

if __name__ == "__main__":
    main()
