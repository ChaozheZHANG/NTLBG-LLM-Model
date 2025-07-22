import os
import subprocess
import shutil
from huggingface_hub import snapshot_download
import time

def check_space():
    total, used, free = shutil.disk_usage('.')
    free_gb = free / (1024**3)
    print(f"💾 可用空间: {free_gb:.1f} GB")
    return free_gb

def download_dataset(repo_id, local_dir, name, priority="high"):
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 5:
        print(f"✅ {name} 已存在，跳过")
        return True
        
    print(f"📥 下载 {name}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset",
            resume_download=True,
            max_workers=8
        )
        
        size = subprocess.check_output(["du", "-sh", local_dir], text=True).split()[0]
        print(f"✅ {name} 完成: {size}")
        return True
        
    except Exception as e:
        print(f"❌ {name} 失败: {e}")
        if priority == "high":
            print(f"🔄 重试 {name}...")
            time.sleep(5)
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    repo_type="dataset",
                    resume_download=True,
                    max_workers=4
                )
                print(f"✅ {name} 重试成功")
                return True
            except:
                print(f"❌ {name} 重试也失败")
        return False

def main():
    print("🚀 开始下载所有数据集...")
    
    # 检查空间
    if check_space() < 200:
        print("⚠️ 磁盘空间不足200GB，请清理后重试")
        return
    
    # 所有数据集（按重要性排序）
    datasets = [
        # 必需数据集
        ("longvideobench/LongVideoBench", "data/longvideobench", "LongVideoBench", "high"),
        ("lmms-lab/Video-MME", "data/video_mme", "Video-MME", "high"), 
        ("MLVU/MLVU", "data/mlvu", "MLVU", "high"),
        
        # 重要数据集
        ("microsoft/MSR-VTT", "data/msrvtt", "MSR-VTT", "medium"),
        
        # 其他数据集
        ("ActivityNet/ActivityNet", "data/activitynet", "ActivityNet", "medium"),
    ]
    
    success_count = 0
    
    for repo_id, local_dir, name, priority in datasets:
        if download_dataset(repo_id, local_dir, name, priority):
            success_count += 1
        
        # 检查空间
        remaining = check_space()
        if remaining < 100:
            print("⚠️ 空间不足，停止下载")
            break
    
    print(f"\n🎉 下载完成: {success_count}/{len(datasets)}")
    
    # 生成报告
    print("\n📊 数据集状态:")
    total_size = 0
    for _, local_dir, name, _ in datasets:
        if os.path.exists(local_dir):
            try:
                size_str = subprocess.check_output(["du", "-sh", local_dir], text=True).split()[0]
                files = len([f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))])
                print(f"✅ {name}: {size_str}, {files} 文件")
            except:
                print(f"❌ {name}: 检查失败")

if __name__ == "__main__":
    main()
