import os
import subprocess
from huggingface_hub import snapshot_download

def download_mlvu():
    print("📥 下载MLVU数据集...")
    
    try:
        # 正确的MLVU repo ID
        snapshot_download(
            repo_id="MLVU/MVLU",  # 注意这里是MVLU不是MLVU
            local_dir="data/mlvu",
            repo_type="dataset",
            resume_download=True,
            max_workers=4
        )
        
        size = subprocess.check_output(["du", "-sh", "data/mlvu"], text=True).split()[0]
        print(f"✅ MLVU下载完成: {size}")
        return True
        
    except Exception as e:
        print(f"❌ MLVU下载失败: {e}")
        
        # 备用方法：使用git clone
        try:
            print("🔄 尝试备用下载方法...")
            os.makedirs("data/mlvu", exist_ok=True)
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/datasets/MLVU/MVLU",
                "data/mlvu_temp"
            ], check=True)
            
            # 移动文件
            subprocess.run(["rsync", "-av", "data/mlvu_temp/", "data/mlvu/"], check=True)
            subprocess.run(["rm", "-rf", "data/mlvu_temp"], check=True)
            
            print("✅ MLVU备用下载成功")
            return True
            
        except Exception as e2:
            print(f"❌ MLVU备用下载也失败: {e2}")
            return False

if __name__ == "__main__":
    download_mlvu()
