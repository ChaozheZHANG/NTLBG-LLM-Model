import os
import subprocess
import urllib.request
import zipfile
from huggingface_hub import snapshot_download

def download_msrvtt():
    print("📥 下载MSR-VTT数据集...")
    
    os.makedirs("data/msrvtt", exist_ok=True)
    
    # 方法1: 尝试正确的repo
    alternative_repos = [
        "microsoft/MSR-VTT",
        "MSR-VTT/MSR-VTT", 
        "datasets/MSR-VTT"
    ]
    
    for repo in alternative_repos:
        try:
            print(f"尝试从 {repo} 下载...")
            snapshot_download(
                repo_id=repo,
                local_dir="data/msrvtt",
                repo_type="dataset",
                resume_download=True
            )
            print(f"✅ MSR-VTT从 {repo} 下载成功")
            return True
        except Exception as e:
            print(f"❌ {repo} 失败: {e}")
            continue
    
    # 方法2: 下载原始数据文件
    print("🔄 尝试下载MSR-VTT原始文件...")
    try:
        urls = [
            "https://www.robots.ox.ac.uk/~vgg/data/msrvtt/train_val_videodatainfo.json",
            "https://www.robots.ox.ac.uk/~vgg/data/msrvtt/train_val_annotation.json"
        ]
        
        for url in urls:
            filename = url.split('/')[-1]
            try:
                urllib.request.urlretrieve(url, f"data/msrvtt/{filename}")
                print(f"✅ 下载 {filename}")
            except:
                print(f"❌ 下载 {filename} 失败")
        
        return True
    except Exception as e:
        print(f"❌ MSR-VTT原始文件下载失败: {e}")
        return False

def download_activitynet():
    print("📥 下载ActivityNet数据集...")
    
    os.makedirs("data/activitynet", exist_ok=True)
    
    # 尝试不同的源
    sources = [
        "activitynet/ActivityNet",
        "ActivityNet/ActivityNet-Captions",
        "datasets/ActivityNet"
    ]
    
    for source in sources:
        try:
            print(f"尝试从 {source} 下载...")
            snapshot_download(
                repo_id=source,
                local_dir="data/activitynet", 
                repo_type="dataset",
                resume_download=True
            )
            print(f"✅ ActivityNet从 {source} 下载成功")
            return True
        except Exception as e:
            print(f"❌ {source} 失败: {e}")
            continue
    
    print("❌ ActivityNet所有源都失败")
    return False

def download_msvd():
    print("📥 下载MSVD数据集...")
    
    os.makedirs("data/msvd", exist_ok=True)
    
    try:
        # 下载MSVD标注文件
        urls = [
            "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/train_val_test.json",
            "https://raw.githubusercontent.com/zhegan27/MSVD-StackDecoder/master/data/msvd_corpus.json"
        ]
        
        for url in urls:
            filename = url.split('/')[-1]
            try:
                urllib.request.urlretrieve(url, f"data/msvd/{filename}")
                print(f"✅ 下载 {filename}")
            except:
                print(f"⚠️ 下载 {filename} 失败，继续...")
        
        return True
        
    except Exception as e:
        print(f"❌ MSVD下载失败: {e}")
        return False

def main():
    print("🚀 开始下载补充数据集...")
    
    success_count = 0
    
    # 下载各数据集
    datasets = [
        ("MSR-VTT", download_msrvtt),
        ("ActivityNet", download_activitynet), 
        ("MSVD", download_msvd)
    ]
    
    for name, func in datasets:
        print(f"\n{'='*40}")
        if func():
            success_count += 1
            print(f"✅ {name} 完成")
        else:
            print(f"❌ {name} 失败")
    
    print(f"\n🎉 补充下载完成: {success_count}/{len(datasets)}")

if __name__ == "__main__":
    main()
