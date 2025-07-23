from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
import os

def download_models():
    models_dir = "data/models"
    os.makedirs(models_dir, exist_ok=True)
    
    models = [
        ("openai/clip-vit-large-patch14", "clip-vit-large"),
        ("microsoft/DialoGPT-medium", "dialogpt-medium"),
    ]
    
    for model_name, local_name in models:
        local_path = f"{models_dir}/{local_name}"
        if os.path.exists(local_path):
            print(f"✅ {local_name} 已存在")
            continue
            
        print(f"📥 下载 {local_name}...")
        try:
            if "clip" in model_name:
                model = CLIPModel.from_pretrained(model_name)
                processor = CLIPProcessor.from_pretrained(model_name)
                model.save_pretrained(local_path)
                processor.save_pretrained(local_path)
            else:
                model = AutoModel.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model.save_pretrained(local_path)
                tokenizer.save_pretrained(local_path)
            print(f"✅ {local_name} 下载完成")
        except Exception as e:
            print(f"❌ {local_name} 下载失败: {e}")

if __name__ == "__main__":
    download_models()
