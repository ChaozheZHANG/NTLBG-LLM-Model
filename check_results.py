"""
快速检查NTLBG-LLM实验结果
"""
import json
import os
from pathlib import Path

def check_results():
    print("🔍 检查NTLBG-LLM实验结果...")
    
    # 检查训练结果
    if os.path.exists("outputs/models/best_fixed_ntlbg_llm.pth"):
        print("✅ 训练模型权重已保存")
        
        if os.path.exists("outputs/fixed_training_results.json"):
            with open("outputs/fixed_training_results.json", "r") as f:
                train_results = json.load(f)
            print(f"   最佳训练准确率: {train_results.get('best_accuracy', 0):.4f}")
    else:
        print("❌ 训练模型权重不存在")
    
    # 检查评估结果
    results_dir = Path("paper_results/real_longvideobench_final")
    if results_dir.exists():
        print("✅ 评估结果目录存在")
        
        files_to_check = [
            ("detailed_results.json", "详细结果"),
            ("aaai_2026_table.tex", "LaTeX表格"),
            ("aaai_2026_summary.json", "实验摘要"),
            ("aaai_2026_paper_sections.txt", "论文章节"),
            ("ntlbg_real_evaluation.png", "结果图表")
        ]
        
        for filename, description in files_to_check:
            if (results_dir / filename).exists():
                print(f"   ✅ {description}: {filename}")
            else:
                print(f"   ❌ {description}: {filename} (缺失)")
        
        # 读取并显示关键结果
        if (results_dir / "detailed_results.json").exists():
            with open(results_dir / "detailed_results.json", "r") as f:
                eval_results = json.load(f)
            
            print(f"\n📊 评估结果摘要:")
            for result in eval_results:
                variant = result.get('variant', 'Unknown')
                accuracy = result.get('accuracy', 0)
                reps = result.get('num_representatives', 0)
                print(f"   {variant}: {accuracy:.3f} (K={reps})")
    else:
        print("❌ 评估结果目录不存在")
    
    # 检查项目结构
    print(f"\n📁 项目结构检查:")
    key_dirs = ["src/models", "outputs", "paper_results", "data"]
    for dir_name in key_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ (缺失)")

if __name__ == "__main__":
    check_results()
