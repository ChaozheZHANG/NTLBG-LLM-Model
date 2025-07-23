import torch
from ntlbg_llm_adapter import create_ntlbg_adapter

print("🔍 测试训练器processor使用...")

# 创建模型
model = create_ntlbg_adapter('qwen2vl')

# 模拟训练器中的数据
questions = ["What is shown in the video?"]
options = [["A) A car", "B) A tree", "C) A house", "D) A person"]]

text_inputs = []
for i, question in enumerate(questions):
    full_text = f"Question: {question}\nOptions: " + " ".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(options[i])])
    text_inputs.append(full_text)

try:
    # 测试processor调用
    processed_inputs = model.processor(
        text=text_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    print(f"✅ processor调用成功: {list(processed_inputs.keys())}")
    
    # 测试是否能传递给模型
    with torch.no_grad():
        outputs = model(**processed_inputs)
    print(f"✅ 模型调用成功")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("🔍 测试完成")
