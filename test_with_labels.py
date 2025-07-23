import torch
from ntlbg_llm_adapter import create_ntlbg_adapter

print("🔍 测试带labels的训练...")

# 创建模型
model = create_ntlbg_adapter('qwen2vl')

# 创建带labels的输入
inputs = {
    'input_ids': torch.randint(0, 1000, (2, 10)),
    'attention_mask': torch.ones(2, 10),
    'pixel_values': torch.randn(2, 3, 8, 224, 224),
    'labels': torch.tensor([0, 1])  # 添加标签
}

try:
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✅ 前向传播成功")
    print(f"   损失值: {outputs.loss}")
    print(f"   logits形状: {outputs.logits.shape}")
    print(f"   损失是否为0: {outputs.loss.item() == 0.0}")
    
    if outputs.loss.item() > 0:
        print("✅ 损失计算正常！")
    else:
        print("❌ 损失仍为0，需要检查labels处理")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("🔍 测试完成")
