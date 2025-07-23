import torch
from ntlbg_llm_adapter import create_ntlbg_adapter

print("🔍 测试NTLBG基础功能...")

try:
    # 1. 测试模型创建
    print("1️⃣ 创建模型...")
    model = create_ntlbg_adapter('qwen2vl')
    print(f"✅ 模型创建成功")
    
    # 2. 测试tokenizer
    print("2️⃣ 测试tokenizer...")
    if hasattr(model, 'tokenizer'):
        print(f"✅ tokenizer存在")
        if hasattr(model.tokenizer, 'pad_token'):
            print(f"   pad_token: {model.tokenizer.pad_token}")
        else:
            print("❌ 没有pad_token属性")
    else:
        print("❌ 没有tokenizer属性")
    
    # 3. 测试processor
    print("3️⃣ 测试processor...")
    if hasattr(model, 'processor'):
        print(f"✅ processor存在")
        
        # 测试简单文本处理
        test_text = ["Hello world"]
        try:
            inputs = model.processor(text=test_text, return_tensors="pt", padding=True)
            print(f"✅ 文本处理成功: {list(inputs.keys())}")
        except Exception as e:
            print(f"❌ 文本处理失败: {e}")
    else:
        print("❌ 没有processor属性")
    
    # 4. 测试前向传播
    print("4️⃣ 测试前向传播...")
    try:
        # 创建简单输入
        batch_size = 1
        seq_len = 10
        inputs = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'pixel_values': torch.randn(batch_size, 3, 8, 224, 224)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ 前向传播成功: {type(outputs)}")
        
        if hasattr(outputs, 'loss'):
            print(f"   损失值: {outputs.loss}")
        else:
            print("   没有损失值")
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("🔍 测试完成")
