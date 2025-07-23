import re

# 修复适配器
with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 在初始化后添加pad_token和device修复
fix_code = '''
        # 修复pad_token问题
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer:
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                print("✅ 设置pad_token为eos_token")
        
        # 确保tokenizer指向processor
        self.tokenizer = self.processor'''

content = content.replace(
    'print("✅ NTLBG-Qwen2VL适配器初始化完成")',
    fix_code + '\n        print("✅ NTLBG-Qwen2VL适配器初始化完成")'
)

# 在forward方法中确保输入在正确设备上
device_fix = '''        # 确保所有输入都在模型设备上
        device = next(self.base_model.parameters()).device
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to(device)
        '''

content = content.replace(
    'outputs = self.base_model(',
    device_fix + '\n        outputs = self.base_model('
)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 修复pad_token和device问题")
