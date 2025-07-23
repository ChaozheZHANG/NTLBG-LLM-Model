import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 修复forward方法中的变量名 - 应该是**kwargs不是inputs
device_fix_correct = '''        # 确保所有输入都在模型设备上
        device = next(self.base_model.parameters()).device
        for key in kwargs:
            if torch.is_tensor(kwargs[key]):
                kwargs[key] = kwargs[key].to(device)
        '''

# 替换错误的inputs为kwargs
content = content.replace(
    '''        # 确保所有输入都在模型设备上
        device = next(self.base_model.parameters()).device
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to(device)''',
    device_fix_correct
)

# 同时修复pad_token设置
content = content.replace(
    'self.tokenizer = self.processor',
    '''# 修复pad_token
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer and not hasattr(self.processor.tokenizer, 'pad_token') or self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            print("✅ 设置pad_token为eos_token")
        
        self.tokenizer = self.processor'''
)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 修复forward方法和pad_token")
