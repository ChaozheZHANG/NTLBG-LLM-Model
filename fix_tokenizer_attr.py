import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 在processor创建后添加tokenizer属性
tokenizer_fix = '''        
        # 为兼容性添加tokenizer属性
        self.tokenizer = self.processor
        
        # 修复pad_token问题
        if hasattr(self.processor, 'tokenizer'):
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        '''

content = content.replace(
    'print("✅ NTLBG-Qwen2VL适配器初始化完成")',
    tokenizer_fix + '\n        print("✅ NTLBG-Qwen2VL适配器初始化完成")'
)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 修复tokenizer属性")
