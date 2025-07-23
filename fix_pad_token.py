import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 在初始化最后添加pad_token设置
pad_fix = '''
        # 修复pad_token问题
        if hasattr(self.processor, 'tokenizer'):
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                print("✅ 设置pad_token")'''

content = content.replace(
    'print("✅ NTLBG-Qwen2VL适配器初始化完成")',
    pad_fix + '\n        print("✅ NTLBG-Qwen2VL适配器初始化完成")'
)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 修复pad_token")
