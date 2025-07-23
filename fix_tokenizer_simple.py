import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 替换复杂的pad_token逻辑为简单版本
simple_fix = '''        # 简单修复pad_token
        if hasattr(self.processor, 'pad_token') and self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token
            print("✅ 设置pad_token")
        
        self.tokenizer = self.processor'''

# 找到并替换复杂的条件
content = re.sub(
    r'# 修复pad_token.*?self\.tokenizer = self\.processor',
    simple_fix,
    content,
    flags=re.DOTALL
)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 简化tokenizer修复")
