import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 在模型初始化后添加pad_token设置
pad_token_fix = '''                # 添加pad_token如果不存在
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print("✅ 设置pad_token为eos_token")'''

# 在tokenizer创建后插入
content = content.replace(
    'self.tokenizer = AutoProcessor.from_pretrained(',
    '''self.tokenizer = AutoProcessor.from_pretrained('''
)

# 添加pad_token设置
content = content.replace(
    'print("✅ NTLBG-Qwen2VL适配器初始化完成")',
    '''# 修复pad_token问题
        if hasattr(self.tokenizer, 'tokenizer'):
            if self.tokenizer.tokenizer.pad_token is None:
                self.tokenizer.tokenizer.pad_token = self.tokenizer.tokenizer.eos_token
        elif hasattr(self.tokenizer, 'pad_token'):
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ NTLBG-Qwen2VL适配器初始化完成")'''
)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 修复tokenizer问题")
