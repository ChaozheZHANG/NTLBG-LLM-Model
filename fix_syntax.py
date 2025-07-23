import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 找到processor创建的地方，在其后正确添加tokenizer
content = re.sub(
    r'(self\.processor = AutoProcessor\.from_pretrained\([^)]+\))',
    r'\1\n        self.tokenizer = self.processor',
    content
)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 语法修复完成")
