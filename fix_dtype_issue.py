import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 修复logits创建中的dtype问题
old_logits = '''                    # 映射到4个选择
                    self.logits = torch.nn.functional.linear(pooled, torch.randn(4, pooled.size(-1), device=pooled.device))'''

new_logits = '''                    # 映射到4个选择 - 修复dtype
                    weight = torch.randn(4, pooled.size(-1), device=pooled.device, dtype=pooled.dtype)
                    self.logits = torch.nn.functional.linear(pooled, weight)'''

content = content.replace(old_logits, new_logits)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 修复dtype问题")
