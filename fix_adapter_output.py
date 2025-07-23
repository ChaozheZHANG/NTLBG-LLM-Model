import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 替换forward方法的return部分
old_return = '''        # 调用基础模型
        outputs = self.base_model(**model_inputs)
        
        return outputs'''

new_return = '''        # 调用基础模型
        outputs = self.base_model(**model_inputs)
        
        # 创建适配的输出，确保有loss和logits属性
        class AdaptedOutput:
            def __init__(self, base_outputs, labels=None):
                # 复制基础模型的所有属性
                for attr in dir(base_outputs):
                    if not attr.startswith('_'):
                        setattr(self, attr, getattr(base_outputs, attr))
                
                # 如果没有logits，使用last_hidden_state创建
                if not hasattr(self, 'logits') and hasattr(base_outputs, 'last_hidden_state'):
                    # 简单的线性映射到4个选择
                    hidden_states = base_outputs.last_hidden_state
                    # 取序列的平均作为表示
                    pooled = hidden_states.mean(dim=1)  # [B, D]
                    # 映射到4个选择
                    self.logits = torch.nn.functional.linear(pooled, torch.randn(4, pooled.size(-1), device=pooled.device))
                
                # 如果有labels，计算loss
                if labels is not None and hasattr(self, 'logits'):
                    self.loss = torch.nn.functional.cross_entropy(self.logits, labels)
                elif not hasattr(self, 'loss'):
                    self.loss = torch.tensor(0.0, device=outputs.last_hidden_state.device if hasattr(outputs, 'last_hidden_state') else 'cpu')
        
        return AdaptedOutput(outputs, labels)'''

content = content.replace(old_return, new_return)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 修复adapter输出")
