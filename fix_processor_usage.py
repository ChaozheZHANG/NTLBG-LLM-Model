import re

with open('ntlbg_longvideobench_trainer.py', 'r') as f:
    content = f.read()

# 修复_prepare_inputs方法中的processor调用
# 将images参数改为videos或直接移除
old_processor_call = '''processed_inputs = self.model.processor(
                text=text_inputs,
                images=frames[0] if frames[0] else None,  # 使用第一个样本的帧
                return_tensors="pt",
                padding=True,
                truncation=True
            )'''

new_processor_call = '''processed_inputs = self.model.processor(
                text=text_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True
            )'''

content = content.replace(old_processor_call, new_processor_call)

with open('ntlbg_longvideobench_trainer.py', 'w') as f:
    f.write(content)

print("✅ 修复processor调用")
