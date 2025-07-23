import re

with open('ntlbg_longvideobench_trainer.py', 'r') as f:
    content = f.read()

# 确保在_prepare_inputs中正确设置labels
old_labels = '''            # 添加标签
            if for_training:
                processed_inputs['labels'] = batch['answers'].to(self.device)'''

new_labels = '''            # 添加标签（确保正确传递）
            if for_training and 'answers' in batch:
                labels = batch['answers']
                if torch.is_tensor(labels):
                    processed_inputs['labels'] = labels.to(self.device)
                else:
                    processed_inputs['labels'] = torch.tensor(labels, device=self.device)
                print(f"🏷️ 设置标签: {processed_inputs['labels']}")'''

content = content.replace(old_labels, new_labels)

with open('ntlbg_longvideobench_trainer.py', 'w') as f:
    f.write(content)

print("✅ 修复训练器标签传递")
