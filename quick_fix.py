import re

with open('ntlbg_longvideobench_trainer.py', 'r') as f:
    lines = f.readlines()

# 找到并修复variants定义
new_lines = []
skip_until_brace = False

for line in lines:
    if 'variants = {' in line:
        new_lines.append('        variants = {\n')
        new_lines.append("            'NTLBG-LLM (6 Representatives)': {\n")
        new_lines.append("                'base_model_type': 'qwen2vl',\n")
        new_lines.append("                'num_representatives': 6,\n")
        new_lines.append("                'max_frames': 32,\n")
        new_lines.append("                'description': '标准NTLBG配置'\n")
        new_lines.append("            }\n")
        new_lines.append("        }\n")
        skip_until_brace = True
    elif skip_until_brace and line.strip() == '}':
        skip_until_brace = False
        continue
    elif not skip_until_brace:
        new_lines.append(line)

with open('ntlbg_longvideobench_trainer.py', 'w') as f:
    f.writelines(new_lines)

print("✅ 修复完成")
