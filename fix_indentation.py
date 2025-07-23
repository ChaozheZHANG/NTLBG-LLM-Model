with open('ntlbg_longvideobench_trainer.py', 'r') as f:
    lines = f.readlines()

# 找到出错的行并修复缩进
fixed_lines = []
for i, line in enumerate(lines):
    if "'NTLBG-LLM (12 Representatives)':" in line:
        # 确保正确的缩进（12个空格）
        fixed_lines.append("            'NTLBG-LLM (12 Representatives)': {\n")
    elif "'NTLBG-LLM (6 Representatives)':" in line:
        fixed_lines.append("            'NTLBG-LLM (6 Representatives)': {\n")
    else:
        fixed_lines.append(line)

with open('ntlbg_longvideobench_trainer.py', 'w') as f:
    f.writelines(fixed_lines)

print("✅ 缩进修复完成")
