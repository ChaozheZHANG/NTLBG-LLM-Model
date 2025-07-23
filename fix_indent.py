import re

# 读取文件
with open('ntlbg_longvideobench_trainer.py', 'r') as f:
    content = f.read()

# 找到并修复variants部分的缩进
variants_section = '''        variants = {
            'NTLBG-LLM (6 Representatives)': {
                'base_model_type': 'qwen2vl',
                'num_representatives': 6,
                'max_frames': 32,
                'description': '标准NTLBG配置'
            },
            'NTLBG-LLM (12 Representatives)': {
                'base_model_type': 'qwen2vl', 
                'num_representatives': 12,
                'max_frames': 32,
                'description': '增加代表点数量'
            }
        }'''

# 替换整个variants定义
content = re.sub(
    r'        variants = \{.*?\}',
    variants_section,
    content,
    flags=re.DOTALL
)

# 保存
with open('ntlbg_longvideobench_trainer.py', 'w') as f:
    f.write(content)

print("✅ 缩进修复完成")
