# 修复训练器中的配置
import re

with open('ntlbg_longvideobench_trainer.py', 'r') as f:
    content = f.read()

# 修复配置，只使用Qwen2VL
content = content.replace(
    "model_type = self.config.get('base_model_type', 'qwen2vl')",
    "model_type = 'qwen2vl'  # 固定使用Qwen2VL"
)

# 简化变体评估
simplified_variants = """        variants = {
            'NTLBG-LLM (6 Representatives)': {
                'base_model_type': 'qwen2vl',
                'num_representatives': 6,
                'max_frames': 32,  # 减少帧数加快测试
                'description': '标准NTLBG配置'
            },
            'NTLBG-LLM (12 Representatives)': {
                'base_model_type': 'qwen2vl',
                'num_representatives': 12,
                'max_frames': 32,
                'description': '增加代表点数量'
            }
        }"""

content = re.sub(
    r"variants = \{.*?\}",
    simplified_variants,
    content,
    flags=re.DOTALL
)

with open('ntlbg_longvideobench_trainer.py', 'w') as f:
    f.write(content)

print("✅ 训练器修复完成")
