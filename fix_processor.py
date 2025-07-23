# 修复 longvideobench_processor.py
import re

with open('longvideobench_processor.py', 'r') as f:
    content = f.read()

# 简化官方加载器检查
content = content.replace(
    'OFFICIAL_LOADER_AVAILABLE = True',
    'OFFICIAL_LOADER_AVAILABLE = False  # 简化处理'
)

with open('longvideobench_processor.py', 'w') as f:
    f.write(content)

print("✅ 数据处理器修复完成")
