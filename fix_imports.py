# 修复 ntlbg_llm_adapter.py 中的导入
import re

# 读取原文件
with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 替换有问题的导入
new_imports = """from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor,
    Qwen2VLForConditionalGeneration, Qwen2VLProcessor
)"""

# 替换导入部分
content = re.sub(
    r'from transformers import \(.*?\)',
    new_imports,
    content,
    flags=re.DOTALL
)

# 注释掉LLaVA相关的类
content = content.replace(
    'class NTLBGLLaVAAdapter',
    '# class NTLBGLLaVAAdapter  # 暂时禁用\nclass _NTLBGLLaVAAdapter'
)

# 修复create_ntlbg_adapter函数
create_adapter_fix = '''def create_ntlbg_adapter(base_model_type="qwen2vl"):
    """创建NTLBG适配器"""
    if base_model_type.lower() == "qwen2vl":
        return NTLBGQwen2VLAdapter()
    else:
        print(f"⚠️ 暂时只支持Qwen2VL，使用默认配置")
        return NTLBGQwen2VLAdapter()'''

content = re.sub(
    r'def create_ntlbg_adapter.*?raise ValueError.*?\)',
    create_adapter_fix,
    content,
    flags=re.DOTALL
)

# 保存修复后的文件
with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 修复完成")
