import re

with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 在forward方法开始处添加参数提取
param_extraction = '''        # 从kwargs中提取参数
        input_ids = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask') 
        pixel_values = kwargs.get('pixel_values')
        labels = kwargs.get('labels')
        
        # 确保tensor在正确设备上
        device = next(self.base_model.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if labels is not None:
            labels = labels.to(device)
        '''

# 找到最后一个forward方法（适配器的main forward）并添加参数提取
lines = content.split('\n')
new_lines = []
in_main_forward = False
added_extraction = False

for i, line in enumerate(lines):
    if 'def forward(self, **kwargs):' in line and 'NTLBGQwen2VLAdapter' in content[content.rfind('class', 0, content.find(line)):content.find(line)]:
        new_lines.append(line)
        new_lines.append('        """前向传播"""')
        new_lines.extend(param_extraction.split('\n'))
        in_main_forward = True
        added_extraction = True
    elif in_main_forward and line.strip().startswith('# 确保所有输入都在模型设备上'):
        # 跳过旧的device代码
        continue
    elif in_main_forward and 'for key, value in kwargs.items():' in line:
        # 跳过旧的循环
        continue  
    elif in_main_forward and 'kwargs[key] = value.to(device)' in line:
        # 跳过旧的设备移动
        continue
    else:
        new_lines.append(line)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("✅ 修复forward方法参数提取")
