import re

# 读取文件
with open('ntlbg_llm_adapter.py', 'r') as f:
    content = f.read()

# 添加NTLBGSelector类定义
ntlbg_selector_class = '''
class NTLBGSelector(nn.Module):
    """NTLBG代表点选择器"""
    
    def __init__(self, input_dim, num_representatives=6):
        super().__init__()
        self.input_dim = input_dim
        self.num_representatives = num_representatives
        
        # 简化的选择器
        self.attention = nn.Linear(input_dim, 1)
        
    def forward(self, video_features, query_embedding=None):
        """选择代表点"""
        B, T, D = video_features.shape
        
        # 简单的注意力机制选择代表点
        attention_scores = self.attention(video_features).squeeze(-1)  # [B, T]
        
        # 选择top-k个代表点
        _, indices = torch.topk(attention_scores, self.num_representatives, dim=1)
        
        # 提取代表点特征
        representative_features = torch.gather(
            video_features, 1, 
            indices.unsqueeze(-1).expand(-1, -1, D)
        )
        
        return representative_features, {'representative_indices': indices}

'''

# 在类定义之前插入NTLBGSelector
content = content.replace(
    'class NTLBGQwen2VLAdapter',
    ntlbg_selector_class + '\nclass NTLBGQwen2VLAdapter'
)

with open('ntlbg_llm_adapter.py', 'w') as f:
    f.write(content)

print("✅ 添加NTLBGSelector类")
