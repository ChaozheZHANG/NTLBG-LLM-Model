#!/bin/bash

echo "🚀 创建H200训练会话"

# 创建tmux会话
tmux new-session -d -s ntlbg-h200

# 主训练窗口
tmux rename-window -t ntlbg-h200:0 'training'
tmux send-keys -t ntlbg-h200:0 'cd /workspace/NTLBG-LLM' C-m
tmux send-keys -t ntlbg-h200:0 'conda activate ntlbg-llm' C-m

# 监控窗口
tmux new-window -t ntlbg-h200:1 -n 'monitor'
tmux send-keys -t ntlbg-h200:1 'watch -n 1 nvidia-smi' C-m

# 日志窗口
tmux new-window -t ntlbg-h200:2 -n 'logs'
tmux send-keys -t ntlbg-h200:2 'cd /workspace/NTLBG-LLM' C-m
tmux send-keys -t ntlbg-h200:2 'conda activate ntlbg-llm' C-m

# 数据分析窗口
tmux new-window -t ntlbg-h200:3 -n 'analysis'
tmux send-keys -t ntlbg-h200:3 'cd /workspace/NTLBG-LLM' C-m
tmux send-keys -t ntlbg-h200:3 'conda activate ntlbg-llm' C-m

echo "✅ tmux会话创建完成"
echo "🖥️  使用方法:"
echo "   tmux attach -t ntlbg-h200  # 连接会话"
echo "   Ctrl+B + D               # 断开连接" 
echo "   tmux kill-session -t ntlbg-h200  # 删除会话"

# 自动连接
tmux attach -t ntlbg-h200
