import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 确保reshape后的维度正确
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
    
    def forward(self, q, k, v, use_cache=False, past_k=None, past_v=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.W_q(q)  # [batch_size, seq_len, d_model]
        k = self.W_k(k)
        v = self.W_v(v)
        
        # 分离多头
        q = self.split_heads(q)  # [batch_size, num_heads, seq_len, d_k]
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # 处理 KV Cache - 移到split_heads之后
        if use_cache and past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=2)  # dim=2 是序列长度维度
            v = torch.cat([past_v, v], dim=2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 计算输出
        context = torch.matmul(attention_weights, v)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # 最后的线性变换
        output = self.W_o(context)
        
        if use_cache:
            return output, attention_weights, k, v
        return output, attention_weights

def test_attention():
    # 设置参数
    batch_size = 2
    seq_len = 4
    d_model = 64
    num_heads = 8
    
    # 创建模型
    attention = MultiHeadAttention(d_model, num_heads)
    
    # 生成随机输入
    x = torch.rand(batch_size, seq_len, d_model)
    print(x.shape)

    # 测试普通前向传播
    output, weights = attention(x, x, x)
    print("\n普通注意力计算结果:")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 测试 KV Cache
    print("\nKV Cache 测试:")
    past_k = past_v = None
    for i in range(seq_len):
        # 模拟逐步输入
        current_input = x[:, i:i+1, :]
        output, weights, new_k, new_v = attention(
            current_input, current_input, current_input,
            use_cache=True, past_k=past_k, past_v=past_v
        )
        past_k, past_v = new_k, new_v
        print(f"步骤 {i+1}: 输出形状 {output.shape}, 累积的K形状 {new_k.shape}")

if __name__ == "__main__":
    test_attention()
