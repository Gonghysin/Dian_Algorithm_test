import torch
import torch.nn as nn
import math

class StandardMultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)  # num_heads个Q头
        self.W_k = nn.Linear(d_model, d_model)  # num_heads个K头
        self.W_v = nn.Linear(d_model, d_model)  # num_heads个V头
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        
        # 线性变换和分头 [batch_size, seq_len, num_heads, d_k]
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k)
        
        # 转置 [batch_size, num_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous()
        output = self.W_o(context.view(batch_size, -1, self.d_model))
        
        return output, attention_weights

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8, num_kv_heads=2):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)  # num_heads个Q头
        self.W_k = nn.Linear(d_model, d_model * num_kv_heads // num_heads)  # num_kv_heads个K头
        self.W_v = nn.Linear(d_model, d_model * num_kv_heads // num_heads)  # num_kv_heads个V头
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.W_k(k).view(batch_size, -1, self.num_kv_heads, self.d_k)
        v = self.W_v(v).view(batch_size, -1, self.num_kv_heads, self.d_k)
        
        # 转置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 扩展k和v以匹配查询头数
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous()
        output = self.W_o(context.view(batch_size, -1, self.d_model))
        
        return output, attention_weights

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)  # num_heads个Q头
        self.W_k = nn.Linear(d_model, d_model // num_heads)  # 1个K头
        self.W_v = nn.Linear(d_model, d_model // num_heads)  # 1个V头
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.W_k(k).view(batch_size, -1, 1, self.d_k)
        v = self.W_v(v).view(batch_size, -1, 1, self.d_k)
        
        # 转置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 扩展k和v到所有头
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous()
        output = self.W_o(context.view(batch_size, -1, self.d_model))
        
        return output, attention_weights

def compare_attention_mechanisms():
    # 设置参数
    batch_size = 2
    seq_len = 4
    d_model = 64
    num_heads = 8
    num_kv_heads = 2  # GQA的KV头数
    
    # 生成随机输入
    x = torch.rand(batch_size, seq_len, d_model)
    print(f"输入张量形状: {x.shape}")
    
    # 初始化三种注意力机制
    mha = StandardMultiHeadAttention(d_model, num_heads)
    gqa = GroupQueryAttention(d_model, num_heads, num_kv_heads)
    mqa = MultiQueryAttention(d_model, num_heads)
    
    # 计算并比较结果
    print("\n标准多头注意力 (MHA):")
    mha_output, mha_weights = mha(x, x, x)
    print(f"输出形状: {mha_output.shape}")
    print(f"注意力权重形状: {mha_weights.shape}")
    print(f"参数数量: {sum(p.numel() for p in mha.parameters())}")
    
    print("\n分组查询注意力 (GQA):")
    gqa_output, gqa_weights = gqa(x, x, x)
    print(f"输出形状: {gqa_output.shape}")
    print(f"注意力权重形状: {gqa_weights.shape}")
    print(f"参数数量: {sum(p.numel() for p in gqa.parameters())}")
    
    print("\n多查询注意力 (MQA):")
    mqa_output, mqa_weights = mqa(x, x, x)
    print(f"输出形状: {mqa_output.shape}")
    print(f"注意力权重形状: {mqa_weights.shape}")
    print(f"参数数量: {sum(p.numel() for p in mqa.parameters())}")
    
    # 计算KV Cache大小比较
    mha_kv_size = 2 * batch_size * num_heads * seq_len * (d_model // num_heads)
    gqa_kv_size = 2 * batch_size * num_kv_heads * seq_len * (d_model // num_heads)
    mqa_kv_size = 2 * batch_size * 1 * seq_len * (d_model // num_heads)
    
    print("\nKV Cache大小比较:")
    print(f"MHA KV Cache大小: {mha_kv_size} 个元素")
    print(f"GQA KV Cache大小: {gqa_kv_size} 个元素")
    print(f"MQA KV Cache大小: {mqa_kv_size} 个元素")
    print(f"GQA相比MHA节省: {(1 - gqa_kv_size/mha_kv_size)*100:.1f}%")
    print(f"MQA相比MHA节省: {(1 - mqa_kv_size/mha_kv_size)*100:.1f}%")

if __name__ == "__main__":
    compare_attention_mechanisms() 