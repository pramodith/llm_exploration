import torch
import torch.nn as nn

class FFN(nn.Module):
    pass

torch.nn.MultiheadAttention

class SelfAttention(nn.Module):
    
    def __init__(
            self, 
            hidden_size: int,
            num_heads: int,
            num_kv_groups: int,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert hidden_size % num_heads == 0, "Hidden dimension size should be divisible by num attention heads"
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, num_kv_groups * hidden_size // num_heads)
        self.v = nn.Linear(hidden_size, num_kv_groups * hidden_size // num_heads)
        self.attention_scaler = (hidden_size // num_heads) ** 0.5
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.cache = {}

    def forward(self, x: torch.Tensor) -> torch.tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # TODO: Read k,v from cache
        query = self.q(x).view(batch_size, seq_len, self.num_heads, -1)
        key = self.k(x).view(batch_size, seq_len, self.num_kv_groups, -1)
        # TODO: ROPE for query and key
        value = self.v(x).view(batch_size, seq_len, self.num_kv_groups, -1)
        
        self.cache["key"] = key
        self.cache["value"] = value
        
        if self.num_kv_groups != self.num_heads:
            key = torch.repeat_interleave(key, self.num_heads//self.num_kv_groups, 2)
            value = torch.repeat_interleave(value, self.num_heads//self.num_kv_groups, 2)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key.transpose(-2, -1))/self.attention_scaler
        attention_weights = torch.softmax(attention, -1)
        value = value.permute(0, 2, 1, 3)
        output = torch.matmul(attention_weights, value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output += x
        return output



class TransformerBlock(nn.Module):
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.pre_norm = nn.RMSNorm(hidden_size)
        self.attention_layer = SelfAttention()
        self.post_norm = nn.RMSNorm(hidden_size)

class TransformerModel(nn.Module):

    def __init__(
        self, 
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim=hidden_size, padding_idx=len(vocab_size))
        self.transformer_block = [TransformerBlock() for _ in range(num_layers)]

if __name__ == "__main__":
    s_attn = SelfAttention(768, 12, 4)
    random_inputs = torch.rand((2, 5, 768))
    s_attn(random_inputs)