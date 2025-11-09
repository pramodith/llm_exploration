import torch
import torch.nn as nn
from typing import Optional

class RMSNorm(nn.Module):
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        output = x / rms * self.weight
        return output
        

class FFN(nn.Module):
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.ffn(x)
        output += x

@staticmethod
def create_causal_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: [B, S]
    _, seq_len = attention_mask.shape
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_mask.device, dtype=attention_mask.dtype))  # [S, S]
    # Broadcast attention_mask to [B, S, S]
    mask = attention_mask.unsqueeze(-1) * causal_mask.unsqueeze(0)  # [B, S, S]
    return mask

class MoERouter(nn.Module):
    
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.linear = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x, attention_mask):
        router_logits = torch.masked_fill(self.linear(x), attention_mask!=1, value=float("-inf"))
        top_k_logits, chosen_expert_indices = torch.topk(router_logits, self.top_k, -1)
        expert_weights = torch.softmax(top_k_logits, -1)
        return expert_weights, chosen_expert_indices

class MoEMLP(nn.Module):
    
    def __init__(self, hidden_dim, num_experts, expert_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.down_proj = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_dim, self.expert_dim)
        )
        self.up_proj = nn.Parameter(
            torch.randn(self.num_experts, self.expert_dim, self.hidden_dim)
        )
    
    
    def forward(self, x, attention_mask, expert_weights, chosen_expert_indices):
        """_summary_

        Args:
            x (_type_): _description_
            attention_mask (_type_): _description_
            expert_weights (_type_): [B, S, K]
            chosen_expert_indices (_type_): [B, S, K]
        """
        batch_size, seq_len, _ = x.shape
        # MLPs are agnostic to sequence positions/length so simplify shape to [B*S, -1]
        x = x.view(-1, self.hidden_dim)
        expert_weights = expert_weights.view(x.shape[0], -1)
        chosen_expert_indices = chosen_expert_indices.view(x.shape[0], -1)
        attention_mask = attention_mask.view(x.shape[0], -1)
        
        mlp_output = torch.zeros_like(x)
        for i in range(self.num_experts):
            # Check if the router has chosen this specific expert for each token
            is_expert_chosen = torch.any(chosen_expert_indices == i, -1, keepdim=True) * attention_mask
            is_expert_chosen = is_expert_chosen.squeeze().bool()
            # Get the tokens that choose this expert
            tokens_for_expert = x[is_expert_chosen] # [B, U, H]
            # Get the weight coefficients
            weights_for_expert = expert_weights[chosen_expert_indices==i]
            # MLP
            expert_x_down = tokens_for_expert @ self.down_proj[i] # [B, U, E_D]
            expert_x = nn.functional.relu(expert_x_down) @ self.up_proj[i] # [B, U, H]
            # Scale expert outputs by the router assigned weights.
            expert_x = weights_for_expert.unsqueeze(1) * expert_x
            # Aggregate results from each expert
            mlp_output[is_expert_chosen] += expert_x
        
        mlp_output = mlp_output.view(batch_size, seq_len, -1)
        return x + mlp_output
            
            
class MoE(nn.Module):
    
    def __init__(self, hidden_dim, num_experts, expert_dim, top_k):
        super().__init__()
        self.moe_router = MoERouter(
            hidden_dim,
            num_experts,
            top_k
        )
        self.moe_mlp = MoEMLP(
            hidden_dim,
            num_experts, 
            expert_dim
        )
    
    def forward(self, x, attention_mask):
        expert_weights, chosen_expert_indices = self.moe_router(x, attention_mask)
        output = self.moe_mlp(x, attention_mask, expert_weights, chosen_expert_indices)
        return output
        


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
        self.o = nn.Linear(hidden_size, hidden_size)
        self.attention_scaler = (hidden_size // num_heads) ** 0.5
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups

    def forward(self, x: torch.Tensor, attention_mask: torch.tensor) -> torch.tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        query = self.q(x).view(batch_size, seq_len, self.num_heads, -1)
        key = self.k(x).view(batch_size, seq_len, self.num_kv_groups, -1)
        # TODO: ROPE for query and key
        value = self.v(x).view(batch_size, seq_len, self.num_kv_groups, -1)
        
        if self.num_kv_groups != self.num_heads:
            key = torch.repeat_interleave(key, self.num_heads//self.num_kv_groups, 2)
            value = torch.repeat_interleave(value, self.num_heads//self.num_kv_groups, 2)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key.transpose(-2, -1))/self.attention_scaler
        attention = attention * attention_mask.unsqueeze(1)
        attention_weights = torch.softmax(attention, -1)
        value = value.permute(0, 2, 1, 3)
        output = torch.matmul(attention_weights, value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output = self.o(output)
        output += x
        return output



class TransformerBlock(nn.Module):
    
    def __init__(
        self, 
        hidden_size: int, 
        num_attention_heads: int, 
        num_kv_groups: Optional[int],
        intermediate_size: int
    ):
        super().__init__()
        self.pre_norm = nn.RMSNorm(hidden_size)
        if num_kv_groups is None:
            num_kv_groups = num_attention_heads
        self.attention_layer = SelfAttention(
            hidden_size, num_heads=num_attention_heads, num_kv_groups=num_kv_groups
        )
        self.post_norm = nn.RMSNorm(hidden_size)
        self.ffn = FFN(hidden_size, intermediate_size)
        
    def forward(self, x: torch.tensor, attention_mask: torch.tensor) -> torch.Tensor:
        attention_output = self.attention_layer(x, attention_mask)
        attention_output = self.pre_norm(attention_output)
        ffn_output = self.ffn(attention_output)
        ffn_output = self.post_norm(attention_output)
        return ffn_output
        
        

class TransformerModel(nn.Module):

    def __init__(
        self, 
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_kv_groups: Optional[int],
        intermediate_size: int,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim=hidden_size, padding_idx=vocab_size-1)
        self.transformer_blocks = [
            TransformerBlock(hidden_size, num_attention_heads, num_kv_groups, intermediate_size) for _ in range(num_layers)
        ]
    
    def forward(self, input_ids: torch.tensor):
        x = self.token_embeddings(input_ids)
        attention_mask = torch.ones_like(input_ids)
        attention_mask = create_causal_attention_mask(attention_mask)
        for layer in self.transformer_blocks:
            x = layer(x, attention_mask)
        return x    

        

if __name__ == "__main__":
    # s_attn = SelfAttention(768, 12, 4)
    # random_inputs = torch.rand((2, 5, 768))
    # s_attn(random_inputs)
    
    # model = TransformerModel(100, 10, 1, 10, None, 20)
    # model.forward(torch.randint(0, 100, size=(2, 8)))
    
    moe = MoE(64, 4, 16, 2)
    out = moe(torch.randn((2, 6, 64)), torch.ones((2, 6, 1)))
    print(out)
