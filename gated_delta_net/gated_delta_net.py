import torch

from torch import nn

class GatedDeltaNet(torch.nn.Module):

    def __init__(self, num_heads, hidden_size):
        super(GatedDeltaNet, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.alpha_gate = nn.Linear(hidden_size, num_heads, bias=False)
        self.beta_gate = nn.Linear(hidden_size, num_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        self.A_log = nn.Parameter(torch.empty(num_v_heads).uniform_(0, 16).log())

    def forward(self, x, q, k, v):
        """
        x: (B, S, hidden_size)
        q, k, v: (B, S, num_heads, head_dim)
        """
        alpha = self.alpha_gate(x) # (B, S, num_heads)
        beta = self.beta_gate(x) # (B, S, num_heads)
        batch, seq_len, _ = x.shape
        _, _, num_heads, head_dim_k = k.shape
        _, _, num_heads_v, head_dim_v = v.shape

        # Alpha should be between 0 and 1
        alpha = - self.A_log.float().exp() * (nn.functional.softplus(alpha.float() + self.dt_bias)) # (B, S, num_heads)
        state = torch.zeros(batch, num_heads, head_dim_k, head_dim_v, device=x.device) # (B, num_heads, head_dim, head_dim)
        output = torch.zeros(batch, seq_len, num_heads, head_dim_v, device=x.device, dtype=x.dtype)
        
        for i in range(seq_len):
            q_i = q[:, i] # (B, num_heads, head_dim)
            k_i = k[:, i] # (B, num_heads, head_dim_k)
            v_i = v[:, i] # (B, num_heads, head_dim_v)

            decay = alpha[: i] # (B, num_heads)
            state = state * decay.unsqueeze(-1).unsqueeze(-1) # (B, num_heads, head_dim, head_dim)
            overwrite_factor = beta[:, i] # (B, num_heads)

            # S * k
            v_old = state * k_i.unsqueeze(-1).sum(dim=-2) # (B, num_heads, head_dim_v)
            diff = (v_i - v_old) * overwrite_factor.unsqueeze(-1) # (B, num_heads, head_dim_v)
            state = state + k_i.unsqueeze(-1) * diff.unsqueeze(-2) # (B, num_heads, head_dim_k, head_dim_v)
            output[:, i ] = (state * q_i.unsqueeze(-1)).sum(dim=-2) # (B, num_heads, head_dim_v)
        
        return output