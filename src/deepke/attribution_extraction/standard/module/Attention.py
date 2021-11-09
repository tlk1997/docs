import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DotAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super(DotAttention, self).__init__()
        self.dropout = dropout

    def forward(self, Q, K, V, mask_out=None, head_mask=None):
        """
        Generally, when inputting information X, assume K = V = X

        att_weight = softmax( score_func(q, k) )
        att = sum( att_weight * v )

        Args :
            Q: [..., L, H]
            K: [..., S, H]
            V: [..., S, H]
            : [..., 1, S]
        Returns:
            attention_out: Attention
            attention_weight: Weight after attention
        """
        H = Q.size(-1)

        scale = float(H)**0.5
        attention_weight = torch.matmul(Q, K.transpose(-1, -2)) / scale

        if mask_out is not None:
            while mask_out.dim() != Q.dim():
                mask_out = mask_out.unsqueeze(1)
            attention_weight.masked_fill_(mask_out, -1e8)

        attention_weight = F.softmax(attention_weight, dim=-1)

        attention_weight = F.dropout(attention_weight, self.dropout)

        
        if head_mask is not None:
            attention_weight = attention_weight * head_mask

        attention_out = torch.matmul(attention_weight, V)

        return attention_out, attention_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, output_attentions=True):
        """
        Args:
            embed_dim: The input dimension must be divisible by num_heads
            num_heads: The number of attention
            dropout: float。
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.output_attentions = output_attentions
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        assert self.all_head_dim == embed_dim, logger.error(
            f"embed_dim{embed_dim} must be divisible by num_heads{num_heads}")

        self.q_in = nn.Linear(embed_dim, self.all_head_dim)
        self.k_in = nn.Linear(embed_dim, self.all_head_dim)
        self.v_in = nn.Linear(embed_dim, self.all_head_dim)
        self.attention = DotAttention(dropout=dropout)
        self.out = nn.Linear(self.all_head_dim, embed_dim)

    def forward(self, Q, K, V, key_padding_mask=None, attention_mask=None, head_mask=None):
        """
        Args:
            Q: [B, L, Hs]
            K: [B, S, Hs]
            V: [B, S, Hs]
            key_padding_mask: [B, S] , Mask is required where it is 1/True
            attention_mask: [S] / [L, S] The specified position mask is dropped, and the mask is needed where it is 1/True
            head_mask: [N] Specify the head mask off,the specified position mask is dropped, and the mask is needed where it is 1/True
        """
        B, L, Hs = Q.shape
        S = V.size(1)
        N, H = self.num_heads, self.head_dim

        q = self.q_in(Q).view(B, L, N, H).transpose(1, 2)  # [B, N, L, H]
        k = self.k_in(K).view(B, S, N, H).transpose(1, 2)  # [B, N, S, H]
        v = self.v_in(V).view(B, S, N, H).transpose(1, 2)  # [B, N, S, H]

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.ne(0)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

        if attention_mask is not None:
            attention_mask = attention_mask.ne(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
            else:
                raise ValueError(f'attention_mask dim must be 1 or 2, can not be {attention_mask.dim()}')

        if key_padding_mask is None:
            mask_out = attention_mask if attention_mask is not None else None
        else:
            mask_out = (key_padding_mask + attention_mask).ne(0) if attention_mask is not None else key_padding_mask

        if head_mask is not None:
            head_mask = head_mask.eq(0)
            head_mask = head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        attention_out, attention_weight = self.attention(q, k, v, mask_out=mask_out, head_mask=head_mask)

        attention_out = attention_out.transpose(1, 2).reshape(B, L, N * H)  # [B, N, L, H] -> [B, L, N * H]

        # concat all heads, and do output linear
        attention_out = self.out(attention_out)  # [B, L, N * H] -> [B, L, H]

        if self.output_attentions:
            return attention_out, attention_weight
        else:
            return attention_out,

