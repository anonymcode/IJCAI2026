import time
import math
import numpy as np
import torch
from torch import nn
import torch.nn.utils.parametrize as P
import torch.nn.functional as F
from torch.nn.utils import parametrize


class SharedKernel(nn.Module):
    def __init__(self, maxlen: int, config: dict):
        super().__init__()
        self.maxlen = maxlen
        self.fix_first_to_one: bool = config.get('kernel_fix_first_to_one', False)
        self.type_of_kernel_initialization = config.get('type_of_kernel_initialization', 'normal')
        # Initialize shared latent kernel vector g
        # g_init = torch.reciprocal(torch.arange(1, maxlen + 1, dtype=torch.float32))  # harmonic initialization: g[k]=1/(k+1)
        # g_init = torch.empty(maxlen, dtype=torch.float32).normal_(0.0, 0.02)
        # g_init = torch.randn(maxlen, dtype=torch.float32) * 0.02  # standard random initialization (small Gaussian)
        # g_init = torch.uniform(maxlen, dtype=torch.float32) * 0.02  # standard random initialization (small Gaussian)
        # g_init = torch.zeros(maxlen, dtype=torch.float32)
        # g_init[0] = 1.0  # fix first element for stable start (A ~ I)

        if self.type_of_kernel_initialization == 'harmonic':
            g_init = torch.reciprocal(torch.arange(1, maxlen + 1, dtype=torch.float32))  # harmonic initialization: g[k]=1/(k+1)
        elif self.type_of_kernel_initialization == 'normal':
             # Standard PyTorch initialization (as in regular layers) with positive weights
            g_init = torch.empty(maxlen, dtype=torch.float32)
            # Use normal_ (standard initialization for weights) with small std
            nn.init.normal_(g_init, mean=0.0, std=0.02)
            # Ensure positive values: take absolute value
            g_init = torch.abs(g_init)
        elif self.type_of_kernel_initialization == 'ones':
            g_init = torch.ones(maxlen, dtype=torch.float32) * 0.5
        # elif self.type_of_kernel_initialization == 'random':
        #     g_init = torch.randn(maxlen, dtype=torch.float32) * 0.02  # standard random initialization (small Gaussian)
        # elif self.type_of_kernel_initialization == 'uniform':
        #     g_init = torch.uniform(maxlen, dtype=torch.float32) * 0.02  # standard random initialization (small Gaussian)


        if self.fix_first_to_one:
            self.register_buffer('_g0', torch.ones(1)) # frozen to one
        else:
            self._g0 = nn.Parameter(torch.Tensor([g_init[0]])) # leanabe 

        self._g_rest = nn.Parameter(g_init[1:])
        print(f"g_init: {self.g.detach().cpu().numpy()}")


    @property
    def g(self) -> torch.Tensor:
        """Returns full kernel vector g by concatenating fixed and learnable parts"""
        return torch.cat([self._g0, self._g_rest], dim=0)

class StraightAFromKernel(nn.Module):
    def __init__(self, kernel: SharedKernel, upper_triangular: bool = False):
        super().__init__()
        self.kernel = kernel
        self.maxlen = kernel.maxlen
        self.upper_triangular = upper_triangular

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        g = self.kernel.g
        # For lower triangular matrix need to flip kernel for correct convolution
        if not self.upper_triangular:
            g = torch.flip(g, dims=[0])
        return g.view(1, 1, -1, 1)


# class InvAFromKernel(nn.Module):
#     def __init__(self, kernel: SharedKernel):
#         super().__init__()
#         self.kernel = kernel
#         self.maxlen = kernel.maxlen
#         # Precompute Toeplitz indexing and mask for lower-triangular build
#         i = torch.arange(self.maxlen)
#         idx = i[:, None] - i[None, :]  # shape (n, n)
#         mask = idx >= 0                # lower triangle (including diagonal)
#         self.register_buffer('idx_lower', idx.clamp_min(0).long())
#         self.register_buffer('mask_lower', mask)

#     def forward(self) -> torch.Tensor:
#         g = self.kernel.g
#         # g = self.kernel.full_g()
#         A_lower = g[self.idx_lower] * self.mask_lower.to(g.dtype)
#         e1 = torch.zeros(self.maxlen, 1, device=A_lower.device, dtype=A_lower.dtype)
#         e1[0, 0] = 1.0
#         x = torch.linalg.solve_triangular(A_lower, e1, upper=False)
#         return x.view(1, 1, -1, 1)

class Custom_A_Conv(nn.Module):
    """Original Custom_A implementation using Conv2d"""
    def __init__(self, maxlen, hidden_dim, config): 
        super(Custom_A_Conv, self).__init__()
        self.use_custom_A_layernorm = config.get('use_custom_A_layernorm', False)
        self.upper_triangular = config.get('upper_triangular', False)
        if self.upper_triangular:
            print("upper_triangular")
            self.padding = nn.ZeroPad2d((0, 0, 0, maxlen-1))  # bottom padding
        else:
            self.padding = nn.ZeroPad2d((0, 0, maxlen-1, 0))  # top padding


        
        # Standard Conv2d layer
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(maxlen, 1), 
            bias=False
        )
        
    def forward(self, seq):
        if len(seq.size()) == 3:  # (batch, seq_len, hidden_dim)
            seq = seq.unsqueeze(1)  # add channel dimension
            seq = self.padding(seq)
            
            if self.use_custom_A_layernorm:
                # 1. Get kernel weights
                weight = self.conv.weight
                
                # 2. Compute L2 norm of entire weight tensor
                norm = torch.norm(weight) + 1e-10  # protection against division by zero
                
                # 3. Normalize weights
                normalized_weight = weight / norm
                
                # 4. Apply convolution with normalized weights
                seq = F.conv2d(
                    input=seq,
                    weight=normalized_weight,
                    bias=None,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups
                )
            else:
                # Without normalization
                seq = self.conv(seq)
                
            seq = seq.squeeze(1)  # remove channel dimension
        else:
            raise ValueError(f"Invalid input shape: {seq.size()}")
        return seq


class Custom_A_FullMatrix(nn.Module):
    """Custom layer with full matrix (not Toeplitz)"""
    def __init__(self, T, hidden_dim, config): 
        super(Custom_A_FullMatrix, self).__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.use_custom_A_layernorm = config.get('use_custom_A_layernorm', False)
        self.upper_triangular = config.get('upper_triangular', False)
        self.type_of_matrix_initialization = config.get('type_of_matrix_initialization', 'identity')
        
        # Create triangular matrix with learnable parameters
        # Simple initialization:
        # - Diagonal: ones (identity for stability)
        # - Triangular part (excluding diagonal): small random values (learnable)
        # - Elements outside triangle: zeros
        
        # Create triangular mask
        if self.upper_triangular:
            mask = torch.triu(torch.ones(T, T, dtype=torch.float32))  # Upper triangular
        else:
            mask = torch.tril(torch.ones(T, T, dtype=torch.float32))  # Lower triangular
        
        # Start with identity matrix (diagonal = 1.0)
        if self.type_of_matrix_initialization == 'identity':
            A_init = torch.eye(T, dtype=torch.float32)

            noise = torch.zeros(T, T, dtype=torch.float32)
            nn.init.normal_(noise, mean=0.0, std=0.02)
            noise = torch.abs(noise)
            noise.fill_diagonal_(0.0)  # Keep diagonal at 1.0
            A_init = A_init + noise
        elif self.type_of_matrix_initialization == 'small_random':
            A_init = torch.zeros(T, T, dtype=torch.float32)
            nn.init.normal_(A_init, mean=0.0, std=0.02)
            A_init = torch.abs(A_init)
        else:
            raise ValueError(f"Invalid type of matrix initialization: {self.type_of_matrix_initialization}")
        

        
        # Apply mask to zero out elements outside triangle
        A_init = A_init * mask
        
        # Store as parameter - all elements in triangular part are learnable
        self.A = nn.Parameter(A_init)
        
        # Store mask for efficient forward pass
        self.register_buffer('triangular_mask', mask)
        self.use_inverse = False  # Will be set if this layer uses inverse
        
    def forward(self, seq):
        """
        Apply full matrix transformation to sequence.
        The matrix A (T x T) is applied to the temporal dimension of the sequence.
        Args:
            seq: (batch, seq_len, hidden_dim)
        Returns:
            transformed sequence: (batch, T, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = seq.size()
        
        # Get the matrix (T x T) and apply triangular mask
        # This ensures all elements in triangular part are learnable
        # Mask is applied in forward so gradients flow to all triangular elements
        A = self.A * self.triangular_mask
        
        if self.use_custom_A_layernorm:
            # Normalize only the triangular part of the matrix
            # Compute norm only over triangular elements
            triangular_elements = A[self.triangular_mask.bool()]
            norm = torch.norm(triangular_elements) + 1e-10
            A = A / norm
        
        # Handle case when seq_len != T
        # If seq_len < T, pad with zeros; if seq_len > T, truncate
        # For lower triangular (causal), pad at the beginning to preserve causality
        # For upper triangular (anti-causal), pad at the end
        if seq_len < self.T:
            padding_size = self.T - seq_len
            padding = torch.zeros(batch_size, padding_size, hidden_dim, 
                                device=seq.device, dtype=seq.dtype)
            if self.upper_triangular:
                # Upper triangular: pad at the end (anti-causal)
                seq = torch.cat([seq, padding], dim=1)
            else:
                # Lower triangular: pad at the beginning (causal)
                seq = torch.cat([padding, seq], dim=1)
            seq_len = self.T
        elif seq_len > self.T:
            # Truncate sequence to match T
            seq = seq[:, :self.T, :]
            seq_len = self.T
        
        # Apply matrix multiplication to temporal dimension
        # seq: (batch, seq_len, hidden_dim) where seq_len == T
        # A: (T, T) - triangular matrix
        # 
        # For causal transformation (matching Conv2d behavior):
        # - Lower triangular A: A @ seq gives causal transformation
        #   output[i] = sum(A[i,j] * seq[j]) for j <= i (depends only on past/present)
        # - Upper triangular A: A @ seq gives anti-causal transformation
        #   output[i] = sum(A[i,j] * seq[j]) for j >= i (depends only on present/future)
        # 
        # To apply A @ seq where seq is (batch, T, hidden_dim):
        # We want: output[b, t, d] = sum_j(A[t, j] * seq[b, j, d])
        # This can be done using einsum: 'ij,bjd->bid'
        
        # Apply A @ seq using einsum for correct broadcasting
        # A: (T, T), seq: (batch, T, hidden_dim)
        # Result: (batch, T, hidden_dim)
        transformed = torch.einsum('ij,bjd->bid', A, seq)  # (batch, T, hidden_dim)
        
        return transformed


# ==================== RoPE: Rotary Position Embedding ====================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_qk(q, k, cos, sin):
    """
    Applies RoPE to Q and K.
    
    Args:
        q: (batch, seq_len, dim) or (batch*heads, seq_len, head_dim)
        k: (batch, seq_len, dim) or (batch*heads, seq_len, head_dim)
        cos: (1, seq_len, dim)
        sin: (1, seq_len, dim)
    
    Returns:
        q_embed, k_embed with applied RoPE
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """
    RoPE: Rotary Position Embedding
    
    Rotates Q and K embeddings based on position in sequence.
    Unlike absolute positional embeddings, RoPE encodes
    relative positions through rotation in the complex plane.
    
    Advantages:
    - Relative positional encoding
    - Extrapolation to long sequences
    - No learnable parameters required (can be made learnable)
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # inv_freq: frequencies for each pair of dimensions
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin for efficiency
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)
    
    def _set_cos_sin_cache(self, seq_len, device=None):
        self.max_seq_len_cached = seq_len
        
        # t: positions [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # freqs[i, j] = position_i * inv_freq_j
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        
        # Duplicate for full dimension: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin: (1, seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)
    
    def forward(self, q, k, seq_len=None):
        """
        Applies RoPE to Q and K.
        
        Args:
            q: (batch*heads, seq_len, head_dim)
            k: (batch*heads, seq_len, head_dim)
            seq_len: optionally, sequence length
        
        Returns:
            q_embed, k_embed: with applied RoPE
        """
        if seq_len is None:
            seq_len = q.shape[1]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=q.device)
        
        # Get cos/sin for required length
        cos = self.cos_cached[:, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :seq_len, :].to(dtype=q.dtype, device=q.device)
        
        # Apply RoPE
        q_embed, k_embed = apply_rotary_pos_emb_qk(q, k, cos, sin)
        
        return q_embed, k_embed


class MultiheadAttentionRoPE(nn.Module):
    """
    MultiheadAttention with RoPE (Rotary Position Embedding).
    
    RoPE is applied to Q and K before computing attention weights.
    This provides relative positional encoding without additional parameters.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_sequence_length=200):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Q, K, V projections (packed)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # RoPE module
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_sequence_length
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None,
                need_weights=True, average_attn_weights=True):
        """
        Args:
            query: (seq_len, batch, embed_dim)
            key: (seq_len, batch, embed_dim)
            value: (seq_len, batch, embed_dim)
            attn_mask: (seq_len, seq_len) — causal mask
            key_padding_mask: (batch, seq_len)
        
        Returns:
            attn_output: (seq_len, batch, embed_dim)
            attn_output_weights: (batch, seq_len, seq_len) or None
        """
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        # Q, K, V projection
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.unflatten(-1, (3, embed_dim)).permute(2, 0, 1, 3).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Reshape for multihead: (seq_len, bsz, embed_dim) -> (bsz*num_heads, seq_len, head_dim)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # ========== KEY DIFFERENCE: RoPE is applied to Q and K ==========
        q, k = self.rope(q, k, seq_len=tgt_len)
        
        # Prepare masks
        combined_mask = None
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            
            # Convert boolean mask to float with -inf
            if attn_mask.dtype == torch.bool:
                combined_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(
                    attn_mask, float('-inf')
                )
            else:
                combined_mask = attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)
            key_padding_mask = key_padding_mask.reshape(bsz * self.num_heads, 1, src_len)
            key_padding_mask = torch.zeros_like(key_padding_mask, dtype=q.dtype).masked_fill_(
                key_padding_mask, float('-inf')
            )
            if combined_mask is None:
                combined_mask = key_padding_mask
            else:
                combined_mask = combined_mask + key_padding_mask
        
        # Compute attention weights: Q @ K^T / sqrt(d)
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.bmm(q, k.transpose(-2, -1)) / scale
        
        # Apply mask
        if combined_mask is not None:
            attn_weights = attn_weights + combined_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Return attention weights
        if need_weights:
            attn_weights_out = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_weights_out = attn_weights_out.mean(dim=1)
            return attn_output, attn_weights_out
        else:
            return attn_output, None


# ==================== CAPE: Context-aware Position Encoding ====================

class CAPE(nn.Module):
    """
    CAPE: Context-aware Position Encoding
    
    Context-dependent positional encoding that computes
    "effective positions" based on attention weights.
    
    Algorithm:
    1. G = 1 - sigmoid(attention_weights) — "gates" showing contextual distance
    2. P = cumsum(G) — cumulative sum for effective positions
    3. Interpolation of positional embeddings based on P
    4. Result is added to attention weights before softmax
    """
    def __init__(self, dim, max_len=None, embedding_dim=None):
        """
        CAPE: Context-aware Position Encoding
        
        Args:
            dim: dimension per head (head_dim)
            max_len: maximum context length
            embedding_dim: input embedding dimension (if projection is needed)
        """
        super().__init__()
        self.max_len = max_len if max_len is not None else dim
        self.pos_emb = nn.Parameter(torch.zeros(1, dim, self.max_len))
        
        # Query projection if embedding_dim is provided (as in reference)
        if embedding_dim is not None:
            self.pre_proj = nn.Sequential(
                nn.Linear(embedding_dim, dim),
                nn.SiLU()
            )

    def forward(self, query, attention_weights, causal_mask=None):
        """
        Args:
            query: (B*H, N, C) — query after reshape for multihead attention
            attention_weights: (B*H, N, N) — Q @ K^T / sqrt(d), before softmax (already with mask)
            causal_mask: (1, N, N) or None — causal mask (True = masked positions)
        
        Returns:
            E: (B*H, N, N) — positional encoding to add to attention_weights
        """
        # Step 1: Compute "gates"
        # G[i,j] is large if tokens i,j are NOT connected (attention is small)
        G = 1 - torch.sigmoid(attention_weights)
        
        # CRITICAL FIX: Zero out G for masked positions
        # so they DON'T affect cumsum (otherwise information leak from future!)
        # For causal attention: G[i,j] = 0 for j > i
        if causal_mask is None:
            # Create causal mask if not provided
            N = attention_weights.size(-1)
            causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=attention_weights.device), diagonal=1)
        G = G.masked_fill(causal_mask, 0.0)
        
        # Step 2: Compute effective positions
        # P[i,j] = sum of G[i,k] for k from j to i (now correct!)
        # Thanks to zeroing G for j > i, cumsum now counts only valid positions
        P = G.flip(-1).cumsum(dim=-1).flip(-1)
        P = P.clamp(max=self.max_len - 1)
        
        # Step 3: Interpolate positional embeddings
        P_ceil = P.ceil().long()
        P_floor = P.floor().long()
        
        # Optional query projection (as in reference)
        if getattr(self, 'pre_proj', None) is not None:
            query = self.pre_proj(query)
        
        # E[i,j] = query[i] @ pos_emb — depends on token content
        # query: (B*H, N, C), pos_emb: (1, C, max_len)
        E = torch.matmul(query, self.pos_emb)  # (B*H, N, max_len)
        
        # Get embeddings for ceil and floor positions
        E_ceil = E.gather(-1, P_ceil)    # (B*H, N, N)
        E_floor = E.gather(-1, P_floor)  # (B*H, N, N)
        
        # Linear interpolation between floor and ceil
        P_frac = P - P_floor  # fractional part of position
        E = P_frac * E_ceil + (1 - P_frac) * E_floor
        
        return E


class MultiheadAttentionCAPE(nn.Module):
    """
    MultiheadAttention with CAPE support (Context-aware Position Encoding).
    
    Based on standard torch.nn.MultiheadAttention, but adds
    CAPE positional encoding to attention weights before softmax.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_sequence_length=200):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Q, K, V projections (packed for efficiency)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # CAPE module
        self.cape = CAPE(
            dim=self.head_dim,
            max_len=max_sequence_length,
            embedding_dim=self.head_dim  # query already in head_dim after reshape
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, 
                need_weights=True, average_attn_weights=True):
        """
        Args:
            query: (seq_len, batch, embed_dim)
            key: (seq_len, batch, embed_dim)
            value: (seq_len, batch, embed_dim)
            attn_mask: (seq_len, seq_len) or (batch*num_heads, seq_len, seq_len)
            key_padding_mask: (batch, seq_len)
            need_weights: if True, returns attention weights
            average_attn_weights: if True, averages weights across heads
            
        Returns:
            attn_output: (seq_len, batch, embed_dim)
            attn_output_weights: (batch, seq_len, seq_len) or None
        """
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        # Q, K, V projection
        # Packed projection for efficiency
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.unflatten(-1, (3, embed_dim)).permute(2, 0, 1, 3).contiguous()  # (3, tgt_len, bsz, embed_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Reshape for multihead: (seq_len, bsz, embed_dim) -> (bsz*num_heads, seq_len, head_dim)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Prepare masks (as in reference: combine attn_mask and key_padding_mask BEFORE CAPE)
        combined_mask = None
        
        # Process attn_mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # (1, tgt_len, src_len)
            
            # CRITICAL FIX: Convert boolean mask to float mask with -inf
            # Boolean True → -inf (blocked), Boolean False → 0 (allowed)
            if attn_mask.dtype == torch.bool:
                combined_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(
                    attn_mask, float('-inf')
                )
            else:
                combined_mask = attn_mask
        
        # Process key_padding_mask and combine with attn_mask (as in reference)
        if key_padding_mask is not None:
            # key_padding_mask: (bsz, src_len) -> (bsz*num_heads, 1, src_len)
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)
            key_padding_mask = key_padding_mask.reshape(bsz * self.num_heads, 1, src_len)
            # Convert bool mask to float (-inf for padding)
            key_padding_mask = torch.zeros_like(key_padding_mask, dtype=q.dtype).masked_fill_(
                key_padding_mask, float('-inf')
            )
            if combined_mask is None:
                combined_mask = key_padding_mask
            else:
                combined_mask = combined_mask + key_padding_mask
        
        # Compute attention weights: Q @ K^T / sqrt(d)
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.bmm(q, k.transpose(-2, -1)) / scale  # (bsz*num_heads, tgt_len, src_len)
        
        # Create boolean causal mask for CAPE (True = masked positions)
        # This mask is needed so CAPE doesn't use information from future when computing cumsum
        causal_mask_bool = torch.triu(
            torch.ones(tgt_len, src_len, dtype=torch.bool, device=q.device), 
            diagonal=1
        )  # (tgt_len, src_len), True for j > i
        
        # Apply combined mask BEFORE CAPE
        if combined_mask is not None:
            attn_weights = attn_weights + combined_mask
        
        # Apply CAPE: add context-dependent positional encoding
        # Pass causal mask so CAPE zeros out G for future positions
        cape_bias = self.cape(q, attn_weights, causal_mask=causal_mask_bool)
        attn_weights = attn_weights + cape_bias
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)  # (bsz*num_heads, tgt_len, head_dim)
        
        # Reshape back: (bsz*num_heads, tgt_len, head_dim) -> (tgt_len, bsz, embed_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Prepare attention weights for return
        if need_weights:
            attn_weights_out = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_weights_out = attn_weights_out.mean(dim=1)  # (bsz, tgt_len, src_len)
            return attn_output, attn_weights_out
        else:
            return attn_output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        
    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, config, maxlen, device): # maxlen is the length of the sequence
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.config = config  # Save config for access to manual_seed in run_epoch
        
        if 'use_pos_emb' in config:
            self.use_pos_emb = config['use_pos_emb']
        else:
            self.use_pos_emb = False
            
        # ALL-APE: adds positional embeddings in each attention block (individual for each block)
        self.use_pos_emb_all_layers = config.get('use_pos_emb_all_layers', False)
        # ALL-APE-shared: one shared positional embedding for all blocks
        self.use_pos_emb_all_layers_shared = config.get('use_pos_emb_all_layers_shared', False)
        # CAPE: Context-aware Position Encoding
        self.use_cape = config.get('use_cape', False)
        # RoPE: Rotary Position Embedding
        self.use_rope = config.get('use_rope', False)
            
        self.use_custom_A = config.get('use_custom_A', False)
        self.use_kimi_attention = config.get('use_kimi_attention', False)

        self.type_of_custom_A = config.get('type_of_custom_A', None) # type of custom A formula
        self.type_of_trinagularity = config.get('type_of_trinagularity', None) # type of triangularity of A matrix
        self.type_of_connection = config.get('type_of_connection', 0) # type of connection between A1 and A2
        
        # Conflict check
        if self.use_kimi_attention and self.use_custom_A:
            raise ValueError("Cannot use both kimi_attention and custom_A at the same time")


            
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num, config['hidden_units'], padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen, config['hidden_units']) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=config['dropout_rate'])

        # ALL-APE: list of positional embeddings for each block
        if self.use_pos_emb_all_layers:
            self.pos_emb_layers = torch.nn.ModuleList()
        # ALL-APE-shared: one shared positional embedding for all blocks
        elif self.use_pos_emb_all_layers_shared:
            self.pos_emb_shared = torch.nn.Embedding(maxlen, config['hidden_units'])
        elif self.use_custom_A:
            self.custom_A1_layers = torch.nn.ModuleList()
            self.custom_A2_layers = torch.nn.ModuleList()
            
            # Create shared full_matrix blocks if needed (one for all blocks in the loop)
            # A1 and A2 are independent - each can have its own type
            config_a1, config_a2 = self._configure_triangularity_configs(config)
            
            # Create shared A1 block if needed (full_matrix or conv with A1_shared flag)
            if config_a1['A1_block_type'] == 'full_matrix':
                self.shared_A1_full_matrix = Custom_A_FullMatrix(maxlen, config['hidden_units'], config_a1)
                self.shared_A1_conv = None
            elif config_a1.get('A1_shared', False):
                # Shared conv block for A1
                self.shared_A1_conv = Custom_A_Conv(maxlen, config['hidden_units'], config_a1)
                self._apply_parametrization_to_conv_block(self.shared_A1_conv, maxlen, config_a1, config_a1['upper_triangular'])
                self.shared_A1_full_matrix = None
            else:
                self.shared_A1_full_matrix = None
                self.shared_A1_conv = None
            
            # Create shared A2 block if needed (full_matrix or conv with A2_shared flag)
            if config_a2['A2_block_type'] == 'full_matrix':
                self.shared_A2_full_matrix = Custom_A_FullMatrix(maxlen, config['hidden_units'], config_a2)
                self.shared_A2_conv = None
            elif config_a2.get('A2_shared', False):
                # Shared conv block for A2
                self.shared_A2_conv = Custom_A_Conv(maxlen, config['hidden_units'], config_a2)
                self._apply_parametrization_to_conv_block(self.shared_A2_conv, maxlen, config_a2, config_a2['upper_triangular'])
                self.shared_A2_full_matrix = None
            else:
                self.shared_A2_full_matrix = None
                self.shared_A2_conv = None


        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        # ALL-APE: list of positional embeddings for each block


        self.last_layernorm = torch.nn.LayerNorm(config['hidden_units'], eps=1e-8)

        for _ in range(config['num_blocks']):
            # ALL-APE: create positional embeddings for each block
            if self.use_pos_emb_all_layers:
                new_pos_emb = torch.nn.Embedding(maxlen, config['hidden_units'])
                self.pos_emb_layers.append(new_pos_emb)
            elif self.use_custom_A:
                # Create A1 block based on block type and shared flag
                if config_a1['A1_block_type'] == 'full_matrix':
                    A1 = self.shared_A1_full_matrix  # full_matrix always shared
                elif config_a1.get('A1_shared', False):
                    A1 = self.shared_A1_conv  # shared conv block
                else:
                    # Create new conv block for each layer
                    A1 = Custom_A_Conv(maxlen, config['hidden_units'], config_a1)
                
                # Create A2 block based on block type and shared flag
                if config_a2['A2_block_type'] == 'full_matrix':
                    A2 = self.shared_A2_full_matrix  # full_matrix always shared
                elif config_a2.get('A2_shared', False):
                    A2 = self.shared_A2_conv  # shared conv block
                else:
                    # Create new conv block for each layer
                    A2 = Custom_A_Conv(maxlen, config['hidden_units'], config_a2)

                # Apply parametrization only to NEW Conv blocks (shared blocks already parametrized)
                # Parametrization logic: if both are conv and not shared, handle connection; otherwise apply independently
                a1_needs_param = config_a1['A1_block_type'] == 'conv' and not config_a1.get('A1_shared', False)
                a2_needs_param = config_a2['A2_block_type'] == 'conv' and not config_a2.get('A2_shared', False)
                
                if a1_needs_param and a2_needs_param and self.type_of_connection == 3:
                    # Special case: shared kernel for both A1 and A2
                    shared_kernel = SharedKernel(maxlen, config_a1)
                    parametrize.register_parametrization(
                        A1.conv, 'weight', 
                        StraightAFromKernel(shared_kernel, upper_triangular=config_a1['upper_triangular'])
                    )
                    parametrize.register_parametrization(
                        A2.conv, 'weight', 
                        StraightAFromKernel(shared_kernel, upper_triangular=config_a2['upper_triangular'])
                    )
                else:
                    # Apply parametrization independently to each NEW conv block
                    if a1_needs_param:
                        self._apply_parametrization_to_conv_block(A1, maxlen, config_a1, config_a1['upper_triangular'])
                    
                    if a2_needs_param:
                        self._apply_parametrization_to_conv_block(A2, maxlen, config_a2, config_a2['upper_triangular'])
        
                self.custom_A1_layers.append(A1)
                self.custom_A2_layers.append(A2)

            new_attn_layernorm = torch.nn.LayerNorm(config['hidden_units'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            if self.use_kimi_attention:
                # Use KimiDeltaAttention instead of MultiheadAttention
                # new_attn_layer = KimiDeltaAttention(config, layer_idx=len(self.attention_layers))
                print(f"not implemented yet")
                exit()
            elif self.use_cape:
                # CAPE: Context-aware Position Encoding
                new_attn_layer = MultiheadAttentionCAPE(
                    embed_dim=config['hidden_units'],
                    num_heads=config['num_heads'],
                    dropout=config['dropout_rate'],
                    max_sequence_length=maxlen
                )
            elif self.use_rope:
                # RoPE: Rotary Position Embedding
                new_attn_layer = MultiheadAttentionRoPE(
                    embed_dim=config['hidden_units'],
                    num_heads=config['num_heads'],
                    dropout=config['dropout_rate'],
                    max_sequence_length=maxlen
                )
            else:
                new_attn_layer = torch.nn.MultiheadAttention(config['hidden_units'],
                                                                config['num_heads'],
                                                                config['dropout_rate'])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(config['hidden_units'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config['hidden_units'], config['dropout_rate'])
            self.forward_layers.append(new_fwd_layer)

            # overall architecture of one block:
            # 1. (attention) LayerNorm
            # 2. MultiheadAttention
            # 3. (forward) LayerNorm
            # 4. PointWiseFeedForward

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

        self.save_attention_weights = False # Flag for managing hooks
        # --- Hooks for saving attention matrices --------------------------
        self.saved_attn = []  # list: [layer0_attn (L,S), layer1_attn, ...]
        self.saved_attn_counts = []  # list of counters for correct online averaging across batches

        def _make_save_hook(layer_idx):
            """
            Returns hook that saves attention matrix.
            If average_attn_weights=True, returns attention weights averaged across heads of shape (L,S) when input is unbatched or (N,L,S)
            If average_attn_weights=False, returns attention weights per head of shape (num_heads,L,S) when input is unbatched or (N,num_heads,L,S).
            so hook averages only across batch.
            """
            def _hook(module, inputs, outputs):
                if self.save_attention_weights == False:
                    return
                # print(f"Hook called for layer {layer_idx}")
                # print(f"module: {module}")
                # print(f"inputs: {inputs}")
                # print(f"outputs[0]: {outputs[0].shape}") # torch.Size([maxlen, batch_size, hidden_dim])
                # print(f"outputs[1]: {outputs[1].shape}") # torch.Size([batch_size, maxlen, maxlen])
                # print(torch.equal(outputs[1][0], outputs[1][2]))


                # for atnlr in outputs[1]:
                #     print(f"atnlr: {atnlr.shape}")
                #     if outputs[1][0] == atnlr:
                #         print(f"outputs[1][0] == atnlr")
                #     else:
                #         print(f"outputs[1][0] != atnlr")
                # exit()

                # MultiheadAttention output: (attn_output, attn_output_weights)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    attn_output_weights = outputs[1]

                    # Initialize structures on first call
                    if len(self.saved_attn) <= layer_idx:
                        self.saved_attn.extend([None] * (layer_idx - len(self.saved_attn) + 1))
                    if len(self.saved_attn_counts) <= layer_idx:
                        self.saved_attn_counts.extend([0] * (layer_idx - len(self.saved_attn_counts) + 1))

                    # attn_w has size (N, L, S), where N=batch_size. Average across batch.
                    attn_avg = attn_output_weights.mean(dim=0)  # -> (L,S)

                    current = attn_avg.detach().cpu()
                    if self.saved_attn[layer_idx] is None:
                        # First sample for this layer
                        self.saved_attn[layer_idx] = current
                        self.saved_attn_counts[layer_idx] = 1
                    else:
                        # Online averaging: m_{n+1} = m_n + (x - m_n)/(n+1)
                        n = self.saved_attn_counts[layer_idx]
                        self.saved_attn[layer_idx] = self.saved_attn[layer_idx] + (current - self.saved_attn[layer_idx]) / (n + 1)
                        self.saved_attn_counts[layer_idx] = n + 1
                    
            return _hook
        # self.attention_layers[0].register_forward_hook(_make_save_hook(0)) # register hook for first layer

        for idx, mha_layer in enumerate(self.attention_layers): # register hook for every layer
            mha_layer.register_forward_hook(_make_save_hook(idx)) # register hook for each layer in attention_layers
            # break

    def _configure_triangularity_configs(self, config):
        """
        Configures A1 and A2 configurations depending on triangularity type.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            tuple: (config_a1, config_a2) - configured configurations for A1 and A2
        """
        # Create new config dictionaries (don't clone original config)
        config_a1 = {}
        config_a2 = {}
        
        # Copy only needed keys from original config
        # if 'use_custom_A_layernorm' in config:
        #     config_a2['use_custom_A_layernorm'] = False
        # if 'type_of_matrix_initialization' in config:
        #     config_a1['type_of_matrix_initialization'] = 'identity'
        #     config_a2['type_of_matrix_initialization'] = 'identity'
        # if 'type_of_kernel_initialization' in config:
        #     config_a1['type_of_kernel_initialization'] = 'normal'
        #     config_a2['type_of_kernel_initialization'] = 'normal'
        
        # Block type selection for A1 and A2 (can be 'conv' or 'full_matrix')
        # A1 and A2 are independent - each can have its own block type
        # Default to conv for backward compatibility
        # config_a1['A1_block_type'] = 'conv'
        # config_a2['A2_block_type'] = 'conv'
        
        # 1st d of model
        if self.type_of_trinagularity == "ll":
            config_a1['A1_block_type'] = 'conv'
            config_a1['upper_triangular'] = False
            config_a1['use_custom_A_layernorm'] = True
            config_a1['kernel_fix_first_to_one'] = True
            config_a1['type_of_kernel_initialization'] = 'normal'


            config_a2['A2_block_type'] = 'conv'
            config_a2['upper_triangular'] = False
            config_a2['use_custom_A_layernorm'] = True
            config_a2['kernel_fix_first_to_one'] = False
            config_a2['type_of_kernel_initialization'] = 'normal'

        # elif self.type_of_trinagularity == "ul":
        #     config_a1['upper_triangular'] = True

        #     config_a2['upper_triangular'] = False
        # elif self.type_of_trinagularity == "lu":
        #     config_a1['upper_triangular'] = False

        #     config_a2['upper_triangular'] = True
        # elif self.type_of_trinagularity == "uu":
        #     config_a1['upper_triangular'] = True

        #     config_a2['upper_triangular'] = True
        elif self.type_of_trinagularity == "al": # 5
            config_a1['A1_block_type'] = 'full_matrix'
            config_a1['use_custom_A_layernorm'] = True
            config_a1['type_of_matrix_initialization'] = 'identity'

            config_a2['A2_block_type'] = 'conv'
            config_a2['upper_triangular'] = False
            config_a2['use_custom_A_layernorm'] = True
            config_a2['kernel_fix_first_to_one'] = False
            config_a2['type_of_kernel_initialization'] = 'normal'


        elif self.type_of_trinagularity == "la": # 4
            config_a1['A1_block_type'] = 'conv'
            config_a1['upper_triangular'] = False
            config_a1['use_custom_A_layernorm'] = True
            config_a1['kernel_fix_first_to_one'] = True
            config_a1['type_of_kernel_initialization'] = 'normal'


            config_a2['A2_block_type'] = 'full_matrix'
            config_a2['use_custom_A_layernorm'] = True
            config_a2['type_of_matrix_initialization'] = 'identity'
        
        # ========== SHARED VERSIONS ==========
        # ll-shared: both L matrices (conv) shared between blocks
        elif self.type_of_trinagularity == "ll-shared":
            config_a1['A1_block_type'] = 'conv'
            config_a1['A1_shared'] = True  # shared between blocks
            config_a1['upper_triangular'] = False
            config_a1['use_custom_A_layernorm'] = True
            config_a1['kernel_fix_first_to_one'] = True
            config_a1['type_of_kernel_initialization'] = 'normal'

            config_a2['A2_block_type'] = 'conv'
            config_a2['A2_shared'] = True  # shared between blocks
            config_a2['upper_triangular'] = False
            config_a2['use_custom_A_layernorm'] = True
            config_a2['kernel_fix_first_to_one'] = False
            config_a2['type_of_kernel_initialization'] = 'normal'

        # al-shared: A1=full_matrix (already shared), A2=conv becomes shared
        elif self.type_of_trinagularity == "al-shared":
            config_a1['A1_block_type'] = 'full_matrix'
            config_a1['use_custom_A_layernorm'] = True
            config_a1['type_of_matrix_initialization'] = 'identity'

            config_a2['A2_block_type'] = 'conv'
            config_a2['A2_shared'] = True  # shared between blocks
            config_a2['upper_triangular'] = False
            config_a2['use_custom_A_layernorm'] = True
            config_a2['kernel_fix_first_to_one'] = False
            config_a2['type_of_kernel_initialization'] = 'normal'

        # la-shared: A1=conv becomes shared, A2=full_matrix (already shared)
        elif self.type_of_trinagularity == "la-shared":
            config_a1['A1_block_type'] = 'conv'
            config_a1['A1_shared'] = True  # shared between blocks
            config_a1['upper_triangular'] = False
            config_a1['use_custom_A_layernorm'] = True
            config_a1['kernel_fix_first_to_one'] = True
            config_a1['type_of_kernel_initialization'] = 'normal'

            config_a2['A2_block_type'] = 'full_matrix'
            config_a2['use_custom_A_layernorm'] = True
            config_a2['type_of_matrix_initialization'] = 'identity'

        else:
            raise ValueError(f"Invalid type of trinagularity: {self.type_of_trinagularity}")
        
        print(f"UUUUUU config_a1: {config_a1}")
        print(f"LLLLLL config_a2: {config_a2}")
        
        return config_a1, config_a2

    def _apply_parametrization_to_conv_block(self, conv_block, maxlen, config, upper_triangular):
        """
        Applies parametrization to conv block through SharedKernel.
        
        Args:
            conv_block: Custom_A_Conv block with .conv attribute
            maxlen: maximum sequence length
            config: configuration for kernel
            upper_triangular: upper triangularity flag
        """
        kernel = SharedKernel(maxlen, config)
        parametrize.register_parametrization(
            conv_block.conv, 
            'weight', 
            StraightAFromKernel(kernel, upper_triangular=upper_triangular)
        )

    def log2feats(self, log_seqs): # log_seqs shape: (U, T) (128, 200)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev)) # seqs shape: (U, T, C) (128, 200, 50)
        seqs *= self.item_emb.embedding_dim ** 0.5 # normalization of embedding vectors



        if self.use_pos_emb:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]) # strange, starts from 0 not from 1 as in the paper (U, T)
            # positions = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1]) # starts from 1 as in the paper 
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
            
        seqs = self.emb_dropout(seqs)

        # -----
        # log_seqs shape: (U, T) (128, 200)
        # seqs shape: (U, T, C) (128, 200, 50)

        # Original user sequence (log_seqs)
    #   log_seqs = [0, 0, 0, 0, 399, 229, 1743, 613, 3, 1119]
        #           ↑  ↑  ↑  ↑   ↑                            
        #        padding    real items

        # timeline_mask will be:
    #   timeline_mask = [True, True, True, True, False, False, False, False, False, False]
        #                ↑                        ↑
        #             padding=True           real items=False

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev) # (U, T)
        # print(f"timeline_mask: {timeline_mask.shape}") # timeline_mask: torch.Size([128, 200])
        # print(f"timeline_mask: {timeline_mask}")
        before_mask = seqs.clone()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim (U, T, C) essentially zero out all embedded padding elements, now padding element is zero vector of dimension (1,C)
        # print('seqs[0]', seqs[0])
        # print('seqs[0].shape', seqs[0].shape) # torch.Size([200, 64])

        # -----
        tl = seqs.shape[1] # hidden dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev)) # (tl, tl) (200, 200) lower triangular matrix of ones
        
        # For KimiDeltaAttention need padding_mask: 1 for valid tokens, 0 for padding
        # timeline_mask: True for padding, False for valid tokens
        # Invert for padding_mask (only if there is real padding)
        kimi_padding_mask = None
        if self.use_kimi_attention:
            # Check if there is any padding
            if timeline_mask.any():
                # There is padding - create mask (1 = valid, 0 = padding)
                kimi_padding_mask = (~timeline_mask).to(seqs.dtype)
            # If there is no padding, pass None (more efficient)
        
        for i in range(len(self.attention_layers)):
            # ALL-APE: prepare positional embeddings for this block (isolated)
            if self.use_pos_emb_all_layers:
                positions = torch.arange(log_seqs.shape[1], device=self.dev).unsqueeze(0).expand(log_seqs.shape[0], -1)
                pos_emb_i = self.pos_emb_layers[i](positions)
                pos_emb_i = pos_emb_i * (~timeline_mask.unsqueeze(-1))  # zero out padding
            # ALL-APE-shared: one shared positional embedding for all blocks
            elif self.use_pos_emb_all_layers_shared:
                positions = torch.arange(log_seqs.shape[1], device=self.dev).unsqueeze(0).expand(log_seqs.shape[0], -1)
                pos_emb_i = self.pos_emb_shared(positions)
                pos_emb_i = pos_emb_i * (~timeline_mask.unsqueeze(-1))  # zero out padding
            
            # if self.use_kimi_attention:
            #     # KimiDeltaAttention works directly with (batch, seq_len, hidden_dim)
            #     # No need to transpose and don't use Q/K/V separately
            #     # 
            #     # IMPORTANT: KimiDeltaAttention has built-in causal mask,
            #     # so attention_mask is not needed (pass only padding_mask)
                
            #     # Pre-norm (LayerNorm before attention)
            #     normed_seqs = self.attention_layernorms[i](seqs)
                
            #     # ALL-APE / ALL-APE-shared: add positional embeddings only for attention (isolated)
            #     if self.use_pos_emb_all_layers or self.use_pos_emb_all_layers_shared:
            #         normed_seqs = normed_seqs + pos_emb_i
                
            #     # KimiDeltaAttention forward
            #     mha_outputs = self.attention_layers[i](
            #         hidden_states=normed_seqs,
            #         attention_mask=kimi_padding_mask,  # padding mask (1=valid, 0=pad) or None
            #     )
                
            #     # Residual connection (use original seqs without positional embeddings)
            #     seqs = seqs + mha_outputs
            else:
                # Standard path with MultiheadAttention
                if self.use_custom_A:  # 3rd d of model
                    if self.type_of_custom_A == 1:
                        seqs_Q = self.custom_A1_layers[i](seqs)
                        seqs_K = self.custom_A2_layers[i](seqs)
                        seqs_V = seqs
                    elif self.type_of_custom_A == 2:
                        seqs_Q = seqs
                        seqs_K = self.custom_A2_layers[i](seqs)
                        seqs_V = seqs
                    elif self.type_of_custom_A == 3:
                        seqs_Q = seqs
                        seqs_K = seqs
                        seqs_V = self.custom_A2_layers[i](seqs)

                    elif self.type_of_custom_A == 4:
                        seqs_Q = self.custom_A1_layers[i](seqs)
                        seqs_K = seqs
                        seqs_V = self.custom_A2_layers[i](seqs)
                    

                    elif self.type_of_custom_A == 5: # correct
                        seqs_Q = seqs
                        seqs_K = self.custom_A1_layers[i](seqs)
                        seqs_V = self.custom_A2_layers[i](seqs)
                    
                    elif self.type_of_custom_A == 7: # correct
                        seqs_Q = seqs
                        seqs_K = self.custom_A2_layers[i](self.custom_A1_layers[i](seqs)) # A2(A1(seqs))
                        seqs_V = seqs

                    # elif self.type_of_custom_A == 41: # wrong
                    #     seqs_Q = seqs
                    #     seqs_K = self.C(seqs)
                    #     seqs_V = seqs
                    # elif self.type_of_custom_A == 42: # correct
                    #     seqs_Q = seqs
                    #     seqs_K = seqs
                    #     seqs_V = self.C_T(seqs)
                
                    # elif self.type_of_custom_A == 43: # correct
                    #     seqs_Q = seqs
                    #     seqs_K = self.C_T(seqs)
                    #     seqs_V = seqs



                    # # 62,73,41 are anti variants of 42,51,43, respectively
                    # elif self.type_of_custom_A == 62: # correct
                    #     seqs_Q = seqs
                    #     seqs_K = seqs
                    #     seqs_V = self.C(seqs) # C = alpha I + A_l A_u
                    
                    # # elif self.type_of_custom_A == 73: # equivalent to 51
                    # #     seqs_Q = seqs
                    # #     seqs_K = self.custom_A1_layers[i](seqs) # both are upper triangular
                    # #     seqs_V = self.custom_A1_layers[i](seqs) # both are upper triangular


                    # elif self.type_of_custom_A == 11: # equivalent to 51
                    #     seqs_Q = seqs
                    #     seqs_K = self.A1(seqs) # both are upper triangular
                    #     seqs_V = self.A1(seqs) # both are upper triangular
                    # elif self.type_of_custom_A == 12: # equivalent to 51
                    #     seqs_Q = seqs
                    #     seqs_K = self.A1(seqs) # both are upper triangular
                    #     seqs_V = self.A1(seqs) # both are upper triangular

                else:
                    seqs_Q = seqs
                    seqs_K = seqs
                    seqs_V = seqs 
                
                seqs = torch.transpose(seqs, 0, 1)
                seqs_Q = torch.transpose(seqs_Q, 0, 1)
                seqs_K = torch.transpose(seqs_K, 0, 1)
                seqs_V = torch.transpose(seqs_V, 0, 1)
                
                # LayerNorm for Q (Pre-LN architecture)
                seqs_Q = self.attention_layernorms[i](seqs_Q)
                
                # ALL-APE / ALL-APE-shared: add positional embeddings to Q, K, V (isolated within block)
                # Positional embeddings are added AFTER LayerNorm for Q, and to K, V directly
                if self.use_pos_emb_all_layers or self.use_pos_emb_all_layers_shared:
                    pos_emb_i_t = torch.transpose(pos_emb_i, 0, 1)  # transpose for compatibility
                    seqs_Q_with_pos = seqs_Q + pos_emb_i_t
                    seqs_K = seqs_K + pos_emb_i_t
                    seqs_V = seqs_V + pos_emb_i_t
                else:
                    seqs_Q_with_pos = seqs_Q
                
                mha_outputs, _ = self.attention_layers[i](seqs_Q_with_pos, seqs_K, seqs_V, 
                                                attn_mask=attention_mask)
                                                # key_padding_mask=timeline_mask
                                                # need_weights=False) this arg do not work?
                if self.use_custom_A and (self.type_of_custom_A == 2 or self.type_of_custom_A == 3):
                    seqs = seqs_Q + self.custom_A1_layers[i](mha_outputs)
                    # seqs = seqs_Q + mha_outputs
                else:
                    # Residual connection: use seqs_Q (without pos_emb) + output attention
                    seqs = seqs_Q + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode = 'BCE'): # for training
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        
        if mode == 'BCE': 
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
            return pos_logits, neg_logits # pos_pred, neg_pred
        
        elif mode == 'CE': 
            return torch.matmul(log_feats, self.item_emb.weight.transpose(0, 1))
        
        

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

    # def score(self, seq): # OG score
    #     maxlen = self.pos_emb.num_embeddings
    #     log_seqs = np.zeros(maxlen, dtype=np.int32) # 0 is padding item
    #     log_seqs[-len(seq):] = seq[-maxlen:]

    #     log_feats = self.log2feats(np.array(log_seqs, ndmin=2, copy=False))
    #     final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

    #     item_embs = self.item_emb.weight # (U, I, C)
    #     logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    #     return logits # preds # (U, I)
    

    def score(self, seq): # score from Scalable SASRec
        '''
        Takes 1d sequence as input and returns prediction scores.
        '''
        maxlen = self.pos_emb.num_embeddings
        log_seqs = torch.full([maxlen], 0, dtype=torch.int64, device=seq.device) # 0 is padding item. in Scalable SASRec it is item_num + 1
        log_seqs[-len(seq):] = seq[-maxlen:]
        log_feats = self.log2feats(log_seqs.unsqueeze(0))
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight  # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits  # preds # (U, I)


    def check_hit(self, target_item, seq, topn):
        with torch.no_grad():
            # predictions = self.score(seq)[0]
            predictions = self.score(torch.LongTensor(seq))[0]


        seen_items = torch.LongTensor(seq).to(self.dev)
        predictions = predictions.put_(
            seen_items, torch.full(seen_items.size(), predictions.min()-1, device=seen_items.device)
        )
        _, predicted_items = torch.topk(predictions, topn)
        (hit_index,) = torch.where(predicted_items == target_item)
        return hit_index, predicted_items.cpu().numpy()
