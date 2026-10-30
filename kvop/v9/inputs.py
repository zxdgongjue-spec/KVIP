# SPDX-License-Identifier: Apache-2.0
"""Adapt supported model heads to the V9 Q4/64-head compute tile.

The additional heads have zero query and weight and unit dequant scale.
They contribute exactly zero to sum_h(weight_h * relu(q_h @ k)); real
head order and values are unchanged. Never reshape a 32-head row as 64.
"""

KERNEL_HEADS = 64
HEAD_DIM = 128
SUPPORTED_HEADS = (32, 64)


def pack_query_heads(torch, query, weights, scales, tokens):
    if query.ndim == 2 and query.shape[1] == HEAD_DIM and query.shape[0] % tokens == 0:
        query = query.reshape(tokens, query.shape[0] // tokens, HEAD_DIM)
    if query.ndim != 3 or query.shape[0] != tokens or query.shape[2] != HEAD_DIM:
        raise ValueError("V9 query must have shape [B*4, H, 128]")
    heads = query.shape[1]
    if heads not in SUPPORTED_HEADS or weights.numel() != tokens * heads or scales.numel() != tokens * heads:
        raise ValueError("V9 requires 32/64 heads and matching weights/scales")
    if query.dtype != torch.float8_e4m3fn:
        raise ValueError("V9 query must use float8_e4m3fn")
    weights = weights.reshape(tokens, heads).to(torch.bfloat16)
    scales = scales.reshape(tokens, heads).float()
    if heads == KERNEL_HEADS:
        return query.contiguous(), weights.contiguous(), scales.contiguous()
    packed_query = torch.zeros((tokens, KERNEL_HEADS, HEAD_DIM), dtype=torch.uint8, device=query.device).view(
        query.dtype
    )
    packed_weights = torch.zeros((tokens, KERNEL_HEADS), dtype=torch.bfloat16, device=query.device)
    packed_scales = torch.ones((tokens, KERNEL_HEADS), dtype=torch.float32, device=query.device)
    # Byte copies also support backends without FP8 strided-copy kernels.
    packed_query.view(torch.uint8)[:, :heads].copy_(query.contiguous().view(torch.uint8))
    packed_weights[:, :heads].copy_(weights)
    packed_scales[:, :heads].copy_(scales)
    return packed_query, packed_weights, packed_scales
