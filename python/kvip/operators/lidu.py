"""Original LIDU Python binding."""


def lidu(*args, **kwargs):
    """Call the original out-parameter LIDU ABI on Torch's current NPU stream."""
    import nanovllm_dsa_lidu_fused_batch_get_prepare_mtp

    return nanovllm_dsa_lidu_fused_batch_get_prepare_mtp.lidu_fused_batch_get_prepare_mtp_out(*args, **kwargs)
