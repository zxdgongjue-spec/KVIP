# 来源

迁入自 vLLM-DSA 提交 `769ca3c` 的
`csrc/attention/nanovllm_a5_ops/a5_lidu_fused_batch_get_prepare_mtp`。
保留原生 kernel、OPP 构建和独立 Torch 注册，不使用 Python 模型替代性能基线。
沿用仓库 Apache-2.0 许可证；本目录仅用于独立基线，不替换框架已有注册。
