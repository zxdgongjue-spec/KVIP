# A5LiduFusedBatchGetPrepareMtp — V2 低 miss 优化（classify/compact 融合 + 流水重叠）设计方案

> 评审用设计说明（已按最新重构架构 `csrc/operators/lidu` 重新同步）。
> 本 MR 仅包含本算子（`csrc/operators/lidu`）的 kernel/host 修改与文档，不改变其他算子与对外 ABI。

## 1. 背景与目标

`A5LiduFusedBatchGetPrepareMtp` 是 Ascend 950（A5）上的 MTP 融合算子：每个请求 2--4 条 packed query 行，
对 TopK 候选做 miss/hit 分类、miss 去重、slot 解析与映射安装、以及 BatchGet 描述符构造。
本优化针对 `unique_miss <= 2048` 的常规工作点（低 miss 为主），在不改变 ABI/正确性契约的前提下降低该
算子的 device 时间，重点减少重复 GM 往返与串行等待。

范围边界：本候选不含 cursor MTE3 扩展路径（`>2048` 的额外选槽逻辑沿用既有语义，不在本次优化与验证
声明范围内）；dispatch 需在进入 kernel 前按 launch 级决定。

## 2. 相对重构主线（当前 main 基线）的修改点

保留重构主线已包含的改动（`SOURCE_SCRATCH_STORAGE_WORDS` 扩展、`IsMtpGraphPaddingRow`、放宽的
`MAX_CACHE_TOKENS/MAX_SOURCE_CAPACITY`、`MIN_QUERIES_PER_REQUEST=1`、scratch 复用等），在其之上叠加：

1. **分类与压缩融合（ClassifyAndCompactQuery）**
   - 基线：先 `ClassifyQuery` 分类到独立 GM 区，再单独压缩（compacted tokens 另占 GM）。
   - 本优化：每 lane 一次遍历完成 classify + 原地 compact，分类结果留 UB，按 query 原地压缩，
     **不再需要独立 GM classification/compacted 分片**；同 pass 发布 per-query miss count 供 leader 归并。

2. **前缀构造与 clear/install 流水重叠（Overlap schedule）**
   - 各 AIV 先构造私有 compact 前缀 `[0, ceil(B/2))` 行，**与在飞 clear VF 并行**；
     随后在 **install VF 在飞**时构造剩余行（`LaunchClearPreparedRequest(...)` 异步发起 clear + 事件 `clearDone`）。
   - refill 同步后，所有 core **先发射 descriptor VF 再统一等待**；**core 0 在其 descriptor VF 运行期间**
     用 Scalar/MTE3 写 `miss_counts`、resident lengths、eviction cursors，再与其他 core 一起 join。

3. **前缀控制缓冲独立（prefixControlBuf_）**
   - 新增专用 `prefixControlBuf_`（每行 64B 的独立 UB 缓冲），使 refill 继续独立使用 `controlBuf_`；
   - `op_host` 与 kernel 的 UB 计算相应改为 `2 * REQUEST_CONTROL_WORDS + SOURCE_SCRATCH_STORAGE_WORDS`
     （新增 1 份 request-control 承载前缀控制，同时保留重构主线的扩展 scratch 大小）。

4. **profiling 增补（不影响功能）**
   - 32-field per-lane profiling ABI 增补 `PROFILE_PREFIX_CLEAR / PROFILE_PREFIX_INSTALL / PROFILE_PREFIX_TOTAL`
     （fields 19--21）；timing 结构体字段初始化为 0。

保持不变：输入/输出张量签名；`miss_counts/destination_slots/resident_seq_lengths` 语义；descriptor 公式
（`indices/offsets/output_indices`）；hit-suffix 不重建；clear-before-install / install-before-refill 的全局
屏障顺序；以及 `<=2048` 之外的行为边界。

## 3. 正确性

- 复用既有 golden oracle 与 v9 harness 完整契约校验：miss_counts 每行/总数、destination miss-prefix/hit-suffix、
  descriptor 集合与有效前缀、forward/reverse 映射与唯一 slot、eviction cursor 与 resident length、descriptor tail poison。
- 已在新架构下通过独立 standalone 编译（`build_lidu_fused_batch_get_prepare_mtp.sh`，SOC=ascend950dt_9582，
  compile EXIT=0、Torch 注册 OK）。
- 真机正确性/性能数据沿用同一 kernel 逻辑在 A5 950DT 上的既有验证（见下）：Base/V2 v9 performance 口径均 30/30 PASS；
  契约用例 6/6 PASS。（注：本次移植后建议在空闲卡复跑一次 v9 perf + contract 冒烟确认，见“复现”。）

## 4. 性能（A5 950DT 单卡，同 kernel 逻辑实测）

- 算子级 kernel-only（msprof op_summary Task Duration，B=1/4/8/12）：
  | B | Base(us) | V2(us) | 降幅 |
  |---:|---:|---:|---:|
  | 1 | 27.62 | 20.99 | +24.0% |
  | 4 | 33.11 | 30.70 | +7.3% |
  | 8 | 38.51 | 36.00 | +6.5% |
  | 12 | 39.10 | 35.78 | +8.5% |
  | 整体 | 34.59 | 30.87 | **+10.8%** |
- 整链整步（fused_us，相对 Base 平均）：约 +1.26%（B1..B8 普遍 +1%~3%；B=12 持平）。该融合算子仅占整步的一小部分
  （约 32us/步；链路主耗时在 KVAF Get/Put、LI/QuantLightningIndexer、TensorMove 与 index cache 维护）。

## 5. 影响面与风险

- 仅改本算子实现（`op_kernel`/`op_host`），无 ABI 变化；其他算子未改动。
- `<=2048` 范围；`>2048` 沿用既有语义，不作为本候选验证/集成声明。
- 若仓库对设备源有 baseline gate，本 MR 会报告设备源差异，属预期。

## 6. 验证/复现

- 构建：`bash csrc/operators/lidu/build_lidu_fused_batch_get_prepare_mtp.sh`
- CPU 契约测试：使用仓库既有 lidu tests。
- 真机：v9 `benchmark_contract.py --case-suite performance --intermediate-fetches 8 --qsfa off`（Base vs V2）。
