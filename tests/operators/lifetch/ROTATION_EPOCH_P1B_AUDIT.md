# P1-B | V9 Rotation Epoch 执行审计（仅测试链路）

基线：`8b4069161c3001c7741685f0c3c9801125991b52`，
A5 Ascend950DT_9582 上 B4 / H8192 / R2048 / G32 / Prefetch64，
两轮混合 Q/K 均 PASS，16/16 Event complete、10/16 skipped。

## 为什么新增审计

此前可以确认每个 Event 的 `core,epoch,complete,skipped`，
但原始 CMake **`-DV9_INTERMEDIATE_PROFILING=0`**。
因此 Event 的 `candidateCount/postedCount` 即使为零也不代表没有工作，
不能据此推断零 Miss 或 Prefetch 覆盖率。

本轮只允许独立 `--li-only-varlen --prefetch-tokens 64` Fixture
启用 `--epoch-audit-profile`，把 CMake 编译参数
`V9_INTERMEDIATE_PROFILING` 改为 1。生产编译默认仍为 0。

## 读取 Event ABI

`context.h` 中 `Event` 固定 128 字节，按 `[rounds,lanes,128]`
保存。设备端即使没有编译 Profiling，也会填
`core,epoch,complete,skipped`：

| Offset | 字段 | 是否必须编译 Profiling |
|---|---|---|
| 0/8 | begin/end，uint64 | 是 |
| 16/20 | core/epoch，uint32 | 否 |
| 24/28 | complete/skipped，uint32 | 否 |
| 32 | waitBegin，uint64 | 是 |
| 40/48/56 | scanTicks/candidateTicks/postTicks，uint64 | 是 |
| 64/72/80 | bitmapBytes/candidateCount/postedCount，uint64 | 是 |

只有 `skipped=0` 且 Profiling 确认开启，才读取下方计数。
`skipped=1` 的记录不代表曾执行 `FastService`，对应
`candidateCount` 和 `postedCount` 直接使用 JSON null。

`postedCount` 是 FastService 选入发起路径的条目累计数，
**不代表成功 DMA、CQ 完成或最终数据写入**。

每轮生成两个文件（写完后才确认 PASS）：
- `epoch-audit-mixed_qk_round1.json` 和 `.txt`
- `epoch-audit-mixed_qk_round2.json` 和 `.txt`

验证：Event 数量、`complete==1`、`skipped\in(0,1)`、
预期固定 Core 分配、`recorded_epoch`、
`selected_for_post_count <= candidate_count`，保留原来的
Native/V9 TopK、Padding、Row Error 和 CQ 检查。

## A5 验证命令

```bash
cd /home/z00945955/Fuse/gitcode_proj/third_party/KVIP
git fetch origin
git switch --track -c experiment/lifetch-rotation-epoch-audit-p1b \
  origin/experiment/lifetch-rotation-epoch-audit-p1b
source scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=6,7
export KVAF_SERVER=127.0.0.1:28999

python -m pytest tests/operators/lifetch/test_epoch_audit.py \
  tests/operators/lifetch/test_rotation_epoch_schedule.py \
  tests/operators/lifetch/test_varlen_lengths.py -q

python tests/operators/lifetch/test_lifetch.py \
  --transport urma --device 0 --kvaf-server "$KVAF_SERVER" \
  --batches 4 --history 8192 --resident 2048 --index-heads 32 \
  --miss-percent 0 --prefetch-tokens 64 --intermediate-fetches 8 \
  --warmup 0 --iterations 1 --trace-rounds 0 \
  --li-only-varlen --epoch-audit-profile \
  --output /tmp/kvip-rotation-p1b-profile
```

本轮需重新编译测试用 V9 LI Kernel 一次，不能套用
`V9_INTERMEDIATE_PROFILING=0` 的旧二进制；编译选项与 SHA256
会写入报告或执行日志。新选项只在本次测试启用，不覆盖运行中的
vLLM/KVAF 安装。

## P1-C 准入设计（此轮不执行）

先用零 Miss/Profiling 确定各 Epoch 的计算与服务覆盖，
再创建不影响 P1-A 的独立 **真实 Miss** Fixture：

1. 生成固定、可验证的 DDR KV payload 和物理地址映射，保证
   每个 Batch 的有效 K、Query Ends、历史边界一致。
2. 在 TopK 中注入可控 Non-resident Token（从单个请求、
   单个窗口、少量 Miss 开始）；确认 Index/Resident/Source
   的真实关系。
3. 比较 Prefetch Candidate / Selected / CQ Completed；
   不得将 `postedCount` 当作 DMA 成功。
4. 检查返回 HBM Slot 的完整 payload，缺失 Fetch 由
   Resolver/Final Fetch 兜底，保证 Index Update 正确。
5. 再测试跨轮状态复用和 B4 全混合变长，最后才接入 GLM-5.3
   的 4 层 Index Share。

边界：P1-B **不修改** Epoch 映射、TopK、KVAF 传输、Resolver，
不验证真实 Fetch、TPOT 或动态工作迁移性能。
