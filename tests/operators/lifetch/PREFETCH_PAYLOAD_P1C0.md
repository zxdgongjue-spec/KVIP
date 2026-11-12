# P1-C0 — V9 混合变长 Prefetch CQ / Mapping / DDR-HBM Payload 验证

基线：P1-B `779bdec53cb23a965c714199e7dfb041d6d504e2`，A5
Ascend950DT_9582 上已经通过两轮 `B4/H8192/R2048/G32`、
`prefetch=64`、`miss-percent=0` 的 LI-only 验证：
每轮 `16/16` Epoch-Lane 完成，其中 `10/16` skipped；
Event 每轮 `candidateCount=6144`、`postedCount=24`。
这些统计 **尚未证明 DMA Token 数据正确**。

## 本轮代码影响范围

仅新增独立 CPU 审计和 CLI 测试开关；不改动 AscendC LI、
Rotation Epoch、KVAF Transport、Resolver、Index Update、生产服务。

`--prefetch-payload-audit` 只能在
`--li-only-varlen --epoch-audit-profile --prefetch-tokens 64` 时启用。
保留原来严格的 `--miss-percent 0`，不直接升级到真实 Final Miss。

### P0：确保物理地址映射有定义

原 LI-only Fixture 没有显式填充 `rotating.lru_slots[B,2048]`，
但设备端 `PhysicalSlot()` 对前 2048 个 Speculative Slot 会查询这张表。
默认零张表不能作为多条 DMA 的不同物理目的地。

测试专用修复：每个请求初始化恒等物理 victim 映射
`lru_slots[b,i]=i`，并在完整两轮测试之间保持不变。
它**仅用于可复现的数据链路测试**，不声称是生产 LRU eviction 策略；
某些预取写入 Resident Victim 区域是允许的，最终缓存一致性需要
Resolver/Index Update 才能判断。

### P1：CQ 与 Row/Mapping 三方对账

每轮 `v9_wait + Stream Synchronize` 后必须检查：

1. Prefetch CQ Status 魔数及 `result=0`；Row `error=0`、
   `pending=0`。
2. 每个请求 `Status.processed_request_count = Row.requests`，
   `Row.slots = Row.requests = sum(Row.epochCount[e])`。
3. `mapping[b,token] >= 0` 的数量等于 `Row.requests`；
   `token < actual_key_length[b]`；没有重复逻辑 Slot 或物理 Slot。
4. Event 中 `sum(postedCount)` 与 `sum(Row.requests)` 一致。
5. `completed_batch_count` 计的是 **CQ Group 数量**，不是 Token；
   每个有请求的 Row 应观察到至少一个成功完成的 CQ Group，
   不要求它等于 `Row.requests`。

记录 `row`、`token`、logical/physical slot、hot/cold 分类、
`epoch_count`、`requests`、`completed_groups`。
`mapping` 中存放的是**逻辑 Slot**：
当 `resident <= slot < resident+2048` 时，
`physical_slot=lru_slots[b,slot-resident]`；其他 Slot 直接映射。
校验每个物理 Slot 在同一个请求内唯一，避免写入覆盖。

### P2：仅对已确认完成的 Token 逐字节比较

对上述映射所标识的每个 Token，在 CQ 完成之后：

```text
mapping[b, token]  ->  logical_slot
                          |
                          v
                  PhysicalSlot(lru_slots)
                          |
                          v
owner.output[b, physical_slot, :656] [HBM]
                          ==
owner.session.cpu_payload[b, token, :656] [DDR 源 Fixture]
```

只把已映射的少量 Token（当前预期每轮 24 个）从 NPU 复制到
CPU；不读取整个 26 MiB 输出 Buffer。
每个请求记录 SHA256；字节不一致打印第一处
`row/token/physical_slot/byte_offset/HBM/DDR`。
若 Mapping/CQ 本身失败，拒绝执行 Payload 读取。

异常时先保留诊断 JSON。CQ 未确认完成不能提前释放 KVAF 资源；
本轮在已有 CQ 错误检查通过之后调用
`owner.mark_completed(completion_verified=True)`，
然后只读核验 Payload。若 Payload 校验失败，标记 FAIL，但由于
CQ 已确认完成，可以安全进入原有释放流程。

## 设备验证命令

```bash
cd /home/z00945955/Fuse/gitcode_proj/third_party/KVIP
git fetch origin
git switch --track -c experiment/lifetch-prefetch-payload-p1c0 \
  origin/experiment/lifetch-prefetch-payload-p1c0
source scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=6,7
export KVAF_SERVER=127.0.0.1:28999

python -m pytest \
  tests/operators/lifetch/test_prefetch_payload_audit.py \
  tests/operators/lifetch/test_epoch_audit.py \
  tests/operators/lifetch/test_rotation_epoch_schedule.py \
  tests/operators/lifetch/test_varlen_lengths.py -q

python tests/operators/lifetch/test_lifetch.py \
  --transport urma --device 0 --kvaf-server "$KVAF_SERVER" \
  --batches 4 --history 8192 --resident 2048 --index-heads 32 \
  --miss-percent 0 --prefetch-tokens 64 --intermediate-fetches 8 \
  --warmup 0 --iterations 1 --trace-rounds 0 \
  --li-only-varlen --epoch-audit-profile --prefetch-payload-audit \
  --output /tmp/kvip-prefetch-payload-p1c0
```

本轮产生：
- `prefetch-payload-mixed_qk_round1.json`
- `prefetch-payload-mixed_qk_round2.json`
- 既有 `epoch-audit-*.json` 和 `li-varlen-stage.jsonl`

每轮通过标准：`Event.SelectedForPost=Row.requests=Mapping.count`，
每个非零请求 CQ Group 完成、无 Row/CQ 错误，物理 Slot 唯一，
全部已映射 Token 的 **656 字节 Payload 精确一致**。
两轮还必须保持原有 Native/V9 TopK、Padding、Q/K Buffer 地址等价。

本轮未验证：真实 Final Miss、Resolver/Final Fetch、Index Update、
4 层 Index Share、缓存最终一致性、性能收益。不能用本轮结果
代表它们已经通过。
