# 真实 score trace：解析与回放导出

在 KVIP 仓库根目录执行，仅需 Python 3.11+ 和 NumPy，不启动 NPU。
工具复用 `tests/operators/lifetch/support/workloads.py` 的 schema-4 校验和回放装配。
不支持把只有分数的记录转换成真实 Q/Index。

## 1. 找到实际可用的样本

```bash
python scripts/lifetch_score_trace.py \
  --root /home/c00982601/CacheOS/ubmem-validation
```

可传整个数据父目录，也可直接传含 `dp0/series.json` 等文件的 `trace` 目录。
打印每个 `TRACE_ROOT` 下保存真实 Q/Index 的 round、group、DP、合计 BS 和原始 token 字节数。
此步骤只读取元数据；文档中的 24 节点目录不一定存在于当前机器。

说明文档列出的代表样本如下，最终以脚本重算为准：

| 来源 | 采集 round | group | 文档 cold miss |
|---|---:|---:|---:|
| real-input-r51-g2g7 | 2 | 2 | 2.12% |
| real-input-r51-g2g7 | 2 | 7 | 19.82% |
| score-trace | 51 | 10 | 3.33% |
| real-input-r51-miss5-miss10 | 2 | 17 | 5.77% |
| real-input-r51-miss5-miss10 | 2 | 4 | 9.81% |

## 2. 解析和导出

将 `trace` 设置为第一步打印的真实目录，并选择该目录列出的 round/group。
以下以低 miss G2 为例：

```bash
trace=/实际路径/real-input-r51-g2g7.dKZUcF/trace
python scripts/lifetch_score_trace.py --root "$trace" --round 2 --group 2
python scripts/lifetch_score_trace.py --root "$trace" --round 2 --group 2 \
  --layer-index 0 --output /tmp/lifetch-g2-real
```

默认选中所有 DP；DP8、每卡 BS2 会在回放入口合并为单卡 BS16。
可用 `--dp-ranks 0 1` 只选两个 DP。合并后的 BS 必须不超过 16。
`--layer-index 0` 将合成 KV 投影到单层 656B；不指定则保留原始 656B/2624B。
真实 Q、Index、权重、分数、初始缓存状态和页槽编号不改动，不重算分数。
归档中的历史搬运记录仍是原始层组口径，新的回放搬运量由运行时重新统计。

输出逐请求统计和总计：

- Demand：Q1～Q4 的 captured TopK 并集，跨 Q 去重。
- Hot：调用前有效 hot 槽中的需求；ABC-only：仅 ABC 尾部命中的需求。
- Cold：Demand − Hot − ABC-only；总 miss 率按总 cold / 总 demand 计算。
- Candidate hot：有效 hot 命中中位于 LRU 候选槽的数量，不是实际补取次数。
- Cold w/o DDR：cold 需求中没有有效 DDR 地址的数量；非零不能导出成功。

转换校验包括归档哈希、数组形状、TopK 与原始分数的一致性、逻辑映射和页共享装配。
不同 DP 的页地址空间隔离，同一 DP 的页别名保留。
每个输出是独立前态的一轮，归一化为回放 round 1；原始采集轮和调度轮保留在元数据中。
`conversion.json` 保存来源、统计和 COMPLETE/FAILED 状态。输出目录必须是新目录。

## 3. 当前统一入口回放

```bash
python tests/operators/lifetch/test_lifetch.py diagnose --level 2 \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 \
  --input real --trace-root /tmp/lifetch-g2-real --group 2 \
  --round-range 1:1 --intermediate-fetches 4 \
  --prefetch-tokens 1024 --prefetch-rank-start 1024 \
  --warmup 4 --iterations 20 --timeout-seconds 600
```

切换数据时对应修改 trace-root/group；不传 `--miss-percent`，命中状态来自采集。
各样本比较时保持窗口、预算、设备和计时参数一致。真实 KV 未采集，搬运验证使用
确定性合成 C8 KV；真实 LI 输入与选集保留。CPU 转换成功不代替真机正确性验证。
