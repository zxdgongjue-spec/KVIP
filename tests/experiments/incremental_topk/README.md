# 单核增量 TopK 实验

独立于正式 LIFetch 的实验，不连接 KVAF、不改生产 kernel。默认单 AIV 顺序处理
2 请求 × Q4，共 8 行。每行输入已有 Top2048 分数和原 token ID，加上 12288 个
新增分数，输出新的 Top2048 分数及原 token ID。复用生产 `topk::LITopk<uint16_t>`，
不重新实现筛选算法；现有 TopK 从 16384 个历史分数中产生。

## 真机

在 KVIP 根目录运行（只依赖 CANN，不依赖 Python、ACL Python 或 KVAF 服务）：

```bash
source /usr/local/Ascend/cann/set_env.sh
build="$PWD/tests/results/incremental-topk/$(date +%Y%m%d_%H%M%S)"
cmake -S tests/experiments/incremental_topk -B "$build" \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599
cmake --build "$build" --parallel 2
"$build/incremental_topk" --device 4 --blocks 1 --rows 8 \
  --new-scores 12288 --warmup 4 --iterations 50 | tee "$build/result.log"
```

`--rows` 是每核顺序处理的 Q 数，默认 8；`--blocks` 是 AIV 数，默认 1。
例如 `--blocks 8 --rows 8` 是 BS16、每核2请求的独立并行实验。
`--new-scores` 支持 256～16384，256 对齐；12K 指 12288。

终端先验证每行的分数多重集合、token ID、ID/分数对应关系及唯一性，再输出
无核内打点的 graph mean/p50/p99（µs）。计时含单 kernel 图执行，不含输入生成、
输入拷贝、校验和 graph capture。计时后再校验，并单独执行一次带打点版本，输出
各行加载、筛选、ID映射、写回的原生 `GetSystemCycle` ticks，以及整个核的跨度。
不假定 ticks 频率；软件 Model 的时间读取 profiler 时间轴，不用 CPU 墙钟换算。

默认输入为固定 seed=43 的有限 BF16 sortable keys 均匀分布，含重复分数。
这是一项筛选开销实验，不能直接代表真实预取收益。可用 `--seed` 改变输入，
或 `--input FILE` 提供 little-endian uint16 sortable keys，shape 为
`[blocks*rows, 16384+new_scores]`：前16384项用于构造旧Top2048，后面为新增分数。
该格式是实验分数输入，不是正式 replay 格式。输出新增 token ID 从16384开始。

## 本地 CANN Model

用独立构建目录设置 `-DRUN_MODE=sim`，然后运行：

```bash
msprof op simulator \
  --application="$build/incremental_topk --sim --rows 8 --new-scores 12288" \
  --soc-version=Ascend950PR_9599 --output="$build/profile" --timeout=10
```

`--sim` 只运行一次带打点 kernel 和结果校验，避免在软件模型中重复50轮。
输入加载量：每Q 28 KiB分数 + 8 KiB旧ID，共36 KiB；输出12 KiB。
8个Q合计读288 KiB、写96 KiB，不包含打点和控制开销。

## 本地结果（2026-11-16）

CANN 9.1，Ascend950PR_9599 软件 Model，1 AIV、8 Q、每Q旧2048＋新增12288：

| 分数输入 | 整个 kernel | 平均每Q | 正确性 |
| --- | ---: | ---: | --- |
| 均匀 sortable BF16 keys，seed43 | 29.14 µs | 3.64 µs | PASS |
| 当前紧凑生成器分数，seed43 | 29.16 µs | 3.65 µs | PASS |

第二组使用生成器的 Q/Index 计算分数并量化为 BF16 sortable keys，
4Q复制为2请求；没有按指定 miss 率重新排列 hot 槽。
这些是带打点版本的 Model 时间，不是真机测量；本次 Model 的
GetSystemCycle 返回0，因此分段 ticks 不用于估时。真机应使用上面的 graph
测时入口。实验包含 GM 输入加载、筛选、ID映射和结果写回，不包含 LI 打分与通信。
