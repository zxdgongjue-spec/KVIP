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

## 稀疏候选：十组对比

编译一次后运行下面的循环；仍为单AIV、2请求×Q4。

```bash
for mode in bitmap compact; do
  for pct in 1 5 10 20 50; do
    "$build/incremental_topk" --device 4 --blocks 1 --rows 8 \
      --new-scores 12288 --mode "$mode" --candidate-percent "$pct" \
      --warmup 4 --iterations 50 | tee "$build/${mode}-${pct}.log"
  done
done
```

- `bitmap`：读完整新增分数和LI原生predicate位图（每token占2 bit，偶数bit为有效位），
  每128分数直接加载32字节mask，用向量Squeeze压缩分数与原位置，
  再调用公共TopK。计时包含位图展开与压缩。
- `compact`：直接读已压缩分数及原token ID。计时不包含上游生成压缩输入的成本。
- 两版使用相同seed、相同分数和随机分散的候选位置。保留比例表示可能参与TopK的
  新分数比例，不是最终入选比例，更不是缓存miss率。
- 1/5/10/20/50%分别保留123/614/1229/2458/6144项；筛选长度向256对齐，尾部填最小sortable key。
  所有排除项严格低于旧Top2048最低分；候选项严格高于该分数。CPU参考仍使用全部新增分数，
  不依赖位图过滤，核验输出分数集合、原ID、唯一性及对应关系。
- 核内`select_ticks`在bitmap模式包含压缩和TopK。模型时间从profiler读取，
  真机graph计时使用无打点版本。默认不指定`--mode`仍为原dense实验。

每Q逻辑输入（含旧Top2048和ID）：bitmap恒定39.00 KiB；compact五档分别为
13.50/16.50/19.50/27.00/48.00 KiB（含256对齐padding）。两版均输出12 KiB。

### 旧实验记录（93069d6，已被向量压缩替换）

以下为旧1-bit位图＋逐bit SIMT收集实现，不能代表当前LI原生位图的开销。
本地CANN 9.1 / Ascend950PR_9599 Model，seed43，单AIV顺序处理8Q，六组全部校验PASS：

| 候选比例 | 新候选/Q | bitmap总耗时 | compact总耗时 | bitmap平均/Q | compact平均/Q |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10% | 1229 | 35.71 µs | 18.29 µs | 4.46 µs | 2.29 µs |
| 20% | 2458 | 46.76 µs | 20.58 µs | 5.85 µs | 2.57 µs |
| 50% | 6144 | 84.39 µs | 26.67 µs | 10.55 µs | 3.33 µs |

取profiler的duration_time（包含模型stall），不是running_time。
旧bitmap收集实现没有带来收益；已压缩输入能降低筛选耗时，但上游压缩代价仍需在集成时计入。
前面的29.14 µs全量结果使用原始均匀分数；本组为了精确控制候选比例修改了门槛以下分数，
因此六组内部是配对对比，与旧全量结果的差值只作参考。

### LI原生位图＋向量压缩（当前实现）

CANN 9.1 / Ascend950PR_9599 Model，单AIV顺序处理8Q；10组全部通过完整输入TopK校验。

| 候选比例 | 新候选/Q | bitmap总耗时 | compact总耗时 | bitmap平均/Q | compact平均/Q |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 123 | 26.47 µs | 16.50 µs | 3.31 µs | 2.06 µs |
| 5% | 614 | 26.91 µs | 17.39 µs | 3.36 µs | 2.17 µs |
| 10% | 1229 | 27.37 µs | 18.26 µs | 3.42 µs | 2.28 µs |
| 20% | 2458 | 28.46 µs | 20.50 µs | 3.56 µs | 2.56 µs |
| 50% | 6144 | 31.44 µs | 26.59 µs | 3.93 µs | 3.32 µs |

bitmap使用与LI `ClassifyScore` / `StoreAlign(hits, selected)`一致的uint16 predicate存储：
每128个分数32字节，token i对应 `word[i/16]` 的 `2*(i%16)` bit。
`Compact`直接加载原生mask，无需转成1-bit格式；读入某个LI窗口时，调用方按原query stride和窗口偏移定位即可。
压缩过程使用两次向量遍历，分别输出分数和uint16窗口内位置，再将最终TopK位置映射回token ID。

新fixture使用严格大于hot阈值的候选，与LI的GT条件一致；旧实验候选允许等于阈值，
因此旧表保留为历史参考，当前两种输入的五档结果才是同一份输入的配对对比。
向量压缩的独立VF时间从trace提取，不包含加载、TopK、最终ID映射和输出写回。
五档向量压缩VF的8Q合计分别为4.67/4.68/4.68/4.68/4.68 µs（平均每Q约0.58 µs）。
旧逐bit收集在10/20/50%时分别为14.62/24.62/59.38 µs；该慢路径已删除。
