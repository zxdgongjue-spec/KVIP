# 用近期 Index 补齐融合算子的早阈值

本次在最新 `main` 的融合执行路径上，增加一种早阈值输入：resident 前缀加上近期 Index，并减少准备这一阈值时的重复搬运、TopK和打分工作。URMA 与 UBmem 均保留当前串行实现和融合调度。`--study threshold`比较是否加入近期Index，`--study pipeline`在相同resident+recent候选上比较两项实现优化，`--study skip`单独比较是否复用已提前计算的近期分数。离线工具读取以前采集的数据，历史记录中的优化参数只用于校验和溯源，不用于启用旧实现。

当前已经完成 CPU 侧数据契约、统计和输入合并测试。100 轮统计结果及 NPU 耗时需要在持有原始数据的服务器运行后才能确定，本文不预设收益。

## 为什么要补这一段

融合算子先用一小部分 Index 分数估计最终 TopK 的第 2048 大分数，再用这个早阈值决定哪些 KV 值得提前拉取。当前基线先计算 resident 物理前缀，也就是为常驻 hot cache 保留的 Index 位置；这个前缀按实际实现全部参与打分，不按 `hot_valid` 过滤。

最近产生的 token 由另一块本地尾部缓存保存，代码称它为 ABC。它们的 Index 可以位于 resident 前缀之外。因此，“前轮 TopK 不在 resident 前缀内”并不自动表示 KV 已经丢失：这些 token 可能一直在本地 ABC 中。

本次把前一轮 ABC 起点到本轮有效 query 末尾的 Index 也加入早阈值计算。使用前一轮起点，是为了包含刚从当前 ABC 范围退出的那一段。只用当前 ABC 起点，会再次漏掉这些位置。

每个 query 的候选集合为：

```text
resident 物理前缀
    ∪ [前轮 ABC 起点，本轮该 query 的有效末尾)
```

两部分重叠的位置只算一次；未来 token 不参与。对这些候选使用本轮 query 重新打分，再取第 2048 大值。最终 TopK 仍完整扫描历史，KV、最终选择和缓存更新的正确性要求不变。

这一步可能提高早阈值，减少后来未被最终 TopK 选中的预取。收益必须扣除新增 Index 打分和合并成本；统计上少了多少候选，不能直接换算成时延下降百分比。

## 数据与当前实现的边界

`python/kvip/diagnostics/` 中的读取器兼容历史 schema-3 token-ID trace 和 schema-4 score trace。它们校验归档摘要、数组类型、请求身份、分数排序、TopK 和实际搬运记录，既不重新采集数据，也不修改服务配置。

原始数据保存在服务器的 `/home/c00982601/CacheOS/ubmem-validation/` 下：

| 数据根目录 | 本次使用内容 |
| --- | --- |
| `score-trace-141.62.38.24-7lg40xtj/trace` | DP0–7；前驱轮1，加100轮统计；G10真实输入使用采集轮51 |
| `real-input-r51-g2g7.dKZUcF/trace` | G2、G7；采集轮1为前驱，轮2保存真实输入 |
| `real-input-r51-miss5-miss10.hj_6zeym/trace` | G17、G4；采集轮1为前驱，轮2保存真实输入 |

这里的轮号是 `series.json` 中的采集轮号。补采目录中的轮2并不等于原100轮目录中的轮51；工具保留各自的实际 scheduler 轮号和来源，比较实际测得的 miss。

五组回放把8个 DP 各2个请求合并为单卡 BS16、每请求4个 query（MTP3）。每组保留真实 Q、Index keys/scales、weights 和原始分数参考；KV 内容由确定性字节生成。选择 `--layer-index 0` 后，每个 token 的 KV 宽度为656B，测量的是共享组中一层的拉取负载。原来的 Index 输入、TopK、miss身份和DP内共享页关系保持不变；四层组的原始 KV 字节跨度会按单层布局重算。

生成的快照额外包含 `pre.threshold_bounds`，类型为 `int32[B,2]`。每行保存前轮 ABC 起点与本轮最大有效 query 末尾。每个 query 再按自己的有效末尾裁剪。工具按请求ID、owner、DP和层组匹配前驱，要求相邻 scheduler 轮次；遇到缺失前驱、请求复用、跳轮或近期逻辑ID不再位于同号物理Index位置时，直接报告错误，不猜测范围，也不悄悄截短。

快照中的 `pre.reserved_hot` 仍作为最新执行路径需要的输入恢复。它用于保护可能被预取覆盖的 hot 槽，两个对照都按相同方式准备；本次不会省略它来缩短基线或实验路径。

## 先运行100轮CPU统计

以下命令只读原始 trace，不使用 NPU，也不启动端到端服务。结果目录必须是新目录。

```bash
(
    set -euo pipefail
    repo=/home/c00982601/CacheOS/vLLM-DSA
    data=/home/c00982601/CacheOS/ubmem-validation
    trace="$data/score-trace-141.62.38.24-7lg40xtj/trace"
    result="$data/recent-index-survey-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    export PYTHONPATH="$repo/third_party/KVIP/python${PYTHONPATH:+:$PYTHONPATH}"

    /usr/local/python3.12.13/bin/python3 -B -m kvip.diagnostics.threshold_survey \
        --root "$trace" \
        --result-dir "$result" \
        --workers 4 \
        --expect-dp 0 1 2 3 4 5 6 7 \
        --expect-rounds 101 \
        --expect-layers 78
)
```

首轮只提供前驱状态，统计覆盖采集轮2–101。组内共享Index的各层使用同一选择统计；组汇总每个请求只计一次，展开到逐层表时按层数拆分字节，避免把四层组误算成四份独立请求。

`summary.json` 使用schema2。终端显示每组的主要结果，详细数据保存在 `requests.csv`、`queries.csv`、`layers.csv`、`round_layers.csv` 和 `miss_bins.csv`：

| 指标 | 含义 |
| --- | --- |
| `gap_mean`、`recent_gap_mean` | 最终真实阈值减去resident或resident+recent早阈值；只能在同组分数量纲内比较 |
| `gap_equal_percent`、`recent_equal_percent` | 早阈值与最终阈值完全相等的query比例 |
| `hot_extra_candidates`、`recent_extra_candidates` | 通过严格 `score > threshold` 判断，但未出现在该请求实际Q4 TopK并集中的去重候选数 |
| `*_extra_bytes` | 上述数量乘该组实际token宽度；`*_extra_656_mib_per_request`统一换成单层656B口径 |
| `recent_extra_reduction_percent` | 相对resident策略减少的多余候选比例；它是流量候选变化，不是时延收益 |
| `retired_abc_coverage_percent` | 近期范围覆盖前轮ABC起点到当前ABC起点之间Index位置的比例 |
| `previous_topk_recent_coverage_percent` | resident+recent集合覆盖前轮Q4 TopK有效位置的比例；它不是hot命中率 |
| `recent_added_positions_per_query`、`recent_range_positions_max` | 平均新增打分位置数及最大完整近期区间长度，用来判断新增计算量 |

统计使用捕获的sortable BF16整数分数做精确排序，避免负数排序或浮点转换改变边界。同分且已被实际TopK选中的token不会被算成多余候选。Q4内去重，不同请求分别计数。

候选模型会应用当前因果边界、可读DDR地址、ABC排除和可回收hot槽等限制，但不模拟窗口时序及预取数量上限。`actual_*` 是历史运行真正完成的搬运，可能含重复；`hot_*`、`recent_*` 是没有窗口配额的候选集合。二者的差异不能直接解释为某个新策略的真实传输量。保留的 `seed_hot`、`rolling` 列是CPU分析参考，既不启用旧内核，也不作为本轮native对照开关。

## 合并五组单层回放输入

以下仍是CPU操作。每组输出一个独立快照；8个DP的DDR槽编号会分配独立命名空间，DP内部原有共享页别名保留。目录不存在或归档内容与预期不符时会明确失败。

```bash
(
    set -euo pipefail
    repo=/home/c00982601/CacheOS/vLLM-DSA
    data=/home/c00982601/CacheOS/ubmem-validation
    out="$data/recent-index-five-inputs-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    export PYTHONPATH="$repo/third_party/KVIP/python${PYTHONPATH:+:$PYTHONPATH}"
    mkdir -p "$out"

    for group in 2 7 10 17 4; do
        case "$group" in
            2|7)
                trace="$data/real-input-r51-g2g7.dKZUcF/trace"
                round=2
                ;;
            10)
                trace="$data/score-trace-141.62.38.24-7lg40xtj/trace"
                round=51
                ;;
            17|4)
                trace="$data/real-input-r51-miss5-miss10.hj_6zeym/trace"
                round=2
                ;;
        esac
        /usr/local/python3.12.13/bin/python3 -B -m kvip.diagnostics.score_replay \
            --root "$trace" \
            --round "$round" \
            --group "$group" \
            --dp-ranks 0 1 2 3 4 5 6 7 \
            --numa-node 2 \
            --layer-index 0 \
            --output "$out/g${group}-input" \
            --quiet
    done
    printf 'FIVE_INPUTS_READY=%s\n' "$out"
)
```

`--numa-node 2`记录24节点NPU4的本轮目标设置。它是生成快照的元数据，不会替CPU服务选择分配器或保证物理页落在该节点。换卡或换机器时，应按实际DDR亲和关系设置，不能从逻辑device编号推导NUMA节点。

## 使用当前统一构建做设备对照

`compare` 的当前接口只接受一个 `--build-dir`，两个后端共用当前支持URMA和UBmem的KVAF `build-npu`。两种后端都需要本地KVAF CPU服务。先加载当前KVIP的 `scripts/env.sh`，确认该CPU服务已启动，并把 `RECENT_INPUT_ROOT` 设置为上一节输出的 `FIVE_INPUTS_READY` 完整目录；此处不创建或选择历史构建。

如需启动CPU服务，在单独终端加载同一份 `scripts/env.sh` 后运行以下命令，并保持该终端运行。这里省略 `--ub-device`，由当前URMA实现自动枚举设备；不要从物理NPU编号硬编码网卡名称。端口必须空闲，已有服务时应核对其构建与配置，不要重复启动。

```bash
(
    set -euo pipefail
    kernel="${KVAF_BATCH_GET_KERNEL:?先加载当前KVIP/scripts/env.sh}"
    build="${kernel%/bin/kvaf_execute_batch_get_kernels.o}"
    test "$build" != "$kernel"
    service="$(dirname "$build")/build-cpu/runtime/kvaf_cpu_service"
    test -x "$service"
    exec numactl --membind=2 "$service" \
        --listen 127.0.0.1 --port 28999 --memory-page-mode thp
)
```

当前KVAF的URMA与UBmem使用不同的CPU内存分配路径。`--memory-page-mode thp`控制URMA的CPU分配策略；UBmem通过HAL申请Host P2P DDR并导出共享句柄，未读取这个选项。UBmem源码请求huge-page属性并按2MiB对齐分配长度，这不能代替实测物理页大小或THP覆盖率。`numactl --membind=2`设置CPU服务的进程内存策略，也不能据此断言驱动内部HAL分配一定落在NUMA节点2。本轮不引入旧THP分配器，实际页属性与HAL NUMA放置仍未测量。

每个后端的融合resident/recent对照在同一进程中复用同一Runtime、DDR导入和arena，因此这组A/B不因重新分配DDR而改变页布局。串行与融合以及两个后端之间分别创建Runtime，不能声称它们共享同一物理arena；跨后端耗时包含当前各自分配路径的影响。

以下命令保持物理NPU4、逻辑device0，对五组各运行URMA串行、URMA融合resident/recent、UBmem串行、UBmem融合resident/recent。融合对照只改变早阈值范围，两个串行基线都用32个worker，URMA使用当前默认post size 128和inflight 2048。

```bash
(
    set -euo pipefail
    repo=/home/c00982601/CacheOS/vLLM-DSA
    input_root="${RECENT_INPUT_ROOT:?设置为FIVE_INPUTS_READY输出的完整目录}"
    kernel="${KVAF_BATCH_GET_KERNEL:?先加载当前KVIP/scripts/env.sh}"
    build="${kernel%/bin/kvaf_execute_batch_get_kernels.o}"
    test "$build" != "$kernel"
    test -s "$build/bin/kvaf_execute_batch_get_kernels.o"
    kvaf_server="${KVAF_SERVER:-127.0.0.1:28999}"
    result="$input_root/compare-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    export PYTHONPATH="$repo/third_party/KVIP/python${PYTHONPATH:+:$PYTHONPATH}"
    export ASCEND_RT_VISIBLE_DEVICES=4

    for group in 2 7 10 17 4; do
        /usr/local/python3.12.13/bin/python3 -B -m kvip.diagnostics.compare \
            --snapshot "$input_root/g${group}-input" \
            --build-dir "$build" \
            --kvaf-server "$kvaf_server" \
            --device 0 \
            --soc Ascend950DT_9582 \
            --serial-workers 32 \
            --warmup 2 \
            --iterations 10 \
            --result-dir "$result/g${group}"
    done
)
```

终端保留各路径耗时、miss、预取量、无效预取量及阈值变化。成功时临时native日志会移除；失败时显示必要的日志尾部。每组 `summary.json` 保留比较结果。耗时包含完整设备算子链及新增近期Index打分与合并，不包含快照恢复、CPU调度和输入H2D；`graph_payload_GBps`以整条设备链为分母，不能当作Fetch阶段的独立带宽。

## 分别验证流水复用和统一TopK

近期Index与resident使用同一份query、weights和query scale。原实现先完成近期段，再释放并重新加载这些输入来计算resident段；流水复用让它们跨这两个物理区间保持有效。Key和key scale仍在各物理段末尾归还缓冲，跨核完成信号与最终历史扫描保持原职责。

原早阈值准备先对resident取TopK，再把近期分数与这2048项合并。统一TopK将resident和近期两个不重叠分数区间接成一个候选流，用现有精确TopK处理；它仍按每个query的因果末尾裁剪，范围不满足条件时保留原路径。完整历史打分、ScoreGM写入和最终TopK没有从计时中删除。

`--study pipeline`自动运行以下三档；三档均启用resident+recent，预取预算均为512、每窗口64，并显式关闭`recent_skip`，保持原三档的对照语义：

| 结果名 | `KVIP_RECENT_PIPELINE` | `KVIP_RECENT_UNIFIED_TOPK` | 对照目的 |
| --- | --- | --- | --- |
| `before` | 0 | 0 | 本次近期阈值实现优化前的执行方式 |
| `pipeline` | 1 | 0 | 单独计量跨段输入复用 |
| `unified` | 1 | 1 | 在复用基础上计量统一TopK的增量 |

这里的`before`已经计算近期阈值，不是只用resident的基线。每个后端的三档共享同一Runtime和DDR arena，各档在计时外恢复相同输入。两项编译开关进入构建缓存标识，加载后还会读取native getter核对实际开关。原始有效LI分数、早阈值及最终阈值要求逐bit相同，三档最终TopK要求顺序一致；合成KV字节与Index联动继续由原oracle验证。

已有上一节五组输入和CPU服务后，可直接运行以下命令；无需重做CPU统计或输入合并：

```bash
(
    set -euo pipefail
    repo=/home/c00982601/CacheOS/vLLM-DSA
    input_root="${RECENT_INPUT_ROOT:?设置为FIVE_INPUTS_READY输出的完整目录}"
    kernel="${KVAF_BATCH_GET_KERNEL:?先加载当前KVIP/scripts/env.sh}"
    build="${kernel%/bin/kvaf_execute_batch_get_kernels.o}"
    test "$build" != "$kernel"
    test -s "$build/bin/kvaf_execute_batch_get_kernels.o"
    kvaf_server="${KVAF_SERVER:-127.0.0.1:28999}"
    result="$input_root/pipeline-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    export PYTHONPATH="$repo/third_party/KVIP/python${PYTHONPATH:+:$PYTHONPATH}"
    export ASCEND_RT_VISIBLE_DEVICES=4

    for group in 2 7 10 17 4; do
        /usr/local/python3.12.13/bin/python3 -B -m kvip.diagnostics.compare \
            --snapshot "$input_root/g${group}-input" \
            --build-dir "$build" \
            --kvaf-server "$kvaf_server" \
            --device 0 \
            --soc Ascend950DT_9582 \
            --serial-workers 32 \
            --study pipeline \
            --warmup 2 \
            --iterations 10 \
            --result-dir "$result/g${group}"
    done
)
```

每组包含两个当前串行基线和六个融合结果。`pipeline`收益以`before`为分母，统一TopK的增量以`pipeline`为分母，组合收益以`before`为分母；与串行的比较另列。三档固定顺序执行，短测仍可能受时钟、温度及外部负载影响。收益尚需服务器实测，不能把CPU模型或编译通过解释为性能通过，也不能将该单组设备链的变化直接换算成服务TPOT变化。

## 单独验证近期分数复用

计算早阈值时，近期区间已经产生了本轮真实分数。`recent_skip`让后续历史扫描复用这些分数，减少同一query与同一Index的重复计算；对应的分类、进度发布和窗口握手仍需完成。该开关保持原窗口协议与预取预算，不把需要等待的传输移到计时区间之外。

阈值确定后，先用已有分数写出近期位置的物理候选位图，再扫描剩余区间。计算和分类共同采用向外对齐的128-token块，首尾多出的有效位置也一并处理，未来位置明确写零。只有本轮已经完整计分、分类的块才会被跳过；即使某个窗口全部跳过，仍在原窗口位置发布完成进度并执行Fetch服务。当前快路径覆盖完整history的Q4布局，其他布局保留原重算路径。近期KV并未因此成为hot hit，缺失数据仍按原Fetch协议拉取。

`--study skip`只运行以下两档融合对照。两档都启用resident+recent、流水复用和统一TopK，只有`recent_skip`不同：

| 结果名 | `recent_pipeline` | `recent_unified_topk` | `recent_skip` |
| --- | --- | --- | --- |
| `before` | true | true | false |
| `after` | true | true | true |

每个后端两档共享同一Runtime和DDR arena，恢复同一份冻结输入，继续验证全部有效分数、初始和最终阈值、每轮有序TopK及KV/Index输出。`kvip_recent_skip()`返回值必须与请求一致，防止加载旧二进制。原`pipeline`研究显式关闭该开关；`threshold`研究使用当前默认开启值，不再代表本节的skip-off基线。

使用现有五组输入与CPU服务运行：

```bash
(
    set -euo pipefail
    repo=/home/c00982601/CacheOS/vLLM-DSA
    input_root="${RECENT_INPUT_ROOT:?设置为FIVE_INPUTS_READY输出的完整目录}"
    kernel="${KVAF_BATCH_GET_KERNEL:?先加载当前KVIP/scripts/env.sh}"
    build="${kernel%/bin/kvaf_execute_batch_get_kernels.o}"
    test "$build" != "$kernel"
    test -s "$build/bin/kvaf_execute_batch_get_kernels.o"
    kvaf_server="${KVAF_SERVER:-127.0.0.1:28999}"
    result="$input_root/skip-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    export PYTHONPATH="$repo/third_party/KVIP/python${PYTHONPATH:+:$PYTHONPATH}"
    export ASCEND_RT_VISIBLE_DEVICES=4

    for group in 2 7 10 17 4; do
        /usr/local/python3.12.13/bin/python3 -B -m kvip.diagnostics.compare \
            --snapshot "$input_root/g${group}-input" \
            --build-dir "$build" \
            --kvaf-server "$kvaf_server" \
            --device 0 \
            --soc Ascend950DT_9582 \
            --serial-workers 32 \
            --study skip \
            --warmup 2 \
            --iterations 10 \
            --result-dir "$result/g${group}"
    done
)
```

每组显示六行：两个当前串行基线及URMA、UBmem各自的before/after。`skip`收益以同后端before的整图平均耗时为分母，与串行的收益另列。实际预取量、final读取量及无效预取量仍从完成记录统计：计算推进速度改变后，异步预取与最终读取的分工和实际流量可能变化，不能预先假定两档搬运字节完全相等。整图计时包含复用所需的读取、分类、同步和完成处理；没有NPU实测前，不宣称达到35%目标。

## 判定是否值得保留

CPU统计先回答“加入近期位置后，早阈值是否更接近真实阈值，多余候选减少多少”。随后在同一输入上运行最新融合路径的resident与resident+recent两个版本，设备端必须计入新增打分和合并工作，并验证全部有效LI分数、最终TopK与合成KV字节。

原始分数读回按有效因果前缀逐bit比较；未来及padding区域不参与比较。真实KV没有保存，因此该回放验证确定性合成KV的搬运正确性，并保留逻辑访问负载；它不还原原机器的物理页分布或服务并发干扰。

如果早阈值改善明显，但完整融合调用没有变快，说明新增计算成本或其他阶段抵消了省下的搬运。此时应依据实际耗时保留或撤回优化，不能用候选下降替代35%时延目标的验证。两组对照使用相同的最新main执行方式，确保测得的是近期Index早阈值这一项变化的净效果。
