# 用近期 Index 补齐融合算子的早阈值

本分支以独立核对的 KVIP `main`（`13d2898892652382ea36a0652030874efc31097d`）为基础，增加一种早阈值输入：resident 前缀加上近期 Index，并减少准备这一阈值时的重复搬运、TopK和打分工作。URMA 与 UBmem 均保留当前串行实现和融合调度。`--study threshold`比较是否加入近期Index，`--study pipeline`在相同resident+recent候选上比较两项实现优化，`--study skip`单独比较是否复用已提前计算的近期分数。离线工具读取以前采集的数据，历史记录中的优化参数只用于校验和溯源，不用于启用旧实现。

后续比较默认只运行 URMA；UBmem 需显式选择。已有16节点五组结果验证了上一版跳过近期重算的正确性，但URMA耗时反而增加1.35%–4.10%。本次修正其执行位置与循环结构，收益仍以新的实卡结果为准。

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

默认每组包含URMA串行和三档URMA融合结果；显式选择两个后端时才同时测试UBmem。`pipeline`收益以`before`为分母，统一TopK的增量以`pipeline`为分母，组合收益以`before`为分母；与串行的比较另列。三档固定顺序执行，短测仍可能受时钟、温度及外部负载影响。收益尚需服务器实测，不能把CPU模型或编译通过解释为性能通过，也不能将该单组设备链的变化直接换算成服务TPOT变化。

## 单独验证近期分数复用

计算早阈值时，近期区间已经产生了本轮真实分数。`recent_skip`让后续历史扫描复用这些分数，减少同一query与同一Index的重复计算；对应的分类、进度发布和窗口握手仍需完成。该开关保持原窗口协议与预取预算，不把需要等待的传输移到计时区间之外。

上一版在阈值确定后，先用已有分数写出近期位置的物理候选位图，再扫描剩余区间；下节说明URMA对此执行顺序的修正。计算和分类共同采用向外对齐的128-token块，首尾多出的有效位置也一并处理，未来位置明确写零。只有本轮已经完整计分、分类的块才会被跳过；即使某个窗口全部跳过，仍在原窗口位置发布完成进度并执行Fetch服务。当前快路径覆盖完整history的Q4布局，其他布局保留原重算路径。近期KV并未因此成为hot hit，缺失数据仍按原Fetch协议拉取。

`--study skip`只运行以下两档融合对照。两档都启用resident+recent、流水复用和统一TopK，只有`recent_skip`不同：

| 结果名 | `recent_pipeline` | `recent_unified_topk` | `recent_skip` |
| --- | --- | --- | --- |
| `before` | true | true | false |
| `after` | true | true | true |

每个后端两档共享同一Runtime和DDR arena，恢复同一份冻结输入，继续验证全部有效分数、初始和最终阈值、每轮有序TopK及KV/Index输出。`kvip_recent_skip()`返回值必须与请求一致，防止加载旧二进制。原`pipeline`研究显式关闭该开关；`threshold`研究显式指定其skip选项，不代表未改动的主线。

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

默认每组显示三行：URMA串行及其融合before/after；显式选择两个后端时才显示六行。`skip`收益以同后端before的整图平均耗时为分母，与串行的收益另列。实际预取量、final读取量及无效预取量仍从完成记录统计：计算推进速度改变后，异步预取与最终读取的分工和实际流量可能变化，不能预先假定两档搬运字节完全相等。整图计时包含复用所需的读取、分类、同步和完成处理；没有NPU实测前，不宣称达到35%目标。

## 判定是否值得保留

CPU统计先回答“加入近期位置后，早阈值是否更接近真实阈值，多余候选减少多少”。随后在同一输入上运行最新融合路径的resident与resident+recent两个版本，设备端必须计入新增打分和合并工作，并验证全部有效LI分数、最终TopK与合成KV字节。

原始分数读回按有效因果前缀逐bit比较；未来及padding区域不参与比较。真实KV没有保存，因此该回放验证确定性合成KV的搬运正确性，并保留逻辑访问负载；它不还原原机器的物理页分布或服务并发干扰。

如果早阈值改善明显，但完整融合调用没有变快，说明新增计算成本或其他阶段抵消了省下的搬运。此时应依据实际耗时保留或撤回优化，不能用候选下降替代35%时延目标的验证。分支内开关对照用于定位单项变化；未修改主线的绝对基准使用下述独立上游源码运行，不能由关闭某个开关替代。


## 独立主线对照与 URMA 近期分数按窗口复用

KVIP的主线版本必须直接从KVIP远端main核对，不能使用vLLM-DSA安装脚本里可能滞后的固定版本。此次核对的main是`13d2898892652382ea36a0652030874efc31097d`，也是本功能分支的祖先。后续main前进时，应重新核对差异并更新基准，不把旧SHA继续称为最新。

过去`--study skip`中的`before`已经启用近期阈值、共享Q/weights/query scale加载及统一TopK，只关闭近期分数复用。它是上一实现的对照，不是原主线。串行生产算子未改动，但32 workers、post=128、inflight=2048是本次统一回放配置，不等于所有服务的默认配置。

新`--study upstream`使用独立、干净的上游checkout。主线串行及融合在独立子进程中加载该checkout的生产Python模块；融合库由该checkout原有构建入口从原生源码编译。诊断包继续负责输入恢复、计时和校验，禁止用功能分支的生产模块补齐上游导入。已有LIDU二进制只在其源文件与上游完全一致的前提下复用。KVAF使用相同已构建运行时，因此两边不会因更换Get实现而形成虚假收益。

输出包含四档：主线串行、主线融合、近期阈值旧实现、近期阈值新实现。前两档记录上游SHA、实际模块路径及原生库摘要；后两档共享一次DDR导入和arena。跨主线与功能分支的独立进程不共享同一arena，结果会明确保留此限制。

本轮URMA优化解决两处具体问题：

1. 旧实现为了省掉近期区间的第二次打分，先给该区间生成预取候选标记，然后才允许首批Fetch启动。近期区间通常在历史尾部，这项工作却挡在开头。新实现先发布resident准备完成；扫描真正走到近期区间时，直接读取早先保存的本轮分数，完成该窗口的分类，然后照常发布进度。首批Fetch因此无需等待尾部标记。
2. 旧实现让每一个冷区间tile都判断是否跳过，并反复修改循环范围。新实现分成普通前段、近期复用段、普通后段；普通段恢复原有连续循环，只有近期段做分数读取和分类。减少的是遍布全历史的调度开销，候选阈值和最终TopK没有改变。

近期分数从已保存的score缓冲区读取，不能假设早TopK的临时UB仍然有效：中间Fetch服务会复用那块临时空间。复用段按原Fetch窗口切分，每个窗口仍完成分类、进度发布和握手；即使窗口中没有新的矩阵计算，也不能跳过这些工作。读取已有分数、分类与同步均位于完整设备链计时内。

本轮仅更改URMA的近期复用路径。生产构建默认关闭`recent_skip`，由回放的after显式开启，待实测收益确认后再决定是否默认启用。不会因本轮URMA试验改变UBmem执行方式。

验证要求：主线与新实现的有效LI分数及有序TopK一致；新旧近期实现的初始阈值也逐bit一致；每档继续校验合成KV字节及Index更新。主线只用resident，因此其初始阈值允许与近期版本不同。串行与融合仍单独报告已有的BF16同分边界差异。

完整设备链包括必要的设备准备、早阈值计算、LI、TopK、Fetch、Index更新和输出，不包含快照读取、状态恢复、CPU调度或输入H2D。该数字是单层656B回放耗时，不是端到端TPOT。URMA异步完成时序可能改变预取/最终读取分工，所以实际读字节和无效预取仍逐档统计，不假定减少重复打分必然减少总时延。

## host16 实测回顾与精确 Kth 路径

`host16-urma-main-recent.2nmYkZ`使用同一组BS16、Q4、656B冻结输入。所有融合版本的有效分数、最终有序TopK、合成KV及Index校验通过；两个近期版本的早阈值也一致。表中均为关闭插桩的完整设备链平均微秒：

| 组 | main串行 | main融合 | 近期阈值＋重算 | 近期阈值＋晚分类复用 | main/近期重算的误预取占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G2 | 332.419 | 404.669 | 409.800 | 418.078 | 38.99% / 3.20% |
| G7 | 750.155 | 686.115 | 668.649 | 679.349 | 36.28% / 24.75% |
| G10 | 379.537 | 435.066 | 432.880 | 446.639 | 32.88% / 9.07% |
| G17 | 545.685 | 469.118 | 456.807 | 469.469 | 40.58% / 21.09% |
| G4 | 473.576 | 429.739 | 438.018 | 446.579 | 23.37% / 9.45% |

误预取占比定义为`unused_prefetch / prefetch`，不是cold miss率。近期阈值将误预取数量减少34.27%～94.82%，总读字节减少6.38%～35.89%；但晚分类相对直接重算又增加8.278～13.759微秒，而误预取量基本不变。因此URMA继续默认关闭`recent_skip`，下一实验不再将这条已劣化路径作为优化后的实现。

省传输不一定省同等时间：URMA预取本来可以与LI计算重叠，减少的传输可能原本就在被掩盖的部分；增加的近期打分及早阈值准备则在首个有效阈值之前。晚分类还增加分数GM读回、分类、bitmap写回及同步，替代的只是少量已处于矩阵流水中的QK tile。现有总耗时不能把这10.792微秒平均额外开销精确归因到某一条同步指令。

新`KVIP_RECENT_THRESHOLD_ONLY`开关默认关闭，仅在URMA回放after中启用。原早TopK已用四个直方图VF求得精确第K大值，随后却继续筛选GT/EQ候选、整理编号、Gather分数、Cast/Remap，再对2048分数取最小值。预取实际只消费第K大值、保护bitmap及进度，生产路径不消费早期2048个分数与编号；最终TopK仍从完整scoreGM独立计算。

新路径复用main的相同四个VF、排序编码、零填充与排名公式，直接将精确Kth写入原阈值bank。删除后续早期名单工作及BS16×Q4共0.75MiB的GM名单写出；不改变prefix布局、阈值发布、hot槽保护、Fetch窗口或最终TopK。联合区间超出已有UB容量、resident过短或特殊dtype时走原精确路径，不截断输入。此实现不从旧优化分支搬入代码，也不更改KVAF、串行或UBmem。

`--study threshold-only --transports urma --upstream-root ... --upstream-revision ...`显示四档，名称与含义固定：

- `main-serial`：独立main生产代码的串行路径。
- `main-resident`：独立main融合，只用resident计算早阈值。
- `recent-recompute`：近期阈值、共享query流水、统一TopK，保留历史扫描中的近期重算。
- `recent-kth-only`：同一近期实现，只删除未使用的早期名单工作。两档近期实现共享Runtime/DDR arena。

因此`upstream-fused`与旧`recent-before`从来不是同一实现；新名字避免将功能分支里的before误认为主线。summary保留机器可读的before/after字段，同时增加明确显示名。

`--diagnostic-replays 1`在正式计时完成后另做一次带计数的重放，不把它混入10次完整耗时。LI时间槽12/13记录早阈值选择及发布，14/15记录最终TopK；早阈值区间不包含前面的QK矩阵计算，最终TopK区间止于最终分类之前。已有计数用于最终分类、mapping等待、Fetch提交/CQ等待和Index阶段。输出原始device cycles及可用性，不猜测时钟换算，也不将并行core阶段相加冒充完整时延。未修改main没有新增标记，相应项显示不可用。诊断中的局部流水join会扰动重叠，仅用于定位。

通过标准仍是同输入的分数、初始/最终阈值、TopK和KV/Index输出一致，且关闭插桩的完整链有实测收益。CPU模型与CANN编译只能证明部分正确性和可构建性，不能提前宣称达到相对串行35%的目标。

## 仅求 Kth 的实测与早分数留片上

host16、NPU4的`host16-urma-exact-kth.WwL6oz`结果显示，精确Kth确实减少了早阈值选择工作，但完整调用尚未超过main融合：

| 组 | 早阈值median cycles：原名单→仅Kth | 完整链：原名单→仅Kth，μs | main融合，μs |
| --- | ---: | ---: | ---: |
| G2 | 8860→6590 | 424.803→422.359 | 405.637 |
| G7 | 9458→7008 | 689.554→695.578 | 691.071 |
| G10 | 9320→7306 | 452.350→451.178 | 433.835 |
| G17 | 9904→7461 | 472.817→470.843 | 467.587 |
| G4 | 10006→7102 | 457.591→455.289 | 428.355 |

早阈值的22%～29%周期降低来自独立诊断重放；该区间不包含前面的近期/resident矩阵计算，也不包含ProcessTopK入口的第一次MTE3→MTE2等待。因此不能据此说整个早期准备快了25%。完整时间中四组只减少1.2～2.4微秒，G7增加6.0微秒；一次诊断样本不足以解释该变化。所有融合分数、有序TopK、KV/Index输出通过校验，近期版本初始阈值逐bit一致。

本轮目标收敛为resident＋近期方案优于未修改main融合，不再同时研究串行、UBmem或传输队列。新`--study seed-ub`只测以下四档融合：

| 显示名 | 精确Kth | 编译新增TopK诊断分支 | 早分数直接留在现有UB |
| --- | --- | --- | --- |
| main-resident | main原实现 | 无新增分支 | main原实现 |
| current-kth | 是 | 是，运行时关闭 | 否 |
| release-kth | 是 | 完全不编译 | 否 |
| seed-ub | 是 | 完全不编译 | 是 |

这里UB指NPU片上计算缓冲区，不是UBmem传输；四档都使用URMA。本研究拒绝额外diagnostic replay，避免诊断重放改变后续档硬件状态。主线独立加载自己的生产模块，三个近期版本共享一次DDR恢复和arena。所有档保持相同窗口预算、QK计算和最终TopK。

原来的resident/近期分数VF先写小输出银行，再DMA写入scoreGM；求早阈值时又从GM回读到TopK合并银行。新路径在首次Fetch服务前借用原本空闲的合并银行，让同一分数VF直接写入连续的resident＋近期区域。完整scoreGM仍照常写出，保留最终TopK和验证所需的每个分数；早Kth直接读取已经在片上的数据。仅resident部分，BS16×Q4即可减少约2MiB的GM回读，另省近期分数回读及相应加载等待。没有增加Index或KV副本，也没有把准备移出计时。

快路径限每AIV一个query、完整history、resident按128对齐且联合输入不超过既有trunk＋K容量。近期计算的首尾对齐位置、超出本query因果终点的位置和padding，仅在GM写出完成后从UB候选中清零；GM原始分数不受影响。精确Kth消费后立即使该片上缓存失效，后续Fetch仍可复用原空间。不满足条件时走现有精确Kth路径。回放根据实际geometry和阈值边界报告eligible query数量，避免把完全未进入快路径的测试当成优化结果。

`current-kth→release-kth`单独衡量删除新增诊断检查的影响；`release-kth→seed-ub`衡量保留早分数的增量。诊断关闭时原MarkTopK没有执行barrier，但仍读取GM参数并作分支；收益不能提前归因于某项指令或假定能解释全部回退。只有关闭诊断的完整链测量能判定新实现是否超过main。

每档输出三项带宽统计：总搬运字节/完整设备链时间、有用搬运字节/完整设备链时间、以及有用/总搬运字节的比例。前两项单位为十进制GB/s，第三项为有效传输占比；这些数据不能当作Fetch独占带宽或硬件峰值利用率。无匹配峰值和纯传输计时证据时，硬件利用率显示N/A。减少无效预取会令总搬运等效带宽下降，优化目标仍是同等正确输出下的完整耗时。
