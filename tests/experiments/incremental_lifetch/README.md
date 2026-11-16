# 增量位图 TopK：URMA 端到端 A/B

基线保留当前融合 LIFetch。实验将每个预取窗口的新增位图候选与已有 Top2048
合并，保存分数、原 token ID 和每个序列分片的处理边界。最终 TopK 接续处理
尚未纳入的区间，不重新扫描已处理历史。中间预取按更新后的每 Q 阈值过滤候选；
等于阈值的候选保留，最终 TopK 的并列分数按现有正确性检查验证。

压缩和 ID 还原直接复用 `incremental_topk` demo 抽出的
`csrc/operators/common/lifetch/incremental_compact.h`；TopK 仍调用
`topk::LITopk<uint16_t>`。LI 位图编码、核分配、Resolver、Index 更新和 KVAF
提交/完成路径保持现有实现。原始分数和位图只读取已发布区间，不改写生产者位图。

跨窗口保存在原 hot TopK 的 GM 区域，原 hot 阈值另存且不修改。每 Q 另有
128 字节的来源指针、双分片游标及当前阈值，随 round init 清零。窗口服务核
沿用当前请求分配；前序服务未完成时立即跳过，不推进消费游标。后续窗口跨过
已跳过的事件，从最近实际完成的游标接续；最终 TopK 等待仍在途的状态并补齐。
原始分数按现有 TopK scratch 可用容量分块加载，下一 Q 预加载最多12288项。
压缩候选跨分片累积，
在候选缓冲装满、相对 ID 跨度超过65536或窗口结束时才做 TopK。压缩前过滤低于
已有 TopK 阈值的候选。核心 tiling 不变。

额外 UB 保存 token ID 映射、1 KiB 元数据，以及预取阶段的下一 Q 输入。
原 LI 缓冲区偏移不变。BS16、heads32 的预取阶段原分配177 KiB，追加39 KiB输入
后为216 KiB，低于218 KiB预算。输入包括旧 TopK 分数4 KiB、ID 8 KiB、新分数
24 KiB和原生位图3 KiB。其他几何根据实际分配计算余量，不足时不启用预加载。
最终 TopK 阶段不分配输入缓冲，复用分类输出缓冲作为旧 ID 缓冲，含元数据共
222240字节。没有逐 bit 标量展开，也没有逐候选 GM 原子操作。

## 运行

在 KVIP 根目录，初始化 CANN 和现有依赖环境后：

```bash
bash tests/experiments/incremental_lifetch/compare.sh \
  /vllm-workspace/vllm-ascend/third_party/KVAF \
  4 127.0.0.1:28885 1,2,4,8,16 10
```

参数依次为 KVAF 根目录、device、server、BS 列表、单个 miss 百分比。
脚本为实验复制独立源码目录；不会覆盖原算子。两边均运行统一入口的
`diagnose --level 3`，包含无打点完整图计时、正确性检查、独立打点 replay
和阶段日志。每版编译缓存跨 BS 复用，结果位于 `tests/results/incremental-lifetch/`。

本实验入口用于 URMA、Q4、heads32、H131072/R16384 的合成负载对比；
不是 UBmem 或模型服务适配。本地增量方法已做 CANN 模型验证，真机完整图正确性和性能由上述 A/B 验证。

## 提炼已有日志（无需 NPU）

```bash
# 默认最近一次 A/B，也可在末尾指定某次结果目录。
python tests/experiments/incremental_lifetch/summarize.py
```

输出正确性、完整图 mean/p50/p99、阶段耗时 A/B 差值、预取/补取/无用读取和
冲突数量，保存为该次结果目录的 `key_results.log`。阶段使用全部有效打点轮次
的平均值（包括 round 0）；独立于无打点图计时。中途失败时保留已有一侧结果。

新版本汇总还显示每窗口 active/skipped lane、TopK 调用总数和最慢 lane 的增量处理时间，来自独立打点轮次。

### 同口径 TopK／预取打点

A、B 都从独立源码副本编译，使用相同的诊断记录和计时边界：

- 最终 TopK：每请求每 Q 的等待、计算段、内部 TopK 调用次数，以及逻辑 block 的累计时间。
  等待包括入口流水事件和增量状态释放；计算段包含输入搬运、候选压缩、合并与索引恢复，
  不包含后续分类。计算段尾部在诊断回放中等待流水完成，避免只记录指令发射时间。
- 中间窗口：尝试、跳过、实际服务次数，增量 TopK 调用数、增量处理耗时。
- 原有预取服务：前序服务等待、服务总耗时，并列出位图扫描、候选处理、提交子阶段。
  基线中间增量处理为零；两边预取服务都使用已有 Event 的相同字段。

窗口耗时展示执行事件的 mean/max，次数按每 replay 汇总后求平均；并行核时间不相加
解释为端到端时间。最终 TopK 另外按 block 累计其顺序处理的 Q。
记录保存在各 case 的 `full_attribution.json`，离线汇总写入 `key_results.log`。
新记录出现缺 Q、非法时间或缺时钟频率时，汇总直接报错，不生成看似完整的比较。

## 跨 Q 输入/输出重叠的本地验证

`local_model.py` 从融合实验的 `methods.h` 原样提取预加载、增量处理、TopK与写回
方法，保留原生位图和GM状态格式。预留融合LI的其他存活UB，按相同顺序和偏移分配，使用同一218 KiB限制
编译；`--serial` 仅关闭重叠，分配和输入保持一致。生成目录中的 CMake 使用
`-DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599` 构建。

```bash
python tests/experiments/incremental_lifetch/local_model.py /tmp/incremental-pipeline
python tests/experiments/incremental_lifetch/local_model.py /tmp/incremental-serial --serial
# 在装有 CANN 的环境中对生成目录分别执行：
cmake -S /tmp/incremental-pipeline/tests/experiments/incremental_topk \
  -B /tmp/incremental-pipeline/build -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599
cmake --build /tmp/incremental-pipeline/build -j2
msprof op simulator \
  --application="/tmp/incremental-pipeline/build/incremental_topk --sim --rows 8 --new-scores 12288 --mode bitmap --candidate-percent 1" \
  --soc-version=Ascend950PR_9599 --output=/tmp/incremental-pipeline/profile --timeout=10
```

2026-11-16模型结果：单AIV、8Q、每Q旧Top2048+新增12288、123候选/Q。
计时为同一核首条指令至末条指令结束，不是CPU运行时间或真机性能。

| 方式 | 8Q模型时间 µs | 校验 |
| --- | ---: | --- |
| 前一版串行 | 30.82 | PASS |
| 同融合LI的UB偏移、关闭重叠对照 | 30.69 | PASS |
| 下一Q预加载＋当前Q异步写回，同融合LI的UB偏移 | 21.67 | PASS |

输出写回复用已结束TopK的merge前缀和旧ID缓冲，不再申请额外UB。
下一Q在进入TopK、复用这些缓冲前等待写回完成；窗口在发布元数据和进入预取服务前
回收所有输入/输出事件。下一Q输入在当前Q压缩完成后加载，因而同一缓冲不会边读边写。
稠密输入继续使用原有分块、容量检查和TopK；只提前读取第一个可用块。

Trace示例（以上21.67 µs运行，核起点归零）：Q1写回4.55–4.94 µs，
Q2过滤/压缩约4.66–5.47 µs；Q3预加载5.47–6.44 µs，
Q2 TopK/ID还原约5.53–6.73 µs。两段重叠分别在MTE3/V和MTE2/V上发生。
完整融合LI的AIC/AIV及host已编译通过；本地方法模型不包含URMA网络时序。

额外边界校验：每Q新增32768、5个候选，8Q PASS；每Q新增32768、100%候选，
2Q PASS；每Q新增12288、零候选，8Q PASS。这三项使用初版相同UB总量的布局，
分别为41.49、38.58、14.24 µs，主要用于验证分块和异步写回边界，不混入上表。

本次本地构建日志、A/B运行日志与MTE2/MTE3/ID还原时间线保存在
`tests/results/local-q-pipeline-20261116/`（本地结果，不提交仓库）。

## 只增加窗口 TopK 的开销对照

```bash
bash tests/experiments/incremental_lifetch/compare_window_topk.sh \
  /vllm-workspace/vllm-ascend/third_party/KVAF 4 127.0.0.1:28885 16 1
```

A为原LIFetch，B只在原窗口服务前调用上述增量TopK（包含跨Q预加载和异步写回）。
新增结果保存到独立GM状态，仅供下一窗口续算，不更新原hot TopK、预取阈值、
候选选择、映射或最终输出。最终TopK依旧完整执行原实现，Resolver、Index update、
Fetch不修改。不引入跳过窗口补跑策略；沿用原服务的前序完成依赖。
没有新增区间或候选时沿用计算方法的空任务处理，汇总实际TopK调用次数，不假定每窗口固定次数。

A/B都有相同的私有状态初始化及诊断记录，避免将初始化差异算成中间TopK开销。
B新增计算可能改变原预取观察到的进度，因此同时展示实际DDR读取数量，不宣称两个
异步运行的预取数量必然完全相同。两版都执行原有正确性检查。

结果在 `tests/results/window-topk-only/<时间>/`，运行结束自动输出 `key_results.log`：
完整图mean/p50/p99、阶段耗时差值、每窗口实际TopK次数/扫描量/候选量/耗时、
原预取服务耗时、最终TopK耗时，以及预取/补拉/无用读取数量。

```bash
# 不重跑NPU，重新提炼某次结果：
python tests/experiments/incremental_lifetch/summarize.py \
  tests/results/window-topk-only/<时间> --require-detailed
```

两个对比脚本均接受第六个位置参数 `WINDOWS`，默认8。例如
`compare_window_topk.sh KVAF_ROOT DEVICE SERVER 16 1 4` 为BS16、miss1%、4个窗口。
32 AIC预算下，每窗口8个AIC对应16个服务AIV；每个服务AIV处理一个请求的4Q。
预取总预算仍为512，默认单窗口预算随窗口数从64调整为128。

`summarize.py` 不传路径时默认读取 `tests/results/window-topk-only/` 下最新的实验目录。汇总旧增量实验时，显式传入对应结果目录。

## 最终 TopK 增量对比

```bash
bash tests/experiments/incremental_lifetch/compare_final_topk.sh \
  /vllm-workspace/vllm-ascend/third_party/KVAF 4 127.0.0.1:28885 16 1 4
```

A/B 都执行同样的中间窗口增量 TopK，使用独立的状态缓冲区。
A 保留原全历史最终 TopK；B 接续中间结果，只处理尚未覆盖的分数。
原预取筛选、最终分类、Resolver、Index 更新和 Fetch 保持原样。
不满足增量输入对齐条件的 query 保留原最终 TopK。

脚本自动校验、采集 level3、汇总完整图以及最终 TopK 的计算/等待/调用次数。
结果位于 `tests/results/final-topk-only/`；重新汇总时向 `summarize.py` 传入该次目录。
最终 TopK 的约 22 µs 是之前真机测量的参考，本次实际收益以 A/B 测量为准。

## 最新 TopK 驱动预取

```bash
bash tests/experiments/incremental_lifetch/compare_topk_prefetch.sh \
  /vllm-workspace/vllm-ascend/third_party/KVAF 4 127.0.0.1:28885 16 1 4
```

A 固定为中间与最终增量 TopK、原位图预取；B 使用以下融合流程：

- epoch 0 的 hot 预取保持原位图路径。
- 中间窗口在每 Q 增量 TopK 的 ID 恢复后，直接复用 UB 输出，SIMD 压缩 cold ID。
- 小候选列表暂存到已有 locations 工作区；计数使用已有增量元数据空位。
- 预取服务只加载压缩结果，跨 Q 去重后直接输出候选，不扫描完整历史位集。
- 后续映射过滤、预算、选槽、WQE 构造及提交复用原代码；只查一次映射。

删除每窗口 hot_valid/reserved 重读、完整 TopK ID 重读、全历史位集展开。
本窗口没有新增分数时，从已保存的 TopK 导出候选，不能误用已被提交阶段覆盖的旧列表。
不增加 GM 工作区或核数；实验限定 H131072/R16384。预算外的 token 后续可再次考虑，
最终 Resolver 继续负责补拉；普通 hot 命中不参与中间 cold 预取。

保留 level3 打点：B 的 Incremental 包括 cold ID 导出；中间 Scan 为压缩列表读取及去重，
hot 窗口 Scan 保持原口径。比较整个 Incremental + Service、LI/完整图、实际读取和无效预取，
不能只用 Scan 的下降代替总收益。脚本自动汇总至 `tests/results/topk-prefetch/<run>/key_results.log`。

### 本地完整窗口验证

```bash
source /usr/local/Ascend/cann/set_env.sh
python tests/experiments/incremental_lifetch/model_window.py \
  --kvaf-root /work/KVAF --output /tmp/lifetch-window-model
```

直接提取 A/B 生成源码的增量方法，并调用实际 FastService；没有另写简化版 TopK 或提交逻辑。
模型为 8 窗口中的第一个 cold 窗口：单 AIV、2 请求、8Q，每 Q 新增16384个分数、12个候选，
包含跨 Q 重复及已有预取映射。A/B 均提交46条WQE；校验完整TopK分数/ID、去重、映射、
目标槽、WQE、队列头和门铃，再复用同一状态运行一次，确认无新增分数时不重复提交。
模拟器门铃写入普通设备内存，不模拟远端DDR/CQ完成；该窗口本身只提交异步预取。

优化前的本地实测（固定参考版本 `3d5a10e`，Ascend950PR_9599，1800 cycles/µs）：

| 区间 | A cycles | B cycles | A µs | B µs |
| --- | ---: | ---: | ---: | ---: |
| 增量TopK及其窗口控制，B含候选导出 | 64798 | 68626 | 36.00 | 38.13 |
| 完整预取服务 | 34719 | 27929 | 19.29 | 15.52 |
| **合计** | **99517** | **96555** | **55.29** | **53.64** |

同样提交数量下合计降低 **2.98%**。脚本保留trace、模型日志及result.json，
且B合计不低于A时返回失败；真机端到端效果使用上面的A/B命令验证。

### B 增量窗口优化

A 保留原实现。B 的 `window_fast.h` 直接进入生成的融合 LI，继续调用原 `LITopk`，
没有另写排序算法；`window_fast_vector.h` 负责位图稀疏块压缩及输出映射。

- 每服务核批量读取所负责请求的进度，再处理各 Q，避免反复读取进度缓存行。
- 新分数预加载容量从 12288 扩至 16384；旧 ID 独立加载，与位图筛选重叠。
- 原生位图先标出非空的 128-score 块，只压缩这些块；稠密输入保留原处理流程。
- ID 映射改为向量查表，并一起计算新阈值、压缩 cold ID。
- cold ID 写回与下一 Q 的加载/筛选重叠，在缓冲区复用前等待完成。
- 最后一次合并不再清空下一次调用会重新初始化的候选区。

同一 8Q 固定负载，最终版本本地实测如下（1800 cycles/µs）：

| 区间 | 优化前 B cycles / µs | 优化后 B cycles / µs |
| --- | ---: | ---: |
| 增量 TopK、候选导出及窗口控制 | 68626 / 38.13 | 34946 / 19.41 |
| 后续预取服务 | 27929 / 15.52 | 27207 / 15.12 |
| 完整窗口 | 96555 / 53.64 | 62153 / 34.53 |

增量区间降低 **49.08%**。TopK 分数/ID、46 条 WQE、去重、映射、槽位和重复窗口检查通过；
稠密及不一致进度测试单独验证正确性，不计入上述稀疏固定用例时间。
这是真实实现的模拟器结果，真机 LI/完整图收益需另行实测。

单 AIV 模型的 UB 占用为 222208 字节，低于原 223232 字节上限；核数、GM 工作区不增加。
候选容量不足、跨越 uint16 相对位置范围或进度不一致时继续使用完整处理，不丢弃任务。
没有新增可处理分片时，也等待旧 TopK 输入加载完成后才归约和写回。

完整验证（增加稠密候选、低于阈值的候选、乱序旧 TopK、不一致的生产者进度及关闭预加载）：

```bash
python tests/experiments/incremental_lifetch/model_window.py \
  --kvaf-root /work/KVAF --output /tmp/lifetch-window-fast --stress
```

固定的 8Q/每 Q 新增 16384 位置、12 个候选用例，B 的增量窗口必须低于 36000 cycles，
即该模型时钟下的 20 µs；超出则脚本报错。这一门槛不适用于稠密边界测试或更大的窗口。
真机仍使用上面的 `compare_topk_prefetch.sh`，保留完整图及 level3 打点。
