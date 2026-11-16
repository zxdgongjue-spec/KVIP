# LIFetch 统一测试

唯一入口：`test_lifetch.py`。URMA、UBmem 共用 `check`、`bench`、`diagnose`，
通过 `--transport` 选择。调用仓库正式算子，不保留历史优化版本选择开关。
完整接口及日志字段见[测试设计](../../../docs/performance/lifetch-tests.md)。

## 代码结构

```text
lifetch/
  test_lifetch.py
  README.md
  support/
    workloads.py       # 配置、合成负载、真实输入及转换、轮间前态
    execution.py       # 初始化、共享资源、正式算子执行、进程管理
    correctness.py     # 数值参考、状态约束、固定检查用例
    performance.py     # 图计时、串行/融合配对、Fetch-only
    diagnostics.py     # 算法统计、阶段/逐核打点、带宽、失败现场
    reporting.py       # JSON、CSV、终端表格
```

## 准备

在 KVIP 根目录运行，使用已构建的 KVIP/KVAF 和已启动的对应 CPU service：

```bash
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh
```

默认 `--device 0 --kvaf-root third_party/KVAF`；其他安装或设备显式指定。
URMA 和 UBmem 命令相同，但 KVAF 目录及服务必须与所选 transport 对应。

## check：正确性

```bash
python tests/operators/lifetch/test_lifetch.py check \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 --timeout-seconds 600
```

默认覆盖初始化、BS1/2/4/8/16、全命中/全 miss、预算为零、跨 Q 重复/空补拉、
冷启动、冲突、LRU 候选外兜底、padding、连续多轮和非法参数。
`--suite basic --cases basic_bs1` 可筛选；`--suite contract` 不需要 NPU。
冲突/兜底/padding 的实际覆盖证据写入 `coverage.json`，不以普通用例 PASS 代替触发证据。

## bench：配对性能

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 \
  --batches 1,2,4,8,16 --miss-percent 3,5,10,15 \
  --warmup 4 --iterations 50 --timeout-seconds 3600
```

同一输入对比原生串行 LI/LIDU/Get 与正式融合完整图。终端只展示最终表格，
按命令的 miss、BS 顺序排列；保存 mean/p50/p99、原始样本、正确性和实际读取量。
默认合成器为内置标量参数的 `compact_builtin_v1`，无需 trace 或外部参数文件。
目标 miss 调整 hot residency；实际逐 Q miss 写入报告。check 仍使用可精确控制分支的固定 fixture。

一个 worker 共用最大 BS 的 DDR、runtime 和 HBM 参考 KV；DDR Put 一次，后续 case 不重复 Put。
构建使用运行时 shape 缓存，不按 BS/miss 编译。每组独立恢复 Index/hot/LRU，恢复及校验在计时外。
bench 不运行诊断打点或 check 的边界专项；资源准备、复用次数及墙钟成本保存在结果目录。
合成 URMA 的 `bench` 和 `diagnose` 支持 `--fetch-jettys-per-aiv {1,2}`，默认 2。
该参数控制融合最终 Fetch 每个 AIV 使用的 Jetty 数，每 Q 仍启动两个 Fetch AIV。
`fetch_only.json` 分别记录 native 和 fused 的 Jetty 数，native 使用两条。

## diagnose：算法与时间拆解

```bash
python tests/operators/lifetch/test_lifetch.py diagnose \
  --transport urma --device 4 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 \
  --batches 4 --miss-percent 10 --level 3 --iterations 3 --timeout-seconds 600
```

各等级均按“算法/KV 统计”和“时间拆解”组织：

- level 1：总需求、miss、预取/补拉/unused、读取字节、完整图及阶段时间、区间有效带宽。
- level 2：逐请求/逐 Q/逐窗口数量、活动核、Fetch 分组计数与时间。
- level 3：逐核时间线、内部步骤和等待、LI 分类细节、可用的 Fetch-only 配对。

`--stages li,resolve,update,fetch` 筛选展示，不跳过执行依赖。性能图与打点图分开采样。
URMA 合成负载支持 Fetch-only；UBmem 和真实回放仍使用完整图诊断。
`all` 执行默认 check 后，用同一次配对测试生成 bench 和 level 1 诊断，不另跑 report。

### 性能对比 CSV

合成负载的 `results.csv` 每行对应一组 BS / miss。`bench` 和 `diagnose` 都写入
整图耗时与实际传输量。`serial_read_entries` 是 LIDU 去重后交给 BatchGet 的条数。
`fused_prefetch_*`、`fused_final_*`、`fused_total_*` 分别统计预取、最终补拉和总读取量，
包含每轮平均、最小和最大条数及平均字节数。`fused_read_count_samples` 是统计所用的计时轮数。

`diagnose --level 1` 已能在同一张表写出阶段数据。时间列含 `mean_us`、`p50_us`、`p99_us` 和 `samples`。

- `serial_get_only_*` 是独立 BatchGet 图的事件计时。`serial_profiled_*` 给出 LI、LIDU（含描述符准备）、Get、核间空隙和事件边界余量。
- `fused_profiled_*` 给出 Init、LI 和 Resolve/Update/Fetch 所在的融合 kernel 时间。`resolver_span`、`output_pack_span`、`index_apply_span` 和 `finalize_span` 取各阶段首尾标记的全局区间，可能互相重叠。预取、最终 Get 和二者合并区间例如 `fused_profiled_final_get_span_p50_us`。
- 整图有效带宽用 `serial_read_bytes / serial_mean_us / 1000` 或 `fused_total_bytes_mean / fused_mean_us / 1000`。分段传输的 `GBps_p50` 先用同一轮的字节数除以该轮区间，再取各轮中位数。零条传输的带宽留空，缺少打点的阶段也留空。

`--stages` 筛选诊断文本展示。CSV 保留已捕获的各个阶段。真实回放的分阶段记录按源轮次保存在 `diagnostic.json`。

## 真实回放与旧 trace 转换

已有 schema-4 `dp*/series.json` 可直接连续回放，必须包含真实 Q、Index keys/scales、weights、query scale：

```bash
python tests/operators/lifetch/test_lifetch.py bench --input real \
  --trace-root /path/to/trace --group 10 --round-range 1:16 \
  --transport urma --device 4 --kvaf-server 127.0.0.1:28885
```

串行与融合从相同初态各自演进；完整序列重复时才恢复初态。
Q/Index 来自采集，KV 使用可校验的身份数据；此项验证算子回放，不是模型服务 TPOT。
形状从 trace 读取，不再传 synthetic 的 BS/history/miss 参数。

历史 `pid*-sample*.pt` 或之前转换出的 analysis bundle 使用同一入口转换，仅需 CPU、NumPy、PyTorch：

```bash
python tests/operators/lifetch/test_lifetch.py convert-trace \
  /vllm-workspace/vllm-ascend/experiments/kvaf/results/lifetch-inputs-dp2-bs4 \
  --output /tmp/lifetch-real-replay
```

这替代了原 `convert_trace.py` 加 `convert_replay.py` 两步命令。
输出 `capture-00000` 等独立快照和 `conversion.json`；终端显示各快照的 BS、group 和校验结果。
旧采集不能证明跨轮稳定请求身份，因此每个快照独立运行，`--round-range 1:1`，group 取转换输出：

```bash
python tests/operators/lifetch/test_lifetch.py bench --input real \
  --trace-root /tmp/lifetch-real-replay/capture-00000 --group 10 --round-range 1:1 \
  --transport urma --device 4 --kvaf-server 127.0.0.1:28885
```

每次运行结果由唯一目录保存，失败也保留已完成 case。
