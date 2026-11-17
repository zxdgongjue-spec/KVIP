# MR14：关键通路优化候选与对比

日期：2026-11-17。基准为`056fe62`原staged行为及`04ddfaa`上一版Cold predicate并行实现。本次继续减少分核通路固定开销，保留默认legacy。本文记录`aa1ea91`提交时的CPU依赖检查与实际host代码分发/载荷仿真，**没有设备微秒数或整链提速结论**。后续已找到本地CANN9.1/950 Model环境，最新参数探索及分层验证见[本轮报告](optimization_exploration_20261117.md)；本文数值不改称设备测量。

## 改动

1. Hot/Cold独立history分片。减少短Hot shard重复复制Q/weight/scale和θ，同时让Cold保留单独并行度；0沿用自动分配，显式值受实际预算限制。
2. phase prepare：Hot与Cold各一次prepare_all，Cold准备留在Hot Service后；保留invocation对照，以比较少一个启动与提前准备Cold对首次Fetch的权衡。
3. inline ready：前序same-stream kernel完成epoch bitmap之后，由Service AIV只发布自身owner的四Q ready，再运行原Service。每epoch省一个progress kernel，包含Hot epoch。

Cold direct canonical＋predicate来自上一版，并继续生效。没有将某history shard的局部θ用作全局θ；没有分散owner的slot/victim规划者。保留Hot全覆盖与四Q保护、Final分类别名等待、safe/conflict Fetch、CQ join、Output旧映射消费后Index commit。段级descriptor与Service并发、Final私有buffer拆分不在这次实现内。

私有Plan64B/ABI3，公开Context/RuntimeShape ABI11。resident TopK与Final TopK各Q仍由单AIV沿原算法处理；`P_hot`、`P_cold`与Fetch宽度可独立设置。本次实现的是可调候选，需要A5网格实测选择参数，不预先宣布最优。

## 仿真结果

固定B1、Q4、H131072、R16384、G32、C28/V56、Fetch2。工具从git读取不可变before文件，与当前候选分别编译实际host dispatch代码，以launch记录桩执行；task布局使用真实Python调度器。准备payload按task与真实输入类型计算，每个方向表示读或写Q/weight/qscale的请求载荷，排除cache sector、padding与其他算子流量。设备数学、ISA、CQ与带宽并发未仿真。

| W | 配置 | dispatch | 准备payload/方向（B） |
|---|---|---:|---:|
| 2 | 原staged separate/wave/Pauto | 15 | 960512 |
| 2 | 上一版direct/invocation/Pauto | 13 | 960512 |
| 2 | 新direct/phase/inline/Ph4/Pc8 | 12 | 205824 |
| 2 | 新direct/invocation/inline/Ph4/Pc8 | 11 | 205824 |
| 8 | 原staged separate/wave/Pauto | 45 | 3842048 |
| 8 | 上一版direct/invocation/Pauto | 31 | 3842048 |
| 8 | 新direct/phase/inline/Ph4/Pc8 | 24 | 1029120 |
| 8 | 新direct/invocation/inline/Ph4/Pc8 | 23 | 1029120 |

W8 phase候选相对原staged减少46.7% dispatch、73.2%准备payload；相对上一版direct/invocation减少22.6% dispatch。独立inline控制在相同direct/invocation/Pauto下31→23，恰好省W8的8个progress启动。

Cold score载荷保持上一版的3,670,016→917,504B（减少75%）；本轮没有再减。Cold predicate比较次数仍458752。Ph4/Pc8下W8的θ padded DMA从Pauto的200704B降至57344B，重复加载减少，score数学量不变；这不是全GM流量，也不是实测DMA时间。

W2：Hot每MIX组4096keys，Cold每组14336keys；W8：Hot4096、Cold2048。减小P会增大task循环长度，可能降低并行算力，也可能减少重复准备与短任务固定成本。必须用真机测量这个权衡。**不能将755.9µs按dispatch或payload比例换算成新延迟，也不能由staged改善推导它优于P2。**

可重现命令：

```bash
python3 tests/operators/lifetch/cold_parallel/verify_cpu.py
python3 tests/operators/lifetch/cold_parallel/simulate_schedule.py \
  --output /tmp/mr14_schedule_simulation.json
```

原始结果：[critical-path-simulation.json](results/critical-path-simulation.json)。文件记录source SHA、before commit、逐项dispatch sequence及排除项。旧版`04ddfaa`需保留在git历史中。

## 验证与真机验收

CPU检查覆盖实际host失败即停、phase指针偏移与先Hot Service顺序、多owner/multiwave只在epoch末发布、真实Context/Progress/ServiceLane的独占ready写入、bitmap sector独占与tail mask、动态输入prepare、ABI版本与公共入口、控制命令及完整实验汇总。CANN元数据检查的负例拒绝Service错误MIX与缺失score_direct MIX1:2。60项公共runtime/binding/kernel contract测试，以及score/staged数学与binding回归通过；CPU/mock不能证明新kernel的A5编译、异步URMA或实际性能。

A5命令见[实验README](../../../tests/operators/lifetch/cold_parallel/README.md)。先W2同有效窗口，再W8策略对照；同轮同prefetch/Fetch/实际预算/二进制比较`prototype_with_update`：

- 相对legacy（库上P2）小于1：该场景整链收益。
- 仅相对原staged小于1：只说明修复MR14劣化，尚不能替代P2。
- `native_chain`为SDK QLI→LIDU prepare→KVAF串行参考，其不变不能证明分核通路不影响后处理。

每个实验完整执行一轮丢弃＋三轮交错测量，保留PASS、sample、receipt与log SHA。`best_candidates.json`给出本次网格的最快候选，选定后独立复跑，避免把一次搜索结果当稳定最优。普通计时与msprof分开；使用首次Fetch、未覆盖join、ConsumerReady及完整graph时间解释收益来源。当前没有精确设备埋点，相关指标仍需A5 trace或后续埋点。
