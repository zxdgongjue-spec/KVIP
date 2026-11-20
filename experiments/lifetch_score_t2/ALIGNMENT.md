# 独立 LI 与 LIFetch 的执行策略对齐

本实验的目标是把 MR14 history 分核策略融合进 MR23 LIFetch，并对比三条融合路线。`mr23` 是固定提交的原融合基线；`window_budget` 是 MR23＋MR14 窗口分核 Score，保留原 Service/Final；`aligned_budget` 是 MR23＋MR14 全 history 分核 Score，接原全量 Top2048/classify，再接原 Resolve/Fetch/Index。MR14 策略融合后的结果均在 `Fused` 列，`SerialBefore` 始终是 SDK LI＋LIDU＋完成 BatchGet 的串行参考，本任务无需新增独立 MR14 SerialAfter 路线。

window 只对齐了 MR14 history 分片和 canonical GM 写回，没有对齐整条 LI。此前 B1/P16 direct 的 52.14µs 来自无 resident/Service/Fetch/Index 的两个 kernel，统计为多轮 p50 均值；此前 SerialAfter 的 LI 约62–65µs还包含图内的 query 准备。当前 LIFetch 汇总使用同图 event 均值，window 的 LI 列包含窗口、Service、Final，aligned 的 LI 列包含全量 Score＋TopK/classify，均不含另列的 Resolve/Get。不能直接相减判断 Score 是否退化。

| 路线 | 数值/后处理链 | LI launches | 预取 |
|---|---|---:|---|
| MR14 独立 direct | 全H/P分片Score → 每Q原版全H Top2048 | 2；串行adapter另含query准备 | 无 |
| MR23 | Hot → θ → Cold/增量TopK/Service → Final | 4 | 原MR23 |
| window_budget | 每波分核Score → 增量TopK/rank/union/post，最后Final | B1=6，B≥2=10 | 原MR23 |
| full_budget（旧） | 全H Score → θ → 原T2 Cold消费者 → 增量Final | 4 | 原MR23 |
| aligned_budget（新增） | 全H分核Score → 每Q原版全H Top2048+classify → 原Resolve/完成Fetch/Index | 2，另有Resolve/Get | 显式0 |

新增路线的 kernel 链为 `v9_li_kernel_score_aligned → v9_li_full_topk(classify) → 原 Resolve/Fetch/Index`。它采用原版16K trunk radix TopK，不做history局部TopK再合并；每Q一个AIV，最多64AIV。fullHistoryTopk_仅由新增入口设置，跳过FinalIncremental并不分配其不用的UB scratch。所有原MR23/window默认入口保持原策略。Score保留实际packed Q、page table、key scale索引和canonical地址；并未复用旧实验的虚拟行/query复制。因此这是 MR23＋MR14 策略融合的算法/调度对齐，不是 MR14 staged Context/lease/ABI 的整包或二进制精确复制。

用户A5 miss10中，B1 window Score=31.53µs、Service=109.76µs、θ+Final=24.20µs，合计LI=165.49µs。主要差额在新增后处理，不能把165.49当纯Score与旧27.98比较。B16相应Score=180.55µs、Service=164.38µs、θ+Final=40.49µs，合计385.42µs；这与此前SerialAfter LI约184–186µs并不矛盾，新增Service已经足以造成大差额。不同实验的输入、资源、编译和计时协议还须配对控制。

同batch、同miss百分比不能确保同工作量。miss按selected-union驻留控制，fixture重排Hot/Cold keys；θ、Cold候选比例、每Q miss、候选重复度都会影响窗口增量TopK和提交。pf是预算而非实际Fetch量。Score几乎不随miss变，Service强烈随miss变，是现有数据的主要证据。真正搬运多少和覆盖多少须读posted/useful/final-miss等计数，不能只用预算推断。

## A5控制实验

```bash
# 先将本轮修改同步到 20 机候选仓库，再在现有容器内执行。
cd /home/w00568585/li-aligned-a5/KVIP
bash experiments/lifetch_score_t2/run_aligned_a5.sh \
  --device 0 --kvaf-root /home/w00568585/li-aligned-a5/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28985 \
  --batches 1,2,4,8,16 --miss-percent 5,10,15,20,25,30 \
  --intermediate-fetches 4 --rounds 3 --warmup 4 --iterations 20
```

当前机器为 a5-20，之前是 a5-12。当前容器只映射物理 card5 时使用逻辑 `--device 0`；其它映射按容器内设备编号选择。容器使用已配置的 CANN/torch 环境执行构建和实验，宿主机负责代码下载和容器管理。

包装脚本默认三路、B1/2/4/8/16、miss5/10/15/20/25/30、pf0、r0＋3个测量轮，共60个任务；每个任务内部运行全部6种 miss。用户末尾参数可覆盖默认值，如 `--batches 1,2 --miss-percent 0,10 --rounds 1` 共12个任务。miss 支持互不重复的有限0..100百分比。三路使用相同fixture、DDR池容量、核预算、同图event均值；r0丢弃。MR23/window同样使用pf0作控制。新增路线不把prefetch请求静默忽略，显式拒绝非0预算。pf0窗口路线仍会执行增量TopK等工作，不能视作对齐路线。需要另跑pf1024实验评估预取的净价值。

输出新增FullScore和FullTopK+Classify两个同图组件，完整链继续包含Resolve/Fetch完成和Index提交。`route_description` 和 manifest 的 `route_descriptions` 明确标识 MR23＋MR14 策略融合路线。结果只在执行COMPLETED、correctness PASS和输入hash一致时汇总，也不是全模型vLLM端到端。

a5-20 的 `lifetch_score_t2_20261120_112037_5_5i87ok` 已完成 pf0、B1/B2、miss0/10 的三路初测，丢弃r0后只有1个测量轮。aligned 的 LI 为55.38–60.25µs，其中 FullScore 23.05–27.50µs，TopK/classify 32.28–32.85µs；完整 Fused 为94.82–130.57µs，较配对 MR23 降低52.8%–60.6%。该结果支持分核与后处理结构对齐产生收益，不能证明与旧52.14µs跨机跨协议精确相等，也不代表B4/8/16或更高miss已经验证。另一次B16尝试在r0 MR23基线失败后停止，aligned尚未执行，不能从该日志判断新路线B16性能或正确性。

## 本地验证

```bash
python3 -m unittest discover -s experiments/lifetch_score_t2 -p 'test_*.py' -v
python3 experiments/lifetch_score_t2/run_aligned_model.py \
  --mode sim --kvaf-root /path/to/patched/KVAF --batches 2,4
```

Model numeric/classify在相同H32768/R16384、四Q sortable scores上检查Top2048的值、ID唯一性、输出hash一致、slot分类和protection bitmap，并检查guard及不改写Row/Event。score路线另用H8192/R2048验证真实FP8全history分片生产者和TopK/classify。周期跨度是TopK局部工作，不含Score、kernel启动、Fetch或完整链；不将其换算为A5收益。

本轮已通过：B2 numeric/classify的选中hash一致；B4 classify的标量oracle/slot/bitmap检查通过；B2/P2 score→TopK/classify通过，检查全部65536个canonical score和16384个选中ID。完整NPU库以CANN9.1/9599编译、链接和新增入口符号检查通过，53个设备源码文件与编译快照hash一致。36项CPU实验检查和3项构建恢复检查通过。

B2/P16的Model尝试受本地Docker虚拟机内存限制被OOM终止，未获得正确性或性能结果；不将其计作算子校验失败。用同一生产者P2覆盖跨request寻址与数值，A5默认P16/16/8/4/2保持原设定。复跑该小规模生产者检查可增加`--routes score --score-parts 2 --batches 2`。详见[aligned_validation.json](aligned_validation.json)与[evidence/aligned_li](evidence/aligned_li)。Model局部验证与上述A5单轮初测作用不同，后续完整矩阵仍须检查所有测量轮正确性与波动。
