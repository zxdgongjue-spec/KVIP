# 独立 LI 与 LIFetch 的执行策略对齐

当前 window 只对齐了 MR14 history 分片和 canonical GM 写回，并未对齐整条 LI。此前 B1/P16 direct 的 52.14µs 来自无 resident/Service/Fetch/Index 的两个 kernel，统计为多轮 p50 均值；此前 SerialAfter 的 LI 约62–65µs还包含图内的 query 准备。当前 LIFetch 汇总使用同图 event 均值，LI 列包含窗口、Service、Final，却不含另列的 Resolve/Get。不能直接相减判断 Score 是否退化。

| 路线 | 数值/后处理链 | LI launches | 预取 |
|---|---|---:|---|
| MR14 独立 direct | 全H/P分片Score → 每Q原版全H Top2048 | 2；串行adapter另含query准备 | 无 |
| MR23 | Hot → θ → Cold/增量TopK/Service → Final | 4 | 原MR23 |
| window_budget | 每波分核Score → 增量TopK/rank/union/post，最后Final | B1=6，B≥2=10 | 原MR23 |
| full_budget（旧） | 全H Score → θ → 原T2 Cold消费者 → 增量Final | 4 | 原MR23 |
| aligned_budget（新增） | 全H分核Score → 每Q原版全H Top2048+classify → 原Resolve/完成Fetch/Index | 2，另有Resolve/Get | 显式0 |

新增路线采用原版16K trunk radix TopK，不做history局部TopK再合并；每Q一个AIV，最多64AIV。fullHistoryTopk_仅由新增入口设置，跳过FinalIncremental并不分配其不用的UB scratch。所有原MR23/window默认入口保持原策略。Score保留实际packed Q、page table、key scale索引和canonical地址；并未复用旧实验的虚拟行/query复制，因此属于算法/调度对齐，不能称为旧二进制的精确复现。

用户A5 miss10中，B1 window Score=31.53µs、Service=109.76µs、θ+Final=24.20µs，合计LI=165.49µs。主要差额在新增后处理，不能把165.49当纯Score与旧27.98比较。B16相应Score=180.55µs、Service=164.38µs、θ+Final=40.49µs，合计385.42µs；这与此前SerialAfter LI约184–186µs并不矛盾，新增Service已经足以造成大差额。不同实验的输入、资源、编译和计时协议还须配对控制。

同batch、同miss百分比不能确保同工作量。miss按selected-union驻留控制，fixture重排Hot/Cold keys；θ、Cold候选比例、每Q miss、候选重复度都会影响窗口增量TopK和提交。pf是预算而非实际Fetch量。Score几乎不随miss变，Service强烈随miss变，是现有数据的主要证据。真正搬运多少和覆盖多少须读posted/useful/final-miss等计数，不能只用预算推断。

## A5控制实验

```bash
git fetch origin codex/lifetch-score-t2
git pull --ff-only origin codex/lifetch-score-t2
bash experiments/lifetch_score_t2/run_aligned_a5.sh \
  --device 5 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28885 \
  --batches 1,2,4,8,16 --miss-percent 0,3,5,10,15,20 \
  --intermediate-fetches 4 --rounds 3 --warmup 4 --iterations 20
```

三路使用相同fixture、DDR池容量、核预算、同图event均值；r0丢弃。MR23/window同样使用pf0作控制。新增路线不把prefetch请求静默忽略，显式拒绝非0预算。pf0窗口路线仍会执行增量TopK等工作，不能视作对齐路线。需要另跑旧pf1024实验评估预取的净价值。

输出新增FullScore和FullTopK+Classify两个同图组件，完整链继续包含Resolve/Fetch完成和Index提交。结果只在执行COMPLETED、correctness PASS和输入hash一致时汇总。SerialBefore仍为SDK链，新增aligned不是MR14 SerialAfter适配器，也不是全模型vLLM端到端。

## 本地验证

```bash
python3 -m unittest discover -s experiments/lifetch_score_t2 -p 'test_*.py' -v
python3 experiments/lifetch_score_t2/run_aligned_model.py \
  --mode sim --kvaf-root /path/to/patched/KVAF --batches 2,4
```

Model numeric/classify在相同H32768/R16384、四Q sortable scores上检查Top2048的值、ID唯一性、输出hash一致、slot分类和protection bitmap，并检查guard及不改写Row/Event。score路线另用H8192/R2048验证真实FP8全history分片生产者和TopK/classify。周期跨度是TopK局部工作，不含Score、kernel启动、Fetch或完整链；不将其换算为A5收益。
