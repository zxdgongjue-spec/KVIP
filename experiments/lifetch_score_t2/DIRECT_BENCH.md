# 统一 LIFetch bench 入口

当前机器是 a5-20，之前是 a5-12。以下命令在已经配置好 CANN、torch_npu 和 KVAF 的实验容器中，从 KVIP 仓库根目录执行。当前容器将物理card5映射为逻辑device0，KVAF server是127.0.0.1:28985；下面使用20机已经构建好的pinned依赖目录。

20机主命令使用参数包装脚本，它仍调用下述统一bench入口：

```bash
cd /home/w00568585/li-aligned-a5/KVIP
bash experiments/lifetch_score_t2/run_b124_a5.sh \
  --device 0 \
  --kvaf-root /home/w00568585/li-aligned-a5/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28985
```

包装脚本固定本轮B1/2/4、三条路线、miss5/10/15/20/25/30、预取512、窗口4、warmup2、iterations10和timeout3600。末尾参数可覆盖这些默认值。它不创建Docker、联网或另换计时链。

单独运行当前默认融合链的直接用例：

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --input synthetic --transport urma --device 0 \
  --kvaf-root /home/w00568585/li-aligned-a5/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28985 \
  --batches 1,2,4 --miss-percent 5,10,15,20,25,30 \
  --intermediate-fetches 4 --warmup 2 --iterations 10 \
  --timeout-seconds 3600
```

该命令默认预取预算512/请求、初始准入rank1024、Q4/H131072/R16384/head32/seed43，history分核参数为0。它测量 SDK LI＋LIDU＋完成 BatchGet 的 `SerialBefore` 与当前 checkout 的默认融合链。

本轮只跑B1/2/4，在同一入口对比三条融合路线：

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --input synthetic --transport urma --device 0 \
  --kvaf-root /home/w00568585/li-aligned-a5/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28985 \
  --batches 1,2,4 --miss-percent 5,10,15,20,25,30 \
  --intermediate-fetches 4 --warmup 2 --iterations 10 \
  --timeout-seconds 3600 \
  --compare-routes mr23,window_budget,query_budget
```

`mr23` 导出固定提交 `a56154c37bebb0928da4a4095e9670898e6b21c7` 的原融合链；`window_budget` 使用当前 checkout 中 MR23＋MR14 history分核策略融合的窗口候选，保留原 θ、Service、Final 和完成 Fetch/Index。新增 `query_budget` 使用同一分核Score生产者，以 `window_query` 让每个Q分别计算窗口TopK，再由原request owner执行去重、准入和post，最后接原Final与完成Fetch/Index。这里接入已有query方案，不宣称尚未测得的NPU收益。

两种候选自动使用 B1/2/4 对应 P16/16/8。三路保持本命令的512预取预算、4个请求窗口、warmup2、iterations10和timeout3600，仍以 SDK链为串行参考。窗口query方案增加每窗口TopK/post的启动边界，完整测量包含这些成本。

每条路线、每轮只启动一个包含全部三个batch的底层bench worker。三路都按maxB4配置共享DDR池及HBM reference（H131072时有效payload为343,932,928字节），URMA Runtime为40 workers/80 jettys。共享池初始化在计时外；每个batch实际活跃LI核数和窗口仍由该batch及设备有效AIC预算决定。32AIC下请求4个窗口时，B1实际为2，其它batch为4。

默认 `--comparison-rounds 1`，不额外丢弃r0，与原直接命令的一轮测量语义一致。三路共3个worker，每个worker执行三个batch×六miss；汇总共54行。每个case将10份独立cache状态的完整调用捕获进同一张测量图，报告event/calls与同图LI、Resolve/Get阶段。若需要多轮及额外冷启动控制，可在原命令末尾加：

```bash
--comparison-rounds 3 --comparison-discard-first
```

这时三路共12个worker（额外r0＋3个测量轮），r0保留原始记录但不进入收益统计。汇总仍按三路×三个batch×六miss组织。

在三路对比命令末尾加 `--comparison-dry-run` 可生成源码快照及全部worker计划，不初始化NPU或KVAF服务，也不报告性能结果。

直接入口也支持单候选调试。按batch设置P的示例为在用户原命令末尾加：

```bash
--li-score-mode window_query --li-score-parts-map 1:16,2:16,4:8
```

固定 `--li-score-parts N` 继续可用；map与非零固定P互斥。所有请求batch必须有对应P，且任务总核数必须满足有效AIC预算。单候选结果比较SDK串行与该候选；精确MR23及候选之间对比使用 `--compare-routes`。

路线对比要求 KVAF固定为 `1c1310387ebfb6776a07316874e924265b209ada`，并已有 `build-npu` Python模块、Get/Put对象和CMake cache。脚本在初始化NPU之前只读核对版本和构建依赖；不自动fetch、reset、改写或重建已有KVAF。依赖版本不匹配时，先按已有依赖准备流程创建独立pinned目录，再将 `--kvaf-root` 指向它。CANN和headers来自所选KVAF实际构建cache。

新增同图组件与配对结果：

| 字段 | 含义 |
|---|---|
| `query_topk_us` | `query_budget` 同一测量图中每窗口Q并行TopK的均值总和；原window的内联TopK不猜分 |
| `li_kernel_launches` | 实际LI启动数；W2时window/query分别6/8次，W4时分别10/14次，Resolve/Get另计 |
| `over_window_budget` | query完整Fused时间除以同配置window时间，小于1更快 |
| `net_gain_vs_window_us` | window减query的完整Fused时间，正数表示省时 |

vswindow结果要求两路都通过正确性、输入hash、P、实际窗口边界及Service调度配对检查。window失败或该配对条件不满足时，vswindow两列留空；query与有效MR23的比较仍可单独保留。组件从具名kernel及同图调用边界提取，缺失、重复或总和不能对齐LI阶段时不接受该case。

打印的对比结果目录保留路线/轮次计划、源码及输入hash、底层case报告、日志、原始同图trace和 `summary.csv/summary.json`。数值检查失败时保留已完成case，并继续后续路线以便诊断；任一case执行失败、正确性FAIL、源码或fixture无法确认一致时，汇总标为不接受，不计算该组合收益，最终退出码反映失败。只有完整调用完成、correctness PASS且证据一致的配对组合才报告加速比。

`pipeline_counts` 在测量event结束之后读取，记录实际prefetch和Final补取的token请求量，不加入正式计时。原始records按 `(source_round, input, request)` 保存每个测量input中每个request的计数；汇总的 `prefetch_requests_per_call`、`final_get_requests_per_call`、`total_get_requests_per_call` 单位是一次完整batch invocation中的token Get请求数，先合计B个request，再对测量input求均值。它们不是单request计数，也不是配置的预取预算。缺失或不完整证据显示 `MISSING/INVALID` 和空值，不以0代替。固定MR23导出的CPU测试harness也应用相同报告补丁；manifest中的 `baseline_reporting_override` 保存原始/生效SHA和diff。该补丁不改MR23 kernel、launch或 `profile_graph_batch`。

本次bench没有采集首个真实doorbell时间；Service kernel起点也不能替代doorbell时间。`pipeline_counts` 回答提交了多少请求，不能据此推断首发时刻、CQ完成时刻或搬运与计算重叠长度。

完整B1/2/4/8/16仍受支持，候选默认P分别为16/16/8/4/2；本轮主命令只选择前三个batch，不启动B8/B16。历史 [run_a5.sh](README.md) 将batch拆成独立worker，另有外层r0与轮次调度；本入口共享所请求最大batch的池并默认只有一轮，不能混合其统计。历史三路pf0的 `aligned_budget` 保留在 [ALIGNMENT.md](ALIGNMENT.md)，它改变后处理结构且关闭预取。原B1 history_parts七点扫描仍在 [RUN_VALIDATION.md](RUN_VALIDATION.md) 单独复跑，其数值LI/p50不填入本次完整融合链结果。
