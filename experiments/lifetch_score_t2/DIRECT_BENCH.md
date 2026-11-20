# 统一 LIFetch bench 入口

当前机器是 a5-20，之前是 a5-12。以下命令在已经配置好 CANN、torch_npu 和 KVAF 的实验容器中，从 KVIP 仓库根目录执行。`--device 4` 沿用本次请求，是容器内逻辑设备编号；`./third_party/KVAF` 必须指向已构建的依赖目录。

先运行用户指定的直接用例：

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --input synthetic --transport urma --device 4 \
  --kvaf-root ./third_party/KVAF --kvaf-server 127.0.0.1:28885 \
  --batches 1,2,4,8,16 --miss-percent 5,10,15,20,25,30 \
  --intermediate-fetches 4 --warmup 2 --iterations 10 \
  --timeout-seconds 3600
```

该命令默认预取预算512/请求、初始准入rank1024、Q4/H131072/R16384/head32/seed43，history分核参数为0。它测量 SDK LI＋LIDU＋完成 BatchGet 的 `SerialBefore` 与当前 checkout 的默认融合链。

在同一入口增加两路对比，只追加一个参数：

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --input synthetic --transport urma --device 4 \
  --kvaf-root ./third_party/KVAF --kvaf-server 127.0.0.1:28885 \
  --batches 1,2,4,8,16 --miss-percent 5,10,15,20,25,30 \
  --intermediate-fetches 4 --warmup 2 --iterations 10 \
  --timeout-seconds 3600 \
  --compare-routes mr23,window_budget
```

`mr23` 导出固定提交 `a56154c37bebb0928da4a4095e9670898e6b21c7` 的原融合链；`window_budget` 使用当前 checkout 中 MR23＋MR14 history分核策略融合的窗口候选，保留原 θ、Service、Final 和完成 Fetch/Index。候选自动使用 B1/2/4/8/16 对应 P16/16/8/4/2。所有路线保持本命令的512预取预算、4个请求窗口、warmup2、iterations10和timeout3600，仍以 SDK链为串行参考。

每条路线、每轮只启动一个包含全部五batch的底层bench worker。两路都按maxB16配置共享DDR池及HBM reference（H131072时有效payload为1,375,731,712字节），URMA Runtime为40 workers/80 jettys。共享池初始化在计时外；每个batch实际活跃LI核数和窗口仍由该batch及设备有效AIC预算决定。32AIC下请求4个窗口时，B1实际为2，其它batch为4。

默认 `--comparison-rounds 1`，不额外丢弃r0，与原直接命令的一轮测量语义一致。两路共2个worker，每个worker执行五batch×六miss；汇总共60行。每个case将10份独立cache状态的完整调用捕获进同一张测量图，报告event/calls与同图LI、Resolve/Get阶段。若需要多轮及额外冷启动控制，可在原命令末尾加：

```bash
--comparison-rounds 3 --comparison-discard-first
```

这时两路共8个worker（额外r0＋3个测量轮），r0保留原始记录但不进入收益统计。汇总仍按两路×五batch×六miss组织。

在两路对比命令末尾加 `--comparison-dry-run` 可生成源码快照及全部worker计划，不初始化NPU或KVAF服务，也不报告性能结果。

直接入口也支持单候选调试。按batch设置P的示例为在用户原命令末尾加：

```bash
--li-score-mode window --li-score-parts-map 1:16,2:16,4:8,8:4,16:2
```

固定 `--li-score-parts N` 继续可用；map与非零固定P互斥。所有请求batch必须有对应P，且任务总核数必须满足有效AIC预算。单候选结果比较SDK串行与该候选；两路精确MR23对比使用 `--compare-routes`。

两路对比要求 KVAF固定为 `1c1310387ebfb6776a07316874e924265b209ada`，并已有 `build-npu` Python模块、Get/Put对象和CMake cache。脚本在初始化NPU之前只读核对版本和构建依赖；不自动fetch、reset、改写或重建已有KVAF。依赖版本不匹配时，先按已有依赖准备流程创建独立pinned目录，再将 `--kvaf-root` 指向它。CANN和headers来自所选KVAF实际构建cache。

打印的对比结果目录保留路线/轮次计划、源码及输入hash、底层case报告、日志、原始同图trace和 `summary.csv/summary.json`。数值检查失败时保留已完成case，并继续后续路线以便诊断；任一case执行失败、正确性FAIL、源码或fixture无法确认一致时，汇总标为不接受，不计算该组合收益，最终退出码反映失败。只有完整调用完成、correctness PASS且证据一致的配对组合才报告加速比。

历史 [run_a5.sh](README.md) 将batch拆成独立worker，另有外层r0与轮次调度；本入口共享maxB16池并默认只有一轮，不能混合其统计。历史三路pf0的 `aligned_budget` 保留在 [ALIGNMENT.md](ALIGNMENT.md)，它改变后处理结构且关闭预取。原B1 history_parts七点扫描仍在 [RUN_VALIDATION.md](RUN_VALIDATION.md) 单独复跑，其数值LI/p50不填入本次完整融合链结果。
