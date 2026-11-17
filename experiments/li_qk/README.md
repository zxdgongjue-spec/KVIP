# B1/Q4：独立 LI 的 Q/K 分核控制实验

旧 LIFetch 优化保存在提交 `20adb7e` 与本地分支 `codex/mr14-lifetch-saved-20adb7e`。此目录作为 MR14 的独立 LI 控制实验补充，不修改原生产 kernel 或默认路由。

前一轮 `test_lifetch.py bench` 的 `prototype_with_update` 包含 LI、Service、Fetch、Index 更新；`native_chain` 也包含 SDK LI、LIDU prepare、KVAF Fetch。因此二者都不能单独识别 score 分核的收益。

此实验固定 **B1、总 Q4、H131072、G32、D128、FP8 E4M3FN Q/K、BF16 weights、FP32 scales、K=2048**，共 **7个正交参数点**，不运行T与Q的交叉网格：

- T=1/2/4/8/16：沿 history（K token）切分，**不切 D128 的点积归约维度**。
- Qparts=1/2/4：每 task 分别处理 Q4/Q2/Q1，总有效 Q 仍为4。
- K轴固定Qparts=1，扫描T1/2/4/8/16；Q轴固定T=1，扫描Qparts1/2/4。T1/Qparts1是共用基线，只计一次。
- C32/V64 为默认请求预算；所有参数点使用同一预算、同一 fixture、同一二进制。实际可用不足时显式改预算，超过单 wave 的 task 按同一容量串行分波。
- Final 全局 TopK 固定4个AIV，每真实Q一个；不随 T/Qparts 改动，不做局部 TopK/merge。

只调用 `li_qk_score`、`li_qk_scatter`、`li_qk_topk`，不下发 Fetch、resident θ、predicate、Service、LIDU 或 Index kernel。共享 LI 头文件编译仍需现有 KVAF device headers，但不链接 KVAF 执行库。scoreOnly=true 且 KVIP_STREAMING_LI=0；post 为单独 AIV 翻译单元，score 为 MIX1:2。

## 两个直接可比的指标

| 指标 | 串行参考 T1/Qparts1 | 分核候选 |
|---|---|---|
| `score_p50_us` | 一个 MIX1:2 核组扫描全 H/Q4、写 GM score | 所有 score waves，写 task score |
| `li_p50_us` | score → 单独全局 TopK | score waves → device canonical 重排 → 相同全局 TopK |

串行算子代码见 [host.cpp 的 SerialLI](host.cpp)，打分 kernel 为 [score.cpp](score.cpp)，独立重排/TopK 为 [post.cpp](post.cpp)。串行score和单wave的T1/Qparts2布局已是canonical，直接使用同一GM buffer，不收费无意义的复制。K分片或Q1的dummy行需要重排/压紧，计入`li_p50_us`；排除在`score_p50_us`之外。

事件计时覆盖指定 kernel 链及链间调度空隙。输入打包、上传、fixture/oracle、分配、输出回读不在计时内；这些适配成本不能由 score-only 结果推导为生产整链收益。warmup4 后各测20次，两种指标交错执行；整轮 r0 丢弃，之后三轮参数点交错。结果汇总仅7行，不扩展 batch、resident、miss 或 Fetch 网格。

`score_over_serial_mean<1` 表示纯 score 相对同模板单核组加速；`li_over_serial_mean<1` 才表示非融合数字 LI（score+canonical+TopK）加速。两者可能结论不同。此参考不是 SDK 自选 tiling 的 QLI，也不是前一轮带 Fetch 的库上 P2。

可选 `--sdk-reference` 再单独测原生 SDK LI：代码是 [native_li.py 的 native_li_only](native_li.py)，只调用 `torch_npu.npu_quant_lightning_indexer`。仍使用同一个真实 G32 fixture，不补成 G64；SDK 不支持此几何时明确失败。它自行选择分核，所以标为 SDK reference；没有其 score-only 分数接口，不拿它的整 LI 时间与纯 score 列做比值。启用后新增 `sdk_li_p50_mean_us` / `li_over_sdk_mean` 两列，参数点仍为7行。

## Q1 与带宽不能忽略

Q1 使用现有 Q2 的最小 M/Fixpipe 物理对齐布局，actual qlen=1。Cube 沿用2G对齐，偶数 AIV 归约完整一个 Q 的 G32 heads；奇数 AIV 仅维持 CV/VC 握手，不输出 dummy score。Q1 的额外 Cube 对齐工作和闲置 AIV 显式保留，不把它伪装成原生无填充的单Q块。未使用分片局部 θ，不涉及跨分片数值累加。

全 H 的有效 key 请求载荷随 Qparts 从16→32→64 MiB；这是请求载荷，缓存命中后的实际 DRAM 流量仍须 profiler。有效 score 输出始终1 MiB。Q1 的 scratch分配为2 MiB，Q2/Q4为1 MiB；所有 dummy Q、行padding、score/TopK guard 均核验。

| 用例 | T | Qparts | 每task有效Q | 每task H长度 | tasks/waves | 有效AIV峰值 |
|---|---:|---:|---:|---:|---|---:|
| 共享基线 | 1 | 1 | 4 | 131072 | 1/1 | 2 |
| K2 | 2 | 1 | 4 | 65536 | 2/1 | 4 |
| K4 | 4 | 1 | 4 | 32768 | 4/1 | 8 |
| K8 | 8 | 1 | 4 | 16384 | 8/1 | 16 |
| K16 | 16 | 1 | 4 | 8192 | 16/1 | 32 |
| Q2 | 1 | 2 | 2 | 131072 | 2/1 | 4 |
| Q4拆分 | 1 | 4 | 1 | 131072 | 4/1 | 4 |

上表是32/64预算，7个点均为单score wave。Q4拆分占4个AIC/8个AIV位置，但只有4个AIV做有效归约；不能将它解释为8个AIV都在计算有效Q。

## 构建与 A5 执行

宿主机同步入口为`sync_a5_host.sh`。在A5容器外执行`bash sync_a5_host.sh --no-proxy`，脚本自动创建固定目录`/tmp/li_qk_sync`，同步MR14，生成启动脚本和离线包，用`docker cp`复制到目标容器，并在目标容器内验证文件。无需查找临时目录后缀。宿主机也无法联网时，使用`--offline-bundle /path/to/li_qk_mr14_offline.bundle`导入已下载的代码包。`--container`可指定实际容器；同步完成后按打印的两条命令进入该容器并运行实验。

一键脚本为`run_a5.sh`，在原A5容器与Python环境中执行。默认拉取MR14源分支并固定实验提交`0c1d483`，建立新worktree，检查依赖，构建，执行7点，汇总并打包；失败停止并打包已生成日志。原checkout和每次结果均保留。CANN路径优先从当前环境识别；KVAF优先使用原checkout的`third_party/KVAF`。

```bash
bash experiments/li_qk/run_a5.sh \
  --repo /home/w00568585/wt-mr14 \
  --kvaf-include /path/to/KVAF/access_operator/demo/aiv_batch_get/include
```

脚本可单独下载后用`bash run_a5.sh`执行。依赖在默认位置时无需参数；路径不同时传`--cann-root`/`--kvaf-include`。实际可用预算不同时传`--aics 28 --aivs 56`，SDK对照加`--sdk-reference`。使用`--help`查看参数。它记录npu-smi，但不自动推断资源预算或排除其它工具的设备占用；在device 5空闲时执行。

若`fetch.log`报`CONNECT tunnel failed, response 400`，错误发生在Git拉取阶段，尚未构建/运行LI。可给更新后的脚本加`--no-proxy`，仅此次fetch清除代理环境及Git代理配置覆盖；不修改持久配置，也不关闭TLS校验。容器无法直连GitCode时，将提供的`li_qk_mr14_offline.bundle`放入容器并执行：

```bash
bash run_a5.sh --git-bundle /path/to/li_qk_mr14_offline.bundle
```

离线包需要包含`refs/remotes/origin/codex/split-core-unified-comparison`，脚本验证后从本地文件导入。已经具有`0c1d483`时也可用`--skip-fetch`完全跳过联网；缺少该提交则明确失败，不改用旧版代码。下载文件中的`--git-bundle`路径与依赖路径均可自定义。

在包含此目录及 MR14 `20adb7e` LI 源码的 checkout 中，启用原 CANN/TorchNPU 环境。替换 CANN、SoC 与 KVAF include 路径为实际值；使用新构建目录，避免 CANN 增量对象二次转换的问题。

```bash
python3 experiments/li_qk/run.py plan --aics 32 --aivs 64 > /tmp/li_qk_plan.json
python3 experiments/li_qk/test_controls.py

python3 experiments/li_qk/run.py build \
  --cann-root /usr/local/Ascend/ascend-toolkit/latest \
  --soc Ascend950PR_9599 \
  --kvaf-include /path/to/KVAF/access_operator/demo/aiv_batch_get/include \
  --output /tmp/li_qk_build

python3 experiments/li_qk/run.py run \
  --build /tmp/li_qk_build --device 5 --aics 32 --aivs 64 \
  --warmup 4 --iterations 20 --rounds 3 \
  --output /home/w00568585/li_qk_T_Q_controls

python3 experiments/li_qk/run.py summarize \
  /home/w00568585/li_qk_T_Q_controls
```

要直接比较 SDK 独立 LI，给 `run` 加 `--sdk-reference`。默认28个子case（7参数点×r0/r1/r2/r3）；启用SDK后每轮再测一次公共SDK参考，共32个子case。确认32/64真实可用，否则统一改成例如28/56；脚本保留7个点，会增加waves而非偷偷删点。每次选新 output，失败即停止、保留 receipt/log；不同时跑前一轮 LIFetch benchmark。lease只隔离使用同一lease的本脚本进程，不能阻止其它工具占用设备。

`summary.csv/json`保存7组 score/LI微秒数和相对串行比值；`measurements.csv`保留每轮结果。manifest绑定fixture、源码快照、构建binary/library、预算请求、命令、npu-smi输出。完整score key SHA必须等于独立oracle；TopK核验唯一、合法、包含所有严格大于Kth阈值的元素，允许阈值处合法tie，guard和输入只读检查在计时外完成。

本机没有A5。CPU host/stub、源码覆盖和 CANN 编译各自报告，不当作设备数值、同步或性能验收；A5性能列由上述真机脚本生成。普通事件计时与 msprof 分开采集，trace中应只有score、必要的scatter与全局TopK；启用SDK时另有SDK QLI。
