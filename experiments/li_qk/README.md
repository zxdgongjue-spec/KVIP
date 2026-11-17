# B1/Q4 与 B16：独立 LI 的 Q/K 分核控制实验

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

当前服务器已确认`/home/w00568585 → /home/w00568585`共享映射。推荐在宿主机执行`bash update_shared_a5.sh`，固定更新`/home/w00568585/li_qk_shared`，不要求Docker CLI、不编译或运行NPU；容器中执行`bash /home/w00568585/li_qk_shared/run_in_container.sh`，沿用原Python/CANN环境。默认八点；原七点加`--suite orthogonal7`，B16控制组加`--suite batch16`。

可先给容器启动命令加`--verify-only`，确认共享文件和哈希可见；它只验证映射，不跑实验。宿主机更新器支持`--no-proxy`和`--offline-bundle PATH`。使用不可变release目录，容器一次固定一个release，避免宿主机后续更新覆盖正在使用的代码包；宿主机Git缓存不用于容器worktree。脚本使用已有挂载，不创建新映射。

代码准备入口为`sync_a5_host.sh`。执行`bash sync_a5_host.sh --no-proxy`，脚本自动创建固定目录`/tmp/li_qk_sync`，同步MR14，生成启动脚本和离线包。可用容器CLI时自动识别Docker/Podman/nerdctl并复制到目标容器；没有CLI时只准备本地文件，不因缺少Docker而失败。可用`--runtime`显式选择工具，nerdctl可加`--namespace`。`--container`指定实际容器。

当前终端已能访问NPU/CANN时，可直接执行`bash sync_a5_host.sh --no-proxy --run-here`同步并运行，不需要容器管理工具。`run-here`指当前登录的A5真机终端，先检查`npu-smi info`与Model preload，再构建`RUN_MODE=npu`并使用ACL设备事件计时；不切换成Model/CPU计算。设备5是否可用还由实际ACL运行检查。只准备源码用`--sync-only`。运行参数放在`--`后，例如`--run-here -- --kvaf-include /path/to/include`。默认使用原KVIP checkout，缺少该目录时使用新建的bare仓库建立实验worktree。网络不可用时将`--no-proxy`换成`--offline-bundle /path/to/li_qk_mr14_offline.bundle`。

一键脚本为`run_a5.sh`，在原A5容器与Python环境中执行。默认拉取MR14源分支并固定本次FETCH_HEAD，建立新worktree，检查依赖，构建，执行所选套件，汇总并打包；失败停止并打包已生成日志。原checkout和每次结果均保留。CANN路径优先从当前环境识别；KVAF优先使用原checkout的`third_party/KVAF`。

```bash
bash experiments/li_qk/run_a5.sh \
  --repo /home/w00568585/wt-mr14 \
  --kvaf-include /path/to/KVAF/access_operator/demo/aiv_batch_get/include
```

脚本可单独下载后用`bash run_a5.sh`执行。依赖在默认位置时无需参数；路径不同时传`--cann-root`/`--kvaf-include`。八点要求32/64；实际预算例如28/56时使用`--suite orthogonal7 --aics 28 --aivs 56`。SDK对照加`--sdk-reference`。使用`--help`查看参数。它记录npu-smi，但不自动推断资源预算或排除其它工具的设备占用；在device 5空闲时执行。

若`fetch.log`报`CONNECT tunnel failed, response 400`，错误发生在Git拉取阶段，尚未构建/运行LI。可给更新后的脚本加`--no-proxy`，仅此次fetch清除代理环境及Git代理配置覆盖；不修改持久配置，也不关闭TLS校验。容器无法直连GitCode时，将提供的`li_qk_mr14_offline.bundle`放入容器并执行：

```bash
bash run_a5.sh --git-bundle /path/to/li_qk_mr14_offline.bundle
```

离线包需要包含`refs/remotes/origin/codex/split-core-unified-comparison`，脚本验证后从本地文件导入。已有所需新提交时也可用`--skip-fetch --ref COMMIT`完全跳过联网；缺少指定提交则明确失败，不改用旧版代码。下载文件中的`--git-bundle`路径与依赖路径均可自定义。

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

本机没有A5。CPU host/stub、源码覆盖、CANN编译与Model各自报告，不能当作A5性能验收；A5性能列由上述真机脚本生成。普通事件计时与 msprof 分开采集，trace中应只有score、必要的scatter与全局TopK；启用SDK时另有SDK QLI。实际Model验证范围见`EXTENSION_REPORT.md`。

## K32 与 B16/30 AIC 扩展（2026-11-17）

设计要求见 `EXTENSION_PROMPT.md`。原七点仍用 `--suite orthogonal7`，新增八点用 `--suite orthogonal8`；八点必须 32 AIC/64 AIV，K32 的每个 task 是 Q4×H4096。`run_a5.sh` 默认改为八点，并使用本次拉取的 MR14 revision，避免仍执行旧的 `0c1d483`。共享目录 launcher 固定使用发布 release 的 `SOURCE_COMMIT`。

按已经确认的共享挂载，在宿主机运行最新 `update_shared_a5.sh`，随后在 A5 容器运行：

```bash
# B1：原七点 + K32；最后单独采集 K32 msprof
bash /home/w00568585/li_qk_shared/run_in_container.sh --suite orthogonal8 --profile

# B16：三个控制组；只使用 30 个 AIC 核组执行 score
bash /home/w00568585/li_qk_shared/run_in_container.sh --suite batch16 --profile
```

B16 的对照组为 T1（16 task，owner 间并行，每 owner 单核），候选为 T2 普通 waves（32 task，30+2 两波）与 T2 balanced。每轮均按 B16 T1 配对，不使用 B1 时间归一化。上述 `batch16` 自动固定 score 预算为 30 AIC/60 AIV；Final TopK 独立使用 64 AIV，每个真实 query 一核。该预算针对 score，并非整条 LI 都限制在 60 AIV。

balanced 将 32 个逻辑 half-task 展平为 1024 个 2048-key 块，30 个核组分别处理 34/35 块（H69632/H71680）。每核独立消费 owner/half 边界的片段列表，一次 score kernel 启动内完成；不同 owner 的 query/weight/scale、真实输出地址保持分离。**均衡意味着每 owner 的两个逻辑 half 可再细分和迁移到不同物理核；不再固定每 owner 恰好由两个物理核完成。** 要严格固定两核/owner，就必须接受 30+2 调度或另做串行 owner 配对。此实现保留 scatter，片段切换仍有 QLI/TPipe 初始化和 CV/VC drain 成本，不能仅凭块数均衡宣称提速。

`profile.py` 是独立的真机采集入口，要求 NPU binary，保留 `msprof --application`、`--aic-metrics=PipeUtilization`、export 命令、正确性和 trace。profile 的 warmup4/iterations5 LI-only 调用不会写入常规 benchmark summary。需要单独重采时：

```bash
python3 experiments/li_qk/profile.py run --suite batch16 --device 5 \
  --build /home/w00568585/li_qk_结果目录/build \
  --output /home/w00568585/li_qk_b16_msprof_新目录
```

### 仿真动态指令采集

在专用 CANN950 simulator 容器构建 `--run-mode sim`，然后执行（路径按容器内实际位置替换）：

```bash
python3 experiments/li_qk/model.py run --build /work/model-build \
  --output /work/b16-model-new --history 131072 \
  --model-lib /usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_3510/lib \
  --timeout 900
```

默认分别采集三个方案的 score/scatter/TopK；`--stage all` 额外验证真实完整链，`--history 8192` 仅为缩小尺寸 smoke。默认日志只开 physical core0，`--core-mask 0xffffffff` 才采集全部核组，需要足够磁盘。所有 Model 指令占比只针对已记录的核，不能当作全设备时间占比；配置与已记录核覆盖写入 manifest。scatter/TopK 单阶段运行在 GM 中预填相同的 oracle score，不能用它证明 producer/consumer 重叠。

`instructions.py` 按 kernel stage 目录分别统计 AIC/AIV 的动态 issue 指令、pipeline、opcode 和 unknown。不混算 AIC 与 AIV 分母；可用 `--pc-map` 提供经过 ELF symbol/section 与运行时 entry PC 校验的区间。共享 VF 只能在已确认串行 kernel 边界的日志中归属当前 score kernel；没有可靠映射时保留 unmapped。PC 没有可靠源码映射时，不填算法子过程的时间占比。旋转日志、退出失败或超时必须保留并标明；Model ticks 不能换算为 A5 微秒。原始 dump 逐 case 压缩保存。

如需进一步区分带宽与流水线利用率，给 `profile.py` 加 `--metrics Memory` 或 `--metrics ResourceConflictRatio`，分别使用新输出目录。动态指令数量的百分比不等于这些硬件利用率。

自动验证PC映射并统计已采集的Model日志（不会重新执行kernel）：

```bash
python3 experiments/li_qk/map_model_pcs.py /work/model-results/T2_balanced_all \
  --build /work/model-build --objdump /path/to/ccec_compiler/bin/llvm-objdump \
  --output /work/model-results/T2_balanced_all/verified-kernel-instructions
```

要求对应case的原manifest与相同captured build，输出入口/ELF/SHA证明、kernel PC范围及动态统计。独立consumer的image基址会变化，不能复用完整链的硬编码地址。结果中的`capture_complete`反映原case是否完整PASS；部分超时日志不能当完整统计。

本轮实际结果与限制见`EXTENSION_REPORT.md`：B16/H8K三控制组完整LI通过，修复了balanced片段切换时AIV1 credits残留；H128K独立scatter/TopK通过；H128K完整score600秒超时。仅开core0日志，没有全核score统计或真实A5 msprof trace。

## G32 执行粒度与全核 Model 补测

新增结果见 [GRAIN_REPORT.md](GRAIN_REPORT.md)。七次独立 balanced score Model 全部 PASS，覆盖30 AIC/60 AIV；固定H8192的同binary/fixture三点，grain2048→512/128的issue窗口包络缩短14.1%/15.6%。这是Model结果，未换算A5微秒；完整H128K仅补充静态计划，旧完整score超时结论仍有效。

`--work-grain` 仅改变 B16 balanced 的任务分配量子，默认2048，允许128..65536的2次幂，要求H对齐2×grain。普通T1/T2 waves保持原计划。连续块会合并，不是每grain重做初始化；内部计算tile仍是H128。

宿主机照旧用`update_shared_a5.sh`更新，容器执行`run_in_container.sh --suite batch16 --work-grain 512 --profile`即可补测。2048/512/128三个配置各产生独立结果；同轮B16 T1配对，Final固定64 AIV，score固定30/60。

严格控制为同一binary时，先按前文构建一次，再运行以下循环（替换构建与输出路径）：

```bash
for li_grain in 2048 512 128; do
  python3 experiments/li_qk/run.py run \
    --build /path/to/same-npu-build --suite batch16 \
    --aics 30 --aivs 60 --device 5 --history 131072 \
    --work-grain "$li_grain" --warmup 4 --iterations 20 --rounds 3 \
    --output "/home/w00568585/li_b16_grain_${li_grain}_new"
done
```

每次输出必须不存在。确定性fixture生成器保持不变，检查三个manifest的fixture hashes一致。单独`profile.py run`也支持`--work-grain`。

全核七点Model脚本是 [run_granularity_model.sh](run_granularity_model.sh)，汇总为 [grain_summary.py](grain_summary.py)。报告同时解释128计算tile、分配量子、连续片段、canonical与Final，及尚未实施的初始化复用和producer直写方向。
