# 当前可运行范围、最快筛选流程与真机批跑

当前标准组合是H131072、Q4、B1/2/4/8/16/24/32 × miss3/5/10/15/20/50，共42组。**已有42组脚本不等于serial6、Fusion3和新分核三条完整链路都已支持42组。** 本次新增真机分组入口和事件性能汇总，未执行新Model/NPU测试，也未补写三条生产算法。

| 路径 | 当前可执行范围 | 计量与限制 |
|---|---|---|
| `scripts/split_core_sim/customer.sh` | 全42组TopK组件；T1/4/8 | Model ticks，AIC0；没有QK/Service/Resolve/Index/Fetch |
| `scripts/split_core_sim/compare.sh` | 42×3种T×2种V=252条件 | 同上，不是三种完整pipeline对比 |
| `scripts/run_split_core_native.sh`（新增） | 当前完整streaming，默认W2时BS≤16的30组，656B；G32/64、预算和W可配置 | 真机graph-event p50/p99毫秒，保留12行UNSUPPORTED；无硬件不能执行run |
| 历史serial6/Fusion3 | 保留固定版本、固定标准点的证据/重放材料 | Service0与GM-copy mock；不能直接遍历新6种miss，不能当当前真机基线 |
| P32/T8旧原型 | 受限研究入口 | 不含完整生产resident/classification/Service；不能冒充完整分核版 |
| PO/owner-pool新设计 | 方案和结构模型 | 尚无执行实现 |

`compare_pipelines.py`只读取历史结果重建比较表，不会执行三条pipeline。历史harness的standard oracle固定205 miss/query和U410，不能只换miss参数。导出的historical CMake提供history/full研究target，并没有一个可通用于新42组的Fusion3融合target。旧原始Fusion3 A/B来自另一组固定binary。要补三条完整链路，应参数化共同输入/oracle，并在同一当前源码语义下补serial/fused/split入口，保留真实Service、classification、Index和transport。

336行自适应结构模型另展开了C/V、G和656/2624B，不是336个设备执行脚本。新真机入口固定656B，当前公开test_lifetch入口也固定该payload；2624B的独立UBmem smoke已有能力，但尚未纳入这个统一矩阵。

## 先准备命令，再迁移到设备

在没有NPU的本机可以生成完整清单，不加载Torch NPU、不占设备、不生成性能：

```bash
bash scripts/run_split_core_native.sh plan \
  --transport ubmem --budget 28/56 \
  --output tests/results/native-plan
```

输出`results.json/csv`共42行：默认30行NOT_RUN、12行UNSUPPORTED_NATIVE；另有`commands.json`列出6条子命令与`coverage.json`说明各策略缺口。`--case`可只选代表点；其他合法行保留NOT_RUN。显式选择BS24/32或超出角色预算会在创建输出前报错。

迁移到匹配的Ascend设备后，先准备该源码版本、CANN、Torch NPU、自定义算子与KVAF依赖，使用目标机真实SoC/服务地址。准备命令以`python scripts/build.py --help`和该环境的`scripts/env.sh`为准；本脚本不安装环境或启动服务。

下面变量均由目标机提供，`KVIP_KVAF_ROOT`应是匹配、已构建的KVAF checkout；环境已经设置后执行：

```bash
# 单个标准点，先确认环境和输出合同
bash scripts/run_split_core_native.sh run \
  --transport ubmem --soc "$KVIP_SOC" --cann "$ASCEND_HOME_PATH" \
  --kvaf-root "$KVIP_KVAF_ROOT" --server "$KVIP_SERVER" --device 0 \
  --budget 28/56 --case b1-h100-m10 \
  --warmup 4 --iterations 20 --output tests/results/native-first

# 展开当前全部支持场景：30组；仍保留BS24/32的12行不支持记录
bash scripts/run_split_core_native.sh run \
  --transport ubmem --soc "$KVIP_SOC" --cann "$ASCEND_HOME_PATH" \
  --kvaf-root "$KVIP_KVAF_ROOT" --server "$KVIP_SERVER" --device 0 \
  --budget 28/56 --warmup 4 --iterations 20 \
  --output tests/results/native-streaming-30
```

用`--transport urma`及对应服务地址可运行URMA。`--budget 32/64`、`--heads 64`、`--q-workers 1|2|4`分别配置资源/head/Fetch角色，不代表启用了TopK T8或history P32。默认30组以W2计算；W4/V56时BS16的6组也超预算，只执行24组，其余明确标不支持。每个配置输出新目录；没有隐式resume或跳过已存在目录。

## 这个入口为什么能节省实验时间

新入口按miss分组，每组给既有`test_lifetch.py`传`--batches 1,2,4,8,16`：30场景从30个外层子进程降为6个。URMA在同一子进程内复用universal-build，因此由每场景一份构建改成每个miss组一份（6份）；没有将“目录存在”扩展成跨run安全构建缓存。UBmem可利用既有按源码/配置构建的runtime library缓存，同时减少进程和导入开销。编译、初始化、数据准备、验证仍需时间，不能承诺总时间缩短5倍。

默认`trace-rounds=0`，避免全矩阵反复采trace；保留原harness全部正确性、stateful与payload/Index检查，不以删检查换时间。reset与验证在NPU Event区间外，报告是同初始软件状态下的graph replay，并非硬件冷cache测试或完整模型TPOT。

URMA从本轮新目录实际`report.json`读取`prototype_with_update`，核对fixture几何、exact miss、U=2m及B个request的reference miss；记录完整seed、CPU准备文件及binary hash。seed不是完整batch输入，BF16转换、batch展开与remote地址安装后的设备tensor没有直接dump/hash，actual_input_sha256保持null。`native_reference_p50_ms`来自同harness的`native_chain`，它是native LI→LIDU→KVAF reference，**不是旧serial6**。UBmem从单份原始worker log解析时序、每Q实际miss、checked geometry和binary标识；暂存输入已被子程序删除，因此actual_input_sha256明确为null，不拿外层另造的fixture冒充。

子输出的计时单位是微秒，统一结果除1000保存为ms。缺失、重复、非有限、样本数/几何/miss不匹配均不标通过。CSV使用所有行字段并集，选中非首case时性能与证据列仍保留。结果保留p50/p99与样本数；当前harness未导出每个原始event样本，20样本P99仅供探索，不能据此宣称稳定尾延迟。正式统计需补raw samples、增加重复/样本并报告置信区间。

同一NPU/链路域串行执行，不并发争抢带宽。默认租约`/tmp/kvip-npu-DEVICE.lease`与旧矩阵入口一致，可通过`--lease-file`配置共享路径。子harness是唯一timeout管理者，外层不再叠加同一deadline。子进程非零或中断后停止矩阵，并保留租约，因为不能独立证明所有嵌套session/DMA已停止；只在核实本次worker已停止后人工清理租约，不自动抢占或继续运行下一组。该机制是失败时阻止错误并发，不是声称已实现任意进程树的安全取消。

## 最快建立可信比较的顺序

1. **正确性与能力**：本机生成42组计划；Model只跑每条新算法的代表点、边界、非法/依赖用例。保留全42矩阵作为回归，不能把抽样当全量通过。
2. **真机无trace筛选**：先B1/8/16 × miss3/20/50共9组，再跑支持域30组。固定C/V、P/T、输入、初始缓存/Index/LRU、G、payload、Service预算、credits、transport和instrumentation。
3. **精测**：对候选和基线在相同设备做交错重复、较多样本，并导出raw samples。不要同时在同一设备跑三个策略；竞争会改变结果。
4. **少量归因采集**：只给差异最大的点和异常点采msprof。指令数、PIPE/stall、队列等待与字节用于解释性能，不替代端到端时间。

三方案公平比较还缺共同实现：A串行、B融合、C进一步分核应采用相同语义/真实transport。若为Model方便使用mock，则三者必须共同使用相同mock，并标明不含真实网络收益；随后再做真机对照。BS24/32要先完成生产入口、owner池和可前进协议。

## 指令数、Model ticks和真机耗时分别怎么看

| 指标 | 获取方式 | 不可作的推断 |
|---|---|---|
| Whole-Model ticks | 正常Model footer，同binary/输入/启动范围对照 | 不能直接换算真机微秒 |
| 动态指令执行数 | 官方导出的每core `*_instr_exe.csv`中`call_count`求和，记录完整core/subcore与kernel/launch覆盖 | 不能把CSV行数当指令数；选中core不能代表全kernel |
| PIPE区间 | 官方trace，在同一clock域做区间union | 不能把多个PIPE时长相加当总耗时 |
| 真机graph Event p50/p99 | 无trace重放并保留样本/状态恢复合同 | 不能和Model ticks混成同单位排名 |

已有TopK trace只选core0的V0/V1；T1/T8任务映射、总block数量不同，不能将选中core指令数差当作整算子减少量。`--core-id`控制解析选择，不代替采集配置中的logging mask。完整指令trace数据很大，通常比无trace性能执行慢；只有需要归因的代表点才抓全覆盖。现有`scripts/run_lifetch_msprof.sh`的固定标准case另有配置合同，不等于新42矩阵runner。

独立TopK host即使改成RUN_MODE=npu，目前也缺少设备Event、预热与重复；新入口测的是生产Final T1的当前streaming，不能称已支持TopK T1/T8真机性能对照。

字段与覆盖口径参见官方[模拟性能数据](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_performance_data.md)和[模拟指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md)。新版[板上指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)中的InstrTimeline也不保证完整动态计数：SIMT/SIMD VF内部指令不显示，可能丢数且只采样部分core；选项可用性须核对目标机SDK。性能基准与profile replay分开，不能把含重复重放、cache控制或插桩的profile时间当原始event基准。

本次CPU核验与审查见`results/native-runner-validation.json`；没有本轮NPU测量或完整三策略结果。
