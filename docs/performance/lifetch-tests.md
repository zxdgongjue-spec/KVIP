# LIFetch 测试设计与使用说明

> 状态：设计稿，接口待实现并依据正式算子约束修正。命令及日志为设计示例。
> 基准：已合入PR13的KVIP主线 `13d2898`，正式支持BS1～16。

## 1. 组织与统一入口

```text
tests/operators/lifetch/
  test_lifetch.py          # 唯一人工入口
  README.md               # 实现后提供可执行命令
  support/
    workloads.py          # 负载生成、读取、前态管理
    execution.py          # 初始化、正式算子调用、资源释放
    correctness.py        # 参考结果、状态约束、固定用例
    performance.py        # 图计时、串行/融合配对
    diagnostics.py        # 耗时、算法统计、带宽、失败现场
    reporting.py          # JSON、CSV、终端短表
```

按职责复用，URMA、UBmem共用输入、校验和报告，执行层保留必要的传输差异。初始化检查并入统一入口，测试调用正式算子。

```bash
python tests/operators/lifetch/test_lifetch.py <all|check|bench|diagnose> [options]
```

| 模式 | 内容 | 输出 |
| --- | --- | --- |
| all | 依次执行check、bench、diagnose的默认用例，作为默认冒烟流程 | 总结果及各阶段结果 |
| check | 初始化、正确性、边界、多轮状态 | 用例结果与失败现场 |
| bench | 串行/融合配对性能 | 最终性能表 |
| diagnose | bench基础上的分级诊断 | 阶段/逐核/操作耗时、算法统计、带宽 |

all使用所选transport，不自动跑另一种传输；不依赖外部真实数据。默认check跑全部内置用例，bench跑默认合成矩阵，diagnose跑该矩阵的level 1诊断。复用负载、构建和无打点性能结果，不重复跑同一份bench。依赖缺失、非法输入或设备执行异常导致无法继续时，对应后续项标NOT_RUN。仅在设备任务确认完成、地址与状态仍可安全使用时，结果精度失败才不阻止后续bench和diagnose；保留性能及错误现场。越界、未完成传输或状态损坏按执行异常处理，停止依赖该状态的后续项。存在失败项时整次运行非零退出。

每个case立即落盘，结束或可捕获中断时自动汇总。

### 1.1 公共约定

| 参数 | 默认值/含义 |
| --- | --- |
| `--transport` | urma；可选urma/ubmem |
| `--device` | 0，当前进程内逻辑NPU编号 |
| `--kvaf-root` | 当前KVIP仓库根目录下的third_party/KVAF；不存在明确报错，其他目录须显式指定 |
| `--kvaf-server` | CPU service地址，与KVAF目录无关；沿用所选运行时已有配置，未配置则要求显式指定HOST:PORT |
| `--output` | 自动创建唯一结果目录 |
| `--timeout-seconds` | 300，单worker上限 |

记录代码/依赖版本及修改状态、输入校验值、设备、NUMA、配置和二进制指纹。加载错误显示实际依赖路径，不静默使用其他KVAF安装。

图计时包括调度及每轮必需的设备准备、LI、Fetch、Index/输出更新；不包括构造、前态恢复、CPU校验及SFA校验。执行完成状态、正确性、计时完整性和采集完整性分别报告；失败、未执行、不支持不能填成PASS或耗时0。

## 2. 负载数据接口

| 输入 | 使用方式 | 状态与输出 |
| --- | --- | --- |
| synthetic | 生成可执行的Q、Index keys/scales及缓存状态 | 单轮，重复测量前恢复同一前态，汇报统计结果 |
| real | 读取已有真实trace或可转换产物 | 按真实trace逐轮运行，保留轮间状态，汇报每轮及平均结果；允许只有一轮 |

合成负载主要配置目标miss、Q大小和BS。跨Q重合、分数/阈值分布、扫描位置和访问局部性由生成器离线用真实trace校准，默认内置校准配置；记录生成器版本和seed。

真实输入按稳定请求/token身份重建地址和页共享关系。串行与融合从同一初态各自演进，不逐轮灌回采集路径的hot/post状态。

## 3. 接口定义

以下为拟定接口，待实现后按正式能力修正。新增参数应服务于测试选择，不暴露历史优化开关。

### 3.1 check

| 参数 | 默认值/含义 |
| --- | --- |
| `--suite` | all；init/basic/boundary/stateful/contract/real，可逗号组合 |
| `--cases` | 可选，仅执行指定内置case ID |
| `--snapshot` | 可选，定制真实检查的已有可执行输入目录 |

BS、Q、序列长度、预算、窗口、seed及eager/graph覆盖均由内置case定义，不作为check命令参数。suite按现有有效测试规整，具体配置及实际覆盖写入结果。all包含前五类；提供snapshot再加入real，显式选择real却无snapshot时报错。

### 3.2 bench

固定对比同transport的原生串行LI/LIDU/Get与当前正式融合路径，不拉取历史commit。

| 参数 | 默认值/含义 |
| --- | --- |
| `--input` | synthetic；可选synthetic/real |
| `--miss-percent` | synthetic默认10；bench接受逗号列表，如3,5,10,15，与BS组成矩阵 |
| `--queries` | synthetic默认4，仅接受正式算子支持的Q；不支持时明确报错 |
| `--batches` | synthetic默认1,4,8,16 |
| `--seed` | synthetic默认43，用于可复现生成 |
| `--history / --resident / --index-heads` | synthetic默认131072 / 16384 / 32 |
| `--trace-root / --group / --round-range` | real必填：数据目录、层组、源采集轮号闭区间（如51:51）；层投影使用输入数据已有配置 |
| `--prefetch-tokens / --intermediate-fetches` | 默认512 / 8，记录实际窗口数 |
| `--warmup / --iterations` | 默认4 / 20；synthetic按单轮，real按完整序列 |

synthetic参数与real选择参数二选一：real不接受合成负载参数BS/Q/history/resident/heads/seed；形状和计算输入由数据确定，不能用合成参数覆盖。裁剪BS需先用数据转换器生成对应输入。

### 3.3 diagnose

复用bench的公共参数、负载选择、串行/融合对照及计时参数，只额外提供两项：

| 参数 | 默认值/含义 |
| --- | --- |
| `--level` | 默认1；可选1/2/3，等级逐级增加采集内容 |
| `--stages` | all；init/li/resolve/update/fetch/finalize，可逗号组合 |

`--stages`仅筛选诊断采集和展示范围，不跳过完整链路中的执行阶段。用于解释所选阶段的必要依赖事件一并保留；无打点完整图仍按完整链路计时。

## 4. check：固定用例与校验

### 4.1 内置suite

复用并整理现有有效fixture、参考模型和断言，不另建一套测试算法。每个case绑定固定几何、预算和执行方式；框架自动运行对应组合，报告实际配置。

| suite | 内置覆盖 | 校验重点 |
| --- | --- | --- |
| init | 资源建立、少量Put/Get、释放与重建 | 实际字节及错误传播 |
| basic | 常规部分命中，BS1/2/4/8/16的已有合法配置 | 完整链路，保留BS2分片回归 |
| boundary | 全命中、全miss、冷启动、跨Q重复、空Q补拉、预算0、选槽兜底、在途冲突、padding | 每个分支实际触发且结果正确 |
| stateful | A拉入→再次命中→淘汰→再次需要，穿插空任务轮 | 轮间不恢复状态，检查持久映射及计数复位 |
| contract | BS0/17/32及非法预算、窗口、形状 | 预期拒绝，负向用例通过 |
| real | snapshot中的真实计算输入及状态 | 真实LI计算、KV和Index结果；不擅自改变输入 |

边界条件：全miss使用装满其他token的hot，冷启动使用空有效位；全命中零读取断言使用预算0子用例，有预取干扰时检查合法补拉；候选不足时完整hot池仍有合法槽以验证兜底。padding哨兵不覆盖接口允许写入的范围。

### 4.2 统一校验

- TopK位置、数量和选择合法；BF16边界等分允许不同合法选集。
- 根据各路径实际选集比较完整KV字节，身份数据避免重复模式掩盖错位。
- Slot、Index keys/scales、逻辑映射、请求归属及hot有效状态一致。
- Fetch无遗漏，完成或明确报错，零任务正确处理；SFA按记录的容差作补充校验。
- 多轮逐轮检查状态约束，不要求两条路径选择相同的合法淘汰槽。

每case结束立即保存配置、触发计数、校验和错误现场。snapshot缺字段或格式不兼容时明确失败，不替换成合成负载。

## 5. bench：配对运行与结果

### 5.1 synthetic

默认内置校准生成器，用户主要调整miss、Q、BS；输入生成一次，串行与融合使用等价独立初态。实际执行LI，不以预生成TopK替代。

每个case先检查输入和运行结果，再捕获图、预热、恢复正式初态。结果精度不符时记录correctness=FAIL；符合第1节的安全继续条件时，仍完成计时并展示实测性能。每次计时前恢复相同单轮状态，恢复和校验不进入计时。BS变化复用可用编译产物；重新捕获图不等于重新编译。

### 5.2 real

按所选真实轮次执行，两条路径各自维护缓存；只在序列开始及每次完整序列重复前恢复初态。预热以完整序列计，正式测量重新从初态开始。

记录每轮及整体平均，保留源轮号、身份和实际有效history。各轮Q/Index必须与当前路径布局按逻辑token对应。

### 5.3 性能口径

| 指标 | 口径 |
| --- | --- |
| 完整图 | 无详细打点的mean、p50、p99、样本数 |
| 收益 | 1−fused_mean/serial_mean；加速比serial_mean/fused_mean |
| 实际miss | 每请求全部Q需求并集的cold/selected；跨请求/轮先加计数再求比例 |
| 读取量 | 实际预取、补拉及字节；不以cold数代替全部读取 |
| real汇总 | 每轮跨重复统计、所有轮的平均、每次序列内被测图时间之和 |

序列图时间之和不是含CPU间隙的服务耗时。正常完成的计时均展示mean/p50/p99及可计算的对比，即使correctness=FAIL；这类结果明确标为精度失败，不算正确实现的性能验收通过。崩溃、超时或未完成的样本不纳入完整计时统计，保留已完成样本数和失败原因，不用0补齐。URMA与UBmem的bench终端只展示运行进度、失败信息和最终性能表，不展开逐case详细摘要、读取计数或测试框架墙钟统计。已有详细信息保留在结果文件中；算法统计、阶段耗时与归因由diagnose展示，不为bench增加细粒度采集。版本、输入、配置和原始样本自动落盘。

## 6. diagnose：按等级采集

### 6.1 等级与内置内容

每次diagnose先执行同配置bench作为无打点参考，再独立运行诊断。若仅结果精度失败且符合第1节的安全继续条件，继续诊断并保留correctness=FAIL；执行异常时报告已取得的计时和现场，不伪造缺失数据。all中直接复用已完成bench。诊断采样次数沿用iterations，预热沿用warmup，无额外参数。

| level | 时间信息 | 算法信息 | Fetch信息 |
| --- | --- | --- | --- |
| 1 | 实际kernel/阶段起止及mean/p50/min/max，µs | 每轮/请求的初始cold、需求、总预取、补拉、总字节 | 预取/补拉/合并有效带宽（明确事件口径） |
| 2 | level1 加所有活跃核、请求、Q的工作/等待和最晚结束核 | 每窗口候选/阈值/预取、逐Q去重/补拉、unused及跨轮趋势 | 每Q/核提交与完成区间、冲突/在途等待及实际队列/搬运配置 |
| 3 | level2 加所选阶段全部已定义内部步骤的cycles、调用次数、处理量 | 同level2，保留详细身份关联以追踪分类结果 | 细分打包/提交/完成等待；选中fetch时自动追加Fetch-only对照 |

level1逐采样展示每请求的有效hot、唯一需求、初始cold、预取、补拉和KV字节，以及各实际kernel的起止、持续时间、kernel间隙和边界残差。起止以该次路径首个kernel为零点，不能跨串行/融合两次回放拼接。

当前level1带宽复用已有诊断时间戳，因此仍启用共用的设备打点开关；不额外启动Fetch-only、不输出逐核CSV，不声称设备内部细粒度写入已完全关闭。正式性能仍取独立无打点回放。最终Get带宽明确标记调用区间；精确提交到完成跨度尚缺时显示限制。零传输带宽为N/A，有传输却缺失带宽事件则标INCOMPLETE。队列和搬运配置属于level2/3。

示例（数值仅演示格式）：

```text
DIAG level=1 transport=urma trace=COMPLETE
li: mean=210.000 p50=210.000 min=209.000 max=211.000 us n=3
STAGE repeat=0 stage=li start_us=10.000 end_us=220.000 duration_us=210.000
ACCOUNTING repeat=0 gap_us=0.2 boundary_us=4.0
  request=0 initial_hot_valid=16384 selected_unique=2253 initial_cold=410 prefetch=128 final=346 KV_read_bytes=310944
BANDWIDTH repeat=0 metric=final_get_bandwidth bytes=226976 span_us=40.0 GB/s=5.6744
  scope: Global elapsed span includes gaps/dependencies; not first WQE to hardware completion
```

### 6.2 阶段及步骤

level1 按真实kernel边界展示：串行LI、LIDU、地址映射、Get；融合init、LI、resolve_update_fetch。融合内部角色重叠，不能相加当作kernel总时间。

| 阶段 | level 2分段 / level 3细分 | 工作量 |
| --- | --- | --- |
| init | 本轮计数、状态及工作区初始化 | 初始化条目、字节和参与核数 |
| li | 计算、TopK、分类、预取发布 | history、窗口、候选和提交数 |
| resolve | 加载、保护信息、每Q去重、选槽/兜底、发布 | 位置、重复、唯一miss、候选及扫描量 |
| update | 等Q、规划、字段搬运、映射提交 | Index行、字节、失效槽 |
| fetch | 等任务、打包、提交、在途限制、完成等待、有效状态发布 | 条目、KV字节、批次、冲突数 |
| finalize | 依赖等待、计数/错误汇总、写回 | 等待来源、未完成任务和错误码 |

逐Q关联发布→消费→提交→观察完成→有效状态发布及Update事件。level 3记录这些阶段内已有步骤，例如update.plan/copy/commit，不要求用户再选steps。不记录每条硬件指令，不给每个tokenprintf。

所有核明细落盘，保留最大值及对应block/request/Q；block与物理core编号分开。步骤耗时为正式上下文内区间，不将独立抽取成本视为相同指标。

### 6.3 算法统计与趋势

level 2/3的明细维度为“采样重复/源轮号→请求→预取窗口或Q”。

| 位置 | 信息 |
| --- | --- |
| 轮初 | 有效hot、容量、初始cold/selected |
| 每窗口 | 扫描范围、阈值、候选、本地命中、预取去重、预算抑制、实际提交/字节 |
| 每Q | TopK数、跨Q重复、新增唯一需求、本地满足、预取满足、新增补拉/字节 |
| 轮末 | 预取总数、useful/unused/redundant、补拉、重读、有效hot |

false hit统一报告为unused_prefetch：实际预取token不在本轮最终全部Q的需求并集中，不是KV内容错误。

按实际传输记录分类：useful为实际使用的副本；unused为不在最终需求的token；redundant为属于最终需求但该次副本未使用。另报refetched（预取后仍最终补拉），它是交叉指标，不参加前三项求和。

依据token、slot、批次和消费关系归因，传输次数与唯一token数分开；缺证据标N/A，不用总计数相减猜测。展示逐窗口及累计预取/unused、逐Q补拉、真实连续轮次的miss/读取量/hot变化。合成重复仅展示分布，不称作decode演进。

### 6.4 带宽及Fetch-only

所有level自动报告选定传输的有效带宽，使用同次诊断的字节和时间：

| 指标 | 字节 | 时间跨度 |
| --- | --- | --- |
| 预取 | 实际预取KV字节，含重复/unused | 首次预取提交至最后对应观察完成 |
| 补拉 | 实际最终Get字节 | 首次最终提交至最后对应观察完成 |
| 合并 | 两者实际字节之和 | 最早提交至最晚完成 |

GB/s为十进制，全局跨度不累加各核时间；包含间隙/依赖，不代表链路峰值。URMA使用实际提交与CQ观察事件，UBmem使用搬运发起与完成事件；软件观察不等于硬件完成时刻。如果只能取得Get调用起止，则另报call_span及对应有效带宽，不将其标为提交到完成的传输跨度。Index搬运与KV字节分开，零任务或缺失事件显示N/A。URMA记录Jetty/队列/在途配置，UBmem记录搬运流水，不虚构CQ。

level 3且stages包含fetch（包括all）时，自动追加实际任务的Fetch-only对照：正式Fetch worker与相同资源配置的KVAF BatchGet，分别单Q和全部Q。标明描述符准备的计时范围和脱离流水的依赖差异，独立收益不等于完整图收益。后端尚不支持该专项时明确标记UNSUPPORTED。

### 6.5 输出及正确性

终端按level自动显示对应汇总，不再设日志等级。完整事件、时间线、算法统计及配置落盘；日志示例见第8节。

保存原始cycles、频率和时钟域；跨核时钟对齐才给全局时间线，否则仅核内duration。重复采样不拼接、重叠区间不重复相加、未归属时间显式保留。缺事件或缓冲区溢出标INCOMPLETE，缺失不填0。

诊断不在Get前加同步，不改变队列顺序；必要正确性检查在计时外。失败自动保留token/slot、预期/实际、映射、任务与完成状态；安全快照采集失败不覆盖原始错误。

## 7. 命令示例

以下为待实现接口。默认URMA、device0和仓库内KVAF；若运行时未配置服务地址，需要指定kvaf-server。

```bash
# 默认冒烟：全部默认check + 默认合成bench + level 1诊断。
python tests/operators/lifetch/test_lifetch.py all --kvaf-server HOST:PORT

# check用例配置内置；仅初始化可用 --suite init。
python tests/operators/lifetch/test_lifetch.py check --kvaf-server HOST:PORT
python tests/operators/lifetch/test_lifetch.py check --suite real \
  --snapshot /path/to/snapshot --kvaf-server HOST:PORT

# 合成性能，仅调整主要负载参数。
python tests/operators/lifetch/test_lifetch.py bench \
  --miss-percent 10 --queries 4 --batches 1,4,8,16 --kvaf-server HOST:PORT

# 真实数据的形状来自输入，不再指定合成参数。
python tests/operators/lifetch/test_lifetch.py bench --transport ubmem \
  --input real --trace-root /path/to/trace --group 10 \
  --round-range 51:51 --kvaf-server HOST:PORT

# 同一bench负载，增加逐核和算法明细。
python tests/operators/lifetch/test_lifetch.py diagnose \
  --miss-percent 10 --queries 4 --batches 16 --level 2 --kvaf-server HOST:PORT

# 自动采集Update内部各步骤cycle。
python tests/operators/lifetch/test_lifetch.py diagnose --transport ubmem \
  --batches 16 --level 3 --stages update --kvaf-server HOST:PORT
```

## 8. 展示日志示例

**以下数字全部为格式演示，不是真机结果或性能承诺。** 终端短表与完整JSON/CSV使用相同字段、单位和关联ID。

### 8.1 check

```text
CHECK transport=urma suites=init,basic,boundary,stateful,contract
Geometry, seed and execution mode are fixed per case; full configuration in result.json
case                      BS  execution  result  detail
init                       -  -          PASS    put/get bytes matched; resources recreated
cross_query_duplicate      2  graph      PASS    cross-Q duplicates=128; slots/KV matched
empty_query_fetch         16  graph      PASS    Q3/Q4 new_fetch=0; completion checked
stateful                   2  graph      PASS    rounds=4; hit/evict/refetch covered
reject_bs32                -  -          PASS    expected validation error
SUMMARY planned=5 pass=5 fail=0 unsupported=0 not_run=0
Results: <run>/summary.json
```

### 8.2 bench

```text
BENCH transport=urma input=synthetic generator=calibrated-v1 seed=43
Timing: graph event; mean/p50/p99 us; fixture restore and validation excluded
BS | serial mean/p50/p99 | fused mean/p50/p99 | reduction | speedup | correctness
16 | 500/495/530         | 350/345/380        | 30.00%    | 1.429x  | PASS
path   | cold/selected | cold_miss | prefetch | final | KV_read_bytes
serial | 640/6400      | 10.00%    | 0        | 640   | 419840
fused  | 640/6400      | 10.00%    | 400      | 320   | 472320
Results: <run>/results.csv; <run>/summary.json
```
精度失败但运行正常完成时，仍展示性能和对比；correctness明确为FAIL。例如：

```text
BS | serial mean/p50/p99 | fused mean/p50/p99 | reduction | speedup | correctness
16 | 500/495/530         | 350/345/380        | 30.00%    | 1.429x  | FAIL
execution=COMPLETED timing=COMPLETE correctness=FAIL
KV mismatch: request=0 Q=2 token=123 slot=45; details=<case>/failure.json
```

上述30%仍为实测耗时差异，但不能作为精度正确的优化收益验收。
串行实际读取可能包含候选hot重拉，并不保证总Get等于cold；示例仅表示该例两者恰好相等。

### 8.3 diagnose：level 1 / 2 / 3

```text
DIAG level=1 path=fused repeat=1 trace=COMPLETE correctness=PASS
Algorithm: selected_unique=200 initial_cold=40 prefetch=96 final=50 KV_read_bytes=95776
Profiled pass; separate from bench timing
kernel               start_us  end_us  duration_us
init                     0.0    10.0         10.0
li                      12.0   212.0        200.0
resolve_update_fetch   214.0   314.0        100.0
span_us=314.0 kernel_sum_us=310.0 gap_us=4.0

LEVEL 2 excerpt (full table: timeline.csv)
block role          request Q   start_us end_us duration_us
0     resolver      0       all    214.0  260.0        46.0
1     index_update  0       all    214.0  290.0        76.0
32    fetch         0..15   1      214.0  314.0       100.0
last_end_block=32 last_step=fetch.publish_valid
block32: task_wait=20us pack_submit=10us complete_wait=65us publish=5us

LEVEL 3 excerpt; raw cycles retained, clock source recorded in JSON
block request Q step                  calls duration_cycles bytes
1     0       1 update.plan                1            2400     -
1     0       1 update.copy                1            6000 32768
1     0       1 update.commit              1            1800     -
```

level 2/3片段只在相应等级采集时出现。实际日志逐核记录所有区间；示例省略其他核，不代表只采最慢核。

### 8.4 diagnose：算法行为与趋势（level 2/3）

```text
ALGORITHM source_round=51 repeat=1 request=0
initial_hot_valid=16000 capacity=16384 selected_unique=200 cold=40
window candidates local_hit prefetch_dedup budget_suppressed posted useful unused redundant refetched
1              90        20             6                0     64     30     24        10         4
2              80        20            12               16     32     20      8         4         2
cumulative: posted=96 useful=50 unused=32 redundant=14 unused_rate=33.33%
Q  topk cross_q_duplicate new_unique local_served prefetch_served final_new
1   100                 0        100           50              30        20
2   100                40         60           25              15        20
3   100                70         30           15               5        10
4   100                90         10           10               0         0
final_total=50 KV_read_bytes=95776  # (96+50)*656

CONTINUOUS TREND request=0; distinct source rounds, not repeated samples
round cold selected prefetch unused final KV_read_bytes
51      40      200       96     32    50         95776
52      30      200       64     20    24         57728
```

示例TopK数量为便于展示而缩小。真实日志使用实际K。每Q满足来源按新增唯一需求归类，避免将跨Q重复重复计入；cold为轮初状态，不能直接与轮末final_new相减解释预取收益。refetched是交叉指标，不参与posted分类求和。

### 8.5 diagnose：Fetch带宽（所有level）

```text
FETCH BANDWIDTH repeat=1 bytes_per_entry=656 completion=software_observed
class    entries bytes  first_submit_us last_complete_us span_us GB/s
prefetch      96 62976              50.0            150.0   100.0 0.630
final         50 32800             120.0            200.0    80.0 0.410
combined     146 95776              50.0            200.0   150.0 0.639
urma: workers=8 jettys_per_worker=2; queue/inflight settings in diagnostic.json
Metric: effective transfer span, includes gaps/dependencies; not peak link bandwidth
```

### 8.6 失败或采集不完整

```text
FAIL case=stateful source_round=52 request=0 Q=2 token=123 slot=45
KV mismatch; mappings/descriptors saved: <case>/failure.json
TRACE INCOMPLETE missing=fetch.complete_wait.end block=32
execution=COMPLETED correctness=FAIL timing=COMPLETE trace=INCOMPLETE
```

以上示例表示计时已完成，但精度及部分打点失败；若设备执行超时或崩溃，则对应execution=FAILED、timing=INCOMPLETE。

### 8.7 all：默认冒烟汇总

```text
ALL transport=urma device=0 workload=synthetic
check    PASS  suites=init,basic,boundary,stateful,contract
bench    PASS  batches=1,4,8,16; correctness passed; performance in results.csv
diagnose PASS  level=1; trace=COMPLETE
Results: <run>/summary.json
```

all的PASS表示执行、正确性和指定采集通过。精度失败时总结果为FAIL，但仍保留已正常完成的性能和诊断结果。

## 9. 结果目录与验收

```text
run/
  config.json
  summary.log / summary.json / results.csv
  cases/<case-id>/
    result.json
    worker.log
    timing_samples.json       # bench原始样本
    diagnostic_summary.log    # diagnose汇总
    diagnostic.json           # 事件、算法计数、带宽及完整性
    timeline.csv              # 已采集时间线
    failure.json              # 失败时生成
```

check要求正确性及指定分支覆盖通过；bench独立报告执行、计时和正确性，精度失败不隐藏已完成的性能；diagnose独立报告正确性和所选采集项完整性。文件读取、框架校验和报告生成复用同一实现，数据路径和配置随结果保存，避免用户反复手动补日志。

level1终端按阶段耗时、每请求计数、有效带宽三张对齐表展示；不重复打印旧case摘要。
`level1_summary.log`保存同样的简表，`diagnostic_summary.log`和`diagnostic.json`保留原始区间与完整证据。
预取提交时间戳缺失时，仅在已有服务起点与完成观察点有效的情况下提供service-span带宽，明确包含控制开销；不将其标为精确doorbell到CQ跨度。没有有效区间仍标INCOMPLETE。

level2在level1简表之后，按每次采样独立展示：活跃block及最晚结束block、各核工作/等待区间、LI分类、逐Q Resolver/Update、窗口预取/unused、逐Q需求/补拉。`timeline.csv`从level2开始包含全部已定义步骤；`level2_summary.log`保存终端表格。
预取服务记录保留设备core/epoch字段、扫描/候选/提交周期、bitmap字节与计数，不将service core编号冒充请求编号。无有效打点记录显示N/A。URMA KVAF组计数每Fetch block只展示一次；UBmem不展示URMA CQ计数。资源几何取自实际捕获的block数，未采集的Jetty、在途上限及搬运配置明确标N/A。
