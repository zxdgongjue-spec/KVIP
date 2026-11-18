# 本地验证记录（2026-11-17）

- 原 LIFetch 方案保存在 `20adb7e`，实验从该提交建立独立分支。生产源码未修改。
- runner 固定7个点：`(T,Qparts)=(1,1),(2,1),(4,1),(8,1),(16,1),(1,2),(1,4)`。所有点固定 B1/Q4/H131072/G32/D128/TopK2048，默认预算32 AIC/64 AIV。
- 4项 CPU 检查通过：C++/Python计划一致且每个真实Q/H坐标覆盖一次；实际host在ACL stub下正确打包、重排并保护dummy/guard；汇总只接受完整7点及r0证据，拒绝日志改动；launch范围和非法几何检查。
- CANN 9.1.0、`Ascend950PR_9599`、`RUN_MODE=npu`全新目录构建通过，包含device编译和host链接。生成元数据确认score为MIX1:2，scatter/TopK为AIV_ONLY；host stub仅有三个LI入口。
- CPU stub的设备时间为null。测试中的合成时间只验证汇总逻辑；不能用来报告指令数、Model周期或A5加速。
- 当前容器没有NPU。CANN静态注册在main之前调用`aclrtGetSocName()`并获得null，随后构造字符串抛异常；因此这里只完成编译验证，没有执行设备数值、同步或性能验收。A5需使用实际SoC/CANN环境重新构建并运行。
- `score_p50_us`仅含score waves；`li_p50_us`包含必要的重排和同一个全局TopK。没有Fetch/Service/Index。串行基线及单wave的T1/Qparts2无需重排；Qparts4保留物理Q2对齐及dummy行压紧的成本。
- 一键启动脚本`run_a5.sh`通过Bash语法检查及3项编排检查：本地Git拉取/新worktree与原修改保留、固定参数及SDK开关转交、构建/运行失败后停止并保留退出码和打包日志。检查使用模拟执行后端，不执行CANN或NPU；命令为`python3 experiments/li_qk/test_a5_launcher.py`。
- 增补联网处理后，启动脚本6项检查通过：此次Git代理覆盖不写回配置；origin不可达时本地提交/离线bundle仍能进入同一实验流程；本地缺少指定提交时拒绝运行。离线和网络检查仍使用模拟LI后端，不表示A5设备结果。
- 固定目录宿主机脚本`sync_a5_host.sh`通过Bash语法检查及本地Docker验证：从已有离线包自动创建同步目录、重复同步、目标容器中两个文件的SHA一致、容器不存在时停止。未启动LI kernel，不能作为A5算子执行验收。
- 无容器CLI时自动准备源码，`--run-here`在当前A5环境使用NPU路径执行。增补检查覆盖没有CLI时的真实Git/bundle准备、缺少`npu-smi`时提前停止、参数转交和后端失败传播；最后的NPU查询与LI执行在流程检查中使用stub，不产生设备性能结果。原Docker复制流程仍通过本地Docker哈希回归检查。
- 共享目录入口`update_shared_a5.sh`通过Bash语法检查和本地Docker已有挂载验证：宿主机只更新文件、容器从映射目录读取并校验哈希、重复更新保留旧release、容器中损坏的代码包/脚本被拒绝。没有运行NPU kernel。当前部署使用已确认的`/home/w00568585 → /home/w00568585`映射，真机计时只在容器启动脚本正常运行后生成。

本地完整日志、7点计划、构建哈希及元数据保存在`tests/results/li-qk-controls-20261117/`。交付包包含`validation/`中的检查记录，不附带跨CANN版本复用的二进制。A5性能结果由`run.py run`和`summarize`生成。

## K32 / B16 扩展的后续验证

以上是原七点的历史记录。本轮新增验证以`EXTENSION_REPORT.md`为准，证据根为`tests/results/li-qk-extension-20261117/`：

- 六项CPU检查、七项launcher检查通过；B16/H8192和H131072的owner/history覆盖及C++/Python任务表一致。新增套件/profile独立输出与30/60预算转交有编排检查。
- CANN9.1.0/Ascend950PR_9599全新NPU构建通过；最新C++/header SHA与构建receipt相符。
- CANN950 Model B16/H8192完整LI三控制组PASS；balanced先发现错误，再增加片段CV/VC fence并重新完整验证PASS。失败版不作为性能结果。
- B16/H131072独立scatter与TopK PASS，使用GM预填oracle。完整score600秒TIMEOUT；没有完整score正确性或全核指令统计。
- 动态issue日志仅physical core0，经实际入口PC与相同ELF核验；逐kernel/core分类与opcode占比已保存。Model whole ticks不是A5微秒或算法时间占比。
- 新`profile.py`提供物理A5独立msprof collect/export及oracle/预期kernel检查；本地没有A5，尚无真实msprof trace。真机命令见报告。

## G32 / 30核粒度补测

本轮结果见`GRAIN_REPORT.md`，证据根为`tests/results/li-granularity-20261117/`：

- 七项CPU检查、七项launcher检查通过；新增grain128..65536校验、C++/Python一致覆盖、短history仅Model score入口许可、参数与summary转交核验。
- CANN950 Model七次独立balanced score全部PASS，开全核mask0xffffffff，实际30 AIC/60 AIV。四个短H探针，三个固定H8192/grain2048/512/128控制点；固定H使用同binary和相同fixture。
- 全核issue窗口包络40759→35002→34402；缩短14.12%/15.60%。总Init46→58→60、总指令增加。issue窗口含等待，不等于活跃周期/完成时间，也不是A5微秒。
- Model与fresh NPU构建的compiled C++/headers及共享device source SHA均与当前代码一致，binary/library SHA核验一致；NPU receipt所有wrapper SHA与最终代码一致。Model receipt的部分Python wrapper随后补充统计/参数检查，各capture保留实际argv、geometry、Model配置与fixture SHA。
- 原完整H128K score超时状态未改变，未有真实A5 msprof；本轮未实施producer直写canonical、初始化复用或Final并发。

## 2026-11-18 后处理持续分核

当前报告见`TAIL_REPORT.md`，证据根`tests/results/li-tail-20261117/`。

- 七项CPU检查通过：C++/Python几何与唯一覆盖、局部TopK精确合并/tie/无效尾、局部θ不能代替唯一θ、resident候选＋Cold GT θ的Final覆盖性质、B16任务量、汇总拒绝不完整/跨binary/fixture混比、opcode区分GM与UB。
- H32K/R8K/P4的C0/C1/C2/C3完整数值控制均PASS（结构适配resident）；同binary/fixture下C1 whole ticks为C0的0.3997，C2为1.0960。
- 最终代码复用原resident primitive；B1/H128K/R16K/P4完整后处理C0/C1全部PASS，whole ticks1135835/243506。Final＋classification＋并集C0/C2全部PASS，whole ticks83149/96339；resident参考单点PASS。动态日志mask0xffffffff，实际ELF/entry/launch验证。
- B16/H8K/R4K/P2四组正交控制PASS，whole ticks C0/C1/C2/C3=107992/73975/175426/140404；resident_parts=2研究适配器数值PASS（性能成本尚非相同prefix输出控制）。B1/H4096/R2048/P2的C2 Final边界用例PASS，最后分片有效长度2045/2046/2047/2048，验证无效候选、合并、真实token映射和guard。最终代码11项Model capture，与v3四项结构控制分别存档，共15项。
- 共享目录脚本通过真实本地bundle发布/verify/help、脚本/包/commit篡改拒绝、重复发布保留旧release检查；未执行CANN或NPU。生成的七个tail入口元数据均为K_TYPE_AIV。
- compiled C++/headers与CANN950 Model、fresh `RUN_MODE=npu`构建的SHA相符，binary/library SHA也验证。Python统计/脚本随后完善，不沿用旧Python分类标签解释GM访存。
- 模型输入是已有最终score，mapping/LRU固定，只验证隔离后处理契约；不模拟动态Service/URMA/Index，也未实现producer/consumer常驻融合。没有真实A5时间或msprof，脚本供服务器采集。
- 容器入口`run_tail_in_container.sh`补为正式仓库文件；六项本地shell检查通过，覆盖仓库入口/共享入口、显式共享路径、参数转交、缺少release的更新提示及无参数启动。宿主机发布使用同一仓库文件；旧提交保留原生成方式。没有执行NPU。
- 现有`run_tail_a5.sh`新增`--from-repo`，直接构建当前仓库，无需共享release/bundle。五项本地编排检查通过：无共享包时校验、B16参数与bench/profile转交且禁止fetch/bundle/clone/worktree、构建失败退出码23并停止后续执行、模式冲突拒绝、原共享模式兼容。使用真实本地Git与模拟NPU/backend，仅验证编排，不产生设备性能；工作区修改保留，记录源码状态。

## B1 history 1/2/4/8/16/32/30 单轴对比

- 新正式入口`run_history_a5.sh`直接从仓库执行，固定B1/Q4/H131072/G32/D128/TopK2048。七点均一次score启动，实际下发T个AIC、2T个AIV；Final仍四AIV。
- T30按原生H128 tile均衡，26片4352 key、4片4480 key；C++/Python每个Q/history坐标覆盖一次，最大片长差128。原power-of-two分片边界保持不变。
- 新四项CPU检查通过：两个history几何的C++/Python一致与覆盖、真实host在ACL stub下的七点score/scatter/TopK/guard/oracle、汇总同轮配对并拒绝错误核数、正式脚本plan-only和32/64预算校验。原七项score回归检查全部通过。
- 真实shell/local Git配合模拟NPU/backend的编排检查通过：不fetch或导入bundle，转交history7与warmup4/iterations20/rounds3，七点msprof另采，构建失败23停止后续执行并保存日志。没有设备性能数据。
- CANN9.1.0/Ascend950PR_9599 fresh NPU构建与host链接通过。仅在本地无驱动容器链接时，将匹配CANN的devlib追加到已有LD_LIBRARY_PATH；没有使用该构建执行NPU。此前两个因本地驱动/依赖搜索路径失败的构建保留为失败记录，不作为通过结果。
- 证据为`tests/results/li-tail-20261117/history7-plan.json`、`history7-cpu-checks.log`、`history7-launcher-*/receipt.json`及`history7-npu-final/build-manifest.json`。A5时间和msprof需真机执行新脚本采集。

## A5源码快照断链与提前退出修复

- 用户日志确认CPU检查通过，构建前`prepare_score_source.py`复制`lidu/build/.../op_host/liboptiling.so`断链失败，尚未进入NPU编译或性能测量；此前的快速结束是失败退出。
- score/尾部源码快照排除生成目录，并记录`excluded_generated_paths`。新增真实Git/source-copy回归复现同样断链，确认生成目录跳过、所需源码与本地修改保留，必要源码断链仍会报错。history CPU检查现为五项，全部通过。
- 启动脚本实时tee各阶段日志，保留失败退出码和阶段；只在汇总完成后打印成功。四项本地编排检查通过：正常结束、build退出23、后端返回0但缺少summary、vendor环境脚本提前exit0。没有实际NPU计时。
- 新证据在`tests/results/li-tail-20261117/history7-failure-logging-*/receipt.json`；计算kernel和七点几何未修改。

## B16 30/32核的八组调度对比（2026-11-18）

- 正式仓库入口`run_b16_a5.sh`：30/32 AIC（60/64 AIV）各测两半waves、balanced G2048、balanced G128、balanced_owner G128。允许将细粒度量子替换为512；Final始终64 AIV。
- 新五项CPU检查通过：C++/Python两个history与两个细粒度计划一致且唯一覆盖；八组真实host在ACL stub下的打包/score/scatter/Final/guard/oracle；同轮三类基准配对及错误Final预算/不完整r0拒绝；仓库/profile完整计划；真实本地Git与模拟NPU/backend编排，覆盖无联网、参数转交、独立profile、构建退出23传播与停止。CPU stub时间为null，合成汇总值只用于测试。
- 原七项score控制与五项history控制回归全部通过，Bash语法与`git diff --check`通过。新证据`tests/results/li-tail-20261117/b16-cores-cpu-checks.log`和`b16-cores-plan.json`。
- H128K静态计划：C30 G128累计工作量69888–70016 key/核；balanced_owner有效片段60→44，但scatter仍两次。C32四组任务映射相同、每核65536 key；均衡wrapper额外控制可能抵消收益。
- CANN9.1.0/Ascend950PR_9599 fresh NPU构建通过，receipt位于`tests/results/li-tail-20261117/b16-cores-npu-build/build-manifest.json`。当前host.cpp、plan.h、score.cpp、post.cpp SHA与构建记录一致；随后增加的Python编排测试和文档不改变编译计算源码。
- 新owner边界方案没有新增Model运行或A5性能结果；60→44仅为静态片段数量，不是实测动态指令降幅。脚本测量数字LI（score＋必要scatter＋相同Final），不含resident/Cold、Service、Fetch/Index或并发流水。完整策略与服务器执行方法见`B16_CORES_REPORT.md`。
