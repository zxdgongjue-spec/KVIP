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
