> 历史研究归档：测试基线为81b1472＋实验overlay，不代表本MR新流式main的性能。本地.aiinfra路径仅标识原始证据；可移交摘要见 results/model-records.json，当前集成状态见 README.md。

# KVIP 独立 AIC/AIV 分核研究与仿真实验 Prompt

请作为 AscendC、长上下文 Indexer、KV 缓存与异构调度研究工程师，直接完成以下研究及可复现实验，以中文交付。

1. 使用 `${KVIP_ROOT}` 中的 KVIP 主仓，核实远端 main、实际 commit、当前代码树与依赖下载状态。保留原 KVIPAgent 工作目录及其他 session 的未提交代码、容器和资源。复用“复核 ComputeMm1 优化分析”同步的 aiinfra-task、aiinfra-review、aiinfra-benchmark 工作流；未建立接受提交时采用单写入者和只读咨询审查，不伪称正式 freeze/gate/close。
2. **硬件资源预算为 C/V=(32,64) 与 (28,56)，AIC/AIV 没有硬件 1:2 任务配对约束。** 当前 mixed kernel、FIX 双目的、query/subblock 编号和 flag 信用的 1:2 都是可改的软件实现。替换仍须遵守目标API适用条件；MIX1:1类型支持不证明FIX直达UB组合合法。新问题模型分别约束 AIC 和 AIV，总量巧合为 1:2 不能缩小研究空间；明确原型保留的局部软件约束。
3. 用户已指定**先仅在 CANN 仿真环境验证**。使用实际安装的 CANN9.1/Ascend950 CA Model；与其他 session 串行预约，不抢占。无需等待真机信息。记录仿真配置、runtime、源码与 driver 哈希、输入、实际 launch、预热/重复、原始日志和 oracle。CA Model 计数不转换为真机微秒、TPOT 或网络吞吐，不把 CPU 数学验证或编译成功算成设备执行。
4. 分析 BS=1/2/4/8/16、miss=3/5/10/15%、每请求 history=128K×30/50/100% 的60个场景，分别考察两档物理预算。128K=131072；报告 valid token 与分配长度/tile 对齐的区别。Q=4、G32/G64、D128、K2048、resident、page 和 payload 从当前用例提取。miss 是选中 TopK 的 query 级整数，另计 request 去重后的 U；不可用 miss×history 当必要搬运量。
5. 源码走读全算子链与 LI 内部的 batch/query/head/history 切分、MMAD M/N/K、L1/L0/FIX、score、resident 阈值、精确 TopK、hits/Progress、Service 和修复。说明 AIV 按 query 分工而非平分 history；区分逻辑 block、活动预算和物理并发驻留。比较 URMA/UBmem 与真实 vLLM 接入链路，标明实验 mock 的范围。
6. 解释当前分核策略重点解决的问题、目标及局限。建立以 selected payload ConsumerReady 的关键路径为主目标的资源约束 DAG、生产/消费服务率、buffer credit、带宽、阈值前缀、最终选择、传输 completion 和尾部修复模型。决策变量独立包含 AIC producer、AIV score/TopK/Service/tail 配额及时间复用；不要预设 v_score=2c。
7. 给出 query 分组与 history 分片的操作量、K 重读、score/候选流量、raw FP32 QK 桥接流量、scratch、任务粒度与归并成本。先建可复算结构模型，再用真实计数校准；不把 trunk 轮数当周期。对小 batch 补 history 并行，对大 batch 检验带宽与归并代价，尤其分析 C28/B16 的原实现回退。
8. 运行当前标准用例的完整仿真基线，再比较至少独立 AIV 最终 TopK 池、精确 local/merge 分片及跨 AIC 的 history 生产任务。保持同 binary/输入/runtime 的控制对照，保留 S1 消融以区分切分收益与移除 resident/classify/progress、Q2→Q4 的组合变化。先功能 smoke 再标准用例。对每个已测配置报告 scope、次数、Model 正常结束与完整输出 oracle；history 数值路径另与原链路逐 token 比较全部 score，lateTopK 保留原评分路径并核验 selected/outputs；未执行的60场景矩阵标明为解析模型，不能填造实验时延。
9. 对独立 AIV TopK 组件检查每 shard 的范围、唯一性、阈值及所有严格赢家，全局 selected set/score、输入未变、尾 guard；对完整链路保留 Index 全字段、输出关联、有效热槽 payload、miss 去重、排列一致性与 guard。对短 shard、空范围、partial queryEnds、ties 给出显式支持范围，不能让长度小于 K 的无符号运算下溢。
10. 计时分别报告 whole-Model Total ticks 与有流水同步的同 worker end-start；不直接相减不同 AIV 的原始时间戳，也不假定其与 footer 同频同 epoch。保留失败记录、撤回错误计时解释，再用独立构建复测，不替换运行中的 binary。
11. 检索并阅读原始论文和官方接口，整理思想、迁移点与边界。覆盖 Flash-Decoding、FA2/3/4、FlashInfer、NanoFlow、InfiniGen、DeepSeek-V3.2/DSA、精确 streaming TopK、KV offload 以及 Ascend 的独立 AIC/AIV taskflow。论文收益不直接当 KVIP 预测收益；不得把已有双池/队列/双 buffer 说成首创。
12. 提出可证伪的创新假设：LI 阶段驱动 AIV 角色调整、桥接流量感知的局部快通路与任务迁移、range-ready/精确 TopK/预取 deadline 联合计划、静态基础计划与轻量 credit 反馈。每项给出目标、机制、对照、观测指标、失败条件和最小实现路线。

交付：修订后的完整 prompt、源码与依赖状态、中文研究报告、60场景×两档预算的可复算计算模型、实验代码、实际结果 CSV、原始证据索引、独立审查与复现命令。结论以实际完成的运行支持，不把仿真正确性扩展为未测量的真机或 native transport 性能。未经用户明确要求不提交或推送仓库。
