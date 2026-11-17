# LI 单算子 K32 / B16 30 核扩展实验 prompt

在 MR14 的独立 LI 控制实验上实施以下工作，并保留已验证的原七点入口。固定每个 request 为 Q4、H131072、G32、D128、TopK2048；不执行 LIFetch 的 Fetch、Service、predicate、Index 或 CQ。分别计时 score 与 score→canonical→全局 TopK。对照是同模板单核组 LI，不把它称为生产 P2 或 SDK QLI。

1. B1 在原七个正交点之外新增 history_parts=32、query_parts=1（Q4×H4K/task）。保留原七点模式，并提供八点模式。K32 必须申请 32 AIC/64 AIV；记录实际任务图、核预算、wave 数、启动数与源码/二进制 SHA。
2. 新增 B16 固定 Q4 的独立矩阵：T1/30核、T2/30核普通波次、T2/30核均衡调度。T2 有 32 个逻辑 Q4×H64K task，不能声称能同时独占 30 个核。将逻辑 task 按 2048-key 对齐的小块展平，分配到 30 个 MIX1:2 核组，每核总块数差最多 1；owner/half 边界保留独立片段，输出区间唯一覆盖。单次设备 score kernel 内消费每核片段列表，不能在宿主以 30+2 两波冒充均衡。记录新增片段初始化成本、尾部不均衡和 canonical 中转。30 AIC 对应最多 60 AIV；Final TopK 的 AIV 预算单独披露。
3. 所有比较固定输入并验证每 owner/Q 的 canonical score、TopK 阈值、guard 和只读输入。B16 均衡版与普通 T1/T2 使用完全相同的 fixture。常规 A5 性能：warmup4、iterations20、丢弃首轮、三轮交错，按每轮同 batch 基线配对；不能使用 B1 基线解释 B16。
4. 在现有 CANN950 仿真环境中构建并执行 B16/T2 的普通波次与均衡版，另保留 T1 控制。尽量使用完整 H128K；缩小 history 的 smoke 必须单独标明。保留配置、退出码、完整日志、oracle、动态指令日志和反汇编。按 kernel、AIC/AIV、流水线/指令类汇总动态计数及分母；区分指令数量占比、活跃周期占比与关键路径等待。按有证据的源码/PC 映射解释 QK、搬运、CV/VC 同步、向量归约、scatter、TopK；无法映射的指令归为 unknown，不编造算法时间占比。静态反汇编只能作为补充，不代替动态指令统计。
5. 提供 A5 上独立 msprof trace 采集入口，采集 B16 T1、T2 waves、T2 balanced 与 B1 K32。profile 与常规事件计时隔离，输出完整命令、日志、trace 原件和可分析汇总。仿真 trace 不能当作物理 A5 msprof 数据。
6. 完成 planner/调度覆盖和实际 host 负向验证、CANN 构建和可运行的仿真实验。报告真实执行结果与阻塞；如果工具链或 Model 不支持动态日志，保存失败证据并提供 A5 采集脚本，不能用估算的指令数填充实测列。按当前共享目录流程交付更新后的脚本；提交 MR14 时保留并发远端改动，禁止 force push。

交付：实现与脚本、任务图、验证记录、B16 分核瓶颈分析、可下载的运行包；只有真机数据可用于宣称 A5 提速。
