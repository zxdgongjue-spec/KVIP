# MR23 基线复现与 split score → 原 T2 后处理实验

先核对 KVIP main 提交 a56154c37bebb0928da4a4095e9670898e6b21c7 的完整改动、测试入口和调用依赖，再实现一个显式实验入口。目标是确定 LI 的 history 分核能否接入当前 LIFetch，同时避免 MR14 staged 路线的额外分数中转、窗口级 kernel 编排、重复 resident 扫描和分片 TopK 合并抵消收益。

1. 确认 MR23 的真实流程和计时边界，区分 native 串行参考、原融合、优化候选，核对 LI、resident θ、Cold predicate、增量 TopK、Service、Final classification、Resolve、safe/conflict Fetch、CQ join、Output 和 Index commit。明确 SFA 是否在计时内。
2. 从源码确认 T2 的含义、AIC/AIV 角色及原生 score 布局。producer 必须使用真实 packed row/Q/history 地址，resident 分片完成后才能形成唯一 θ；不得使用局部 θ、改变输入 owner，或直接关闭 scoreOnly 来尝试接入后续阶段。
3. 最小候选仅替换 score producer：完整 score 写到原 canonical GM；保留原 T2 resident θ、Cold 分类、增量候选、唯一 Service owner、Final 和 Resolve/Fetch/commit 策略。消除 scratch→scatter 和局部 TopK 合并；保持默认入口。记录完整 score 屏障对 resident 优先和首次 Fetch 提前量的影响。
4. 原 T2 Hot 只写 resident/recent score，完整 score producer 可直接替代其完成边界。候选执行 score → 原 resident θ → T2 Cold 分类/预取 → 原 Final → 原 Resolve/Fetch/commit；不保留多余的空 Hot kernel。
5. 使用同一 MR23 全图计时协议，固定 B1/B16、Q4、H131072、R16384、32 Index heads、同 seed/输入/固定 KVAF。对比精确提交的原融合、B1/P2、B1/P16、B16/P2；保留 native 串行参考监控漂移。miss 仅取 0/3/5/10/15/20%。增加 prefetch=0 控制组以识别重叠贡献；不能把独立 LI microbenchmark 或旧 staged 历史数据当作同场景端到端控制。
6. 预热整图、丢弃首轮、轮内交错，记录每个配置的 effective AIC/AIV、二进制与依赖版本、输入校验和、完整 event 均值、profiler p50/p99、LI/Resolve 阶段和原始 trace。Fetch 和 Index/output 完成必须包含在算子链终点；SFA 校验与模型端到端吞吐另列。
7. 完成分片覆盖/地址/全局 θ/tie/预算/trace 归属校验；能使用 CANN Model 时再编译和仿真一个标准用例。分别报告静态、CPU 模型、CANN 编译、指令仿真及 A5 实测状态，不能把估算写成实测。没有服务器访问时交付可直接在容器执行的仓库脚本，明确待验证项。
8. 交付源码、脚本、分析、提交和可复现命令。若最小候选整链退化，根据新增 Cold GM 读取与预取提前量损失定位原因，再决定是否拆 Hot/Cold producer、继承更多 predicate 并行度或引入段级 ready；不要直接扩大后续分核范围。
