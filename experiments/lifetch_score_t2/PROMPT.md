# 执行需求：MR14 分核并行方案接 MR23 融合后处理

目标：在 MR23 a56154c37bebb0928da4a4095e9670898e6b21c7 的真实 LI/Fetch 通路上，适配 MR14 c684d79f3a4f8432bf0915edeba392c7d54ba289 的 history 分核/direct producer/predicate 并行机制，测量完整融合链收益。之前“完整 H score 分核再接 T2 Cold GM 再读”仅作为控制，不能当作最新 staged/direct 联合候选。

1. 先分核完成 Hot/resident score，随后执行 MR23 原 resident Top2048 与唯一 θ；按原 T2 实际窗口切 Cold history，每个分片在原 score reduction 的 UB 上完成 Cold predicate，直接写原 canonical 和 packed bitmap。使用真实 packed row/Q/owner，不造虚拟输入，不加 scratch/scatter/分片 radix merge。
2. P 分片不可乱序发布连续全局进度。用同 stream 的窗口 producer completion 汇合后，由唯一 Service lane 发布原 T2 两 cursor，执行 MR23 原增量 TopK、跨 Q ranked admission、去重和预取。保留 predecessor fence、Final、Resolve、safe/conflict Fetch、CQ join、Output 和旧映射消费后 Index commit 的依赖。
3. score 并行度与尾部 T2 配置独立。默认对 B1/2/4/8/16 各自比较固定 MR23 与 window_budget，P依次16/16/8/4/2，对应16/32/32/32/32 AIC；B1实际W2，其余请求W4。提供 --batches 和 --parts-map，拒绝B×P>32、不隐式改多波。可选full_budget采用同P表；旧固定P2/P16名称兼容。说明核数比不是加速比，B16不增加计算核数，重新加载Q、短Hot片和更多启动边界可能抵消收益。
4. 固定 Q4/H131072/R16384/head32/seed43、同 KVAF revision、input hashes、miss0/3/5/10/15/20%、pf1024/W4。增加pf0控制观察预取重叠。原 serial LI+LIDU+Fetch 仍测量；SFA 校验在计时外，不伪称模型端到端吞吐。
5. 验证整页切片覆盖、地址、θ/ties、bitmap、两 T2 cursor、budget、CLI 与全图kernel计时归属。复用58aa570新mixed producer/AIV Service的编译与Model证据，分清其范围与未验证的Service/URMA；本轮只改脚本不重跑旧Model。正式A5 summary只接受全部正确性PASS，丢弃r0，各batch单独任务保持同DDR pool容量，同batch路线邻近且轮次反转/旋转顺序。记录版本、实际geometry和trace；新增同图Hot/Cold score、θ、Service、Final拆解，按W/W/1/1契约检查，并与整体LI时间加总一致。
6. 保留固定依赖版本保护和失败build fresh-attempt恢复，不能为了跑通复用错ABI产物。交付仓库内可直接执行的脚本、提交与宿主机同步/容器命令；如没有 A5 访问，明确待真机验证，不生成虚假的加速比或指令占比。
7. 明确SerialBefore是SDK LI+LIDU+完成的BatchGet，本runner不包含同协议MR14 SerialAfter；不能把独立LI的52.14µs套到融合链。说明score/predicate已继承分核，但增量TopK/Service并行度、residentθ、窗口屏障和Fetch依赖仍可能抵消；尤其B1仅一个Cold窗口，B16原score已经32核。summary将相对MR23整链净收益拆成LI、Resolve/Get、gap和其它/边缘四项，正数省时，检查加总相等，保留配对轮次和串行漂移。
