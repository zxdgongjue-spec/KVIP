# 执行需求：MR14 分核并行方案接 MR23 融合后处理

目标：在 MR23 a56154c37bebb0928da4a4095e9670898e6b21c7 的真实 LI/Fetch 通路上，适配 MR14 c684d79f3a4f8432bf0915edeba392c7d54ba289 的 history 分核/direct producer/predicate 并行机制，测量完整融合链收益。之前“完整 H score 分核再接 T2 Cold GM 再读”仅作为控制，不能当作最新 staged/direct 联合候选。

1. 先分核完成 Hot/resident score，随后执行 MR23 原 resident Top2048 与唯一 θ；按原 T2 实际窗口切 Cold history，每个分片在原 score reduction 的 UB 上完成 Cold predicate，直接写原 canonical 和 packed bitmap。使用真实 packed row/Q/owner，不造虚拟输入，不加 scratch/scatter/分片 radix merge。
2. P 分片不可乱序发布连续全局进度。用同 stream 的窗口 producer completion 汇合后，由唯一 Service lane 发布原 T2 两 cursor，执行 MR23 原增量 TopK、跨 Q ranked admission、去重和预取。保留 predecessor fence、Final、Resolve、safe/conflict Fetch、CQ join、Output 和旧映射消费后 Index commit 的依赖。
3. score 并行度与尾部 T2 配置独立。默认比较 MR23、B1/P16 window、B16/P2 window；B16/P2 限制32 AIC/64 AIV，B1实际W2，B16请求W4。额外 full P2/P16 为可选控制。说明 B16 不增加计算核数，重新加载 Q 和更多启动边界可能抵消收益。
4. 固定 Q4/H131072/R16384/head32/seed43、同 KVAF revision、input hashes、miss0/3/5/10/15/20%、pf1024/W4。增加pf0控制观察预取重叠。原 serial LI+LIDU+Fetch 仍测量；SFA 校验在计时外，不伪称模型端到端吞吐。
5. 编译新 mixed producer 与 AIV Service，验证整页切片覆盖、地址、θ/ties、bitmap、两 T2 cursor、budget、CLI 与全图 kernel 计时归属。Model 验证一个标准 score/predicate 用例，分清该范围与未验证的 Service/URMA。正式 A5 summary 只接受全部正确性 PASS，丢弃r0并交错测量，记录版本、实际geometry和trace。
6. 保留固定依赖版本保护和失败build fresh-attempt恢复，不能为了跑通复用错ABI产物。交付仓库内可直接执行的脚本、提交与宿主机同步/容器命令；如没有 A5 访问，明确待真机验证，不生成虚假的加速比或指令占比。
