# KVIP 契约闭环扩展执行 Prompt

以已提取的 KVIP 和候选 vLLM-DSA 适配为基线，修复审计可复现的页面替换与租约
生命周期缺陷，扩展分级测试，确认独立仓、框架镜像和安装 wheel 的公开接口对齐。
保留框架中已有未提交修改，在隔离检出目录实施；本任务不包含生产启用或远端发布。

1. grow 仅允许页表追加，阻止等长/增长时替换旧页；显式 rebind 覆盖 hot/cold/tail、
   H0 失效、非法页写入前拒绝和中途失败隔离。restore 后 owner 不可重用。
2. 从框架现有 CPU BlockTable 生成每 forward 的 request→page_ids 快照，eager/Graph
   准备共用；不增加 NPU 回读或修改 kernel。未知页变化拒绝，不静默损坏或回退。
3. 单 active lease 只能绑定一个 session，显式 acquired stream、generation、closed/
   active 和本地可重入锁；旧 session 不可访问新租约。未知完成保持资源与隔离。
4. Graph 新公共边界明确 current_stream 和保留资源回调；旧框架编排保持，补实际模块
   CPU 接线与共同状态验证。不能把 AST 或事件替身测试声称实际 NPU Graph 验证。
5. 公开稳定 facade 收口所有框架依赖。由于强 owner/Graph 协议不兼容，将公开主版本
   升至 3.0.0、包 0.3.0，设备/传输/writer ABI 不变；按能力组协商未使用的 ABI。
6. 覆盖 CPU consumer 非法/边界调用、完整输出 schema、真实 launcher host binding、
   32/64 heads packing 数值等价、admission 上下界；仅有模型测试不能作为设备正确性。
7. 修复跨 Git 根树/子树基线比较；完整 CPU 入口缺 Torch/compiler/关键测试或出现 skip
   必须失败，纯 host 明确独立子集。安装 wheel 的测试拒绝从源仓误导入 provider。
8. 性能证据同时门控 TPOT/selection 的 median/p95/p99，记录 warmup、样本、重复、
   完整命令和硬件/二进制/日志/功能报告指纹；不自动认证，不用 synthetic 数据宣称性能。
9. 同步两仓代码/规范/覆盖矩阵，构建新 wheel、sdist 和 Git bundle；保护设备源码、
   build/geometry/packing 与 oracle，实际检查全部 ABI 字段和基线 host 用例。

退出条件：所有当前 CPU 可验证项完成、两仓和独立安装包分别通过，提供可复现命令
和精确结果。硬件功能/性能保持 pending，明确新增 host 元数据/锁的开销仍需上板评估。
