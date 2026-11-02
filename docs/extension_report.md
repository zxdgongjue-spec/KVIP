# 契约 3.0 扩展与验收报告

本页保留 0.3 阶段的历史记录。现行 0.4.0 已统一 KVIP/kvip 命名，采用并列 submodule
和 editable 联调；最新命令见 [开发说明](../DEVELOPMENT.md)，实际结果见
[机器验证记录](../manifests/validation.json)。
名称迁移后的保护比较明确记录 namespace 归一化，不再声称全部 Python 文件原始字节相同。

本轮执行需求见 extension_prompt.md，生效规范见 public_contract.md，逐环节覆盖见
conformance.md。完整命令结果与安装产物状态见 [机器验证记录](../manifests/validation.json)；本报告不
将 CPU 验证转换为硬件认证。第一轮报告保留为 implementation_report.md。

## 已补齐的差异

| 审计差异 | 实现与验证 |
| --- | --- |
| grow 可绕过页替换边界 | 已有页前缀不可变、CPU page_ids 证明；跨页追加和非法替换回归 |
| 部分写入后可重试损坏数据 | grow/rebind/restore/write 异常 poison、H0 失效并禁止复用；注入 scale copy 失败回归 |
| restore 后旧 owner 仍可用 | restore 完成后退休；拒绝读取、增长、写入及二次恢复 |
| 同 lease 可绑定多个 session | owner 本地锁、唯一绑定、显式 acquired stream、closed/active 与 generation 校验 |
| 释放失败后可再次释放未知 lease | 框架直接 owner 路径也 sticky quarantine；保留 lease/资源并拒绝再次释放 |
| 框架依赖内部 KVIP 模块 | 所有稳定依赖收口到 kvip.v9.__all__；真实导入测试拦截私有模块依赖 |
| 页表契约只在独立 API 实现 | model_runner metadata hook 从既有 CPU BlockTable 每 forward 按 request ID 快照，eager/Graph prepare 共用 |
| 只有 AST/事件替身测试 | 新增真实模块导入、owner native packing、CPU GraphLayer+真实 OwnedIndex 接线，以及真实 ContractLaunch C++ shim 绑定 |
| 边界/packing/consumer 用例不足 | 32/64 heads 数值等价、容量上下界、零/满 descriptor、多目录/非法参数/保护区 |
| 镜像基线脚本失败 | 按实际 Git 根树/子树归一化；保护 82 文件、全部 launch/completion AST 与 44 host 模型用例 |
| 缺 Torch 时关键用例静默 skip | host 显式纯子集；完整 cpu 缺依赖/关键 suite/任何 skip 失败；安装 provider 路径明确 |
| 性能门槛只比较中位数 | TPOT/selection median/p95/p99 全门控；样本/配置/日志 hash/功能报告共同验证 |

公开 host 契约因 owner 字段和 Graph 回调不兼容升级 3.0.0，包版本 0.3.0；设备 ABI13、
Graph transport ABI1、Index writer ABI2、设备源码、packing、geometry、build、oracle
和低层设备提交/完成逻辑保持基线。eager-only 不再强制协商未使用的 Graph/writer ABI。

## 验证层级与局限

独立完整 CPU 契约和框架镜像分别通过 287 项无 skip 测试；安装 wheel 同样 287 项通过。
框架分别使用镜像源码与非 editable 的 site-packages 安装包执行 103 项
CPU 逻辑/接线场景，工具本身另有 5 项回归。C++ descriptor 为 528 bytes，全部 74
字段偏移匹配；源基线 44 host 用例一致。纯 host 明确排除 Torch 依赖后 190 passed、
4 deselected、0 skipped。最终机器记录见 [validation.json](../manifests/validation.json)。

真实 ContractLaunch 测试使用实际 Torch Tensor/schema、实际指针与 geometry packing，
编译 C++ host shim 并经 ctypes 调用；shim 不计算 LI、执行 DMA 或模拟网络时间。
框架模块测试仅替换重量级 package 初始化、logger、torch.npu 与外部 native/KVAF 边界；
测试实际 backend/graph/config/contracts 和公开 KVIP，仍不等于运行完整 vllm serve。

新增 CPU 页快照、锁和状态校验有 host 成本。一次 CPU dispatch 对照新增约 6.83 us，
仅记录该验证环境的 Python 开销，不能推断 TPOT；稳定 page_ids 的同长检查及页内增长复用
私有页表，不增加 table.clone 设备拷贝。NPU memory、真实 SDK/SoC 构建、KVAF/SQ/CQ/Graph、跨 rank、
QSFA/模型 accuracy、端到端 TPOT/尾延迟和 allocator 实际页面交接仍为 hardware-pending。

该阶段已执行相关 Ruff、空白检查；完整框架 format.sh ci 因缺 pre-commit 失败。
model_runner_v1.py 的 6 项全文件 Ruff 问题在 HEAD 基线已有，本轮新增 metadata hook
没有增加这些问题；未顺带改写无关代码。不得将完整仓格式检查描述为通过。

## 交付位置

独立算子实现以本仓为交付根目录，框架通过 `third_party/KVIP` 消费。
当前源码获取、版本固定和 editable 联调方式见 [开发说明](../DEVELOPMENT.md)；
历史测试数量以本报告记录的阶段为准，最新结果以机器验证记录为准。
框架生产启用前必须执行 [硬件验收](hardware_qualification.md)，不能以本轮 CPU 结果自动认证。
