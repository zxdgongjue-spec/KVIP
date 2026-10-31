# 公开契约隔离实现与验证报告

本文记录第一轮 0.2.0/契约 2.0.0 的实现与当时测试结果，后续审计发现的缺口和
修复以 [3.0 扩展验收报告](extension_report.md) 为准，不能用本文 implemented 状态替代最新验收。

## 交付边界

独立仓：`/path/to/KVOP`，分支 `codex/public-contract-isolation`。
源基线 vLLM-DSA `fae4597f7`，提取子树提交 `2f49ba0e140311aa8efd40dbb539b1169b959747`。
完整任务 Prompt 见 [implementation_prompt.md](implementation_prompt.md)。
独立仓的公开规范见 [public_contract.md](public_contract.md)，审计输入与原稿保存在
`docs/design_reference/`，它们描述修改前基线，当前状态以本报告为准。

框架适配位于 `/path/to/vLLM-DSA`，
原 `/path/to/vLLM-DSA` 的分支和生产代码未改动。
本次代码是可验证的候选实现，尚未合入、启用或发布到生产环境。

## 未实现项的处置与修改合理性

| 差异 | 处置 | 合理性与实际边界 |
|---|---|---|
| G01 公开版本/能力 | implemented | 不可变 Capabilities/LayoutSpec/Requirements、contract2.0、独立 ABI 版本和 fail-closed 协商；真实 SDK 配对仍需设备验收 |
| G02 admission | implemented | 统一 admit_request/raising wrapper，框架接入 host metadata；拒绝 C0/B33/类型/容量/tail/pool/resource；device preflight 保留 |
| G03 直接/间接 alias | implemented | extra 输出与 owned key/scale 可写，cache 完整区间与 tensor/workspace 交叉校验，保留合法只读 alias 和相邻缓冲 |
| G04 生命周期 | implemented | PreparedSession 和 GraphFrameSession、QueueOwnerProtocol，明确在途/排空/业务成功/隔离与 stream；框架 eager 使用 session，原 Graph 编排保持 |
| G05 H0/阈值/extra | preserved | 保留 V9 profile 的 victim hot 和 >= 保护，新增固定语义测试；不改成 legacy 规则 |
| G06 页面交接/身份 | implemented boundary + preserved | 新增显式 checked rebind_pages，保留 hot、cold、tail，原 grow/write 不调用该边界；epoch/身份由现有框架追踪 |
| G07 独立桩 | implemented | CPU-only 九参 FakeNpuRuntime；真实 consumer 方法→fake 测试；V9 延迟 owner 与 Graph frame 专属桩 |
| G08 传统算子全部抽取 | out-of-scope | 按能力组隔离；QSFA/legacy wrapper 仍由框架拥有，未引入全局函数表/timing setter |
| G09 硬件功能/性能 | hardware-pending | 未执行 NPU memory、KVAF eager/Graph、模型端到端；有可执行验收流程和 fail-closed 证据门槛 |
| G10 eager 分配/同步优化 | preserved + measured | 不移除同步、不更改 packing 或默认 kernel 参数；session 有可测 host 成本，不能宣称零影响 |
| G11 独立仓/发布 | implemented local | 独立 Git、wheel、来源/兼容 manifest、契约 CLI、ABI/源码/差分工具；远端发布和生产资格保持待完成 |

C01/C02/C03/C05/C06/C08/C11/C18 的 host 接口和隔离逻辑已实现；C04/C07/C09/C10/C12/C13
保留原设备/框架机制并补边界测试；C14 增加独立 consumer 桩；C15 保留框架职责；
C16/C17 的 host 验证已实施，真实设备与端到端性能要求尚未确认。

## 具体代码

- `kvop/v9/api.py`：公开版本、layout、能力、需求与协商。
- `capabilities.py`：结构化 admission，三参数兼容 wrapper 和可选完整 host geometry。
- `access.py`、`contract.py`、`index_cache.py`：访问模式、完整字节区间、字段/间接缓存校验。
- `session.py`：typed SelectionResult、CompletionResult、eager/Graph frame 状态机和 owner 协议。
- `transport.py`：真实 lease 字节量纲、Graph ABI/方法验证；不伪造未知 SDK fingerprint。
- `owned_index_cache.py::rebind_pages`：仅显式交接边界执行的重定位；不接入 steady hot path。
- `kvop/testing/`：CPU FakeNpuRuntime、DeferredTransport 和 StubQueueOwner，生产发现不选择它们。
- `contract_info.py`、`manifests/compatibility.json`：可查询机器契约与未验收资格状态。
- `tools/`：C++/Python ABI、基线差分、host 派发和真实证据性能门槛。

框架仅修改 `v9_backend.py`、`v9_config.py` 及新增 consumer/协商测试，嵌入 KVOP
镜像与独立仓同源。模型文件、model runner、attention API 和 NPU 设备源码未修改。

## 已确认的功能保持证据

1. 修改前 CPU Torch 基线：42 passed。
2. 44 组 B1/B2、W0/W4、11 类 fixture，baseline/candidate 输出数组逐字节一致。
3. `operators/` 及四份数值 oracle 与基线无差异。
4. 构建、device resources、geometry、packing 源文件与基线无差异。
5. 低层 launch/_call/wait/resolve/fetch/completion 的 AST 与基线一致。
6. 实际 C++ header 编译后与 Python descriptor 同为 ABI13、528 bytes、74 个 offset 一致。
7. 新增回归覆盖五项审计探针漏检、public negotiation、stream/in-flight/release、
   Graph frame、九参实际 consumer、页替换/tail、H0/extra 和发布证据拒绝路径。

这些证据证明 host 行为及源码保持范围；不等于运行了真实设备 kernel 或完整模型。
最终 source suite 90 passed；在源码仓之外安装 wheel 后同样 90 passed；框架相关
host suite 88 passed，均无跳过。五项审计 probe gap flag 全为 false；完整结果保存在
`manifests/validation.json`，测试可通过 `bash tools/check.sh` 重跑。
独立仓 Ruff 及修改文件空白检查通过；完整框架 format CI 因缺少 pre-commit 未执行。

## 性能判断与生产约束

一次 CPU-only 微基准结果：baseline dispatch 约 1.88 us，强 session 约 3.33 us，
增量约 1.45 us/次。它不包含真实 tensor、kernel、网络与 TPOT，既不能证明设备回归，
也不能支持“零开销”。因此未将本次适配应用到原主工作区或生产配置。

`hardware_certified=False` 和 `production_promotion=blocked_pending_hardware_and_performance_qualification`
保持明确；要求硬件已认证的 negotiation 会拒绝。完整验收流程见
[hardware_qualification.md](hardware_qualification.md)。
`tools/qualify_performance.py` 拒绝 CPU/memory 桩、缺功能通过标记、缺 provenance 或
不可比环境；默认允许 TPOT 回归为 0%，失败不会自动认证、改配置或部署。

不要为满足“功能和性能无影响”而移除 completion/guard、改变 H0 或伪造硬件通过。
若没有 A5/KVAF 环境，当前正确的完成状态是代码候选与 host 验证完成、生产验收待完成。
