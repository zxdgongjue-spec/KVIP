# KVOP 公开契约隔离与独立仓实现 Prompt

## 目标和基线

你是负责 Ascend V9 融合算子与框架集成的工程师。以 vLLM-DSA 提交 `fae4597f7`
中的 `third_party/KVOP` 为唯一实现基线，完成公开接口契约的 host 代码、独立桩、
框架适配、回归验证，并生成可单独构建、测试和版本管理的 Git 仓。
契约依据为《融合算子接口契约》2.0.0-draft.1 的 C01..C18 和审计 G01..G11。

源仓有用户未提交工作。使用现有最新代码 worktree，禁止覆盖原工作区、stash、reset
或把旧 `ascend-dsa-fused-ops` 误当生产基线。独立仓目录为
`/path/to/KVOP`，基线子树提交为
`2f49ba0e140311aa8efd40dbb539b1169b959747`；不创建远端、不推送或部署。

## 首先梳理并判定修改合理性

按类别逐项记录：

1. 必须实现：公开版本/LayoutSpec/capabilities 握手、统一 admission、直接与间接
   buffer 别名检查、typed outputs、owner/session 生命周期、stream/完成/释放协议。
2. 测试和隔离：legacy 九参 consumer fake、V9 独立 owner 桩、source/ABI/数值
   基线对照、wheel 独立安装、无框架反向 import、版本错误 fail-closed。
3. 保持已有语义：H0 包含有效 victim 候选、阈值 >=、unique miss、extra 反馈、
   sticky/quarantine、ABI13、kernel 数量及提交顺序、默认参数与同步行为。
4. 硬件发布门槛：NPU memory、真实 KVAF、Graph replay、模型 accuracy/TPOT。

只修改能用 host 回归、字节一致或公开调用测试支持的内容。硬件环境缺失时不声称
功能和性能无影响：交付隔离实现及待合入适配，标为硬件验收待完成，不激活生产配置。
不为满足“完整”擅自扩展 Q2/Q3/变长、改 H0、移除同步或重写 kernel。

## 实施要求

### A. 公开接口

- 无 torch/NPU 初始化的 query_capabilities/版本协商；公开契约 2.0.0 与包版本、
  device ABI13、Graph ABI1、writer ABI2 独立。
- 不可变 LayoutSpec/Capabilities/Requirements，结构化 admission 支持结果；
  明确候选数、有效预算、物理容量、tail、pool row、资源、heads 与 capture/dummy。
- 保持旧 validate_request 的兼容调用形式，新增能力不静默放过非法 host 输入。
- 固定 field access specification 和 SelectionResult，保持底层 tensor 字典和设备布局。
- 不在热路径读取 device count、查 env、编译、全局 mutable registry 或全量 schema 扫描。

### B. 生命周期

- 定义 QueueOwnerProtocol、显式 stream 和 session 状态机。
- prepare 绑定资源；enqueue 前置 in_flight，异常后保留异步可见资源；completion
  分开“排空”与“业务成功”；在途/未知完成拒绝 close，业务失败不再复用 session。
- Graph 使用原有 begin/end/guard 路径；不要让 capture 时的 Python 状态误约束 replay。
- 公开接口可由 standalone stub 实现，独立仓不 import vllm/vllm_ascend。
- 现有低层 ContractLaunch 保持兼容，新 session 作为更强的独立入口；框架仅做
  版本、admission、lease 和已绑定执行调用的薄适配。

### C. 校验补齐

- extra_prefetch_slots/counts 纳入可写字段；检查 cache 完整区间、与 caller tensor/
  workspace/其他 writable cache 的重叠，保留合法只读 alias。
- 修复 C0/B33 等公开 admission 漏检；所有 host 可知错误在 owned Index 重排前拒绝。
- 间接缓存和 owned key/scale 的修改标明访问模式；不声称 Python metadata validator
  已验证 device 动态值或真实 SDK 兼容。
- 明确页面更换的失效/恢复协议，补 same-length rewrite/页替换/epoch/身份边界测试。

### D. 独立桩和验证

- legacy FakeNpuRuntime 严格九参；CPU-only 验证有效 descriptor，按有效地址范围复制
  payload，保护 hit/tail；zero-task 模式不伪造正确性；拒绝在 NPU/capture 中 D2H。
- V9 owner/transport 测试覆盖延迟完成、跨 stream、重复 enqueue、提前释放、未知完成、
  release 异常、drained 非 OK、test binary 隔离。
- 既有 host suite、framework host、ABI结构/字段偏移、有效输入差分结果全部验证。
- 改动前后 `operators/` 及数值 oracle 的 SHA256 相同；新增协议不改变设备参数/调度。
- 在独立临时 Python 环境安装 CPU Torch（可用时），解除前轮 Torch skipped；
  wheel 在框架之外安装并测试，不以 PYTHONPATH 指向源仓代替安装验证。
- host 派发微基准只说明 host 开销，真实 NPU、KVAF、Graph、TPOT 标为未验证。

## 仓库与交付

- 独立 KVOP Git 仓保留子树基线历史及来源/许可；可构建 source wheel，独立运行测试。
- 主仓对应嵌入目录保持同源，提供薄 adapter 改动及测试；不向旧主工作区合并生产代码。
- 提交 prompt、未实现项处置表、公开规范、SDK/SoC 兼容与验证状态 manifest、README、
  可执行验证脚本、适配回滚与硬件验收命令。
- 对 G01..G11/C01..C18 给出 implemented、preserved、out-of-scope、hardware-pending
  状态；不要通过修改文档掩盖未完成要求。
- 不能确认硬件功能/性能时，明确禁止生产兼容声明，不部署、不自动切换开关。

## 退出条件

所有可在当前环境实现的公开契约逻辑、桩和回归完成；独立包 clean checkout 安装/测试
通过；设备源码与基线一致；框架适配通过 host 回归；仅外部硬件验收保留 pending，
并生成可执行验收清单。最后报告两个仓绝对路径、测试数量、diff/源码证据及未验证项。
