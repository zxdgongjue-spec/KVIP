# KVOP 2.0 公开接口契约

包版本 0.2.0；公开契约 2.0.0；设备 ABI13；Graph transport ABI1；Index writer ABI2。
来源为 vLLM-DSA `fae4597f7` 的 KVOP 子树，独立 Git 基线 `2f49ba0`。

## 支持范围与发现

```python
from kvop.v9 import ContractRequirements, negotiate_contract, query_capabilities

capabilities = query_capabilities()  # 不 import torch/vllm/torch_npu/kvaf
negotiate_contract(ContractRequirements(features=frozenset({"owned-index", "fixed-q4-graph"})))
```

公开 profile 为 `v9-fused-selection`；真实执行支持 B1..32、固定 Q4、32/64 model
heads、同批等 candidate/logical length 和 budget。Q1..Q3、mixed Q、变长不在支持集。
硬件资源、独占页、pending restore、真实行与 Graph bucket 等框架条件继续单独检查。
`hardware_certified=False` 表示设备与服务验收未完成；要求已验收能力的 consumer
可通过 `ContractRequirements(require_hardware_certified=True)` fail-closed。

LayoutSpec 明确 656 bytes/token、128 tokens/block、TopK2048、Index dim128、kernel
heads64。公开/包/device/transport/writer 版本分别协商；公开握手不能替代已编译库的
ABI/size/offset/geometry 检查，也不能推断任意 KVAF runtime 与头文件兼容。

`python -m kvop.v9.contract_info` 输出机器可读的完整能力与字段访问规则。
`Geometry.tensor_spec/workspace_spec` 是当前 shape/dtype 机器规范；
`tools/check_contract.py` 对实际 C++ header 编译并验证 Python 的全部字段偏移。

## Admission 与准备

```python
from kvop.v9 import admit_request, validate_request

result = admit_request(
    queries_per_request=4, candidate_lengths=[16384], resident_budgets=[12288],
    logical_lengths=[16513], pool_rows=[0], geometry=geometry, aic_count=28,
)
if not result.supported:
    raise ValueError(result.message)
```

上例的 geometry 由 caller 按实际 tensor 池构造。validate_request 保留原有三参数的
raising 形式并接受同样扩展参数；只传三项是结构校验，不能证明完整 runtime 支持。
全部为现有 CPU metadata，不将设备 count 做 `.item()`/D2H。C0/B33、非整数/bool、
非等长、越容量、tail 超页表、pool 行重复和资源不足在可知时先拒绝。

固定 tensor/output/workspace 和 owned cache 在 prepare 校验同设备、连续、shape、
dtype、直接/间接字节区间与读写冲突。extra outputs 是可写；owned key/scale 是可写；
index_dirty 是只读。只读之间允许 alias，任何 writer 的重叠拒绝，首尾相邻允许。
device init 仍校验真实长度、victim bitmap/list、页表、目录和注册地址范围。

## 标识和结果

batch row、pool row、request identity/generation、layer、token、resident slot、physical
token row、mapping epoch 分开定义。目的物理行由 resident_table 翻译，不能假定连续。

SelectionResult 只绑定八个 caller-owned 输出，不复制内容：topk、destination、
miss_counts、resident_lengths、transfers、status、extra_prefetch_slots/counts。
destination 是每 query 的 miss-first slot，TopK 不保证同下标与其一一配对。
MTP miss_counts 是请求内 unique logical miss，不能用它切每个 query 的前缀。
extra 的 -1 空洞合法，count 是全行非负项数量。transfers 为预取/最终补取/合计/丢弃
的提交口径；未被选中但保留的 extra 不算丢弃，也不能用 transfer 替代 logical miss。

V9 H0 继续包含正反映射一致的 victim 候选，阈值相等也保护；与 legacy bitmap-exclusion
规则不同。此版本不改变算法；接口隔离不是把两种 miss/LRU 规则强行做成相同。

## 独立执行生命周期

低层 `ContractLaunch` 原接口保持兼容，默认仍提交 init→LI→resolve→fetch 四阶段。
新 consumer 应在取得已校验且 idle 的 lease、构造 launcher 后建立 PreparedSession：

```python
from kvop.v9 import PreparedSession

session = PreparedSession(launcher, stream=acquired_stream)
session.enqueue()
completion = session.complete()
if not completion.success:
    session.close()  # 只有 drained 且无未知 DMA 时才能释放
    raise RuntimeError(completion.status)
selection = completion.selection
session.close()
```

owner 实现 QueueOwnerProtocol，负责真实 release 和 quarantine；session 负责在途状态，
launch/complete 的 acquired stream 一致性、重复提交/提前 close 拒绝和资源保留。
PreparedSession 不增加 kernel、拷贝或同步；调用已有 launcher.completion 的同步检查。
未知完成或 release 失败为 QUARANTINED，保留全部异步可见对象并要求 owner 隔离。
已排空但 status 非 OK 为 RETIRED，不能复用，但可安全 release。Drained 不等于成功。

旧 benchmark owner 的 acquire 可能保守设置 in_flight=True，继续用原低层协议；不能
未经 idle 状态转换就包入新 session。Graph 使用 frame begin/end、device guard 与
finish 边界，不捕获 Python eager session 状态机。正常 Graph 路径和二进制保持原实现。

独立 consumer 可使用 `GraphFrameSession(runtime, stream, synchronize=...,
check_completed=..., quarantine=..., keepalive=...)` 管理同样的帧级 begin/complete/abort。
同步和检查回调在 capture/replay 外执行；该对象不释放 runtime 持有的 layer 注册。
现有框架 Graph boundary 保持原路径，避免在未完成硬件验收前更换 replay 编排。

## Index 页面交接

原 grow/write/restore 行为保留。更换已有 owned 页表应在 checked completion 后调用
`rebind_pages(table, completion_verified=True, exclusive_pages=True, logical_length=...)`，
提供包括 tail 的长度；该边界操作在覆写前采集源 cold/tail，保留 hot/ID，并失效 H0 认证。
它会在 CPU 校验页号，只适用于图外所有权交接，不在 steady enqueue/Graph 内调用。
调用方仍须证明独占、先后顺序及请求身份；未知页变更不得用相同 data_ptr 继续认证。
mapping epoch=0 精确检查，未知 writer/溢出不能错误复用正 epoch。

## 桩、兼容与性能

`kvop.testing.transport.FakeNpuRuntime` 为传统 adapter 的 CPU 九参消费桩，显式注册
测试块、校验全局 descriptor 前缀并只复制有效目的行，保持 hit/tail；capture/NPU
不允许进入该桩。zero_task 只证明编排，不能证明 payload 正确。`StubQueueOwner` 与
DeferredTransport 用于独立 V9 session 的延迟完成/异常测试，不作为生产 provider。
独立 NPU memory 桩仍与真实库用 transport 标记强隔离。

源码/Oracle 一致及 host A/B 不代表真实功能/性能无影响。强 session CPU 派发有可测
成本，eager 分配/同步仍存在。本仓不改变设备 kernel、默认窗口、packing、几何或
完成顺序；真实 NPU/KVAF/Graph/模型性能通过前不得合入生产适配或宣称零回归。
SDK、SoC 和资格状态见 `manifests/compatibility.json`。
