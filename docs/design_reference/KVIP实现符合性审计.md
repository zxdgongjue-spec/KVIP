# KVIP 融合算子实现符合性审计与补齐清单

审计日期：2026-10-31。实现基线：`fae4597f7`；契约：
[融合算子接口契约 2.0.0-draft.1](融合算子接口契约.md)。
远端重新 fetch 后未发现该 KVIP 分支的新提交。

## 1. 判断

KVIP **部分符合**本版契约。它已经形成独立 Python 包、设备 ABI13、计算/取数/发布
链路、测试传输隔离和框架生命周期接线；尚未完整实现公开能力协商、统一 host
admission、全面别名校验与可独立约束的 owner/session 协议。不能将它描述为只有
目录搬迁，也不能标记为完整契约已生效。

以下“已符合”表示本次源码和可执行 host 证据覆盖对应机制，不等于真实 NPU/网络
验收。“部分符合”指机制存在但公开表达、校验或测试不完整；“未实现”对应本版目标
要求；“范围外”不应成为要求 V9 补齐全部传统算子的理由。

## 2. 条款追踪矩阵

| 条款 | 判断 | 实现证据 | 待补内容 |
|---|---|---|---|
| C01 profile 和职责 | 部分符合 | `a5_c8_policy.py` 用 v9_result 分支跳过 native prepare/BatchGet；KVIP 独立 import | 正式声明能力组；不要求 V9 全量绑定 legacy/QSFA |
| C02 版本/发现 | 部分符合 | `ContractLaunch.__init__` 查 ABI=13、结构大小、全部 offset、编译几何 | 包公开 SemVer/capability/layout/SDK manifest，启动协商 G01 |
| C03 布局常量 | 部分符合 | `contract.py` 656/128；QueueOwner/GraphQueueOwner 校验 lease geometry | 机器 LayoutSpec 和多布局协商，消除散落重复数值 |
| C04 标识/地址 | 已符合机制 | `contract_access.inc`、`InitResidentState`、`ValidateContractRemote`；resident 全宽表 | NPU 非连续页和真实远端范围验收 |
| C05 admission | 部分符合 | 框架 route、Geometry、device preflight 分层存在 | 公开 helper 覆盖不完整，G02 |
| C06 访问/别名 | 部分符合 | validate_buffers 有同设备、dtype、shape、连续与重叠检查 | 两类可复现漏检 G03；间接可写域机器规范 |
| C07 H0/victim | V9 profile 符合；与旧契约不同 | `frozen_hot`、`eligible_victims`、`InitResidentState` | 明确产品是否要求 legacy miss/LRU 等价，G05 |
| C08 输出/反馈 | 已符合机制 | `StageResidentDestination`、`PublishResidentState`；`record_mtp_layer/commit_lidu_mixed` | typed SelectionResult 和字段语义文档；禁止按 unique miss 切 query 前缀 |
| C09 状态/失败 | 已符合机制 | Status、sticky、completion、Graph guard、KVAF shutdown quarantine | 行错误与 DMA 完成分别报告、设备故障验证 |
| C10 执行顺序 | 已符合机制 | execute 在 launch 前 mark maps/in_flight；fetch 核验后发布 | 独立 session 强化与真实设备顺序证明 |
| C11 owner/transport | 部分符合 | 框架 owner/测试 owner 各自实现；注册范围/queue 结构 packer | 正式协议、stream/in_flight 的独立 API 约束 G04 |
| C12 Index/epoch | 已符合主要机制 | OwnedIndexCache、framework writer、MappingVersions、restore/grow | 页面替换等边界专项覆盖 G06 |
| C13 Graph | 部分符合，设备待验 | 预构建、固定 bucket、rebind、dummy、guard、begin/end | 本机 Graph host 模块无 Torch 跳过；真实 replay 验收 G09 |
| C14 传统 descriptor | 集成存在；V9 内部不适用 | `ops_nanovllm`、prepare `_out`、`KvafAdapter.execute_batch_get` | 独立 legacy consumer fake/signature 测试 G07 |
| C15 attention | 框架范围，非 V9 包缺陷 | metadata 只写 slots，QSFA C8 八参 wrapper | timing context 仍是旧行为；单独抽取时再改 G08 |
| C16 分层测试 | 部分符合 | host oracle/transport、memory 库隔离测试、NPU/真实 KVAF CLI | 真实设备与服务级验收 G09 |
| C17 性能 | 部分符合 | build fingerprint/预构建；Graph 使用固定目标地址 | eager 分配/同步及 packing 分配仍在，G10 |
| C18 版本演进 | 未形成完整公开机制 | ABI13 和发行版本分开存在 | SemVer 策略、machine spec/document CI、release/rollback G11 |

## 3. 整个流程逐阶段核对

| 阶段 | 入口/关键符号 | 当前执行与风险边界 |
|---|---|---|
| 1. 启动与 import | `v9_backend.load_binding` | 从安装的 `kvip.v9` 加载；无 in-tree 自动 fallback。缺公开契约握手，但后续有二进制 ABI 校验 |
| 2. 资源与路由 | `required_workers`、`route`、`v9_config.admission` | 获取 AIC、规划双 jetty 队列；限制 Q4、等长、资源、共享页和 graph 行数 |
| 3. Index 写入与交接 | `_write_index`、`v9_index_writer.write_q4` | native 路径前恢复逻辑布局；已有 owned cache 时 writer 按 bank 写入；失败 retire |
| 4. pending restore | `V9Backend.execute` 开头 | 融合写 resident 前完成恢复；policy 后面的同名调用依赖 pending 标记使其无重复搬运 |
| 5. 绑定输入/身份 | execute 的 Geometry、index_caches | 按 pool/layer 和 request identity/generation 复用；32 头 pack；真实页表宽度保留 |
| 6. 构建与 workspace | `library`、`build_library` | 源/SDK/config 指纹和锁；Graph prebuild 在 watchdog 前；eager 仍可按首次几何构建 |
| 7. acquire 和 launcher | QueueOwner.acquire、ContractLaunch | lease 核验、pack args、ABI/offset/几何/queue view 校验；keepalive 持有 launcher |
| 8. init | `InitResidentState/ValidateContractRemote` | 动态检查等长、pool 唯一、victim/bitmap、目录范围、page、tail；重置 transient；sticky 不可普通 reset 清除 |
| 9. LI 与预取 | `kernel.cpp`、`contract_scores.h`、`window_fetch.h` | 热冷扫描、阈值保护、中途预取；reservation 不是完成证明 |
| 10. final resolve | `ContractFinal/ResolveContract` | 计划 cold miss、去重、最终补取 descriptor、staged destination、extra 和 Index 维护；此时 map 尚不能当成功结果 |
| 11. fetch 和发布 | `v9_fetch_kernel/PublishResidentState` | 汇合 prefetch/final CQ；失败 sticky，成功才安装 forward/reverse/cursor 并发布 destination/count/status |
| 12. eager 完成 | `ContractLaunch.completion` | 同步后检查 pending/queue error；未知完成 quarantine；mark_completed 与 status=OK 分别判断 |
| 13. framework handoff | `_execute_packed_mtp_prepare` | 复制 destination/miss/length，记录 extra；v9_result 存在则跳过传统 prepare 和 execute_batch_get |
| 14. metadata/QSFA/LRU | `ops_nanovllm`、`record_mtp_layer/commit_lidu_mixed` | 拼 tail、消费选中 KV；extra 全行滤 -1、较旧 LRU 反馈；不能把 miss count 当选中槽数量 |
| 15. close、fallback、释放 | owner.close、invalidate_layer、release_pool_row、shutdown | 已完成才释放队列，native 接手先 restore，失效 owner 保留到退出；请求释放与活动请求回退是不同操作 |

### Graph 分支

`prebuild_libraries→prepare_capture→GraphLayer` 预备库、注册、placeholder 和固定绑定。
真实调用在 `V9GraphBackend.begin` 先准备各层请求/Index/generation/epoch，再
begin_fused_graph。device writer、launch、guard 被捕获；模型调用结束走 finish
读取 checks 并 end_fused_graph，异常 quarantine。

`GraphLayer.execute` 中 Python 标记位于 launch 后，但 `PreVictimRuntime.begin` 对
GRAPH_REPLAY 提前设置 `_maps_may_have_changed=True`。结合此处才能判断真实 replay
失败退场，不能截取单个函数次序就报告“Graph 未提前标记 map 变化”的缺陷。

Graph guard 是危险索引防护及错误锁存，不是 rollback。Index 维护可早于 CQ；同批
不同请求也不是一个原子事务。frame 失败后必须拒绝模型结果，保留不能安全释放的资源。

## 4. 需要补齐的差异点

### G01 公开版本与能力握手缺失 P0

证据：`kvip/v9/__init__.py` 导出 Geometry/ContractLaunch 等，无公开 contract version
或 query_capabilities；framework load_binding 直接 import。已有 ContractLaunch
ABI13/size/offset/几何检查应保留，不能用仅包 SemVer 检查替代。

补齐：共同定义 manifest、profile/LayoutSpec/transport ABI；worker 启动匹配所需能力，
prepare 验证具体库。验收缺包、错 contract major、缺 Graph、错 layout、SDK 不兼容
与测试库误加载均在设备持久状态变更前失败；只需新增可选能力时不强制全链 MAJOR。

### G02 公开 admission 不足且存在多处规则 P0

复现：真实 `validate_request` 接受 candidate_lengths=[0] 以及 33 行的等长列表；
两者违反设备要求。helper 未接收 logical_lens、Geometry、page policy 和资源，无法
证明完整支持。框架 `v9_config.admission` 也不是调用该 helper 的单一入口。

设备 preflight 会对相应非法值拒绝，因此这不是证实“非法输入必然被 kernel 正常执行”。
问题是公开 admission 返回通过后，owned cache import/grow 和 lease 准备可能已发生。

补齐：统一机器限制与拒绝原因；区分 structural admission、runtime value validation、
框架业务 eligibility；已知非法 host 值先拒绝，不将所有动态检查搬成 CPU `.item()`。
验收 B0/B33、C0、非整数、Q1..3、非等长 C/L/S、尾部超容量、pool 重复和资源不足。

### G03 别名校验存在可复现漏项 P0

1. `contract.py::validate_buffers` 的 writable 集合不含设备会写的
   extra_prefetch_slots/counts；将任一个地址改为 budgets 地址，shape/dtype 保持
   元数据要求，真实校验函数未拒绝。
2. `index_cache.py::bind_caches` 只检查起始指针不同，未检查完整区间。两个均满足
   长度/128 对齐要求、起始相差 128 字节的 cache 被接受，实际上区间重叠。

这两点已用 metadata facade 调用真实校验函数复现，不需要真实 Torch/NPU。它证明
host guard 不完整，不证明框架正常分配已经产生这种 alias，也不表示真实设备已损坏。

补齐：统一 read/write access table；extra 输出纳入 writable；逐个检查 cache 完整
区间及与 caller tensor/workspace 的冲突。key/scale owned 可写域与间接指针指向空间
也入访问规范；index_dirty 是只读，不应被误列为遗漏写字段。验收 overlap/adjacent/
read-only alias/不同 layer session 复用等边界。

### G04 Owner 协议依赖调用方约定 P0

框架明确在 launch 前置 owner.in_flight，完成后 close；独立包 ContractLaunch 继承的
launch 不自行维护这一状态。`_call` 与 `completion` 使用当时 current_stream，未将
prepare/acquire 的 stream 固定为公开句柄并验证一致性。框架正常调用使用同一流，
但独立调用方切换流或漏置 in_flight 时缺统一保护。

补齐：公开 QueueOwnerProtocol 与 session 生命周期，规定谁负责 in_flight；将 stream
显式绑定至 session，或要求调用参数并校验；unknown completion 必须保留库、workspace、
cache、resident、queue。增加跨 stream 拒绝、重复 enqueue、在途 close、drained 业务失败、
release 异常等 public API 测试。不是要求 KVIP 接收 framework runtime 私有对象。

### G05 H0 与 legacy miss 规则不同 语义决策 P0

V9 `frozen_hot` 和 InitResidentState 以正反映射一致为 hot，包含 victim 候选；
legacy prepare 把 victim bitmap 中 slot 排除。V9 threshold 又可能保护候选 victim。
因此 miss 数、槽位分配与 LRU 可以不同，即使选中 token 的 attention payload 正确。

本版契约明确保留两种 profile。若要求严格 legacy 等价，必须先确定 H0/victim 的
产品语义，修改算法/反馈/oracle 的同一规范；若允许 V9 扩展，增加专门 golden case
证明 victim hot、阈值相等、跨 query unique miss 与 extra 顺序。ties 允许的 TopK
选择差异按数值/排序规范判断，不只比较某一次输出 token 顺序。

### G06 Owned Index 页面变更证据不足 P1

已有 request identity/generation、源 storage、grow、restore 和 epoch 机制。仍需
专测相同地址/相同长度但页表替换、pool 行重用、外部 alias、同长度源改写和 stream
交接；不能仅用 storage.data_ptr 不变作为拥有相同物理页的证明。
本次未认定页面替换必然错误；应把允许变更、失效/迁移责任写入协议并增加边界测试。

### G07 桩覆盖范围与原稿不一致 P1

已有 HostTransport 和独立 owner、memory transport，尚无公开的、覆盖所有传统
adapter 调用的 FakeNpuRuntime contract。HostTransport.submit/complete 不是 NpuRuntime
九参签名。测试层面的两种接口应分别验收，避免把 legacy fake 作为 V9 网络契约证明。
zero-task 只做编排；正确性桩按有效 descriptor 复制 payload 并保护 hit/tail。

### G08 传统 wrapper 抽取与 timing 尚未实现 P1

原稿提出的 `nanovllm_dsa_a5.contract`、全量函数表、显式 capturing 尚未接入最新
framework；C8 LIDU 仍读 forward_context 并在非捕获时同步。KVIP 只抽取 V9 和
native MTP 对照，QSFA/其他 legacy 留在框架不是 V9 违反职责边界。
若继续抽取，按能力组和实例函数集合接入；同步修改单独 profile，禁止全局 timing setter。

### G09 设备、Graph 与真实网络验收未闭环 P0 发布门槛

本机 KVIP host suite 为 40 passed/2 skipped；framework 相关 host suite 为
49 passed/5 skipped，其中 Graph host 模块整体因 torch 缺失跳过。这是 pytest
计数口径，不能解释为“仅五个 Graph 分支没执行、其他 Graph 分支均通过”。

仓内 VALIDATION 记录其他环境构建与测试；本次不将其当作实际 NPU 运行。
补齐单 NPU memory、真实 KVAF eager/Graph、未知 CQ、跨 rank 可见性、模型正确性和
worker 故障退出验收，附源/SDK/library/SoC fingerprint 和运行日志。

### G10 当前实现不满足零额外成本的原稿说法 P1

eager execute 创建 row_modes/profile/index_table、输出和 Index pointer/dirty tensor，
每层 acquire/completion 同步；32 头 packing 分配并复制。Graph execute 的 packing
和 guard 组合也不等于零临时 tensor，只是被捕获的目标绑定保持稳定。
优化方向为 session 复用、预分配 pack buffer 和按全 frame 的完成策略；必须保持
错误门控与原有同步保证，分别测 setup、enqueue、completion 和端到端 TPOT。

### G11 发布/事实源尚未统一 P1

KVIP 是可独立 wheel 的普通目录，未配置独立远端；旧 csrc V9 和上一轮本地
ascend-dsa-fused-ops 不是生产第二入口。以 KVIP 为主线，定义 tag、兼容矩阵、构建
manifest 与不可变 release；历史实现只读或生成镜像。先形成机器规范再生成契约文档，
避免文档、Python、C++ 结构三处手改；保留现有 schema/offset 运行检查。

## 5. 修订后原稿差异

- 原稿的单一全量 L2 接口改为 legacy 与 V9 capability profile。
- 补上 V9 输入/输出表、实际 H0、extra 的非紧凑布局、transfers 的提交口径和 dropped 定义。
- 分开包/public contract/device ABI/graph ABI/writer ABI；公开版本仍是待实现目标。
- 区分 map 发布、DMA 已排空、业务成功、资源可释放；禁止保证完整回滚。
- 删除全局 timing setter、零成本结论和所有校验都只在启动期完成的描述。
- 原稿完整保存在原稿目录，不通过删减历史约定掩盖 ABI13 与 legacy 的语义差异。

## 6. 本轮验证与复现

```bash
PYTHONPATH=third_party/KVIP pytest -q third_party/KVIP/tests/v9
# 40 passed, 2 skipped

PYTHONPATH=third_party/KVIP pytest -q --confcutdir=tests/ut/dsa_offload \
  tests/ut/dsa_offload/test_v9_framework_host.py \
  tests/ut/dsa_offload/test_v9_framework_abi13.py \
  tests/ut/dsa_offload/test_v9_graph_host.py \
  tests/ut/dsa_offload/test_v9_qsfa_loader.py
# 49 passed, 5 skipped

probe=docs/source/developer_guide/Design_Documents/wangyong/contracts_20261031/tools/probe_kvip_contract.py
PYTHONPATH=third_party/KVIP python3 "$probe"
```

最后一个脚本调用真实 metadata validator，控制组通过，五项 gap flag 都为 true：
extra counts alias、extra slots alias、Index cache 区间重叠、零候选长度和 B33。
脚本不分配设备缓冲，不执行设备 kernel，不修改生产逻辑；未来补齐后对应 flag 应为 false。

完整 `bash format.sh ci` 在本机缺少 pre-commit；文档链接/镜像一致性、探针 Ruff 和
修改文件空白检查单独进行，不记为全仓格式检查通过。

## 7. 推荐实施顺序

先补 G03 与 G02 的可复现校验缺口，并用真实公开 API 回归；接着完成 G01/G04
版本、能力和 session/owner 协议；对 G05 作语义决定并固化 golden fixture；再做
G06/G07/G08/G11 边界测试和发布；通过 G09 后才发布生产兼容声明。G10 性能优化
独立测量和验收，避免把弱化完成检查当成性能收益。

本轮仅修订文档、保存可复现探针并完成审计，以上生产补齐任务尚未实施。
