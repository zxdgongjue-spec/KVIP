# KVIP 仓库架构与融合算子执行流程分析

本文梳理 KVIP 代码仓的整体分层、生产 V9 融合链路、原生 LIDU 对照算子的数据流和
潜在性能瓶颈。分析基于当前 ABI 13 实现；真实 A5、KVAF、Graph 和模型端到端性能仍需
以硬件 profiler 结果为准。

## 1. 总体结论

本仓不是单一“大融合 Kernel”，而是包含两条可独立演进的执行链：

- **生产主链 KVIP V9**：将 LI 选择、中途预取、最终槽位规划、KVAF 补取、CQ 回收和
  resident 状态发布组织为 `init -> li -> resolve -> fetch` 四阶段设备流水。
- **原生对照链 native_lidu**：单算子完成 classify、compact、dedup、换槽、映射更新和
  BatchGet descriptor 生成，实际 KV 搬运仍由外部 KVAF 完成。

进一步优化不只等于扩大融合范围。减少重复扫描、降低 GM/UB 搬运、优化并行分工、隐藏
通信延迟、减少 eager 同步、提高预取质量和降低实际 miss 数，通常具有同等或更高价值。

## 2. 代码仓整体架构

```mermaid
flowchart TB
    FW[vLLM-DSA / 调度与 Attention] --> API[kvip.v9 公共 API]
    API --> ADM[Capabilities / Admission / Geometry]
    API --> SES[PreparedSession / GraphFrameSession]
    ADM --> CONTRACT[ABI13 Contract / TensorSpec / Workspace]
    SES --> LAUNCH[ContractLaunch]

    LAUNCH --> INIT[v9_init_kernel]
    LAUNCH --> LI[v9_li_kernel]
    LAUNCH --> RESOLVE[v9_resolve_kernel]
    LAUNCH --> FETCH[v9_fetch_kernel]

    INIT --> STATE[Resident H0 / Owned Index Cache]
    LI --> PRE[TopK + Window Prefetch]
    RESOLVE --> PLAN[最终 Resident Plan + Final GET]
    FETCH --> PUB[CQ Drain + Resident State Publish]

    PRE <--> KVAF[KVAF Device Queues]
    PLAN <--> KVAF
    PUB <--> KVAF
    PUB --> OUT[TopK / Destination / Miss / Extra / Status]
    OUT --> FW

    NATIVE[operators/native_lidu] --> BASE[Legacy LI -> LIDU -> KVAF 对照链]
    EXP[experiments/mtp_prepare] --> NATIVE
    TEST[tests/v9 + tools] --> API
    TEST --> LAUNCH
```

主要目录职责：

| 目录 | 职责 |
|---|---|
| `kvip/v9` | 公开契约、几何、缓冲绑定、生命周期、构建与 transport 适配 |
| `operators/v9` | V9 生产设备实现 |
| `operators/native_lidu` | 原生 `A5LiduFusedBatchGetPrepareMtp` 对照算子 |
| `experiments/mtp_prepare` | native LIDU 实验变体、oracle 和 simulator 性能数据 |
| `tests/v9` | 契约、生命周期、NPU-memory、KVAF、Graph 和基准测试 |
| `tools` | 符合性检查、构建和性能资格门禁 |

## 3. 整体执行时序

```mermaid
sequenceDiagram
    participant F as Framework
    participant A as Admission/API
    participant O as QueueOwner
    participant L as ContractLaunch
    participant I as init
    participant Q as LI
    participant R as resolve
    participant X as fetch
    participant K as KVAF
    participant D as Downstream/QSFA

    F->>A: 查询能力并校验 B/Q/容量/布局
    A-->>F: 支持或副作用前拒绝
    F->>O: 获取独占 Queue lease
    F->>L: 绑定固定 tensors/workspace/index cache
    L->>L: 校验 ABI13、offset、设备库和输出注册范围

    F->>L: launch()
    L->>I: v9_init
    I->>I: 清旧 reservation、动态参数预检
    I->>I: 校验 H0 / 映射 / Index cache
    I-->>L: status=PENDING 或 fail-closed

    L->>Q: v9_li
    Q->>Q: LI 打分、TopK、hot/cold 处理
    Q->>K: 按窗口提交 speculative GET
    K-->>Q: 异步 SQ/CQ 进度

    L->>R: v9_resolve
    R->>R: 跨 Q 去重和最终 resident plan
    R->>R: 分配 miss slot、生成最终 descriptor
    R->>K: 提交 final GET
    R->>R: 维护 owned Index cache

    L->>X: v9_fetch
    X->>K: drain prefetch/final CQ
    K-->>X: 完成数和错误状态
    X->>X: 成功后发布 forward/reverse/cursor
    X->>X: 发布 destination/miss/extra/status
    X-->>F: 设备输出就绪

    F->>L: completion()
    L->>L: stream synchronize + D2H 状态检查
    alt 完成可证明
        L->>O: mark_completed
        F->>D: 消费 resident KV、TopK 和反馈
    else pending/queue error
        L->>O: quarantine 并保留异步资源
    end
```

## 4. V9 融合算子内部处理时序

V9 的“融合”主要是跨计算与传输的流水融合，而不是把全部逻辑合并成一个 Kernel。

```mermaid
sequenceDiagram
    participant Init as Init AIV / request
    participant LI as LI AIC+AIV
    participant Service as Window Service AIV
    participant Final as Final Worker AIV
    participant Fetch as Fetch AIV
    participant GM as GM状态/Workspace
    participant Net as KVAF SQ/CQ

    Init->>GM: 清理上次 reservation dirty set
    Init->>GM: 初始化 status/output/workspace
    Init->>GM: 校验行元数据、victim bitmap、页表和目录
    Init->>GM: 构造或复用 H0 / hot bitmap
    Init->>GM: 对账 Owned Index cache

    LI->>GM: 读取 Q、Index key/scale、index table
    LI->>LI: Lightning Indexer 计算分数与 TopK
    LI->>GM: 写 TopK、scores、阈值和进度

    loop 每个 cold window
        Service->>GM: 扫描已完成的 LI 窗口
        Service->>GM: 聚合 cold candidate / 读取 forward map
        Service->>GM: 预留 eligible victim slot
        Service->>Net: 批量构建并提交 prefetch WQE
        Service->>GM: 写 reservation/mapping/progress
    end

    Final->>GM: DMA TopK 到 UB
    Final->>Final: 跨 4 Query 去重
    Final->>Final: hot/reserved 命中解析
    Final->>Final: victim/cursor 选择最终槽位
    Final->>GM: staged destination / plan / extra prefetch
    Final->>Net: 各 worker 提交 final GET
    Final->>GM: 增量维护或交换 Index cache

    Fetch->>Net: Drain prefetch CQ
    Fetch->>Net: Drain final-worker CQ
    Fetch->>GM: 核验 submitted/completed/pending
    alt 全部完成且无错误
        Fetch->>GM: PublishResidentState
        Fetch->>GM: forward/reverse/cursor 原子发布
        Fetch->>GM: destination/miss/extra/status=OK
    else 失败或完成未知
        Fetch->>GM: status=COMMUNICATION/INVALID/POISONED
    end
```

## 5. 数据流和依赖关系

```mermaid
flowchart LR
    Q[Query FP8] --> LI[LI score/TopK]
    W[Weights BF16] --> LI
    QS[Query scale] --> LI
    K[Index key FP8] --> LI
    KS[Index scale] --> LI
    IT[Index page table] --> LI

    FR[forward/reverse/cursor] --> H0[H0与hot bitmap]
    VIC[victims + victim bitmap] --> H0
    IC[Owned Index cache] --> H0
    EPOCH[mapping epoch] --> H0

    LI --> TOPK[逻辑 TopK]
    LI --> SCORE[scores/threshold]
    SCORE --> WIN[窗口候选扫描]
    H0 --> WIN

    WIN --> RSV[reserved tokens/slots]
    DIR[address + valid table] --> PWD[Prefetch WQE descriptors]
    RSV --> PWD
    RT[resident page table] --> PWD
    PWD --> PQ[Prefetch SQ/CQ]
    PQ --> KV[resident_output KV payload]

    TOPK --> PLAN[Final Resident Plan]
    H0 --> PLAN
    RSV --> PLAN
    VIC --> PLAN
    PLAN --> PT[planned tokens/slots]
    PLAN --> EXTRA[extra prefetch slots]
    PLAN --> DEST_STAGE[staged destination]

    PT --> FWD[Final WQE descriptors]
    DIR --> FWD
    RT --> FWD
    FWD --> FQ[Final SQ/CQ]
    FQ --> KV

    PQ --> GATE{CQ全部完成?}
    FQ --> GATE
    GATE -->|是| PUB[PublishResidentState]
    GATE -->|否| ERR[Sticky failure / quarantine]

    PUB --> FR2[新 forward/reverse/cursor]
    PUB --> DEST[destination]
    PUB --> MISS[miss counts/transfers]
    PUB --> STATUS[status=OK]
    PUB --> IC2[Index cache新状态]

    KV --> ATTN[QSFA/Attention消费]
    DEST --> ATTN
    TOPK --> ATTN
    FR2 -. 下一轮依赖 .-> H0
    IC2 -. 下一轮依赖 .-> H0
```

关键依赖包括：

1. TopK 依赖 Query、Index key/scale 和 Index page table。
2. 是否需要搬运同时依赖 TopK、调用开始时的 H0、forward/reverse 和 speculative reservation。
3. 最终槽位不能在 prefetch reservation 未知时独立确定。
4. 映射和 destination 不能在 CQ 成功前成为权威状态。
5. 下一轮 H0 依赖本轮发布的映射、cursor 和 owned Index cache，形成跨调用状态环。

## 6. native_lidu 单算子内部时序

```mermaid
sequenceDiagram
    participant Lane as Request Lane Group
    participant Leader as Leader Lane
    participant GM as GM Workspace/Maps
    participant UB as UB
    participant Out as 输出

    Lane->>UB: 加载 miss-slot bitmap
    par 每个 Query 分配到 lane
        Lane->>GM: 读取 TopK 和旧 forward map
        Lane->>GM: 写 classification[Q,2048]
    end
    Lane->>Lane: GlobalSync

    loop Q=2..4，由 leader 串行处理
        Leader->>GM: 读取 classification
        Leader->>UB: squeeze miss tokens
        Leader->>UB: squeeze hit tokens/slots
        Leader->>GM: 写 compacted tokens 与 hit suffix
        Leader->>UB: 原子 bitmap 去重 miss token
    end

    alt unique miss > 2048
        Leader->>UB: 合并 miss bitmap 与全部 hit slots
        Leader->>UB: 从 cursor 环形扫描额外可驱逐 slot
        Leader->>GM: 写 resolved miss slots
    end

    Lane->>GM: 校验 destination/resident block
    Lane->>Lane: GlobalSync
    Lane->>GM: Clear victim forward map
    Lane->>Lane: GlobalSync
    Lane->>GM: Install forward/reverse map
    Lane->>Lane: GlobalSync
    Lane->>GM: Refill 每个 Query 的 miss prefix
    Lane->>Out: 写 miss counts/resident length/cursor
    Lane->>Out: 全核并行生成 compact descriptors
```

## 7. 潜在瓶颈

下表是基于代码结构的瓶颈候选，不能替代真实硬件 profiler 排名。

| 瓶颈候选 | 原因 | 主要增长维度 |
|---|---|---|
| LI 全历史打分 | Q、head、candidate 计算及 Index key 读取 | candidate length |
| `init` H0/映射检查 | 扫 forward/reverse、bitmap、Index cache 和页表 | candidate/resident capacity |
| Final plan 重复工作 | 多个 final worker 各自读取完整 TopK 并构造一致 plan | final worker 数、Q |
| 网络与 CQ polling | Fetch 逐 CQE 检查，受 RTT、queue depth 和错误重试影响 | miss 数 |
| 全局发布门槛 | final payload 完成前不能发布权威 map | final miss 数 |
| Workspace 中间落盘 | TopK、scores、reservation、plan、descriptor 多次 GM/UB 往返 | B、Q、miss 数 |
| eager completion | stream synchronize 与多份 D2H 状态读取阻止跨层重叠 | 层数、调用频率 |
| 固定 Q4/等长 | 不规则请求需要 padding 或分 bucket | workload 不规则度 |
| native LIDU leader 串行段 | compact、dedup、slot selection 集中在 lane 0 | miss 数 |
| native LIDU 全局屏障 | classify、prepare、clear、install、refill 之间均有 barrier | batch/波次数 |

已有 simulator 数据显示：

- classify/compact 融合与 output/descriptor overlap 在低 miss case 改善约
  `13.96%~15.57%`；
- multi-lane dedup 在 M2048 改善约 `11.84%`；
- row-control prefetch 在 M1280~M2047 改善约 `11.39%~12.95%`。

上述数据不包含真实网络、CQ、Graph 或模型端到端延迟，实验候选也尚未进入生产路径。

## 8. 优化方向判断

优化不应只围绕“扩大融合”展开，建议分为六条路线：

1. **减少工作量**：改善 pre-victim/预取命中率、降低 logical miss、增量维护 H0 和
   Index cache。
2. **减少重复计算**：重点评估 final workers 重复构建完整 resident plan，研究一次规划、
   多 worker 消费；但必须先解决跨 AIV 调度和完成保证。
3. **计算与传输重叠**：细化 LI window、WQE build、SQ post 和下一窗口打分的流水。
4. **内存路径优化**：减少中间结果写 GM、复用 UB、按 cache line 分区并降低随机依赖链。
5. **传输与队列优化**：调整窗口、prefetch limit、WQE bank、queue depth、final worker 数，
   合并 doorbell 并让 CQ drain 与其他计算重叠。
6. **Host/Graph 优化**：减少 eager 每次 completion 的同步和 D2H，优先验证 Graph frame、
   批量完成、lease 复用和跨层异步。

推荐验证顺序：

```text
真实 A5 profiler 建立分阶段基线
-> 分离 LI / init / resolve / CQ / publish / host-sync
-> 先减少扫描和重复规划
-> 再调窗口、队列与计算通信重叠
-> 最后评估是否合并 Kernel launch
```

`resolve` 与 `fetch` 的阶段分离本身有诊断和流水价值，可以分别测量 final submit 与 CQ
等待。直接合并虽可能减少 launch 开销，但也可能增加 UB 压力、延长 Kernel 驻留，并破坏
通信重叠和失败隔离边界。
