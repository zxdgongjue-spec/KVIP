# Score history / batch-query 分核

生产调用保持 `LIFetchBinding` → `DirectIndexUpdate` → `LIFetchForward`。旧融合入口仍为默认 `score_route="legacy"`；URMA显式传入 `score_route="auto"`、`"history"` 或 `"bsq"` 后，公共Forward自动使用完整staged流程。无需改调用方为另一Forward类。UBmem继续使用原融合入口。

## 当前分核

每个逻辑任务为 `(request_row, query_begin, Hbegin, Hend)`；一AIC配两AIV，物理容量 `Ceff=min(C,floor(V/2))`，每wave至多Ceff个producer。任务超过容量时依次启动wave，不把逻辑任务数当物理核数。

- auto：`2B<=Ceff` 时Q4/history；每个request的P为 `min(floor(Ceff/B),rangeTiles)`。
- 其余auto场景以及bsq：Q2，每request两组query扫描完整当前范围，通过wave覆盖。
- history：强制Q4/history；不足两片时退化P1并继续分wave。

Hot与每个Cold窗口单独按128-token tile切片；每wave在设备刷新live FP8 Q、BF16权重、Qscale、Key/scale指针和页表。各task独占score片段，设备scatter生成完整 `[B,4,H]`，forward期间无score读回或host重排。

C28/V56、H128K/Q4的初始auto策略：

| B | score路线 | P（还受Hot/Cold tile数裁剪） | 最大score waves |
|---:|---|---:|---:|
|1|Q4/history|28|1|
|2|Q4/history|14|1|
|4|Q4/history|7|1|
|8|Q4/history|3|1|
|16|Q2/BSQ|1|2|
|32|Q2/BSQ|1|3|

这是核预算启发式，尚不是实测性能交叉点。resident θ与Final TopK独立采用T1完整BQ扫描，没有接入独立TopK的T4/T8 local+merge。

## 与旧P2及独立实验的区别

旧默认P2需要splitQ资源条件、G32、每request实际query数量4且Key长度H、512-token对齐与最后tile排空条件。两个AIC/request各算完整Q4，分别处理resident以及每个Cold窗口的一半。否则一般query/tile分核。`splitQ=1`本身不代表启用了history P2；各Q的queryEnds仍可以动态裁剪。

`score_schedule.h`和`experiments/split_core/score_*`是当前算子的独立score-only验证adapter，按256-token对齐、使用host重建canonical，不执行完整Service/TopK链路。生产staged按128-token tile切片并在设备完成scatter。历史双版本回放和独立TopK工具已移出当前MR，保留在Git历史；TopK的T与score的P、tail Fetch的W分别控制不同阶段。

完整接入顺序、内存预留和验证边界见 [staged_integration.md](staged_integration.md)。
