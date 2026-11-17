# 当前LIFetch通路的显式score分核

默认融合入口保留。`LIFetchBinding(score_route="auto")` 与原公共 `LIFetchForward(binding, update)` 即可启用当前URMA完整staged入口；`history`/`bsq`可强制对照，`legacy`回到原融合方案。旧 `staged=True` + `StagedLIFetchForward` 仍可用作兼容入口，但不再是唯一调用方式。新路径支持逻辑B≤32，旧默认完整入口仍B≤16。

## 执行顺序与协议

Hot FP8 QK score waves → 设备scatter到完整BQH → 完整resident θ及LRU保护predicate → Hot progress/Service → 每Cold epoch的全部score waves/scatter → 完整θ下cold predicate → 连续range progress/Service → 完整H Final TopK及classification → whole-owner Resolve/Output/Index/Fetch/completion。

每Q的queryEnds动态裁剪θ、Cold predicate和Final。任意一个history片的局部θ不能替代完整resident θ。Final沿用原global token坐标、cutoff tie、locations、query保护bitmap和错误输出。分类kernel按同stream完成后才启动tail。

Resolve保留同owner四Q保护bitmap并集和crossQ dedup，统一分配唯一slot。Output先消费旧映射并发布descriptor，Index按原snapshot/exchange/commit协议执行；Fetch保留safe/conflict phases、prefetch drain和metadata/transport completion双join。submit不等于payload完成。Context320B、RuntimeShape80B、Index update28字段保持原样；sidecar承载分核任务。

独立物理预算 `Ceff=min(C,floor(V/2))`，score每wave≤Ceff；θ/Final每wave≤V。tail owner容量为 `min(16,floor((V-4W)/3))`，每wave驻留 `3n+4W` AIV。Fetch与completion只等待当前owner范围，上一wave完成后才清packed counters并启动下一wave。

CANN legacy CMake按整个编译文件的默认类型生成kernel元数据。纯AIV入口在独立 `staged_vector.cpp` 中编译，score留在MIX文件，避免把θ/Final等错误打包成MIX 1:2并超出AIC预算。`verify_staged_metadata.py --auto-gen <build>/auto_gen/<library>`核对实际生成配置；当前检查还覆盖prepare_all、scatter_predicate、service_ready及score_direct。CPU负例验证不替代本版本实际CANN生成配置验收。

## 关键通路候选：独立分片与ready转交

`score_post="direct"`让Cold producer在最终key reduction后输出真实owner/Q的canonical与predicate。新增`score_hot_partitions`、`score_cold_partitions`，0保留自动策略；显式分片数限定在实际MIX组预算和128-key tile数量内，要求Q4/history（或auto选Q4）。多个owner允许跨wave，但不能提前跨过未完成的epoch。共享workspace预留与实际task布局使用同一分片参数。

`score_prepare="phase"`分别合并Hot与Cold prepare，顺序为Hot prepare/score/scatter→完整resident TopK/Hot predicate→Hot Service→Cold prepare→各Cold epoch。与`invocation`全量prepare在前的控制分开，避免首个Hot服务机会被Cold准备阻塞。所有prepare仍在每次调用读取实时Q、weight、scale和pages，没有冻结模型输入。

`score_service="inline"`要求scatter/direct predicate；Service kernel在自己的lane内为本owner四Q发布ready并clean cache，再调用原Service。跨kernel同stream依赖保证bitmap完整可见；Service lane不读取兄弟lane的ready，不新增跨lane屏障。它消除独立progress启动，仍然等待整个epoch，没有实现段级乱序消费、跨owner Final→tail流水或classification私有scratch。

私有Plan仍64B，末字段承载serviceMode；其版本升为3，Python拒绝旧bridge。公开Context/RuntimeShape ABI11不变。resident/Final TopK的history分片保持1，Fetch宽度单独取1/2/4，默认legacy未切换。vLLM配置可显式传入`lifetch_score_hot_partitions`、`lifetch_score_cold_partitions`、`lifetch_score_prepare`与`lifetch_score_service`，必须在workspace预留前设置。

本轮CPU/source仿真结果与A5执行命令见[critical_path_candidate.md](critical_path_candidate.md)和[Cold对比脚本](../../../tests/operators/lifetch/cold_parallel/README.md)。本版本尚无A5真机延迟或真实URMA新增路径验收，不能宣称相对库上P2已提速。

## 调用与startup

```python
binding = LIFetchBinding(..., score_route="auto", aic_count=28,
    aiv_budget=56, intermediate_fetches=32)
update = DirectIndexUpdate(..., rotating=binding)
forward = LIFetchForward(binding, update, aiv_budget=56)
forward()  # graph稳定；只执行已准备的launch
```

必须分配完整Q4/H页表；queryEnds仍允许非完整有效长度。V≥B，W取1/2/4，URMA队列数覆盖 `2B+8W`。本轮活跃decode rows须使用互不重复的stable cache owner；vLLM selector在每次prepare中校验，忽略空/padded行。构造时不冻结owner映射：owner及页表在每round设备prepare中使用实时值。直接调用binding的调用方应遵守同一active-owner约束。

所有device ABI指针以int64存储原uint64位模式上传，避免NPU的uint64 tensor构造限制。共享score、Q/W/scale/pages scratch在串行层组间复用，每组保留独立live bindings。

vLLM URMA binding读取显式 `config.lifetch_score_route`（缺省legacy）。启用后重绑共享events为 `(windows,2,128)`，并预留所有数值score缓冲。若上游在绑定之前做HBM profiling，应在原rotation workspace分配后、profiling前调用：

```python
workspace.reserve_split_score(torch, resident=R, aics=C, aivs=V,
    windows=windows, route="auto")
reserved_bytes = workspace.nbytes
```

`nbytes`统计共享streaming及score缓冲。绑定Forward复用这些数值地址；每组小型不可变task/length/binding元数据仍在startup创建，调用方应计入组数对应的startup余量。不能把profiling之后才设置开关等同于之前已经预留了score内存。

## 验证

```bash
python3 -m unittest discover -s experiments/split_core/tests -p 'test_staged*.py' -v
python3 -m unittest discover -s experiments/split_core/tests -p 'test_score*.py' -v
python3 experiments/split_core/prepare_staged_source.py --version current --model --output /absolute/new/source
cmake -S experiments/split_core/staged_build -B /absolute/new/build \
  -DSTAGED_SOURCE=/absolute/new/source -DSCORE_VERSION=current \
  -DKVAF_INCLUDE=/absolute/patched/KVAF/aiv_batch_get/include
cmake --build /absolute/new/build --target staged_routes -j2
```

CPU验证包含全部42标准规格、精确Q/H覆盖、物理预算、B24/32 Index ABI、公共Forward和旧融合/UBmem入口、动态owner、共享score与profiling events。CPU/mock不能替代device验证。

main统一测试入口的paired native benchmark保留B≤16，URMA `bench/check --score-route auto/history/bsq`可显式验证新路由；UBmem仅接受legacy。staged的逐核timeline解码尚未接入统一diagnose，因此非legacy的diagnose/all会在配置时拒绝。生产binding与独立staged验证工具仍支持B≤32。

历史r5版本（`ecf4e8e`）的三项Model功能烟测均通过并正常结束。采用H8192/R6400/G32、每Q miss205、C4/V16/W2，同一fixture覆盖B1/history、B1/BSQ、B3/BSQ。B3覆盖两score waves和两个owner waves。driver核验真实score、全resident θ、Final集合、dedup、Index四字段置换、hot656B payload、guards和completion。精简结果和log/binary hash见 `results/score-integration-validation.json`，完整源码/config/fixture/hash与失败记录由该摘要链接到原提交归档。清理并合入main测试重构后的版本另行运行CPU与构建检查，不沿用旧Model通过作为新源码证明。

该小尺寸fixture的每owner cold并集为410；另有563个选中resident token落在预留LRU区。prefetch0且无tail映射时，旧缓存契约将这些reserved hot token也计入补取，因此独立输入oracle要求去重补取总数973。driver同时检查row/worker计数，不能用cold数替代最终补取数；完整Final集合、映射和payload检查仍保留。

Model中prefetch预算为0，使用独立编译的current score wrapper与当前完整tail，最终BatchGet使用受控同步GM-copy；原生staged入口另行编译/链接核验。它不证明真实URMA/CQE/DMA、非零prefetch异步正确性或真机速度。完整42场景Model矩阵以及NPU/KVAF整链验收仍pending；新路径不设为默认。本文没有将独立TopK测量当score或整链提速证据。
