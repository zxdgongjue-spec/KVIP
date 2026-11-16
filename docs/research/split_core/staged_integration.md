# 当前LIFetch通路的显式score分核

默认融合入口保留。`LIFetchBinding(score_route="auto")` 与原公共 `LIFetchForward(binding, update)` 即可启用当前URMA完整staged入口；`history`/`bsq`可强制对照，`legacy`回到原融合方案。旧 `staged=True` + `StagedLIFetchForward` 仍可用作兼容入口，但不再是唯一调用方式。新路径支持逻辑B≤32，旧默认完整入口仍B≤16。

## 执行顺序与协议

Hot FP8 QK score waves → 设备scatter到完整BQH → 完整resident θ及LRU保护predicate → Hot progress/Service → 每Cold epoch的全部score waves/scatter → 完整θ下cold predicate → 连续range progress/Service → 完整H Final TopK及classification → whole-owner Resolve/Output/Index/Fetch/completion。

每Q的queryEnds动态裁剪θ、Cold predicate和Final。任意一个history片的局部θ不能替代完整resident θ。Final沿用原global token坐标、cutoff tie、locations、query保护bitmap和错误输出。分类kernel按同stream完成后才启动tail。

Resolve保留同owner四Q保护bitmap并集和crossQ dedup，统一分配唯一slot。Output先消费旧映射并发布descriptor，Index按原snapshot/exchange/commit协议执行；Fetch保留safe/conflict phases、prefetch drain和metadata/transport completion双join。submit不等于payload完成。Context320B、RuntimeShape80B、Index update28字段保持原样；sidecar承载分核任务。

独立物理预算 `Ceff=min(C,floor(V/2))`，score每wave≤Ceff；θ/Final每wave≤V。tail owner容量为 `min(16,floor((V-4W)/3))`，每wave驻留 `3n+4W` AIV。Fetch与completion只等待当前owner范围，上一wave完成后才清packed counters并启动下一wave。

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

Model功能烟测采用H8192/R6144/G32、每Q miss205、C4/V16/W2，同一fixture覆盖B1/history、B1/BSQ、B3/BSQ。B3覆盖两score waves和两个owner waves。driver核验真实score、全resident θ、Final集合、dedup、Index四字段置换、hot656B payload、guards和completion。原始源码/binary/config/fixture/log hash与结果摘要见 `results/score-integration-validation.json`。

Model中prefetch预算为0，最终BatchGet使用受控同步GM-copy；它不证明真实URMA/CQE/DMA、非零prefetch异步正确性或真机速度。完整42场景Model矩阵以及NPU/KVAF整链验收仍pending；新路径不设为默认。本文没有将独立TopK测量当score或整链提速证据。
