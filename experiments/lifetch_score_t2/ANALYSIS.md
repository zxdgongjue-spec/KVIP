# MR23 复现范围与 score → 原 T2 候选分析

## 基线确认

`a56154c37bebb0928da4a4095e9670898e6b21c7` 是 MR23 合并提交（父提交 b30e873/f35035f，42 文件，+4435/-776），不是只增加一个用例。改动包括 resident TopK 复用、窗口增量候选、按 per-Q 覆盖把 admission rank 从 1024 提升到 2048、排除 hit/已预取并跨 Q 去重、带原始位置的 compact misses、足够 victim 时的稀疏 resolver 及通用回退、移除独立 Init、新的独立状态整图计时。

以下是提交附带的作者真机数据，不是本次重新运行得到的数据，单位 µs/call，graph event 均摊：

| miss | B1 串行 | B1 原融合 | B16 串行 | B16 原融合 | B16 融合耗时降幅 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 261.54 | 302.86 | 346.70 | 337.66 | 2.61% |
| 10% | 269.70 | 335.71 | 395.29 | 360.94 | 8.69% |
| 15% | 272.02 | 327.94 | 449.49 | 418.84 | 6.82% |
| 20% | 275.62 | 338.62 | 486.04 | 524.26 | -7.86% |

这不支持“融合始终比串行快”：B1 全部更慢，B16 在 20% miss 又变慢。B16/10% 的同图阶段数据是串行 LI 225.75 + LIDU 37.10 + Fetch 110.50；融合 LI（含预取）274.52 + Resolve/Update 65.63。串行 LIDU+Fetch 相对融合末段少了约 81.97µs，LI 端额外付出约 48.77µs，净调用内收益约 33.2µs。这说明预取收益已被 LI/Service 成本部分抵消；优化 score 仍有空间。

B16/20% 串行后段 38.42+203.25=241.67µs，融合末段 231.16µs，只剩约 10.51µs 的阶段差；融合 LI 比串行 LI 多约 45.63µs，整链因此退化。末段增加可能来自 miss probe/dedup、保护/victim、fallback、Fetch 冲突/CQ等待等；阶段总量不能证明某一原因，需详细 trace/counters。新候选复用这些算法，不会自动解决高 miss 末段。

提交没有给出可核验的全部原始 argv、device budget 和 raw trace。仓库文档的标准 diagnose 示例为 W4/pf1024，默认 bench 为 W8/pf512。本次脚本固定明确的几何并重新对照，不宣称能原样复现作者每个数字。已有旧 MR14 数据采用不同路径/计时方式，不能直接与本次 event 均值合并。

## 为什么不能直接接 MR14 staged 尾部

原 T2 已有 score 与 resident/Cold/Final 的明确边界，也有 Cold kernel 内的窗口循环和 Service。MR14 history staged 额外引入各窗口 prepare/score/scatter/progress/service；把 Cold predicate 留给少量 AIV 串行再次读分数，并反复做 resident/progress 处理，可能抵消 score 的多核收益。

因此本候选不迁移 staged 尾部，也不扩大 radix 分片合并。只替换 score 输出接口，之后用 MR23 原 T2 调度。score 写原布局，theta/Final/Service 消费原地址，不生成虚拟 row 或分片局部 TopK。

## 本次候选的真实收益点与代价

| 项目 | MR23 T2 | score_t2 |
| --- | --- | --- |
| B1 score 资源 | 2 AIC/4 AIV | P16 时 16 AIC/32 AIV；P2 控制保持 2/4 |
| B16/P2 score 资源 | 32 AIC/64 AIV | 32 AIC/64 AIV，没有增加 score 核数 |
| score 中间布局 | 原 canonical | 同一 canonical，无 scratch/scatter |
| θ/Final | 原算法 | 同一原 kernel，不加分片 merge |
| Cold predicate | 与 score 的 reduction 融合 | 原 T2 AIV 每 tile 再读一次 score 后分类 |
| LI launches | 4 | 4；完整 score 替代 Hot，无额外空 Hot |
| Service/Resolve/Fetch/commit | 原 MR23 | 原 MR23 |
| resident 优先 | Hot 完成即 θ | 完整 H score 完成后 θ，可能推迟首次预取 |

B1/Q4/H128K 的完整 score 是 1MiB（uint16）。Cold 为 Q4×114688×2=896KiB；候选新增这部分 predicate 的 GM 读取，而非新增全量 scratch 的写/读/重写。B16 新增约 14MiB Cold score 读取。此项通常比原 FP8 Cube+head reduction 更轻，但对 B16/带宽场景不能忽略。

Cold predicate **没有继承 P16 的全部并行度**，首版故意保持 T2，用于控制后处理变化。它已省去 scatter/score rewrite，且不再进行 QK/反量化/head reduction；predicate/增量 TopK/Service 的剩余串行成本由真机量化。若后段占比仍大，再考虑 θ 后分核分类与段级 ready，不应宣称本版已继承所有并行度。

`PrepareShape` 会把请求窗口数收敛到 `liCores`。B1 的原 T2 liCores=2，实际 rounds≤2；B16 在 32 AIC 下可保持请求的 W4/W8。score P16 不改变原 T2 shape，仍按两 history owner 发布 ready。脚本保存 requested 参数与候选实际 shape；不得把 requested W8 当作 B1 实际八个 Service epoch。

Final 是完整有效 history 上的数值 Top2048，不是跨 Q 去重；跨 Q 的共享 token/victim 去重在后续 owner 规划中完成。本版保留 Final 在 Cold/Service 后执行，保证 ctx->locations 和其他 scratch 不被并发覆盖。

判断收益必须看完整 event 均值/实际完成终点；score-only 和整链必须分表。`prefetch=0` 控制用来隔离预取重叠。B1/P2 控制能显示“更晚开始 θ + 再读 Cold”的纯代价；B1/P16 再量化新增 score 并行带来的净收益。B16/P2 更可能依靠消除 score/Service 互相停顿获益，同时也可能因新增 Cold 读取和独立后段退化。

## 验证记录与限制（2026-11-19）

- 6 项 CPU 检查通过：真实 C++ 分片头的 B1/P1/2/4/8/16/30/32、B16/P1/P2 覆盖/行地址；预算与空片拒绝；全局 θ/严格 greater-than；新 kernel 同图计时归属；CLI 默认与 transport 门禁；summary 使用整图 event 比值且拒绝 operand 改变。
- A5 runner dry-run 通过，确认从本地 Git 导出精确 MR23、生成交错指令，不需要容器联网/Docker。旧 Python 3.9 缺少 tar extraction filter 的兼容问题已修复。
- CANN 9.1 / `Ascend950PR_9599` 的 NPU 和 sim 目标均编译/链接通过；新入口和两 kernel 符号存在。首次编译发现 VF 循环变量必须 uint16，已修复并重编通过。
- CANN Model 的 B1/Q4/H131072/P16 **score producer** 用例通过，524288 个 key 与精确正 FP8 oracle 一致，score guard、输入数据/guard 校验通过；整次 Model 运行 48726 ticks。详见 `model_score_result.json`。
- Model 实际 SoC 返回 `Ascend950PR_9589`，编译目标为 `Ascend950PR_9599`；该 Model 结果验证逻辑，不能代替目标 A5 芯片性能。ticks 包含整次 Model 运行边界，未作为精确 kernel cycles，也未换算 µs。
- 首次直接加载 npu 产物到 Model 发生启动段 segmentation fault；切换正确 `RUN_MODE=sim` 构建后标准 score 用例成功。未将失败计入性能结果。
- **尚未在 CANN Model 或 A5 上验证完整候选 Cold/Service/Final/URMA 末段**；无新整链加速比、首次 Fetch 或 ConsumerReady 数据。编译通过和 score oracle 通过不代表完整流程正确性已通过。
- 为避免磁盘再次耗尽，本次 Model 关闭 instruction dump，仅保留运行日志、配置和 proof，不提供指令占比。

下一步使用 `run_a5.sh --miss-percent 10 --rounds 1` 先验证标准组全流程，再跑全部 miss≤20% 组。所有配置要 PASS，输入 hashes 相同，native 参考没有明显漂移，才能评估候选；完整测量自动保留 profiler trace。若出现退化，再用 `diagnose --level 2` 定位 resident θ/Cold Service/Resolve/CQ 等待，不猜测加速幅度。

## A5 构建失败与恢复（2026-11-19）

用户运行 `lifetch_score_t2_20261119_204047_27p7vr4k` 时，候选 P2 在 CANN 9.2.0 preprocess 的 AIV 合并阶段报 `indexer.cpp.o: unknown file type`。B1 native event 为 254.86µs，LI/LIDU/Fetch 同图阶段为 220.36/19.14/12.03µs；融合未执行，B16 未完成。这些值只证明 native 路线执行，不能形成候选性能或正确性结论。

在本地 CANN 9.1 对已经成功执行过 `merge_obj_text.sh` 的 `.o` 再次运行相同 `ld.lld -m aicorelinux -Ttext=0`，复现同一 `unknown file type`。对象为 ELF EXEC（type=2），不是编译器最初的 REL（type=1）。CANN 合并脚本原地替换 `.o`，而旧构建器对缺少 `complete.json` 的缓存复用同一 CMake/ExternalProject 目录，因此重试可能把已链接对象再次交给链接器。服务器报错对象未直接检查；新的 ELF 诊断用于确认其实际类型，不能排除其他文件/编译器原因。

恢复方案只修改启动构建：每次未完成构建使用全新 `attempt-*`，保留失败对象和日志；成功后原子发布 `.so` 与完成记录，后续缓存命中不调用 CMake。缓存协议摘要更新会避开旧失败目录。MR23 基线只替换同一启动构建器，算子与计时源码保持原提交，替换 SHA256 和原文件作为实验产物保留。

验证：3 项构建恢复测试通过（失败目录/日志保留、新目录重试后成功复用、configure 失败与缺失库不发布完成记录）；9 项实验控制/依赖隔离测试通过，包含基线构建覆盖记录检查。原有源代码/头文件/共享 selector 摘要更新与缓存复用检查、CMake 源目录检查通过。本地 CANN 9.1 NPU 全新目录编译链接通过，缓存第二次调用没有重编，发布后的库在正确预加载 CANN Model runtime 时动态加载及新入口符号检查通过。

本地没有硬件驱动；直接加载 NPU 库时 SoC 检查出现 `basic_string::_S_construct null not valid`，原构建目录和发布目录一致。正确预加载 Model runtime 后两者的环境问题消失，未将该初始化失败当作候选运行结果。完整候选 A5 Cold/Service/Final/URMA 正确性和收益仍待重跑验证。

## 已执行但 TopK 失败：Cold DMA 同步（2026-11-19）

A5 `r0_pf1024_score_t2_p2` 的 B1/B16 完成执行，均未通过正确性：B1 serial/fused event 255.76/408.90µs，B16 367.41/418.96µs。这里是 P2 控制，不是 B1/P16；runner 在失败处停止，P16 及正式轮没有完成。

同图时间分解：B1 候选 LI 比 native 多 137.32µs，Resolve/Update 比串行 LIDU+Fetch 多 10.38µs；B16 候选 LI 多 127.42µs，而融合末段少 77.11µs。B16 的末段差值被 LI 增量抵消；两路都包含 Fetch 完成。P2/B16 与原 T2 同为 32 AIC；全 history score 完成后才开始 θ/Service，而且 Cold 额外重读 GM。当前失败结果不能用于证明候选的有效性能或推断首次预取时间。

日志的 `score_mismatches=131072` 是审计的单位混用：例如 key15088 解码为 BF16(-2288)，32768 为 +0，贴出的八个样例全部与数值参考一致。这不代表全 history score 已经证明完全正确，也不会消除 TopK 漏选高分/选入低分的失败。审计现在按实际参考格式比较，并记录 resident θ 与物理 Cold bitmap 的差异。

实际消费函数 `ProcessPrecomputedVec1` 存在独立流水线依赖缺口：MTE3→V 等待只能限制 V；MTE2 score DMA 仍可能提前覆盖 V 在读取的 ping-pong UB。新增搬入前 V→MTE2 Set/Wait，事件不借用预先发布的 bank credit。原 canonical、唯一 θ、T2 owner/Service、Final 与 CQ/Index 依赖保持原路径。

使用同一实际消费函数、相同 B1/Q4/H32768/R16384 输入的 CANN Model 对比：保留的修复前二进制有 **37236/65536 个 Cold predicate 错误**；修复后 **0 个错误，PASS**，strict greater-than ties、输入、阈值和 guard 检查通过。NPU/sim 目标共享该消费源文件；本次实际编译运行为 CANN 9.1 sim。详见 `cold_model_result.json`。另一个 H128K 的修复前 probe 也出现大量 predicate 错误；没有将不同几何数据作为同场景比值。

该结果验证了消费路径同步修复，没有执行 Service、Final TopK、URMA 或 Index commit，也未测量 A5 修复后的性能。先运行 `run_cold_contract.sh --device 5 --batches 1,16`，再按原固定参数运行完整 `run_a5.sh`。所有正确性 PASS 后，才评估 P2/P16 的耗时和剩余瓶颈。
