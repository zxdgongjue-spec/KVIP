# MR23 score 分核接原 T2 后处理

实验基线固定为 main 的 MR23 合并提交 `a56154c37bebb0928da4a4095e9670898e6b21c7`。本目录属于 `codex/lifetch-score-t2` 的实验候选；不是 MR14 staged 路线，也没有改默认生产入口。

## 调用与算法差异

| 路线 | LI kernel 链 | LI launches | 后续 |
| --- | --- | ---: | --- |
| MR23 原 T2 | Hot score → resident θ → Cold score/classify/窗口 Service → Final | 4 | 原 Resolve/Fetch/Output/Index commit |
| score_t2 | 全 H 分核 score 直写 canonical → 原 resident θ → T2 Cold classify/窗口 Service → 原 Final | 4 | 同一原 Resolve/Fetch/Output/Index commit |

T2 是每 request 两个 history owner：2 AIC/4 AIV。score 的每 AIV 处理两个 Q；resident θ/Final 的四个 AIV 分别处理一个真实 Q。两个 history owner 分别处理半个 resident、每个 Cold 窗口的一半。不是每 Q 各抢一个 victim。

新 producer 使用每 request `P` 个 AIC、`2P` 个 AIV；以整 256-key 页均分，内部仍为 128-key tile。`B*P <= effective_aic`，P 不能超过非空页数。真实 score 地址为 `(packed_row*4+Q)*H+history_position`，packed row 再通过原输入 block table/rowOwners 对应 cache owner。物理 AIC ID 不作为输出行。

scoreOnly 保留原 FP8 QK、反量化、权重/head reduction、BF16 sortable key 的数学路径；禁止在局部分片中执行 resident/Final TopK 或 Service。完整 score kernel 结束后复用原 resident θ kernel。Cold 消费每个 tile 一次，加载唯一 θ，严格使用 `score > θ`，调用与原 score kernel 共用的 bitmap/progress 发布函数。后续保留原 ranked admission、增量候选、Service predecessor fence、共享四 Q hot/victim 保护及唯一 owner。

全 score 已就绪也不提前发布所有 Cold ready：原 T2 按半窗口发布 bitmap + progress，Service 只消费其合法连续覆盖。Final 仍在 Cold/Service kernel 结束后调用，避免 `ctx->locations`、增量 TopK 和 classification scratch 重用冲突。URMA CQ join、safe/conflict Fetch、Output 消费旧映射、Index commit 均沿用原 resolver。

## 固定 A5 对比

先在可访问 Git 的宿主机更新本候选分支；宿主机和容器共同可见 `/home/w00568585`。容器不需要 Docker 命令，也不需要 Git 网络。进入候选 KVIP checkout，使用已构建且固定在 `1c1310387ebfb6776a07316874e924265b209ada` 的独立 URMA KVAF；不自动切换/覆盖 vLLM 的其他 KVAF。

```bash
cd /home/w00568585/wt-lifetch-score-t2
bash experiments/lifetch_score_t2/run_a5.sh \
  --device 5 \
  --kvaf-root /home/w00568585/kvaf-mr14 \
  --kvaf-server 127.0.0.1:28885 \
  --miss-percent 0,3,5,10,15,20 \
  --prefetch 1024 \
  --intermediate-fetches 4 \
  --rounds 3 --warmup 4 --iterations 20
```

路径请替换为实际已更新的 KVIP 与已构建的固定 KVAF。需要原 SDK `torch_npu` QLI、已注册的 A5 LIDU/SFA vendor，以及 CANN/URMA 环境；沿用之前已正常执行 MR23 bench 的环境。脚本调用新版 `tests/operators/lifetch/test_lifetch.py bench`，不是旧的无子命令 `lifetch.py`。

如果现有 vLLM-DSA/KVAF 的 HEAD 不是 `1c131038`，给同一命令增加 `--prepare-kvaf` 即可：

```bash
bash experiments/lifetch_score_t2/run_a5.sh \
  --device 5 --kvaf-root /home/w00568585/vLLM-DSA/third_party/KVAF \
  --prepare-kvaf --kvaf-server 127.0.0.1:28885 \
  --miss-percent 10 --prefetch 1024 --intermediate-fetches 4 \
  --rounds 1 --warmup 4 --iterations 20
```

该选项把 `--kvaf-root` 当作本地 Git 来源，默认创建 `/home/w00568585/KVAF-score-t2-1c131038`，应用本 KVIP 的四个 URMA 补丁并构建，然后自动使用新目录运行对比。原 vLLM-DSA/KVAF 的 HEAD、编辑和构建产物保留。编译日志为结果目录中的 `dependency-build.log`，创建/补丁应用不联网；实际构建沿用服务 Python/CANN 环境，需要已安装的构建依赖。如果本地 Git 历史缺少该提交，脚本会给出宿主机 fetch 命令；如果找不到本地 UMDK，设置 `KVAF_UMDK_ROOT` 为含 `include/urma_api.h` 的现有目录再运行。UMDK 的新编译产物写入独立 KVAF 目录。

准备完成后的实验可以直接指定 `--kvaf-root /home/w00568585/KVAF-score-t2-1c131038`，不再加 `--prepare-kvaf`。可用 `--pinned-kvaf-root` 自定义新目录。新 LIDU vendor 对两路测试共同使用，所选路径写入 manifest。脚本构建 CPU service 产物，但不会启动或重启已有的 KVAF server。

`run_a5.py` 本地 `git archive` 导出 MR23 基线，避免把未提交编辑或 build 软链接复制进去。基线的算子、launch 和计时源码保持该提交；两路统一使用当前 `python/kvip/runtime/build.py` 的启动构建修复，替换记录和 SHA256 写入 `baseline_build_override.json`/manifest，原文件另存 `baseline-build-original.py`。两个 source 通过源文件摘要缓存分别编译 LIFetch。所有测试在同一个固定 KVAF 上执行；native 串行参考会在每路测试中实际执行。不会直接把不同版本的 build-npu 二进制复制到固定版本目录，也不会跳过版本检查。

### `indexer.cpp.o: unknown file type` 构建恢复

CANN 的 `merge_obj_text.sh` 会把编译产生的 REL 对象原地链接成 EXEC 对象。失败后复用同一 CMake/ExternalProject 目录，可能跳过编译并再次合并该对象，触发 `unknown file type`。本地 CANN 9.1 已复现该错误机制；服务器上具体对象的类型仍需失败诊断确认，不能仅凭这一行认定对象类型或算法错误。

当前构建器对未完成的缓存每次创建全新的 `attempt-*` 目录，不复用对象和 stamp；成功后原子发布动态库与 `complete.json`，之后直接复用完成产物。失败目录保留，缓存根部 `build.log` 指向最近一次日志。出现 `unknown file type` 时还会输出对象大小及 ELF type/machine。缓存协议更新会避开此前目录，不需要删除整个 KVAF 或重新准备依赖。

更新候选分支后，在容器使用已经准备好的固定依赖重跑：

```bash
cd /home/w00568585/wt-lifetch-score-t2
bash experiments/lifetch_score_t2/run_a5.sh \
  --device 5 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28885 \
  --miss-percent 10 --prefetch 1024 --intermediate-fetches 4 \
  --rounds 1 --warmup 4 --iterations 20
```

若全新 attempt 仍失败，保留完整新 `build.log` 及对象诊断，再定位编译器、文件内容或其他原因。`INCOMPLETE/NOT_RUN` 的结果不能计入融合性能对比。

默认交错运行：原 MR23（B1/B16）、score P2（B1/B16）、score P16（仅 B1），r0 丢弃，再测 r1/r2/r3。P2 的 B1 控制组用于判断全 score 屏障/Cold 再读的代价；B16/P2 已使用 32 AIC，没有凭空增加 score 并行度。若设备有效核数小于 32，B16/P2 会拒绝执行，禁止超预算隐式分波。

要单独观察预取重叠贡献，改成 `--prefetch 0,1024`。要复用之前 W8/pf512 几何，使用 `--intermediate-fetches 8 --prefetch 0,512`。这些是不同参数实验，不能与 MR23 提交附表混成同场景复现。

仅跑一个标准控制组便于先确认编译/正确性：

```bash
bash experiments/lifetch_score_t2/run_a5.sh \
  --device 5 --kvaf-root /home/w00568585/kvaf-mr14 \
  --kvaf-server 127.0.0.1:28885 --miss-percent 10 \
  --prefetch 1024 --intermediate-fetches 4 \
  --rounds 1 --warmup 4 --iterations 20
```

独立入口（已有完整环境）：

```bash
python tests/operators/lifetch/test_lifetch.py bench \
  --transport urma --device 5 --kvaf-root /home/w00568585/kvaf-mr14 \
  --kvaf-server 127.0.0.1:28885 --batches 1 --history 131072 \
  --resident 16384 --index-heads 32 --miss-percent 10 \
  --prefetch-tokens 1024 --intermediate-fetches 4 \
  --li-score-parts 16 --warmup 4 --iterations 20 \
  --output /home/w00568585/t2_candidate_b1
```

不加 `--li-score-parts` 是当前 source 的默认入口；严格基线对比请用包装脚本的精确提交 source。`diagnose` 也支持该选项；`--level 2` 单独进行详细打点重放，正式时间仍来自未开启详细打点的整图。

## 产物与时间口径

脚本直接生成 `manifest.json`、`jobs.json`、每路 `.log`、源码 diff、`samples.json` 和 `summary.csv`。原始 profiler chrome trace 位于各路 `fused/diagnose/round-1/trace.json`，可用 Perfetto 查看；不是独立 `msprof` 指令 trace。正式结果使用同一次重放的 graph event/calls，另列 profiler p50/LI/Resolve 与调用内 gap，r0 不纳入均值。对比所有真实 operand `.bin/.npy` 校验和不一致即报错，结果不沿用之前独立 p50 的计时。

`Resolve/Get` 是融合 kernel 区间，包含 Resolve、Fetch join、Output、Index update 的工作与等待；不能直接当成纯 resolve 算法时间。LI 区间包含 Cold Service 内的预取推进，不能当成 score-only。SFA 用于完成后的正确性校验，**不在这个算子链计时内**；该脚本不能报告模型 token 吞吐。

## 当前验证状态

CPU 测试与脚本 dry-run 已执行；CANN 9.1 的 npu/sim 编译链接通过。B1/P16 score producer 的 CANN Model oracle/guard 校验通过，整次 Model 为 48726 ticks。完整 Cold/Service/Final/URMA 候选流程及 A5 硬件收益仍待验证；该 score 用例没有验证整个尾部。详见 `ANALYSIS.md` 和 `model_score_result.json`。候选暂不用于生产性能结论。

### Cold 消费路径的同步修复

`ProcessPrecomputedVec1` 原来只等待 MTE3→V，再由 MTE2 搬入下一块 score。MTE2 是独立流水线，该等待没有限制它覆盖 V 仍在读取的 UB。当前在搬入前补齐平衡的 V→MTE2 Set/Wait，之后仍使用 MTE2→V 和 V→MTE3 发布依赖。

另修复失败审计的单位混用：设备 GM 为 sortable BF16 uint16 key，而 synthetic 参考为数值 score。现在在相同口径比较，并增加 saved θ 与 Cold bitmap 对照。TopK 的真实漏选/低分选入检查继续保留，未放宽正确性标准。

先在 A5 检查实际 Cold 消费函数（无需 KVAF server，不执行 Service/Fetch），再运行完整 `run_a5.sh`：

```bash
cd /home/w00568585/wt-lifetch-score-t2
bash experiments/lifetch_score_t2/run_cold_contract.sh \
  --device 5 --batches 1,16 \
  --kvaf-root /home/w00568585/KVAF-score-t2-1c131038
```

该标准用例固定 Q4/H32768/R16384，包含每 Q/owner 不同 θ、严格 greater-than ties、连续多 tile 和 ping-pong 复用；核数为 B×2 AIC/B×4 AIV。它只验证实际 Cold bitmap、阈值与输入/guard，不提供性能结果，也不能替代完整 H128K、URMA Fetch、Index/Output 的正确性检查。每次构建使用新目录；若 CANN 不在默认位置，增加 `--cann /实际/CANN/目录`。本地 Model 使用 `--mode sim --batches 1 --device 0`，日志详见 `cold_model_result.json`。
