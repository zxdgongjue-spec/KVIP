# LI位图输出与压缩候选输出A/B

固定BS1、Q4、history8192、hot4096、heads32、TopK2048；1 AIC及配套2 AIV。
直接调用当前生产LI，保持M128/N128/K128打分tile、scale、hot TopK和最终TopK。
关闭预取服务，不运行Resolver/Index更新/Fetch。hot TopK产生各Q真实阈值，cold只保留严格大于阈值的项。

`prepare.py`将当前生产源码复制到构建目录，只替换cold候选输出，正式算子不改：

- A：原生2-bit predicate，每128 token输出32字节bitmap。
- B：复用score寄存器和比较mask，向量Squeeze输出分数与tile内位置，向量归约生成数量。
  每tile独立发布，避免跨核原子追加或全局前缀和。记录为544字节：128×uint16分数、
  128×uint16位置、32字节数量区（首uint32有效）。只有前count项有效，位置需加tile起点。
  固定容量记录包括未使用尾部，不是整个窗口无缝紧凑的GM数组；下游可按count读取有效前缀。
  每Q/tile沿用现有进度发布。UB候选缓冲从1KiB增加到4KiB。

两版保留最终TopK所需的完整score写回。B没有再次加载完整score做压缩，也不逐bit收集。
候选输出A共4KiB，B共68KiB；增加的写回、计算、同步和结束等待全部计入完整LI时间。
输入为seed43的FP8 ±1 Q/Index，权重和scale为1；用于隔离代码差异，不代表真实模型分布。

## CANN软件Model运行

在有CANN及Ascend950PR_9599 simulator的环境运行。依赖KVAF头文件，不连接KVAF服务。
需要给出已有的KVAF源码路径：

```bash
source /usr/local/Ascend/cann/set_env.sh
kvaf=/path/to/KVAF
run="$PWD/tests/results/li-candidate-output/$(date +%Y%m%d_%H%M%S)"
for mode in 0 1; do
  cmake -S tests/experiments/li_candidate_output -B "$run/build-$mode" \
    -DRUN_MODE=sim -DCOMPACT_OUTPUT="$mode" \
    -DKVAF_INCLUDE="$kvaf/access_operator/demo/aiv_batch_get/include"
  cmake --build "$run/build-$mode" --parallel 2
  msprof op simulator \
    --application="$run/build-$mode/li_candidate_sim $run/result-$mode.bin" \
    --soc-version=Ascend950PR_9599 --output="$run/profile-$mode" --timeout=10 \
    > "$run/mode-$mode.log" 2>&1
  cat "$run/mode-$mode.log"
done
cmp "$run/result-0.bin" "$run/result-1.bin"
```

每版校验最终TopK值集合、ID唯一性、hot阈值、完整cold区间的候选判定与压缩分数/ID。
`cmp`额外校验A/B完整score、最终TopK ID、还原后的候选集合逐字节一致。
耗时读取profiler的设备模型duration_time，不使用CPU墙钟；此入口没有NPU graph性能计时。

## 本地结果（2026-11-16）

CANN9.1、Ascend950PR_9599软件Model，8,080/16,384个cold Q-token位置入选（49.32%）。
两版校验PASS，输出文件cmp一致。

| 核 | A 位图 | B 压缩候选 | 增量 |
| --- | ---: | ---: | ---: |
| AIC0 | 29.92 µs | 32.26 µs | +2.34 µs |
| AIV0 | 34.55 µs | 36.56 µs | +2.01 µs |
| AIV1 | 34.22 µs | 36.90 µs | +2.68 µs |
| 最慢核 | 34.55 µs | 36.90 µs | +2.35 µs（6.80%） |

每个AIV执行32次cold打分/归约VF，每次处理2Q×128分数。
VF指令持续时间合计：AIV0从6.01增至6.74 µs，AIV1从6.05增至6.76 µs。
这些是指令区间统计，会与其他流水重叠，不能与完整LI时间或MTE时间直接相加。
两核并行，不能把两核的增量相加作为LI延迟。

本次证明固定tile压缩输出在该配置下增加约2.35 µs；没有测下游预取消费、增量TopK或完整LIFetch收益。
