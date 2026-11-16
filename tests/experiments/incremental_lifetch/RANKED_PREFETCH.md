# 按窗口分数阈值和每 Q 配额预取

A 固定为已接受的 ABC + 增量 TopK 基线。B 只调整预取候选的选择；
完整 Top2048、最终 TopK、跨 Q 去重、映射过滤、选槽、KVAF 提交和补拉保持原流程。

四窗口时，每个 Q 使用当前完整 Top2048 的第 1024、1365、1707、2048 名分数作阈值。
先限定 cold，再按阈值过滤，最后保留其中分数最高的至多 32 个
（每请求每窗口 128 / Q4），随后跨 Q 去重。去重或已映射过滤后不补足配额。
同分时优先保留原 TopK 顺序；始终先输出严格高于截断分数的项，再填充同分项。
窗口 0 也使用新规则。最终 TopK 的完整 2048 项不被截断或改写。

实现复用 PR15 的精确 u16 两级直方图选阈值，不排序完整 TopK。
第一版在 2048 项掩码数组中求配额阈值；优化版压缩 cold 分数后求阈值，
不足配额时跳过第二次求阈值。末窗口无需计算第 2048 名阈值。
另去掉了被新筛选替代的旧 cold 压缩。使用原有 TopK 临时 UB，不新增 UB 分配。

## 真机完整图 A/B

```bash
git switch codex/lifetch-prefetch-optimization
git pull --ff-only
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh
bash tests/experiments/incremental_lifetch/compare_ranked.sh \
  /vllm-workspace/vllm-ascend/third_party/KVAF \
  4 127.0.0.1:28885 16 10 4 384
python tests/experiments/incremental_lifetch/summarize.py --experiment ranked-prefetch
```

最后四个参数为 BS、miss%、窗口数、ABC tail 长度。测 1% miss 时把 `10` 改为 `1`。
脚本保留独立的 level3 采样、完整图性能、逐窗口耗时和触发数、最终 TopK、
预取/补拉/无用读取统计，以及 KV/Index/映射/冷启动正确性检查。
新路径不再断言窗口 0 必须预取 hot token；这条旧策略覆盖断言与 cold-only 规则不相容。

## 本地模块验证

在有 CANN simulator 的环境中生成并编译实际共用头文件：

```bash
python tests/experiments/incremental_lifetch/model_rank.py /tmp/rank-model
cmake -S /tmp/rank-model -B /tmp/rank-model/build -DRUN_MODE=sim
cmake --build /tmp/rank-model/build -j8
mkdir /tmp/rank-model/trace
(cd /tmp/rank-model/trace && \
 LD_LIBRARY_PATH=/usr/local/Ascend/cann/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH \
 ../build/rank_model > ../model.log 2>&1)
python tests/experiments/incremental_lifetch/model_rank.py /tmp/rank-model --summarize
```

8 组输入覆盖四个窗口、0/12/64/256/2048 个 cold、全同分、
低分 cold 被窗口阈值排除、非法位置；对照 CPU 精确选择结果。
分别计时第一版、优化版、原 cold 导出，共 24 项。
输入 TopK 已在 UB，计时含向量计算、必要同步与读取计数，不含 GM 输入加载、
跨 Q 去重或网络传输。模型按 1800 cycles/µs 换算；不是 NPU 实测。

## 2026-11-16 本地结果

24/24 选择/原导出校验通过。分数与 ID 的完整输入不改写；CPU 参考按
窗口排名阈值过滤，再稳定选择最高分的至多 32 项。
以下为最终版 simulator cycles / 1800，非 NPU 时间。每个场景采样一次；
4Q 增量按同场景串行四次换算，尚未包含窗口 0 / 无新增分数分支的额外 GM 加载。

| cold/Q | 窗口排名阈值 | 新筛选 µs/Q | 原 cold 导出 µs/Q | 净新增 µs/4Q |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1024 | 0.44 | 0.45 | -0.03 |
| 12 | 1365 | 0.90 | 0.45 | 1.84 |
| 64 | 1707 | 1.04 | 0.45 | 2.36 |
| 256 | 2048 | 0.85 | 0.45 | 1.62 |
| 2048 | 1024 | 1.24 | 0.45 | 3.14 |
| 256，全同分 | 1365 | 1.05 | 0.45 | 2.43 |
| 12，全部低于窗口阈值 | 1707 | 0.89 | 0.45 | 1.79 |
| 32，另有32个无效位置 | 2048 | 0.71 | 0.45 | 1.03 |

64 cold/Q、排名1707场景的优化迭代：

| 实现 | cycles/Q | µs/Q |
| --- | ---: | ---: |
| 第一版：2048项掩码数组求配额阈值 | 3741 | 2.08 |
| 压缩 cold 后求阈值 | 2619 | 1.46 |
| 再消除 PackCold 循环内运行时分支 | 1869 | 1.04 |

最终版的 1869 cycles 分解：

| 步骤 | cycles/Q | µs/Q |
| --- | ---: | ---: |
| cold 分数压缩、计数及同步 | 791 | 0.44 |
| 完整 Top2048 的窗口排名阈值 | 367 | 0.20 |
| cold Top32 配额阈值（含转换和同步） | 282 | 0.16 |
| 按阈值筛选 ID、处理边界并列及计数 | 429 | 0.24 |

实际融合方法的本地提交校验也通过：单 AIV、2请求×Q4、每 Q 12个新增候选，
两个请求各提交23条（跨Q重复和已有映射被剔除），检查了完整TopK、槽位、
源/目标地址、WQE与doorbell；相同进度再次执行无重复提交。
该集成采样在最后一次循环分支优化之前：增量TopK+筛选43953 cycles（24.42 µs），
后续service26039 cycles（14.47 µs）。它不是新筛选的单独耗时，也不是网络完成耗时。
最终版完整融合算子通过 CANN 设备编译。真机完整图 A/B 尚待运行。
