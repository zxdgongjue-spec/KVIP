# MTP prepare 最优 kernel 独立交付包

2026-11-02。目标落点为 KVIP 的 `experiments/mtp_prepare/`，本目录可以完整复制到仓外运行，
不需要 vLLM、KVAF、原仓 `.git` 或历史分支。包内包含完整 operator 源码、仿真 harness、
CPU oracle、全量矩阵脚本和历史对比表。生产资格仍为 `false`。

## 版本选择

| 本地版本名 | 内容 | 选择依据 |
|---|---|---|
| `v0_base` | 与最新 KVIP native 相同的 kernel 基线 | 每项优化必须保留同输入对照 |
| `v2_low` | classify/compact 融合 + output/descriptor overlap | B1/B4/B8 低 miss 当前最优，下降 13.96%/15.57%/14.31% |
| `v4_boundary` | multi-lane dedup + overlap | M2048 当前已测试最优，下降 11.84% |
| `v6_high` | V4 + row-control prefetch | M1280–M2047 当前最优，下降 11.39%–12.95% |
| `p0_extended_reference` | 独立 MTE3 cursor 提交 | `>2048` 的显式正确性参考；低 miss 会回退 |
| `p2_extended` | cursor + compact + overlap | B1/M2049 比 P4 更优 |
| `p4_extended` | cursor + multi-lane + overlap + prefetch | B1/M2081、M4096 比 P2 更优 |

前四项是历史最优候选；后三项是 extended 实验候选。V5 和已知 B4/B8 row0-only 失败的
P1 不进入此包。V2 使用已迁移到最新基线的 P2-low 完整 operator，kernel 与历史 V2
逐字节相同。`variants.json` 固定来源提交，`package_files.json` 固定实际文件 SHA256。

## 目录

```text
variants/<version>/   # 各自完整 op_host / op_kernel / ops.json / torch_extension
harness/             # CANN simulator 可执行程序；每个 active 保存完整输出
oracle/reference.py  # CPU 功能 reference；import 时不需要 NPU
cases/matrix.json    # 38 cases × 7 kernels = 266 个独立比较单元
tools/compare_kernels.py
tools/package_delivery.py
tests/               # 标准库 unittest，无需 pytest
reports/             # 历史结果；新 harness 结果需重新同输入对比
docs/KVIP_UPLOAD.md  # 已核实仓库和提交方式
docs/A5_COMPARISON_PROMPT.md  # A5代理执行的完整真机对比任务
```

## 使用

使用 Python3.10+，宿主机安装 `requirements.txt` 中的 CPU NumPy/Torch。Linux native 还需明确指定当前
CANN 9.1 路径；macOS 使用已有 CANN 9.1 Ascend950 容器。

先检查打包完整性并查看全量计划（不会启动仿真）：

```bash
python3 tools/package_delivery.py --check
python3 -m unittest discover -s tests -v
python3 tools/compare_kernels.py --plan
```

最小真实编译、运行和 oracle smoke：

```bash
python3 tools/compare_kernels.py --docker \
  --variants v0_base v2_low --cases b1_q4_m300 \
  --warmup 0 --active 1 --jobs 1 --output .artifacts/smoke
```

全量同输入矩阵（运行时间较长，可按 case/version 分片到不同 output）：

```bash
python3 tools/compare_kernels.py --docker --warmup 1 --active 3 --jobs 1 \
  --output .artifacts/full
```

Linux native 先执行安装目录的 `set_env.sh`，再用
`--native --cann /actual/cann/path` 替代 `--docker`。native 模式只检查
SDK 配置，不改 CANN 安装；容器模式在一次性容器内配置 `flush_level=2`。
`--profile-enabled 0` 运行完整正确性矩阵但不产生性能排名，便于与 profile on 对照。
`--prepare-only` 可先生成相同输入和清单，`--verify-existing PATH` 可重新审计已有结果。

每个 active 均恢复相同初始 maps/cursor/output poison，并保存、校验全部输出。每格独立
进程；每版本独立 build。检查 `miss_counts[B+1]`、destination miss-prefix/hit-suffix、
descriptor tuple、forward/reverse map、cursor、resident length 和 poison tail。
均匀/非均匀 batch、Q2/Q3/Q4、2048/2049/2080/2081、B33 sequential-wave 都进入矩阵。

所有选中的 kernel 均执行所有选中 case；失败继续记录，最终返回非零。历史 V0 的
extended cursor 失败不会被隐藏。正常 case 的 reference 为 `v0_base`；extended/mixed
case 的 reference 为 `p0_extended_reference`。reference 缺测或失败时收益留空。只有
完整 oracle PASS 且 profile 开启的结果进入排名。Graph padding、alias/invalid layout
需要额外契约 oracle，目前在 matrix 中明确列为未覆盖门禁。

每个 output 包含 source/input/binary/harness hash、环境清单、逐格日志、每 active 输出、
CSV/JSON/Markdown 总表和 Perfetto JSON。Perfetto 阶段起点由 profile duration 重建；
trace 中的 simulator_cycle 不是真实微秒。B33 总周期包含顺序波次，逐 row phase
不能作为完整尾波的独立计时。新 harness 增加 profile 开关读取，可能改变代码生成，
新旧表应分别保留，正式结论使用新 harness 内同一轮 base 对比。

## 单算子构建与提交

每版保持原完整构建入口，例如：

```bash
bash variants/v2_low/build_lidu_fused_batch_get_prepare_mtp.sh
```

需要真实匹配的 CANN、torch_npu 和 NPU 环境。各版本导出相同 Torch 算子名，比较时
使用不同进程和不同扩展产物路径。原 operator 构建脚本会向当前 Python 环境的
site-packages 写入同名 `.pth` 注册文件；运行前应激活该版本专用虚拟环境，避免不同
版本覆盖导入路径。simulator 矩阵脚本不执行这些安装步骤。

KVIP `build.sh native-lidu` 仍只构建 `operators/native_lidu`。实验包通过自己的脚本运行，
不自动切换生产路径。KVIP 保护 `operators/` 的源码基线，直接覆盖会被门禁拒绝。

```bash
python3 tools/package_delivery.py --kvip-root /path/to/KVIP
# 审阅计划后复制到尚不存在的 experiments/mtp_prepare：
python3 tools/package_delivery.py --kvip-root /path/to/KVIP --stage
python3 tools/package_delivery.py --archive dist/mtp-prepare-best-kernels-20261102.tar.gz
```

工具不会 commit/push。具体分支、验收与 MR 步骤见 [KVIP 上传说明](docs/KVIP_UPLOAD.md)。
`experiments/` 可作为 Git 源码提交；现有 KVIP wheel/sdist 默认不收录该目录，本包采用
独立 tar.gz 分发。源码包不包含 build、运行输入/输出、OPPROF 或 trace；运行产物位于
被忽略的 `.artifacts/`，可另行分享。

此处结果仅支持同 simulator 的 device-cycle 比较。`network=false`、
`device_CQ_executed=false`；真机 A5、KVAF/CQ、NPUGraph 和 vLLM 端到端性能仍需后续验收。

本次已完成的实际验证和266格计划的执行边界见
[交付验证记录](reports/DELIVERY_VALIDATION.md)。

A5真机执行使用 [A5对比测试Prompt](docs/A5_COMPARISON_PROMPT.md)。它要求在A5上先补齐
独立硬件runner，再执行正确性与profiler矩阵；现有 `--native` 仍是simulator模式。
