# A5 真机多版本 MTP prepare 对比执行 Prompt

将下面的完整任务交给 A5 机器上的执行代理。工作目录是 KVIP 仓库中的
`experiments/mtp_prepare/`；先固定用户给出的提交或开发分支，再开始执行。

本包当前已经提供 simulator 矩阵和完整 standalone operator，但尚未提供全量 A5
硬件矩阵 runner。`tools/compare_kernels.py --native` 仍然以 `RUN_MODE=sim` 编译，
即使在 A5 服务器运行也不是真机测量。本任务要求执行代理在 A5 端实现独立硬件
runner，先验收它，再采集真实数据。

## 可直接复制的任务

你是一名熟悉 Ascend950/A5、AscendC、CANN、torch_npu 和多核同步的性能工程师。
请在当前 A5 机器上，对 KVIP 的 `experiments/mtp_prepare/` 中7个版本执行真实 NPU
正确性和同输入性能对比。先读取本目录 README、variants.json、package_files.json、
cases/matrix.json 和本 Prompt，然后完成以下全部工作。

### 1. 固定代码、环境与测量范围

记录当前 Git 完整 SHA、分支、clean/dirty 状态、包清单 SHA256，以及实际机器的
SoC、可用 AIV 数、驱动/固件、CANN、Python、Torch、torch_npu、设备编号和设备占用。
使用已有匹配环境，不猜测 CANN 路径，不在服务器上自动升级驱动、固件或运行时。
根据实际环境执行对应 `set_env.sh`，确认目标是支持此算子的 Ascend950/A5。

先执行以下仓内已有命令：

```bash
cd /actual/path/to/KVIP/experiments/mtp_prepare
python3 tools/package_delivery.py --check
python3 -m unittest discover -s tests -v
python3 tools/compare_kernels.py --plan
```

这三项只验证源码、脚本和计划。不得把它们的成功，或 simulator 的成功，记录成
真实 NPU 功能/性能通过。

本次只测 native `A5LiduFusedBatchGetPrepareMtp`。不要求启动 KVAF 服务，也不要求
部署 vLLM。KVIP 的 `docs/hardware_qualification.md` 和 `qualify_performance.py`
用于真实 KVAF/V9/模型端到端资格，包含 transport、selection、feedback、lifecycle、
eager/Graph 等条件；不能把这些条件强行套到本轮 native-only 对比，也不能以本轮
结果解除那些生产资格门禁。

### 2. 独立构建每个版本，证明实际加载来源

版本为：`v0_base`、`v2_low`、`v4_boundary`、`v6_high`、
`p0_extended_reference`、`p2_extended`、`p4_extended`。

各版本导出相同 Torch namespace/op 名，构建脚本还会向当前 Python 环境写入同名
`.pth`。因此每个版本使用专用 Python 虚拟环境、独立构建产物目录和独立进程，不能
在同一进程连续加载多个版本。可在已配置好 Torch/torch_npu 的 A5 Python 环境上
建立 `--system-site-packages` 虚拟环境，继承依赖但把该版本 `.pth` 写入自己的环境。

以下是包内确实存在的 standalone 构建入口；先对 V0/V2 执行，成功后推广到所有版本：

```bash
# 从 experiments/mtp_prepare 执行；当前 shell 已激活匹配的 CANN 环境。
python3 -m venv --system-site-packages .artifacts/a5/venvs/v0_base
NANOVLLM_A5_OPS_PYTHON="$PWD/.artifacts/a5/venvs/v0_base/bin/python" \
  bash variants/v0_base/build_lidu_fused_batch_get_prepare_mtp.sh

python3 -m venv --system-site-packages .artifacts/a5/venvs/v2_low
NANOVLLM_A5_OPS_PYTHON="$PWD/.artifacts/a5/venvs/v2_low/bin/python" \
  bash variants/v2_low/build_lidu_fused_batch_get_prepare_mtp.sh
```

先确认目标虚拟环境不存在或是本任务自己的可复用环境；不覆盖用户其他环境。构建
日志中记录实际 Python、CANN、SoC、命令和错误；保留每版生成的 device kernel、
`libcust_opapi.so` 和 Torch `_C.so` 的路径及 SHA256。不要修改任何 variant 的源码，
不要覆盖生产 `operators/native_lidu`，不要变更全局 Python 导入注册。

特别检查 `oracle/reference.py::load_npu_extension()`：它会先尝试注册
`vllm_ascend`，成功后可能返回框架已经加载的旧算子。因此全量硬件 runner 必须直接
从所选 variant 的 `torch_extension` 绝对路径加载 standalone 扩展，不能盲用该 loader。
加载后断言以下三者都落在当前 variant 目录内：

- `extension.__file__`；
- `extension._C.__file__`；
- `extension._OPAPI` 指向的 `libcust_opapi.so`。

记录 `ASCEND_CUSTOM_OPP_PATH` 实际采用的 vendor 路径。若发现同名算子已被其他来源
注册，终止该进程并修复隔离，不得继续测量后标成选定版本。

### 3. 实现独立 A5 runner，复用准确的功能契约

在 `experiments/mtp_prepare/tools/` 新增真实硬件入口，例如
`compare_kernels_a5.py`，并补充可在 CPU 执行的计划、参数、加载路径、基线选择和
结果汇总测试。此文件当前尚不存在；下文所有该名称的命令均为建议接口，必须先实现。

复用 `oracle/reference.py` 的 `build_case`、`reference_update`、`check_case` 以及
相关 map/descriptor 检查，不修改 golden oracle 以迁就失败结果。加载参考模块后，
由硬件 runner 显式把已经验证来源的 standalone `extension` 传入检查函数。
不要调用会优先尝试框架注册的自动 loader。

原参考脚本的参数真实存在：

```text
--mode check|bench|all
--device npu:0
--batch-size B
--query-counts Q1 Q2 ...
--miss-counts M1 M2 ...
--source-capacity 131072
--cache-tokens 12288
--seed 20261029
--layer-id 2
--num-layers 4
--profile-detail
```

但 `reference.py --mode bench` 当前是 NPU Event 计时，不是本任务要求的 profiler
主测量入口。其 CLI 默认 seed 为 `20260818`，而包内矩阵默认 seed 为 `20261029`；
必须显式读取每个 case 的 seed，不能依赖 CLI 默认值。

硬件 runner 应提供：`--plan`、`--execute`、`--variants`、`--cases`、`--case-file`、
`--device`、`--variant-python-root`、`--output`、`--profile-enabled` 和重复轮次参数。
CPU 计划模式不得 import torch_npu 或触发 NPU 初始化。执行时每个版本/用例使用
独立 worker，失败仍继续其他格，并最终返回非零。

硬件 runner 新增后记录它自身及测试的 SHA256、Git dirty 状态和原包清单 SHA256。
如刷新 `package_files.json` 纳入新增工具，应同时保留原清单与新清单，不能把新增的
A5 代码伪装成原始提交已验证的内容。数据和构建产物始终保留在 `.artifacts/`。

### 4. 先做真实 NPU smoke，再跑全矩阵

先运行 `v0_base` 与 `v2_low` 的 B1/Q4/M300，同输入、同 seed；确认两者独立加载、
完整 oracle 和 profiler 文件均通过，随后运行7版本×38case。以下接口须先由上一节
实现，不能假设包中已经提供：

```bash
# 待实现的 A5 runner 接口示例，不是现有 simulator-native 命令。
python3 tools/compare_kernels_a5.py --plan
python3 tools/compare_kernels_a5.py --execute --device npu:0 \
  --variant-python-root .artifacts/a5/venvs \
  --variants v0_base v2_low --cases b1_q4_m300 \
  --output .artifacts/a5/smoke
python3 tools/compare_kernels_a5.py --execute --device npu:0 \
  --variant-python-root .artifacts/a5/venvs \
  --output .artifacts/a5/full
```

所有版本共享同一份 CPU 输入、输入二进制哈希和层配置。保留正常/异构 query count、
非均匀 miss、2048/2049/2080/2081、mixed paths 和 B33。

真实 host tiling 通过 `GetCoreNumAiv()` 查询核心数，不应照搬 simulator 的32核假设。
保留 B33 用例，并在资源和 ABI 容量允许时额外增加 `B=实际AIV数+1` 的用例，覆盖
真实 sequential-wave。记录每格实际 AIV 数、used core 数和 cores-per-request；
不能仅看到 B33 通过就声称已经覆盖真实顺序波次。

Graph padding、duplicate pool-row、alias 和 invalid layout 目前需要额外 admission
oracle。这38个正常输入 fixture 不构成这些契约的完整覆盖；单列未覆盖项。

### 5. 每次执行恢复状态，逐 active 验证

每次 warmup 和 active 前恢复相同的初始 `token_to_slot_pool`、`slot_to_token_pool`
及 `eviction_cursors`，重置 destination、miss counts、descriptor poison 和 profiling
buffer。不能把第一次 miss 后更新过的 map 直接用于后续计时，使后续迭代变成 hit。

每个 active 均保存并校验完整输出，不能只检查最后一次。检查至少包括：

- `miss_counts[B+1]` 的每行和总数；
- destination 的 miss-prefix 与 hit-suffix；
- descriptor tuple 集合及有效前缀；
- forward/reverse map、逆映射、唯一 slot 分配；
- eviction cursor 和 resident sequence length；
- 未写入 descriptor tail 的 poison；
- 失败状态所要求的未更新 map/cursor 和事务行为。

`profile_enabled=0` 与 `1` 均做正确性和重复执行验证，区分 profiling 的固定开销。
不把数据恢复、H2D/D2H、CPU oracle 或输出采集计入目标 kernel-only 区间。禁止在
kernel-only热路径为了日志频繁 `.item()` 或同步回读。

### 6. 使用真机 profiler 作为主要性能证据

使用 `torch_npu.profiler`，每个 case/variant 独立 profile，固定
`wait=0, warmup=5, active=5, repeat=1`，每步调用 `prof.step()`。
至少进行3轮独立、交错顺序的比较，记录所有轮次与预热边界，不挑选最快一轮。
保持相同设备、流、输入、环境、预热和采样配置，记录温度/频率/占用等可读状态。

从原始 `op_statistic.csv` 和 kernel detail 确认真正执行的融合算子名、task 类型和
调用次数，汇总该算子的全部对应 device task。若某版本存在多个实现 task，应按
同一次公开算子调用归组；匹配不到目标算子或计数不符时，该格性能证据失败。
以实测记录验证分母，不用默认猜测5次就必然有5条相同 kernel 行。

分别报告：

1. 目标融合算子的平均 device us，作为主要对比指标；
2. 包含 reset copy/其他 device 操作的整步设备时间，单列其组成；
3. NPU Event 的辅助延迟分布，每轮预热后至少100次、至少3轮，给出
   p50/p95/p99及全部原始样本。Event不能替代profiler主结论。

若 profiler 的 active 期间也采集到了 reset/copy/D2H，必须保留原始文件，并明确
目标算子筛选规则。不能把整个采样窗口的全部耗时直接冒充融合算子执行时间。
保存可由 Perfetto 打开的真实 profiler trace；与 simulator 按 duration 重建的阶段图
分别命名、分别解释。profile开关影响和采样开销单列。

### 7. 显式基线、失败与最终报告

任一 request 的 unique miss 大于2048时，使用同case的
`p0_extended_reference` 作为 extended 正确性/性能参考；其他case使用 `v0_base`。
reference缺测或oracle失败时，收益百分比留空。不能选首个候选充当base，也不能用
历史V0的错误extended输出作为有效性能分母。

所有选中版本执行所有选中case，历史V0/V2等版本的extended失败也保留在表中。
错误输出即使更快也不得排名。每格记录 build/runtime/oracle/profile 的分别状态，
不能把编译失败、超时或未执行记为0 cycles或0 us。若要修复kernel，先保留失败证据，
以单独补丁/版本名记录，不静默修改候选后沿用原始SHA。

输出 `.artifacts/a5/<本次运行目录>/` 下的：

```text
environment.json
source_input_library_hashes.json
commands_and_logs/
correctness.json
raw_profiler/
traces/
results.csv
results.json
FINAL_REPORT.md
```

总表至少包含 variant、case、dtype、B/Q/unique miss、实际 core geometry、profile
开关、oracle状态、目标kernel平均us、Event辅助p50/p95/p99、reference、下降百分比、
失败原因和原始证据路径。给出低miss、高miss、extended、mixed、sequential-wave
分组下的最优版本，以及是否仍支持 V2/V6 分桶；不要把不同工作量的case互相比快慢。

历史 simulator cycle 表与真机 us 表并列保留，不能把 cycles 直接换算为真机us。
本轮 native-only 测量不证明真实KVAF网络、CQ、NPUGraph或vLLM端到端收益，不自动
修改 production 资格、部署配置、生产默认kernel或框架submodule pin。

完成后先展示本次实际执行/通过/失败/未执行数量、关键对照表和主要结论，再提供
报告与原始trace位置。遇到机器环境或依赖问题，报告具体证据并继续可完成的独立检查，
不得补造性能数据。
