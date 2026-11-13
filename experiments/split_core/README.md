# 分核实验与用例执行

当前集成与限制见 [研究入口](../../docs/research/split_core/README.md)。这里区分当前核心的组件验证、真实设备入口和81b1472历史实验；旧百分比不用于当前流式main。

## 当前42组：冒烟与客户用例

固定**sequence=131072（128K）、每request Q=4**；BS=1/2/4/8/16/24/32，miss目标=3/5/10/15/20/50%，共42组。small/medium/large为12/12/18组。smoke与customer使用相同42组完整清单，不自动抽成少量冒烟；区别仅在用途与结果标签。默认G32、T1、V56，每组Model执行一次，不做预热。

旧60组（history30/50/100%、最大BS16）结果保持归档，不能作为这42组的通过证据。

### 生成、预览、CPU验证

```bash
python3 scripts/generate_split_core_sim.py
KVIP_ACTION=plan bash scripts/split_core_sim/smoke.sh
KVIP_ACTION=cpu bash scripts/split_core_sim/customer.sh
```

生成物为`cases.json`、`smoke.sh`、`customer.sh`及可选`compare.sh`。`plan`只写42行NOT_RUN；`cpu`校验实际FP8 Q/K、score与精确miss，不产生设备时延。CPU fixture为一组Q4种子；Model driver将其复制到每个request，不代表异构客户请求分布。

### 上库前完整冒烟 / 客户仿真

在匹配的Linux CANN Model环境执行：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
bash scripts/split_core_sim/smoke.sh
bash scripts/split_core_sim/customer.sh
# 客户矩阵使用T8，仍为42组：
KVIP_TOPK_SHARDS=8 bash scripts/split_core_sim/customer.sh
# 可选同输入策略/预算对照：42 × T1/T4/T8 × V56/V64 = 252条件
bash scripts/split_core_sim/compare.sh
```

两个主入口各完整遍历42组。默认自动从当前源码构建`topk_sim`，先执行`--capabilities`核对Q4及BS32支持；旧BS16 driver必须重新构建。可用`KVIP_MODEL_DRIVER`指定已经重建的driver、`KVIP_BUILD_DIR`指定独立构建目录、`KVIP_SOC`指定Model匹配SoC、`KVIP_CANN_ARCH`指定SDK平台目录。`KVIP_OUTPUT`必须是全新目录，默认使用UTC时间及PID创建独立结果路径。

只执行一个场景用于调试时，其他41行明确保留NOT_RUN：

```bash
KVIP_OUTPUT=tests/results/debug-b32-m50 bash scripts/split_core_sim/customer.sh --case b32-h100-m50
```

直接Python入口支持`--suite smoke|customer`、`--group small|medium|large|all`、`--shards 1|4|8`、`--budget 28/56|32/64`。每行显式记录Q、T、预算、heads、名义/实际miss、warmup=0、iterations=1和driver SHA。`--warmup/--iterations`只用于native，不会让Model自动重复。

脚本逐case保留FAILED并继续遍历，整轮存在失败时返回非零；不能只凭exit0标PASS，必须同时有唯一正常Model footer与完整local/global oracle、guard及输入未改检查。`compare.sh`也会保留其他配置的执行机会，最后汇总失败退出码；各配置结果分目录保存，不将252次冒充42次。

同一仿真资源必须独占。默认使用`/tmp/kvip-cann91-simulator.lease`，不抢占旧锁；跨容器时应将同一个共享租约路径bind mount，并通过`KVIP_SIM_LEASE_FILE`（直接Python为`--lease-file`）指定。容器各自的`/tmp`不能互斥。不得并发执行smoke/customer/compare或其他Model实验。

### 实测边界

这里仅测**AIV-only精确TopK**，AIC工作数为0；不含QK、完整LI、Service或native transport，也不是生产P32/T8。BS24/32只扩展该组件的query任务轮转，未放宽当前生产streaming的BS16上限。

每query整数miss依次为61/102/205/307/410/1024，实际百分比为`100m/2048`。受控重叠U=2m；50%对应每request唯一冷token2048，恰到当前LRU容量边界。这不代表任意客户50% miss都有相同重叠，也不表示prefetch预算足以覆盖全部miss。

## 当前流式main：真实设备入口

仅供以后装有Torch NPU、KVAF服务和匹配CANN的设备使用；本轮没有运行此入口。当前native完整streaming只支持BS≤16，选择BS24/32会在创建结果/fixture/租约前拒绝；不能用native的`--shards`声称启用T8。`--soc`应填写目标设备实际值。`--q-workers`是Fetch worker，和TopK分片T不同。

```bash
python scripts/run_split_core_cases.py run --backend native --transport ubmem --soc Ascend950PR_9599 --server 127.0.0.1:28880 --group small --budget 32/64 --output tests/results/native-small
python scripts/run_split_core_cases.py run --backend native --transport ubmem --soc Ascend950PR_9599 --server 127.0.0.1:28880 --group medium --budget 28/56 --output tests/results/native-medium
python scripts/run_split_core_cases.py run --backend native --transport urma --soc Ascend950PR_9599 --server 127.0.0.1:28885 --group large --case b16-h100-m10 --budget 28/56 --output tests/results/native-large
```

上面large示例仅选BS16/miss10%一项，其余17行保留NOT_RUN；当前native不能整组运行含BS24/32的新large清单。

原有`test_lifetch.py`保留完整数值/Index/payload检查；结果必须生成本次checked-output.npz，父脚本另验shape、范围、唯一性与hash。依赖目录可用`--kvaf-root`、`--cann`显式指定。未运行的配置保持NOT_RUN；没有NPU时不能用CPU检查替代此验收。

## 历史35次实验：离线核对与源码重建

```bash
python experiments/split_core/verify_results.py
python experiments/split_core/prepare_historical.py --output tests/results/historical-source
cmake -S experiments/split_core/historical -B build/historical \
  -DWORK_ROOT="$PWD/tests/results/historical-source" \
  -DKVAF_INCLUDE="$PWD/third_party/KVAF/access_operator/demo/aiv_batch_get/include" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build build/historical --target history_pool_sim -j2
# lateTopK使用独立构建目录，避免本SDK在多goal增量重建时重入预处理。
cmake -S experiments/split_core/historical -B build/historical-full \
  -DHISTORICAL_VARIANT=full -DWORK_ROOT="$PWD/tests/results/historical-source" \
  -DKVAF_INCLUDE="$PWD/third_party/KVAF/access_operator/demo/aiv_batch_get/include" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build build/historical-full --target full_pool_sim -j2
export KVIP_FIXTURE_DIR="$PWD/tests/results/historical-source/standard-fixture"
```

准备器固定Git对象81b1472，校验完整overlay/harness源码清单，生成与原标准fixture一致的输入；KVAF应按该提交build.py的prepare-only流程准备到固定commit与patch状态。所有device archives从源码编译，不依赖旧机器的.a或driver。须有包含该Git对象的完整clone，不能用无历史zip替代。

例如在上述Model环境中，以不同新目录分别运行同binary对照：

```bash
mkdir -p tests/results/replay-original tests/results/replay-p32t8 tests/results/replay-s4
KVCORE_HISTORY_SHARDS=0 build/historical/history_pool_sim tests/results/replay-original standard 1 32
KVCORE_HISTORY_SHARDS=32 KVCORE_TOPK_SHARDS=8 KVCORE_C=32 KVCORE_V=64 build/historical/history_pool_sim tests/results/replay-p32t8 standard 1 32
KVCORE_POOL_SHARDS=4 KVCORE_V=64 build/historical-full/full_pool_sim tests/results/replay-s4 standard 1 32
```

P28/T8改为P=28、C=28、V=56；smoke使用`zero 1 32`与P2/T1。每次要同时核对score keys、局部候选和完整oracle，局部检查可用`historical/history_oracle.py`。重建路径和链接过程可能改变binary哈希及Model计数，必须建立重建后的同binary基线；不能强行匹配归档ticks。离线核对只验证导出日志和表格一致性，大型原始tensor不在MR中。

重放history后，可直接检查全部score相同、局部候选与全局严格赢家：

```bash
python experiments/split_core/check_history_replay.py --baseline tests/results/replay-original --candidate tests/results/replay-p32t8
```

## 串行/Fusion3/分核统一表

`python3 experiments/split_core/compare_pipelines.py`核对9条历史全链记录、各自对照和1条新main NOT_RUN行。`--write --plot`重建CSV与图（绘图需要matplotlib）。详见[可比范围与结果](../../docs/research/split_core/pipeline-comparison.md)。导入的Fusion3 capture不是本MR新执行的Model，单次whole-Model计数不换算物理时间。
