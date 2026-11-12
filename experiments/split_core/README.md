# 分核实验与用例执行

当前集成与限制见 [研究入口](../../docs/research/split_core/README.md)。这里区分当前核心的组件验证、真实设备入口和81b1472历史实验；旧百分比不用于当前流式main。

## 60个基础场景

BS1/2为small（24组），BS4/8为medium（24组），BS16为large（12组）；miss为3/5/10/15%，sequence为128K的30/50/100%。有效长度39322对应存储39424；另外两档为65536和131072。预算、G32/64、算法和重复次数另算展开维度。

```bash
python scripts/run_split_core_cases.py plan --group all --output tests/results/split-plan
python scripts/run_split_core_cases.py cpu --group all --output tests/results/split-cpu
```

CPU入口生成真实FP8 Q/K，不通过交换预计算score伪造miss；每query整数miss为61/102/205/307，当前受控重叠U=2m。它验证fixture/数学语义，不能产生设备时延。结果明确区分NOT_RUN、PASSED_CPU、PASSED_MODEL、PASSED_NPU和FAILED。`--case`可只执行指定行，其余行仍保留NOT_RUN。

## 当前核心AIV组件：CANN Model

Linux环境需要匹配CANN9.1、Ascend950编译器和Model库；本仓不提供运行库或自动安装系统依赖。

```bash
cmake -S csrc/operators/split_topk -B build/split-topk \
  -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build build/split-topk -j2
export LD_PRELOAD="$ASCEND_HOME_PATH/aarch64-linux/simulator/dav_3510/lib/libruntime_camodel.so"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/aarch64-linux/simulator/dav_3510/lib:$ASCEND_HOME_PATH/aarch64-linux/lib64:$LD_LIBRARY_PATH"
python scripts/run_split_core_cases.py run --backend model --group small --budget 32/64 --shards 8 --model-driver build/split-topk/topk_sim --output tests/results/split-small
python scripts/run_split_core_cases.py run --backend model --group medium --budget 28/56 --shards 4 --model-driver build/split-topk/topk_sim --output tests/results/split-medium
python scripts/run_split_core_cases.py run --backend model --group large --budget 28/56 --shards 1 --model-driver build/split-topk/topk_sim --output tests/results/split-large
```

脚本有排他租约，不抢占其他任务；超时结束自己的进程组。Model必须有唯一正常footer及host oracle，不能仅凭exit0标PASS。该组件只测精确TopK，AIC工作数为0，不包含QK、Service或native transport。完整query/range且每片至少K是当前模块限制；空/短range不能直接调用。history P/T gather实现已收敛到共享核心文件，但尚未接替新流式Final阶段。

## 当前流式main：真实设备入口

仅供以后装有Torch NPU、KVAF服务和匹配CANN的设备使用；本轮没有运行此入口。`--soc`应填写目标设备实际值。`--q-workers`是Fetch worker，和TopK分片T不同。

```bash
python scripts/run_split_core_cases.py run --backend native --transport ubmem --soc Ascend950PR_9599 --server 127.0.0.1:28880 --group small --budget 32/64 --output tests/results/native-small
python scripts/run_split_core_cases.py run --backend native --transport ubmem --soc Ascend950PR_9599 --server 127.0.0.1:28880 --group medium --budget 28/56 --output tests/results/native-medium
python scripts/run_split_core_cases.py run --backend native --transport urma --soc Ascend950PR_9599 --server 127.0.0.1:28885 --group large --budget 28/56 --output tests/results/native-large
```

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
