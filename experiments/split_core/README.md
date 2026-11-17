# 当前 score / staged 验证

生产路由和调用见 [接入说明](../../docs/research/split_core/staged_integration.md)。本目录保留当前实现的独立数学 oracle、分核覆盖/预算测试、完整 tail 的受控 FP8 fixture 与 Model driver。历史源码副本、独立 TopK、native矩阵、trace/adaptive 工具和生成结果可从 `5b006c6` 恢复。

CPU 检查（使用带 NumPy、Torch 的 Python）：

```sh
PYTHONPATH=python python3 -m unittest discover -s experiments/split_core/tests -p 'test_score*.py' -v
PYTHONPATH=python python3 -m unittest discover -s experiments/split_core/tests -p 'test_staged*.py' -v
PYTHONPATH=python python3 -m pytest tests/runtime/test_independent_schedule.py tests/runtime/test_lifetch_binding.py tests/runtime/test_lifetch_kernel_contracts.py
python3 experiments/split_core/run_staged_routes.py plan --output tests/results/staged-plan
```

`cases.py` 固定 H128K/Q4，B=1/2/4/8/16/24/32，miss=3/5/10/15/20/50%，共42条件。计划与CPU通过不代表42组设备通过。独立 score wrapper 采用256-token切片；生产与完整 staged driver 采用128-token切片。

CANN9.1 Model 准备与构建：

```sh
python3 experiments/split_core/prepare_staged_source.py --output build/staged-source --model
cmake -S experiments/split_core/staged_build -B build/staged-model -DRUN_MODE=sim -DSTAGED_SOURCE="$PWD/build/staged-source" -DKVAF_INCLUDE=/path/to/patched/kvaf/include -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
cmake --build build/staged-model --target staged_routes -j2
python3 experiments/split_core/verify_staged_metadata.py --auto-gen build/staged-model/auto_gen/tail
```

匹配Model运行环境中的 `LD_PRELOAD` 应包含 `libruntime_camodel.so`。单点运行：

```sh
python3 experiments/split_core/run_staged_routes.py run --driver build/staged-model/staged_routes --case b1-h100-m10 --route history --output tests/results/staged-b1 --lease-file /shared/kvip-model.lease
```

多个容器必须挂载同一独占租约路径。输出包含fixture哈希、完整 score/θ/Final/Index/payload/guard/completion oracle及正常 Model footer，保存在 `tests/results/`。Model 使用同步GM-copy final BatchGet adapter、prefetch0；不能证明 URMA/CQE/DMA、真实NPU或性能交叉点。score-only driver 由 `prepare_score_source.py` 与 `score_build/` 准备，运行入口为 `run_score_routes.py`。

历史r5三项功能Model与CANN构建结果在[精简摘要](../../docs/research/split_core/results/score-integration-validation.json)中绑定原源码版本；原始失败记录与日志不改写。

## 生产 score 控制与 Model 周期

当前完整driver默认调用生产`v9_staged_score`；`STAGED_SCORE_WRAPPER=ON`才使用独立wrapper，此项仅适合组件研究。driver兼容旧11参数，并接受七个附加参数：

```text
staged_routes FIXTURE OUTPUT B H R C V MISS ROUTE DEVICE \
  WINDOWS POST PREPARE SERVICE P_HOT P_COLD FETCH_WORKERS
```

POST为separate/scatter/direct，PREPARE为wave/invocation/phase，SERVICE为separate/inline；ROUTE另支持完整legacy控制。legacy需要默认post/prepare/service和Ph/Pc=0。每种配置保留全score/θ/Final/Index/payload/guard/completion验证；legacy使用原融合约定，不要求落盘全量canonical。两个共同Model-only clock kernel夹住forward，排除上传和oracle回读，输出raw clock及`forward_model_cycles`。必须确认正常footer、完整oracle和计时有效后才能比较。

clock的scalar GM写入后显式clean/invalidate所属cache line。旧probe曾出现设备SYS_CNT非零、Host回读全零，不能采用那些周期；修复已由独立十次probe的serial/parsim4正常结束验证。probe通过不代表score或整链通过。完整driver在计时外逐字节核对全部Cold bitmap，即使prefetch0时未使用的predicate也验收；失败先写`failure.txt`和已取得的`clock.json`，防止异常清理隐藏原始断言。forward周期包含两个共同probe之间的成本，未扣除估计开销。

CANN9.1设备源发生变化时使用**新的build目录**；本地观察到增量流程会二次link已转换的device object，报unknown file type。保留失败日志，重新准备快照、fresh build，不以旧device library的成功替代新源码编译。

`tests/operators/lifetch/cold_parallel/run_model_controls.py`在已有隔离容器中串行运行生产score完整控制。约定repo→`/repo`只读，指定work→`/work`可写，KVAF include只读；work中准备`source-v2/staged-source.json`、`build/staged_routes`、`fixture-smoke`和`fixture-full`。fixture分别使用H8192/R6400与H131072/R16384，miss10/heads32；可用`staged_routes.fixture`生成。manifest记录binary/library/source/fixture哈希及Model配置，共用宿主全局Model租约。

```bash
python3 tests/operators/lifetch/cold_parallel/run_model_controls.py plan \
  --matrix full --container YOUR_MODEL_CONTAINER \
  --work tests/results/YOUR_MODEL_WORK --output tests/results/YOUR_MODEL_WORK/plan

python3 tests/operators/lifetch/cold_parallel/run_model_controls.py run \
  --matrix full --container YOUR_MODEL_CONTAINER \
  --work tests/results/YOUR_MODEL_WORK --output tests/results/YOUR_MODEL_WORK/full-controls \
  --case legacy_W2 --case staged_before_W2 --case inline_auto_W2 \
  --case inline_Ph4_Pc8_W2 --case inline_Ph26_Pc28_W2 --timeout 7200
```

`--case`重复选点，未选项保留NOT_RUN；`--source-manifest`/`--driver`可调整work内路径。W8另比较staged_before_W8、inline_auto_W8、inline_Ph4_Pc8_W8与inline_Ph26_Pc26_W8。Model仅prefetch0/同步GM final Fetch，H8192 smoke不满足legacy sequence-split长history门禁；只有H128K完整控制能验证该门禁下P2。宿主耗时不是设备耗时，Model周期不是A5/URMA微秒数。实际Model结果和配置见[本轮报告](../../docs/research/split_core/optimization_exploration_20261117.md)。

应用预算可用`--aic-budget 32 --aiv-budget 64`覆盖，必须不超过Model配置和实际launch需求；默认full预算28/56。`--host-relink-receipt`保存host重链接收据路径。接受结果要求唯一正常footer、完整oracle、正确route/fixture/Cold字节数，且`0 < start < end <= whole_model_ticks`、`forward_model_cycles=end-start`。timeout保留TIMEOUT与原日志，未执行项保留NOT_RUN，不将缺测计为通过。
