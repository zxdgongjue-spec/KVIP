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
