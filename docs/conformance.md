# 契约符合性、独立验证和 3.0 迁移

本文件区分源代码/CPU 证明与硬件资格。完整 CPU 测试不表示已执行 NPU kernel、
真实 KVAF、服务模型或性能测试。现行规范为 public_contract.md；design_reference
保留历史设计与当时审计，历史缺口状态不作为当前发布认证。

## 3.0 变更和边界

包 0.3.0、host 契约 3.0.0；device ABI13、Graph transport ABI1、writer ABI2 不变。
2.0 的 owner 不能直接包装新 PreparedSession。需要增加 owner 本地 RLock、active/
closed 状态、单调 lease generation、acquired stream、唯一 session binding，并遵守
release/close 后置条件。GraphFrameSession 新增 current_stream 回调，quarantine
接收构造时的 keepalive tuple。框架旧 Graph 编排保留，另用共同状态场景验证。

OwnedIndexCache 的 NPU 构造和 table refresh 应提供可信 CPU page_ids。grow 只允许
追加页，等长也检查旧前缀；页面替换必须 checked rebind。restore 后对象退休。
按 request identity 关联 host 页快照，不能通过共享 data_ptr 或普通请求 generation
认证页面不变。框架 metadata hook 仅读取既有 CPU BlockTable，不做设备回读；其 CPU
快照和锁开销需纳入真实端到端性能验收，不能凭 kernel 源码不变称零开销。

## 覆盖矩阵

| 契约环节 | 当前可执行证据 | 仍需硬件验证 |
| --- | --- | --- |
| C02/C03 版本、布局、ABI | 能力组协商、公开符号清单、C++ 全字段偏移、旧 major 拒绝 | SDK/runtime/已构建库实际组合 |
| C05/C06 admission、buffer | batch/candidate/budget/tail/pool 边界，dtype/shape/alias/cache 重叠 | device 动态长度/页表/目录 |
| C07/C08 选择和结果 | Python oracle；32/64 heads 原值、中性填充、weighted ReLU 数值等价 | 实际 LI/resolve/extra/映射/payload |
| C09/C11 session | 延迟完成、唯一绑定、关闭/换代旧句柄、stream、异常隔离 | SQ/CQ、真实 DMA 可见性和资源释放 |
| C10 host launcher | 真实 ContractLaunch、实际 Tensor/schema、编译 C++ shim、ctypes 四阶段绑定 | shim 不计算 kernel 或通信 |
| C12 OwnedIndex | 非法 grow 页替换、合法追加、rebind hot/cold/tail、非法页无写入、部分失败 poison、restore 退休 | allocator 真实交接与跨层并发 |
| C13 Graph | 独立帧状态、回调资源留存、框架 CPU 边界接线 | 固定指针、捕获/replay、device guard、bucket 并发 |
| C14 consumer | 严格九参 fake、零/满前缀、多目录、末行、非法调用不写入、hit/tail 保护 | 真 KVAF、跨 rank、网络故障 |
| C16/C17 发布 | 分级无 skip 门槛、基线子树归一化、安装包来源、尾延迟门槛 | NPU memory、KVAF、模型 accuracy/TPOT |

主要测试在 tests/v9/test_public_contract.py、test_inputs.py、test_consumer_fake.py、
test_launcher_binding.py、test_public_session.py、test_owned_boundary.py、test_graph_session.py。
框架用 tests/ut/dsa_offload/test_v9_module_integration.py 实际导入 backend/graph/config/
contracts，使用显式 CPU 宿主替身；它不加载完整 serving/NPU 环境。其余历史 AST UT
继续验证函数逻辑，不能用数量替代端到端覆盖。

## 两侧独立运行

```bash
python -m pip install '.[cpu-test]'
bash tools/check.sh --tier cpu
python tools/run_conformance.py --tier host
```

check.sh 在独立 Git 仓根目录和框架 third_party/KVOP 下均可运行。它分别执行 pytest、
C++ ABI 和保护源码/44 个 oracle 基线比较。source-baseline 需要对应 Git 历史；源码
快照可显式指定 `--baseline-directory`，缺历史不得静默跳过。纯 host 入口主动阻断 Torch，
预先选择不依赖 Torch 的子集；cpu 入口要求 CPU Torch 和 C++ compiler，关键 suite
缺失或任何 skip 均失败。NPU 测试另按 hardware_qualification.md 执行。

安装包测试应在仓库之外进行：

```bash
python -m pip wheel --no-deps --no-build-isolation . -w dist
# 在独立环境安装 wheel，然后从外部目录执行，清除指向源仓的 PYTHONPATH。
python /path/to/KVOP/tools/run_conformance.py --tier cpu --root /path/to/KVOP --provider installed
python /path/to/KVOP/tools/verify_baseline.py --scope installed-package
```

installed-package scope 仅验证已安装公开元数据和打包源 hash；完整测试仍需要外部 tests
树。不能把这个小检查当完整 CPU 契约验收。完整入口报告实际 provider 路径并拒绝回源。
框架接线测试可用已安装的 KVOP，不需要把独立仓源码放到 PYTHONPATH。

## 验收状态

manifests/validation.json 记录本次实际命令结果；硬件未完成时 compatibility.json 的
hardware_certified 始终为 false。任何 CPU 或 synthetic 性能门槛单测都不能改变认证。
性能记录需要真实日志、功能报告及 SHA256，门控 TPOT 和 selection 的 median/p95/p99；
证据格式见 hardware_qualification.md。脚本不能证明证据真实性或统计显著性。
