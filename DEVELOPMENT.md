# KVIP 开发、同步和独立 Demo

远端为 https://gitcode.com/CacheOS/KVIP.git；Python import 为 `kvip`，distribution
为 `vllm-ascend-kvip`。包版本 0.4.0、公开契约 3.0.0、设备 ABI13。
现行代码/目录已统一名称，历史 Git 提交与提取基线保留，不重写历史。

## 独立仓开发

在 Python 3.10+ 环境安装一次，完整 CPU 测试还需要支持 FP8 类型的 CPU Torch 和 C++ compiler：

```bash
git clone https://gitcode.com/CacheOS/KVIP.git
cd KVIP
python -m pip install --editable '.[cpu-test]'
bash tools/check.sh --tier cpu
python tools/run_demo.py --profile host --output results/demo-host
```

editable 让后续 Python 测试/新进程使用当前源码；已有 worker 需正常重启。
改 metadata/dependencies 后重装，改 AscendC 仍需重新编译。日常联调不需手工发布 wheel。
纯 host 使用 `python tools/run_conformance.py --tier host`，它明确排除 Torch 依赖。
完整 CPU 入口拒绝关键 suite 缺失和任何 skip。

## 框架并列子模块

```text
vLLM-DSA/
  third_party/KVIP/  # gitlink 固定一个完整提交
  third_party/KVAF/  # 独立 gitlink，不嵌套到 KVIP
```

框架首次 `git clone --recurse-submodules <framework-url>`，进入框架后执行：

```bash
python tools/operator_workspace.py setup
python tools/check_v9_framework_contract.py --provider checkout --kvip-root third_party/KVIP
```

setup 初始化 gitlink 指定的两份源码并 editable 安装 KVIP，KVAF 使用自己的 CMake 构建，
不能对 KVAF 使用 pip editable。工具只设置本仓 `submodule.recurse=true`，不修改全局 Git。
后续统一更新命令：

```bash
python tools/operator_workspace.py pull
# 已完成父仓 git pull，只需同步其固定提交：
python tools/operator_workspace.py sync
# 已缓存全部对象的离线同步：
python tools/operator_workspace.py sync --offline
```

pull 只允许 clean 仓和 fast-forward，sync 拒绝丢弃子模块本地修改/未跟踪文件；不做
stash/reset/clean，也不使用 --remote 跟随分支头。新 clone 的普通 git pull 不会替你
完成 Python 安装，先运行 setup；metadata 改变时再运行 setup。默认不要浅克隆，历史
source-baseline 验收需要提取基线对象。

联调独立的相邻 checkout 可将其 editable 安装到同一开发环境，再传该绝对路径。
有意使用脏源码时显式加 `--allow-dirty`，结果只标记 development_only，不能当发布资格。
`--provider installed` 继续用于验证正式非 editable 安装产物。

## 独立 NPU Demo 与外部 KVAF

KVIP 自身不拉取 KVAF。可使用框架的并列 KVAF，或单独 clone 官方仓并固定相同提交：

```bash
git clone https://gitcode.com/CacheOS/KVAF.git /path/to/KVAF
git -C /path/to/KVAF checkout --detach 7c54bd5f9d49fb78943a80ed398764053f52e7d8
python -m kvip.v9.kvaf_compatibility --root /path/to/KVAF --profile v9-eager \
  --expected-revision 7c54bd5f9d49fb78943a80ed398764053f52e7d8
python tools/run_demo.py --profile kvaf --kvaf-root /path/to/KVAF \
  --expected-kvaf-revision 7c54bd5f9d49fb78943a80ed398764053f52e7d8 \
  --kvaf-server 127.0.0.1:28999 --dry-run
```

dry-run 只给源文件/导出和命令证据，不运行 NPU/服务。实际执行需要匹配的 CANN、SoC、
设备与 KVAF CPU 服务；分别 `bash build.sh cpu` / `bash build.sh npu` 构建 KVAF，
NPU binding 在同一 checkout 的 build-npu/python。将设备参数改为服务器实际值后去掉
--dry-run；--profile npu-memory 只运行设备本地内存传输，不能证明真实网络性能。

独立 demo 的 --execution graph 使用固定 output 的普通 NPUGraph；它不是框架的
GraphFrame 多层注册协议。官方固定提交有 V9 device headers/eager lease，但缺 frame
四接口。框架用既有 `patches/kvaf/v9_graph_context.patch` 在 build/kvaf-sources 的哈希
副本中构建桥接，不直接修改 KVAF 子模块；记录 base SHA+patch SHA，并在构建后验证
实际 runtime 导出。源码与 CPU fixture 通过都不能代替上板验收。

## 更新与发布

在 KVIP 自己的开发分支修改、独立测试并 push，再由框架显式更新 gitlink 并提交。
主仓拉取新提交后只会使用该 gitlink，KVIP/KVAF 远端分支继续前进不会让框架自动漂移。
两个子模块分别更新；框架不保留第二份移动分支或 SHA 锁文件。
发布 CI 可构建 wheel 并做 installed gate，它是发布验收，不是开发联调前置条件。
硬件功能/性能资格继续以 manifests 和 hardware_qualification.md 为准。
