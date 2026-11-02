# KVIP 提交与上传说明

核对日期：2026-11-02。交付对象为普通 Git 源码目录 `experiments/mtp_prepare/`，
附带可独立解压、校验和执行的 tar.gz。此目录保留多个候选用于实验，不切换默认算子。

## 已核实的仓库

- 正式独立仓：[CacheOS/KVIP](https://gitcode.com/CacheOS/KVIP)。KVOP 是历史命名。
- 本地独立 checkout：`/Users/iantargaryen/Projects/KVIP`。
- origin fetch/push：`https://gitcode.com/CacheOS/KVIP.git`。
- 本地 clean main 和当日 `git ls-remote --heads origin` 均为
  `611ffbfa8cde157a135bd520249dc32a4167d40e`。工作期间远端从初次核验的
  `cc1cc75a20169f9445f7847cd65279ef5072cb3a` 前进到此提交，最终以此为准。
- 框架的 `third_party/KVIP` 为 Git mode `160000` 的 submodule；其 pin 应独立核对，
  不会自动跟随 KVIP main。
- 最新 KVIP 的 native 三个 kernel 源码与本包 `variants/v0_base` 相同；本工具 staging
  前会重新逐文件比较，避免以旧基线替换新实现。
- origin 为 CacheOS 组织仓。用户授权提交后，通过GitCode官方Fork API建立了真实个人
  fork `gcw_7iY8JFPv/KVIP`；本轮使用 `myfork` 推送开发分支，保留组织仓main。

以下机制来自 KVIP 当前源码：`DEVELOPMENT.md` 第85行起要求先在独立 KVIP
开发分支测试、push，再由框架显式更新 gitlink；`build.sh` 的 `native-lidu`
入口固定构建 `operators/native_lidu`。本包各版本必须使用独立构建目录和进程，避免
同一 Torch 进程重复注册同名算子。

## 为什么放 experiments

KVIP 的 `tools/verify_baseline.py` 当前保护整个 `operators/` 及部分 ABI/oracle
文件，并拒绝设备源字节差异。因此把优化候选复制到 `operators/` 的新子目录也会使
现有 source-baseline gate 失败。`experiments/mtp_prepare/` 可保持默认设备和契约
不变，直接作为研究候选提交。正式更换默认 kernel 时，需要单独提交实现、正确性和
性能资格方案，不能通过删除 baseline 检查取得通过。

当前 `MANIFEST.in` 不收录 `experiments/`；本方案会进入 Git clone，但不会自动进入
普通 KVIP wheel/sdist。需要源码发布时使用本工具生成的 tar.gz；若要求进入 sdist，
应在单独审查中显式扩展 MANIFEST。生成的 simulator build、原始 trace、dump 和大型
产物留在 `.artifacts/` 或 `dist/`，不上传 Git。

## 打包、校验、预检和导出

从本包根目录执行：

```bash
python3 tools/package_delivery.py --refresh-manifest --check
python3 tools/package_delivery.py --archive dist/mtp_prepare_sources.tar.gz
python3 tools/package_delivery.py --kvip-root /path/to/KVIP
python3 tools/package_delivery.py --kvip-root /path/to/KVIP --stage
```

`--kvip-root` 默认只打印计划；只有显式 `--stage` 才新增
`experiments/mtp_prepare/`。目标已存在即退出，不覆盖。工具会验证 package 文件全集与
SHA256、每版本11个必要源码、无符号链接依赖、当前 native 基线一致，并验证 staging
前后整个 `operators/` 的文件哈希不变。不会执行 git add、commit 或 push。

tar.gz 有固定 `mtp_prepare/` 前缀、时间戳和权限，附同名 `.sha256`，只包含清单文件，
可在无 vLLM checkout、无历史 Git 对象的机器解压使用。清单生成后修改任意源码都会使
`--check` 失败；确认修改后再显式 refresh。

清单和 archive 均排除 `.artifacts/`、`artifacts/`、`dist/`、`build/`、`build_*/`、
`OPPROF_*/`、`dump/`、`traces/`、Python 缓存和二进制产物
（`.so/.dylib/.o/.a/.bin/.whl`），本地构建后 refresh 不会把这些文件纳入源码包。

导出到临时 KVIP clone 后应运行包内 unittest、`compare_kernels.py --plan` 和 KVIP
既有 `bash tools/check.sh --tier cpu`。Linux/CANN 的 simulator 全量执行命令见本包
README；host 单测和 matrix plan 不证明 kernel 已重新仿真。输出性能表仅排名完整 oracle
通过的记录，`>2048` 采用明确的 P0 reference，不能伪造 V0 extended 分母。

## 审查后上传

遵循本任务 AGENTS 的要求，使用个人 fork 开发分支，再向 CacheOS/KVIP 提 MR。
本轮已经确认的个人fork为 `https://gitcode.com/gcw_7iY8JFPv/KVIP.git`。

```bash
git -C /path/to/KVIP switch -c codex/mtp-prepare-kernel-variants
# 若尚无myfork远端，使用已经确认存在的个人fork。
git -C /path/to/KVIP remote add myfork https://gitcode.com/gcw_7iY8JFPv/KVIP.git
git -C /path/to/KVIP add experiments/mtp_prepare
git -C /path/to/KVIP diff --cached --check
git -C /path/to/KVIP diff --cached --name-only
git -C /path/to/KVIP commit -s -m "perf(mtp): archive kernel candidates and comparison harness"
git -C /path/to/KVIP push -u myfork codex/mtp-prepare-kernel-variants
```

MR 描述包含：候选职责、逐文件 provenance、源码清单、完整 simulator 结果和失败项、
现有 KVIP CPU gate 结果、默认 `operators/` 未改的证据，以及真机资格状态。
此操作只提交研究目录；合入后框架使用 `git submodule update --init --recursive`
取得固定版本。若框架需要使用新提交，另行提交 `third_party/KVIP` gitlink 变更，
再执行框架的 `python tools/operator_workspace.py setup` 和框架契约验收。

本工具本身不执行commit/push；实际提交和推送由已授权的Git步骤完成。
开发分支固定为 `codex/mtp-prepare-kernel-variants`，A5机器应检出该分支并记录实测SHA。
