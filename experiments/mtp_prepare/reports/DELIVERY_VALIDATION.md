# 交付包验证记录

核验日期：2026-11-02。此报告区分脚本/源码资格、已重跑的 simulator 用例和计划矩阵。

## 已完成

- 7 个版本，每版包含完整 operator 的11个必要文件；没有 vLLM/KVAF 运行依赖。
- 38 个 CPU fixture 全部成功构造，并通过 CPU reference 的 map/count/descriptor 不变量。
- 计划矩阵为7×38=266格；正常/extended reference 显式固定，不以首个候选充当基线。
- `--prepare-only` 已实际生成 V0/V2、uniform/heterogeneous/mixed/B33 输入和哈希。
- 使用包内独立脚本复验历史 P2-low 7个用例，全部完整 oracle PASS。
- 包内标准库 unittest 包括逐 active 检查、失败排除、输入篡改、源文件篡改、tar确定性、
  生成产物排除和 KVIP stage 保护等回归，21项全部通过；Ruff与格式检查通过。
- 逐版本对比固定 Git 来源，7×3份 kernel 文件与 `variants.json` 指定提交字节一致。

## KVIP 提交兼容性

在 KVIP `611ffbfa8cde157a135bd520249dc32a4167d40e` 的临时独立 clone 中，实际执行
`package_delivery.py --stage`，目标为 `experiments/mtp_prepare/`。完整 `operators/`
文件哈希未改变，默认 native 三源码与本包V0一致。原用户 KVIP checkout 未写入。

临时 clone 上使用 Python3.12、CPU Torch2.8.0执行 `bash tools/check.sh --tier cpu`：

- 349 tests PASS，0 skip；
- 设备 ABI13、528字节 descriptor、74 offsets匹配；
- 83份受保护文件保留基线语义，44组 host输出一致；
- source-baseline gate与Ruff通过。

最初系统Python3.9不满足KVIP的Python3.10+要求，已改用独立临时Python3.12环境完成
上述验收，没有修改用户全局Python环境。实验包可提交为源码研究目录；该结果不等于
已通过生产kernel替换验收。上传方式见 `../docs/KVIP_UPLOAD.md`。

## 新 harness 的真实 CANN smoke

容器：`quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12`。
镜像摘要：`sha256:556cd79602b2bd97b1d19652432601b9a18e04133a70731c395d44929fc21b8b`。
两版都完成独立编译和 simulator 执行，均验证 `samples/0/` 的全部输出。

| Case | Version | warmup/active | Wrapper device cycles | Oracle | 相对同轮 V0 |
|---|---|---:|---:|---|---:|
| B1/Q4/M300 | v0_base | 0/1 | 61,913 | PASS | 0.00% |
| B1/Q4/M300 | v2_low | 0/1 | 53,920 | PASS | 下降12.91% |

原始摘要和源码/输入/harness hash 见 `smoke_20261102/`；可视化在本地被忽略的
`.artifacts/smoke-20261102/traces/`。相对值用61,913作为同轮分母。
新增 harness 有 profile 开关读取和逐 active 输出采集，代码生成与历史 harness 有差异，
因此不能把本轮53,920直接与历史53,497混算。历史完整矩阵保留在
`all_results_vs_base.csv`；当前 smoke 仅为交付脚本的编译/执行验收，active=1不足以
证明稳定的生产收益。

另外实际编译、运行了 `v2_low / B1/Q4/M1 / active=2 / profile_enabled=0`：两个
active输出均完整oracle PASS，均保存9个输出文件；该格没有性能排名。摘要见
`profile_off_active2_20261102/`，验证了关闭profile和逐次状态恢复/快照路径。

## 全量执行与后续资格

266格新 simulator 矩阵尚未全量执行。执行入口已交付：

```bash
python3 tools/compare_kernels.py --docker --warmup 1 --active 3 --jobs 1 \
  --output .artifacts/full
```

V0/V2的extended已知失败仍应原样记录；不能为使总状态变绿而删掉这些格。只有完整
oracle PASS 的记录进入排名，基线失败时百分比留空。Graph padding、alias/invalid
layout 的 admission 语义需要专门 oracle，目前不属于这38个正常输入用例。

此包输出 `network=false`、`device_CQ_executed=false`，不能用于证明真实A5时延、
KVAF网络、NPUGraph或vLLM端到端性能。KVIP提交资格与生产启用资格分别验收。
