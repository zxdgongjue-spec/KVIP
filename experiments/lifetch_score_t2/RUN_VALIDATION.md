# 20 机复跑：原 history7 与完整五 batch 融合链

当前实验机为 **a5-20**；之前的 **a5-12** 不参与这次运行。下面的性能命令在已经建好的实验容器内执行，宿主机只负责更新共享 checkout。脚本自身不执行 Docker、SSH 或联网下载。

上次只测到 B1/B2，是那次启动选择了 `--batches 1,2 --rounds 1`：三条路线 × 两个 batch × r0/r1 = 12 个任务。脚本支持 B1、B2、B4、B8、B16。新组合入口默认跑完整五个 batch、miss 5/10/15/20/25/30%，保留三个测量轮；r0 丢弃。

原表中的 `history_parts` 是固定 B1 的 history 分核数。它扫描 P1/2/4/8/16/32/30，不是在扫描 batch。组合入口先复跑这张表，再串行运行完整融合实验；两种实验有独立 fixture、manifest 和 summary。

| 实验 | 固定形状/路线 | 测量范围 | 时间统计 |
|---|---|---|---|
| 原 MR14 history7 | B1/Q4/H131072/G32/D128/TopK2048，P1/2/4/8/16/32/30 | score；score + 必要 scatter + 原全局 TopK | 每轮 p50 的均值、同轮 T1 比值 |
| 当前融合链 | 五个 batch × 六个 miss，`mr23,window_budget,aligned_budget` | LI + 完成的 Resolve/Get + 输出/Index | 同一测量图 event mean、同图阶段拆分 |

history7 的 Final 始终为 4 AIV，全部只有一次 score launch。P30 采用 H128 分片粒度，26 核处理 4352 token、4 核处理 4480 token。完整融合实验的 `aligned_budget` 是 MR23 融合流程同步 MR14 全 history 分核方案；`window_budget` 是窗口候选。串行参考仍为 SDK LI + LIDU + 完成的 BatchGet（SerialBefore）；这里没有单独测量 MR14 SerialAfter。

默认预取为 0，这是 `aligned_budget` 的执行约束。要测提前预取，应另跑不包含 aligned 的窗口对照，保留独立结果目录。

本轮脚本通过代码仓的 `codex/lifetch-score-t2` 分支同步。在宿主机更新现有候选 checkout，同时获取独立 history7 所需的 MR14 历史对象；随后进入已有容器：

```bash
git -C /home/w00568585/li-aligned-a5/KVIP fetch origin \
  codex/lifetch-score-t2 codex/split-core-unified-comparison
git -C /home/w00568585/li-aligned-a5/KVIP merge --ff-only \
  origin/codex/lifetch-score-t2
git -C /home/w00568585/li-aligned-a5/KVIP log -1 --oneline
docker exec -it lifetch-aligned-a5 bash
```

容器内，使用已构建的 pinned KVAF。命令不依赖之前 Shell 中的 `LI_WORK` 或 `PYTHON` 变量；默认从 KVAF 的 `build-npu/CMakeCache.txt` 解析实际 CANN 和 SoC：

```bash
cd /home/w00568585/li-aligned-a5/KVIP
bash experiments/lifetch_score_t2/run_validation_a5.sh \
  --device 0 \
  --kvaf-root /home/w00568585/li-aligned-a5/KVAF-score-t2-1c131038 \
  --kvaf-server 127.0.0.1:28985 \
  --batches 1,2,4,8,16 \
  --miss-percent 5,10,15,20,25,30 \
  --rounds 3 --warmup 4 --iterations 20
```

`--device 0` 指容器逻辑卡号。使用与当前 pinned KVAF 构建相同的 CANN/SoC；需要覆盖时传 `--cann-root PATH --soc NAME`。脚本可读取 `/usr/local/dcmi/npu-smi`，因此已挂载 dcmi 而镜像 PATH 中缺少 npu-smi 时无需额外复制文件。完整融合实验需要既有 KVAF server；原 history7 不连接 server。

组合入口优先使用 `--mr14-source PATH` 指定的干净固定 checkout；默认从本地 Git 对象导出 `c684d79f3a4f8432bf0915edeba392c7d54ba289` 到新结果目录的 `mr14-source`。它为这份快照建立独立的本地 Git 元数据，供原 history build 记录准确 source HEAD；不会切换用户 checkout，也不会在容器联网。目标提交不存在时，应先在可联网宿主机更新 MR14 分支。

原 history7 单独复跑以及不初始化硬件的完整计划：

```bash
bash experiments/lifetch_score_t2/run_validation_a5.sh \
  --only history7 --device 0 \
  --kvaf-root /home/w00568585/li-aligned-a5/KVAF-score-t2-1c131038 \
  --profile

bash experiments/lifetch_score_t2/run_validation_a5.sh \
  --kvaf-root /home/w00568585/li-aligned-a5/KVAF-score-t2-1c131038 \
  --dry-run
```

`--only chain` 则只补跑五 batch 融合链。`--profile` 是独立 history7 msprof 采集，原固定 warmup4/iterations5，不混入 bench 统计。组合入口的公共 `--rounds/--warmup/--iterations` 传给两种 bench，默认与原表相同。

已有 KVAF 版本不同或尚未构建时，可以在同一命令加入 `--prepare-kvaf`，把 `--kvaf-root` 改为原 KVAF checkout，并传 `--pinned-kvaf-root` 指定一个独立目录。原 checkout 保留；目标提交和 UMDK headers 必须已经在本机。初次构建没有 cache 时必须显式传实际 `--cann-root/--soc`；`--jobs 2` 控制依赖及 history 编译。

默认共 28 个 history benchmark case（7 点 × r0+r1+r2+r3），再跑 60 个融合任务（三路线 × 五 batch × 四轮）。每个融合任务内部遍历六个 miss，最终融合 summary 为 90 行（三路线 × 五 batch × 六 miss），不是只有 60 个 miss 用例。任何阶段失败都会停止后续阶段，并在组合 manifest 标为 FAILED，保留已经完成的独立结果。

打印的 `lifetch_validation_*` 目录中：

- `manifest.json`、`commands.json`：实际参数、执行顺序与成功/失败阶段。
- `history7-plan.json`：原七点几何；`history7/results/summary.csv`：原表 score/LI p50。
- `history7/msprof/`：可选的独立 profiler 结果。
- `chain/summary.csv`：完整融合收益；`chain/jobs.json`：全部五 batch 任务。
- `history-*.log`、`chain-run.log`：阶段进度和失败诊断。

不要把独立 history 的 p50 填进整链 SerialAfter，也不要将单阶段 p50 相加代替完整融合测量。原表复现值和 12 机旧结果可能受实际 SoC、CANN、频率与共享设备负载影响；这次比较应以 20 机同次复跑的配对结果为准。本地验证覆盖计划与启动契约，未在本机运行 A5 性能测量。
