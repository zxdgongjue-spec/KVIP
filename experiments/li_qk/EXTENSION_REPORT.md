# LI K32 / B16 30 核扩展验证报告

日期：2026-11-17。执行要求先保存在 `EXTENSION_PROMPT.md`，随后实现和验证。此报告对应独立数字 LI 控制实验，固定每 owner Q4/G32/D128/TopK2048，不执行 Fetch、Service、predicate、Index 或 CQ。T1 使用同一 QLI 模板，是实验参考，不能称为 SDK QLI 或库上生产 P2。

## 实现与控制组

原七点保留为 `orthogonal7`；`orthogonal8` 只增加 B1/T32/Qparts1，每 task Q4×H4096，单次 score launch，要求 32 AIC/64 AIV。没有增加 Q/K 交叉网格。

B16 套件固定 H131072、Q4，总 score 预算 30 AIC/60 AIV，包含三个同输入控制组：

| 控制组 | 逻辑 task | 实际有效片段 | score launches | scatter launches | Final TopK AIV | scratch |
|---|---:|---:|---:|---:|---:|---:|
| T1 waves | 16×H128K | 16 | 1 | 0，canonical alias | 64 | 16 MiB |
| T2 waves | 32×H64K | 32 | 2，30+2 | 2 | 64 | 16 MiB |
| T2 balanced | 32 个逻辑 half | 60 | 1，每核独立消费片段 | 2 | 64 | 30 MiB |

balanced 按 2048-key 块展开：16×128K/2048=1024 块，26 核各34块、4核各35块；每核累计 H69632/H71680，工作量差最多一块。遇到 owner/half 边界保留独立输入绑定及输出区间，形成每核两个片段。代码使用 slot-major 绑定表，30 个 MIX1:2 核组在一个设备 kernel 内独立前进。

这里保留的是每 owner **两个逻辑 half**。为了在30核上均衡，它们可再切分到不同物理核；不能声称每 owner 固定独占两个物理核。16×2=32 个核组无法同时独占30核。Final TopK 单独使用64 AIV，因此整个 LI 的峰值 AIV 数没有限制为60。

计时保持两个范围：`score_p50_us` 为全部 score launches；`li_p50_us` 为 score→必要 canonical scatter→相同全局 TopK。B16 每轮按 B16/T1 配对，warmup4、iterations20，丢弃 r0，随后三轮交错。profile 独立运行，不混入 benchmark summary。

## 验证结果与实际范围

环境：专用 CANN 9.1.0 / Ascend950PR_9599 Model 容器，仿真拓扑32 AIC/64 AIV，L2 cache disabled。NPU fresh build 同样通过。没有物理 A5 访问，本轮没有真实 msprof trace 或 A5 微秒数据；服务器应在自己的 CANN 环境重新构建。

| B16 实验 | H | 执行与正确性 | whole Model ticks | 说明 |
|---|---:|---|---:|---|
| T1 waves，完整 LI | 8192 | PASS | 54092 | N=1 smoke |
| T2 waves，完整 LI | 8192 | PASS | 76716 | N=1 smoke |
| T2 balanced，修复后完整 LI | 8192 | PASS | 65013 | 与上两组相同 fixture |
| T2 balanced，独立 scatter | 131072 | PASS | 174318 | GM 预填 oracle；两次 scatter |
| T2 balanced，独立 TopK | 131072 | PASS | 50651 | GM 预填 oracle；64 AIV |
| T2 waves，独立 score | 131072 | TIMEOUT，exit124 | 未获得 | 600秒，只有部分日志 |

这些 ticks 是 Model 从开始到停止的完整运行计数，不能当成精确 kernel 时间，也不能换算 A5 微秒或报告生产加速比例。独立消费者证明全尺寸寻址和算法正确性，不能证明完整 H128K producer、consumer 重叠或整 LI 加速。H8K 下均衡粒度为每核2/3块，与 H128K 的34/35块不同。

实际 host/stub、planner 和 fixture 六项 CPU 检查通过；shell 编排七项检查通过，包括新增套件的预算转交和 profile 独立输出。H8K 与 H128K 的 C++/Python task 图一致，owner/history 唯一覆盖；B16 oracle 改为逐 owner 生成，小尺寸全部 SHA 与原生成器一致。成功 Model case 验证 canonical score SHA、TopK 阈值及合法性、guard、输入只读，退出码0且有正常 Model footer。最新 C++/header 与 fresh NPU build 的 SHA 一致；Python 与 Bash 语法检查通过。

完整 H128K score、全核动态计数、B1/K32 设备正确性与性能、物理 A5 msprof 仍待执行。超时的部分 score 日志不得补齐或按比例外推为完整实测。尝试 PV/cmodel capability fallback 在初始化时 exit139，没有执行 kernel，未当作验证结果。

## 仿真发现并修复的同步错误

初版 balanced H8K 在 owner5/Q2 首个后半段出现 canonical mismatch，退出失败；这份结果不用于收益比较。

原 QLI `ProcessMain` 结束时仅 drain AIV0 的两个 VC credits（0/1）。单次 launch 内重用 QLI 时，AIV1 的16/17 credits 可能残留，较快 AIV 又提前初始化并重新发布，造成下一片段读取/覆盖顺序错误。实验 balanced 入口增加片段边界 fence：AIC 额外 drain16/17，再发布6/22边界 flag；两个 AIV 等待边界后才进入下一片段。修复后的完整 H8K LI 全部通过。普通 score 入口及生产 QLI 源码行为未改。

## 动态指令数量

日志 mask=`0x1`，只记录 physical core0 的 AIC 与两个 AIV。下面按**该 kernel、该 core 的全部已解析动态 issue 事件**做分母；两个 AIV 数据相同，表中仅列 AIV0。AIV 分母包含 scalar/PUSHQ 与展开后的 RVEC issue 事件。跨 pipeline 的 ID 可以乱序，ID下降不用于划分 kernel。

`map_model_pcs.py` 根据实际入口指令与 ELF symbol/.text 范围推导加载基址，并核验 captured build。独立 consumer 的加载地址与完整链不同，不能沿用完整链的硬编码 PC。原始日志、入口/ELF/SHA 证明、逐 opcode CSV 和 unmapped 事件均保留。

| case / kernel / core | issued events | vector math | UB load/store | scalar/address | vector loop/control | flow | sync | data movement | dispatch | Cube | Fixpipe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H8K T2 balanced score / AIC0 | 12705 | — | — | 87.52% | — | 6.62% | 4.06% | 1.04% | — | 0.25% | 0.50% |
| H8K T2 balanced score / AIV0 | 24098 | 47.81% | 22.98% | 20.86% | 4.66% | 1.88% | 1.10% | 0.16% | 0.55% | — | — |
| H128K scatter / AIV0 | 3756 | — | — | 85.73% | — | 3.04% | 7.40% | 3.73% | 0.11% | — | — |
| H128K TopK / AIV0 | 23164 | 47.90% | 31.26% | 8.04% | 10.81% | 0.84% | 0.38% | 0.08% | 0.69% | — | — |

H8K 三控制组的 core0 score 指令：T1 AIC24444/AIV46425；T2 waves AIC25112/AIV47374；修复后的 balanced AIC12705/AIV24098。T1、T2 waves 的 core0 累计处理 H8K，balanced core0 只处理 H4K；**单core数量减少不能解释为全设备指令减少**。balanced 全局初始化从32次增加到60次，收益目标是缩短关键路径与填满核组，可能伴随总指令增加。

可识别的流程与 opcode 证据：

| kernel / 流程 | 源码 | 代表性动态 opcode（单kernel/core分母） |
|---|---|---|
| score AIC：初始化、输入寻址/搬运、QK MMAD、Fixpipe、CV/VC handshake | `score.cpp`、`indexer/indexer.h`、`indexer_cube.h` | H8K balanced：LD_XD_XN_IMM 13.75%，MOV_XD_IMM 11.48%；MMAD 归 Cube 类 |
| score AIV：QK加载、ReLU、weight/scale乘加、G归约、sortable score 写出 | `indexer_vector.h`、`indexer_score.h` | H8K balanced：RV_VLD 21.25%，RV_VRELU 17.00%，RV_VMULA 17.00% |
| scatter：task元数据、owner/Q/history地址计算、GM→UB→GM、事件同步 | `post.cpp` 的 `li_qk_scatter` | H128K：ADD 16.93%，SHL 11.02%，LD_XD_XN_IMM 9.96% |
| Final TopK：完整history分块扫描、直方图选择、压紧、保留TopK与下一块合并、写indices | `post.cpp`、`indexer_vector.h` 的 `ProcessTopK`、`topk.h` | H128K：RV_VLD 16.65%；RV_VSQZ、RV_VSTUR各10.08%；RV_CHISTv2 9.81% |

此表是源码流程解释与 opcode 证据，不是算法子函数的耗时占比。子函数可能内联、共享指令，尚无可靠逐算法 PC/活跃周期映射，相关耗时保留 `NOT_AVAILABLE`。MMAD数量少可对应大量算术；一个等待指令也可能阻塞很久。不能由 scalar 87% 断言 scalar 是时间瓶颈，也不能由 sync数量1%排除等待瓶颈。

## B16/T2 的当前瓶颈与后续方向

1. **普通30+2调度的尾波可由 task 图确认。** 计算量线性近似下，T1关键路径为H128K；T2 waves是H64K+H64K，且第二波只有2核。它没有自然得到二倍缩短，还增加一次 score 启动、两次 scatter及初始化。均衡版每核总H≤71680，理想计算长度约为T1的54.7%；这只是忽略带宽、初始化和同步的调度下界分析。
2. **balanced 的 score 均衡没有自动传到 scatter。** score 两片段在各核内部独立推进；scatter仍按 slot 两次全局启动，第一/第二slot最大长度都可能H64K。`max(first)+max(second)`会丢掉各核互补的长度优势。H128K消费者仿真能确认现有两launch布局，不能单凭whole ticks确定它在A5上的占时。
3. **初始化与GM中转仍存在。** 每片段重建 QLI/TPipe并做 drain/fence；全部 score 输出16MiB，scatter至少额外读取16MiB、写16MiB，balanced padding使scratch分配30MiB。B16 key逻辑请求载荷仍256MiB；实际DRAM流量、缓存和带宽须msprof验证。
4. **Final TopK没有随T2缩短单Q扫描。** 已按64个真实Q分配64AIV，每核仍扫描完整H128K。score变快后它可能成为更显著的尾部，但当前不能凭指令数确定其真机占时。

优先真机比较三控制组，查看 score 的每核区间与尾波，再查看两次 scatter 和 TopK 的关键路径。若 scatter 明显，应增加独立控制组：同核片段列表的 persistent scatter，或 score producer 直接写真实 canonical。后者必须显式区分虚拟输入row和真实owner/Q/history输出寻址，不能简单关闭scoreOnly。复用初始化/预打包元数据可作为后续优化；本轮不混入这些新变量。

## A5 运行与产物

宿主机更新已确认的共享挂载，只同步源码，不调用容器 CLI：

```bash
bash update_li_qk_shared.sh
# Git代理不可用时：加 --no-proxy；完全离线时：
# bash update_li_qk_shared.sh --offline-bundle /path/to/li_qk_mr14_offline.bundle
```

随后在原A5容器、原Python/CANN环境依次执行；两个命令不要同时跑：

```bash
bash /home/w00568585/li_qk_shared/run_in_container.sh --suite orthogonal8 --profile
bash /home/w00568585/li_qk_shared/run_in_container.sh --suite batch16 --profile
```

默认device5，参数路径不同时加 `--kvaf-include PATH` / `--cann-root PATH`。八点使用32/64；B16自动30/60，Final64。常规结果在 `results/summary.csv`；`msprof/` 包含 B1/K32 或 B16 三控制组的原始trace、export、完整命令和oracle。单独换硬件指标可用 `profile.py --metrics Memory` 或 `ResourceConflictRatio`，每次新输出目录。

本地证据根为 `tests/results/li-qk-extension-20261117/`：成功三组smoke、全尺寸consumer、超时/失败日志、verified-kernel-instructions及构建receipt均保留。下载包只收源码bundle、更新器、报告和必要日志/动态指令压缩文件，不打包大fixture或可跨CANN复用的二进制。**本次提供的动态Model日志不是msprof trace；真实trace需要上述A5命令生成。**
