# score_t2 候选正确性排查（2026-11-19）

MR23 原始基线与 `score_t2_p2` 是不同实现。`run_a5.sh` 的 r0 先执行精确 MR23 控制组；任何失败都会立即停止后续 job。用户日志进入 `r0_pf1024_score_t2_p2` 后出现 B1/B16 FAIL，说明该次 MR23 job 已通过，不能把候选错误归到 MR23 原实现。默认入口保留，候选由 `--li-score-parts 2/16` 显式启用。

失败发生在 TopK 正确性检查，不是编译或 KVAF teardown。Q1 的参考 cutoff 为 -988，候选 selected_min 为 -1208/-1200，遗漏约 200 个高分 token。日志给出的遗漏 token 全部在 resident 16384 之外，保留的低分 token 在 resident 内，与 Cold 候选未完整进入 Final TopK 一致。这不证明整个 score 矩阵错误，也不能单凭该日志区分分类、进度或增量 TopK 的具体原因。

原审计把 GM 的 sortable BF16 uint16 key 和 synthetic 的数值分数直接比较，错误报告全 H 的 `score_mismatches`。例如 key 15088 解码为 -2288，15093 为 -2208，32768 为 +0，实际上与样例参考相同。现在审计先解码到数值口径，另记录 saved resident θ、Cold bitmap 置位数量、与 GM score/θ 的逐 token 对照。实际 TopK 验证标准没有放宽。

`ProcessPrecomputedVec1` 新增 GM→UB DMA 后，原先只等待 MTE3→V。该等待限制 V 流水线，不能阻止独立 MTE2 流水线覆盖 V 仍读取的 ping-pong UB bank。修复增加平衡的 V→MTE2 Set/Wait，再搬入 score，保留后续 MTE2→V 和 V→MTE3 发布。仅改变 opt-in 的 precomputed Cold 方法，原 MR23 score/classify 方法保持原样。

## 独立验证入口

```bash
cd /home/w00568585/wt-lifetch-score-t2
bash experiments/lifetch_score_t2/run_predicate.sh \
  --device 5 --kvaf-root /home/w00568585/KVAF-score-t2-1c131038
```

该用例运行真实 `ProcessPrecomputedVec1`，固定 B1/Q4/H131072/R16384/T2，逐项检查 458752 个 Cold predicate、每 Q 不同 θ、等于 θ 的严格排除、连续多 tile/ping-pong 复用、输入和 guard。无 Service/TopK/Fetch，不需要 KVAF server；使用其 device headers。每次独立 fresh build。日志为 `configure.log`、`build.log`、`check.log`。本地 CANN Model 可加 `--mode sim --device 0`。

该用例只能证明 Cold 消费组件的分类与发布，不替代完整 A5 B1/B16 的正确性验证。仍需完整 `run_a5.sh` PASS 后才能比较性能；之前的 r0 FAIL 数字不是候选性能结论。

## H128K Model 结果

本地 CANN 9.1 的上述同输入对照已经完成：修复前 252370/458752 个 Cold predicate 错误；增加 V→MTE2 同步后 0 个错误，strict greater-than ties、输入和 guard 检查 PASS。详见 `predicate_model_result.json`，保存输入模式、源文件及日志 SHA256。整次 Model 为 271721 ticks，不是 A5 延迟，也不作为单 kernel 精确 cycles。编译 SoC 为 Ascend950PR_9599，Model 返回 Ascend950PR_9589；完整目标 A5 路线仍需设备验证。
