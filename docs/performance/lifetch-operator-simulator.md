# LI / 输出发布组合优化

本 MR 基于公开 `main@81b1472`，覆盖更新原 MR !4，保持 Draft，未执行合入。
范围为 URMA LIFetch 算子、编译开关、源体契约测试和本文。保留当前 main 的
UBmem transport、共享 resolver、KVAF lease、公开 ABI 和 launch 顺序。

## 组合实现

- LI resident predicate：保留在 MaskReg，执行严格 GT 和 causal visibility AND，
  消除中间 UB mask 的存取。URMA CMake 显式启用；共享header缺省0，UBmem
  继续原 predicate 路径。
- ScaleTiles8：KScale双bank从16-tile缩到8-tile，每个AIV释放8KiB UB。分配、
  清零、PA/非PA拷贝、窗口编号与消费偏移统一；最后消费者或尾块完成后归还credit。
  候选完整声明64KiB L0B双槽范围，MMAD几何不变。共享header缺省0，只有URMA
  CMake默认启用8，UBmem不隐式启用。
- PublishOutputsBulk：MTE2加载TopK/slot，SIMT执行logical间接查询，MTE3发布输出。
  双24KiB bank复用原Update scratch，让下一query的加载/计算有机会覆盖上一query
  的输出搬运。复用前等待所属bank，尾部drain两bank后才让planner复用scratch。
  inactive行不访问poison unpack/owner源，输出完整写-1。

原SIMT、串行per-query、packed路径均保留。Plan/Apply算法、transport调度和现有
transport入口保持main原样。没有移植私仓stage-trace/诊断ABI、实验Q/K UB中转、
O1–O6变体、服务variant接口或大型trace归档。

## CMake 配置与真机验证

每组须使用独立构建目录和独立进程，避免同名共享库符号混用。Scale=0表示历史16。

| 配置 | REGISTER_PREDICATE | OUTPUT_BULK | LI_SCALE_TILES | OUTPUT_PINGPONG |
| --- | ---: | ---: | ---: | ---: |
| 原始对照 | 0 | 0 | 0 | 0 |
| 原MR LI/F1 | 1 | 1 | 0 | 0 |
| 仅增加Scale8 | 1 | 1 | 8 | 0 |
| 仅增加双bank | 1 | 1 | 0 | 1 |
| **本MR默认组合** | **1** | **1** | **8** | **1** |
| packed候选 | 0 | 2 | 0 | 0 |

完整变量名分别为 `KVIP_OPT_LI_REGISTER_PREDICATE`、`KVIP_OPT_OUTPUT_BULK`、
`KVIP_LI_SCALE_TILES`、`KVIP_OPT_OUTPUT_PINGPONG`。PINGPONG只允许与mode1组合；
切mode0或mode2时，必须显式关闭PINGPONG。

正常URMA构建及既有 `tests/operators/lifetch/test_lifetch.py --transport urma ...`
使用默认组合；现有UBmem选项、生产build cache接口不变。缓存已包含算子源码/CMake
与common headers，源码更新会触发重编译。例如独立编译原MR对照：

```bash
cmake -S csrc/operators/lightning_indexer_with_fetch -B /tmp/lifetch-li-f1 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann \
  -DKVAF_INCLUDE=/path/to/KVAF/access_operator/demo/aiv_batch_get/include \
  -DKVIP_OPT_LI_REGISTER_PREDICATE=1 -DKVIP_OPT_OUTPUT_BULK=1 \
  -DKVIP_LI_SCALE_TILES=0 -DKVIP_OPT_OUTPUT_PINGPONG=0
cmake --build /tmp/lifetch-li-f1 --parallel 2
```

真机比较须固定输入、初始Index/LRU和Fetch预算，比较两侧完整
`prototype_with_update` NPU Event样本；不能用native_chain/fused之间的speedup
代替本次消融收益。先验证CQ、输出、Index与SFA，再判性能。

## 历史软件 Model 证据

以下数字属于先前实验源码，不能当作本MR在最新main或真机上的新收益。

- 原MR在main@46c03fc、B1/Q4/H4352/R4096/heads32、CANN9.1
  Ascend950PR_9599、每方案N=1的六kernel配对中，LI0/F0→LI1/F1总跨度
  403129→355135 native ticks（降低11.9054%）。Outputs+plan降低41.1060%，
  Apply回退10.5269%，已计入总结果；不是纯发布函数单独收益。
- 后续B1/Q4/H2304/R2048/逆序PA、两次重复的LI→Outputs/Plan→Apply有限链路：

| heads | scalar/Scale16 | 双bank/Scale16 | 双bank/Scale8 | 相对scalar降低 | 相对双bank再降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 168086.5 | 105966 | 103556.5 | 38.39% | 2.27% |
| 64 | 170195.5 | 108089 | 105996 | 37.72% | 1.94% |

后者使用实际LI TopK，但slots/update payload为合成输入，未执行Resolve/Fetch。
先前移植组合另有H32/H64严格Model数值smoke通过。Model完成后原采集器收尾停滞，
被终止且保留collector=-15，不能报告采集器正常成功。本次未重测Model性能，旧38%
不外推到现网，也不外推为最新UBmem/共享代码上的收益。

## 当前验证边界

host测试包括独立predicate逻辑等价模型（不执行MicroAPI）、40组旧publisher和
56组双bank输入的实际C++源体、ASan/UBSan、UB/输出边界、事件watermark，以及
漏wait/错ID/漏drain等故障注入。当前公开main上的默认组合和原MR LI/F1回退配置
重新通过CANN9.1编译。物理A5、真实KVAF/CQ、生产大形状与持续graph replay的
完整正确性及收益仍待上机验收，CPU和编译结果不能替代。
