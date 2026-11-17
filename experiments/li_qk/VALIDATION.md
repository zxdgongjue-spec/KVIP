# 本地验证记录（2026-11-17）

- 原 LIFetch 方案保存在 `20adb7e`，实验从该提交建立独立分支。生产源码未修改。
- runner 固定7个点：`(T,Qparts)=(1,1),(2,1),(4,1),(8,1),(16,1),(1,2),(1,4)`。所有点固定 B1/Q4/H131072/G32/D128/TopK2048，默认预算32 AIC/64 AIV。
- 4项 CPU 检查通过：C++/Python计划一致且每个真实Q/H坐标覆盖一次；实际host在ACL stub下正确打包、重排并保护dummy/guard；汇总只接受完整7点及r0证据，拒绝日志改动；launch范围和非法几何检查。
- CANN 9.1.0、`Ascend950PR_9599`、`RUN_MODE=npu`全新目录构建通过，包含device编译和host链接。生成元数据确认score为MIX1:2，scatter/TopK为AIV_ONLY；host stub仅有三个LI入口。
- CPU stub的设备时间为null。测试中的合成时间只验证汇总逻辑；不能用来报告指令数、Model周期或A5加速。
- 当前容器没有NPU。CANN静态注册在main之前调用`aclrtGetSocName()`并获得null，随后构造字符串抛异常；因此这里只完成编译验证，没有执行设备数值、同步或性能验收。A5需使用实际SoC/CANN环境重新构建并运行。
- `score_p50_us`仅含score waves；`li_p50_us`包含必要的重排和同一个全局TopK。没有Fetch/Service/Index。串行基线及单wave的T1/Qparts2无需重排；Qparts4保留物理Q2对齐及dummy行压紧的成本。
- 一键启动脚本`run_a5.sh`通过Bash语法检查及3项编排检查：本地Git拉取/新worktree与原修改保留、固定参数及SDK开关转交、构建/运行失败后停止并保留退出码和打包日志。检查使用模拟执行后端，不执行CANN或NPU；命令为`python3 experiments/li_qk/test_a5_launcher.py`。
- 增补联网处理后，启动脚本6项检查通过：此次Git代理覆盖不写回配置；origin不可达时本地提交/离线bundle仍能进入同一实验流程；本地缺少指定提交时拒绝运行。离线和网络检查仍使用模拟LI后端，不表示A5设备结果。
- 固定目录宿主机脚本`sync_a5_host.sh`通过Bash语法检查及本地Docker验证：从已有离线包自动创建同步目录、重复同步、目标容器中两个文件的SHA一致、容器不存在时停止。未启动LI kernel，不能作为A5算子执行验收。
- 无容器CLI时自动准备源码，`--run-here`在当前A5环境使用NPU路径执行。增补检查覆盖没有CLI时的真实Git/bundle准备、缺少`npu-smi`时提前停止、参数转交和后端失败传播；最后的NPU查询与LI执行在流程检查中使用stub，不产生设备性能结果。原Docker复制流程仍通过本地Docker哈希回归检查。
- 共享目录入口`update_shared_a5.sh`通过Bash语法检查和本地Docker已有挂载验证：宿主机只更新文件、容器从映射目录读取并校验哈希、重复更新保留旧release、容器中损坏的代码包/脚本被拒绝。没有运行NPU kernel。当前部署使用已确认的`/home/w00568585 → /home/w00568585`映射，真机计时只在容器启动脚本正常运行后生成。

本地完整日志、7点计划、构建哈希及元数据保存在`tests/results/li-qk-controls-20261117/`。交付包包含`validation/`中的检查记录，不附带跨CANN版本复用的二进制。A5性能结果由`run.py run`和`summarize`生成。
