# Model evidence

九组B1/B16、Q4/H131072、K2048数值TopK Model执行均PASS。单次完整Model执行（warmup0/iteration1），不是A5微秒或多轮性能统计。

这些是原始receipt、host report及从真实ELF/entry验证后的all-AIV指令汇总；CPU测试数据不包含在此目录。大体积fixture二进制和raw dump保留在本地`tests/results/li-tail-20261117/radix-target-v16/`；本目录保留其hash、每物理AIV entry和指令数证据。raw issued指令日志全部保留。磁盘保护仅丢弃部分B16 CCU/popped辅助日志，详见pruning记录，不参与issued分母。

manifest中的/work路径是当时容器路径，不能在其他服务器直接恢复执行。重新执行使用仓库radix.py/run_radix_a5.sh，生成新的完整结果目录。A5输出会使用device event时间并丢弃预热轮；Model ticks不能换算为A5µs。

Git内CSV使用LF行尾；run.log.gz解压后逐字节保留原log，hash仍对应原log。证据JSON仅压紧展示格式，计数和entry未修改。
