# 当前开发交付

现行包 0.4.0 / import kvip / 公开契约 3.0.0。源码 CPU 349 项、纯 host 252 项通过，
框架 editable 接线 103 项通过；运行方式见 DEVELOPMENT.md，机器记录见 manifests/validation.json。
以下内容保留历史上下文；CPU 或 source-only 证据不代替 NPU/KVAF/Graph/模型资格。

## 历史本地验证记录

日期：2026-10-30。此记录不代表 NPU 上板或真实性能验收。

| 检查 | 结果 |
| --- | --- |
| 独立 KVIP pytest，WSL CPU Torch | 42 passed |
| wheel 安装到框架之外，Windows 无 Torch | 40 passed，2 个 Torch CPU 用例 skipped |
| 框架原有 eager/Graph/ABI13/QSFA loader Host 回归 | 86 passed |
| CPU benchmark CLI full suite | 11/11 pass；data_record.md 7666字符 |
| 迁移设备源文件与生成依赖 | 62个文件与 SOURCE.json 的迁入 SHA256 一致 |
| CANN 9.1.0，950DT，B4/Q4，C131072/S12288，AIC28，W4 | 生产 libv9_bridge.so 编译成功 |
| 相同配置的 NPU 内存传输桩 | 编译成功，导出 v9_test_transport；未上板 |
| Python Ruff、compileall、Shell bash -n、所改框架文件 git diff --check | 通过 |
| 仓库完整 bash format.sh ci | 环境缺少 pre-commit，未执行完整检查 |

生产库编译 SHA256：
`096994ed4bba25db92f9082e40769e9f834150ad605cce60a5970748578cb968`

测试传输库编译 SHA256：
`4807baa43985155c91aca05e45573f6f151018fa610545c429a5254b99602d98`

历史交叉编译使用 WSL 的 ASCII 源码/构建目录，原始路径保留在对应 Git 历史中。
二进制及服务器输出不纳入源码仓。测试库不是生产性能库。

服务器仍需执行 README 中的 NPU 内存桩、真实 KVAF eager/Graph、跨步/跨层/换序/源改写，
随后验证框架 V9 开启/关闭、回退和真实模型 Graph。原生 LIDU 独立扩展需要服务器的 Torch-NPU 构建环境；
本地仅保留同源源码，未将其独立构建或运行记作通过。
