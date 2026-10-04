# 实验脚本

[仓库首页](../README.md) · [复算和环境说明](../docs/README.md#历史实验与结果完整性)

可复用算法位于 [src/palimpsest](../docs/architecture.md)，路径通过[统一配置](../configs/README.md)加载。实验脚本按数据集和用途归类；既有机器结果仍在 `work/`。这些是历史研究代码，运行前先读相应报告和脚本的路径常量。

**统一运行方式：**在仓库根目录使用 `python -m experiments.<主题>.<脚本名>`。PowerShell 下载/续跑脚本用其新文件路径显式执行。

| 主题 | 内容 | 脚本数 |
|---|---|---:|
| [检测 baseline](baselines/README.md) | B-Free、D3、Benford-RF 和评分汇总；官方模型使用独立环境。 | 16 |
| [RRDataset 与早期统计模拟](rr/README.md) | 数据准入、配对、传播统计和历史统计模拟；旧原型不代表真实物理过程。 | 27 |
| [Chimera 拍屏](chimera/README.md) | 外部真实配对、固定检测器迁移、频谱与过程反证。 | 25 |
| [Raw2Event RAW](raw2event/README.md) | 数字源对应、RAW 几何、CFA、计数映射和参数可辨识性。 | 39 |
| [ImageNet-ES 物理干预](imagenet_es/README.md) | 真实光圈/ISO 变化、响应拟合和跨内容留出。 | 7 |
| [Dragotti 拍屏](dragotti/README.md) | 跨相机配对、几何及衍射频率预算。 | 8 |
| [拍屏数值与速度](screen_numerics/README.md) | 虚拟输入上的积分、光学、曝光、对照及性能实验。 | 22 |
| [打印、扫描与拍纸](print_scan/README.md) | DFD、DESCAN、CSGC、DIV2K-SCAN 的数据审计和模型验证。 | 45 |
| [投影相机](projector/README.md) | CompenNet/CompenNet++ 几何、光度和跨设置实验。 | 21 |
| [通用下载工具](downloads/README.md) | 带长度/MD5 检查的 PowerShell 下载器；仅显式执行。 | 1 |

源代码迁移前的提交为 `0f675d6d544de407dae5c162e26100245edf79ef`。旧结果与断点指纹保持原值；不要绕过校验后续跑。
