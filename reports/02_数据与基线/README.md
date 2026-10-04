# 数据与基线

**先读：[RRDataset 三种 baseline 全量对照](RRDataset_三种Baseline全量对照_2026-09-24.md)。**它给出统一测试集、BA/AUC、同源退化和包含解码的速度。

| 文件 | 用途 |
|---|---|
| [数据集整理与准入审计](数据集整理与Baseline准入审计_2026-09-23.md) | 官方包指纹、逐图清单、对应关系与已知重叠。 |
| [B-Free 详细评测](RRDataset_B-Free真实传播Baseline_2026-09-24.md) | 官方权重、推理口径与逐条件误差。 |
| [Benford-RF 详细评测](RRDataset_Benford传统算法全量Baseline_2026-09-24.md) | 传统算法适配版、训练口径与局限。 |
| [B-Free 指标 JSON](RRDataset_BFree_评测指标_2026-09-24.json) | 供程序读取的详细汇总。 |
| [Chimera 真实拍屏 B-Free 配对评测](Chimera真实拍屏_B-Free外部配对Baseline_2026-09-25.md) | RR 之外 1,200 个来源 × 两套真实拍屏；固定阈值、AUC、同源跌幅与速度。 |

原始逐图 CSV、日志和推理脚本位于本机 `work/`，大型数据及清单位于 E 盘。D3 的单图协议与作者 batch=128 协议不同；Benford-RF 不是原论文的逐字复现。更多早期小样本检查见[历史探索](../90_历史探索/README.md)。
