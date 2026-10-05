# Raw2Event 新来源一次外推：BGGR 相位收益复现，几何与光谱仍有缺口

日期：2026-09-25。本文接续[20 组已知源→拍屏 RAW 的前向检验](Raw2Event已知源拍屏RAW前向验证与Bayer相位诊断_2026-09-25.md)。这次目的很窄：在**未参与前一轮权重拟合或相位挑选的新来源**上，检查 BGGR 优于 RGGB 是否复现；不重新校准显示、镜头或传感器，不把 CIFAR-10 当作 AI 来源分类数据。

## 新来源如何冻结和验收

从作者仓库已索引的 5,914 个直列前缀中，先排除之前冻结的全部 35 个角色，再以固定种子 `raw2event-phase-confirmation-2026-09-25-v1` 对每个剩余前缀取 SHA256，每类 CIFAR 选哈希最小的 1 个，合计 **10 个来源**。筛选只看文件名中的类别和索引，没有预览这些新 RAW／RGB；清单 `E:\ai_image_origin_research\data\manifests\raw2event_phase_confirmation_v1.csv` 的 SHA256 为 `c699c6b9ce8ce779f892c5caaab11e19b4c24e0e697ec8116b94115c0175e372`。原先 10 个保留及 3 个跨日期压力保留来源没有开启。

随后按[Raw2Event 官方仓库](https://huggingface.co/datasets/raw2event/raw2event/blob/d400d349292a6ae1c8ea967e9ae99624d8b156cd/README.md)固定修订 `9df99d9ed09e5ed705f50cae011e49cb2af620b9` 下载这 10 组 RAW MKV、ISP-RGB MKV 和元数据 DAT，共 **30 文件、1,892,938,216 字节**；逐文件与发布 LFS SHA256／Git blob 指纹核对。没有因预览结果排除任何来源。

在不对每张命名原图优化四角的统一 AprilTag 几何下，文件名指向的 CIFAR 图在 **6,000 张同类候选**里有 **7/10 排第一、10/10 排前十**；未排第一的是 airplane 第 6、dog 第 8、ship 第 4。这个差异揭示统一 Tag 框在新来源上的泛化不足，不能把之前 20/20 排第一的结果外推到整个目录。随后为了做像素级模拟比较，按上一轮**已固定**的算法，用已知数字原图与该组相机 ISP-RGB 细化内容框，未读取 RAW；细化后的源图相关最小 0.973、中位 0.989。这是模拟对齐所用的条件信息；由于会针对命名原图调框，**不另计作独立身份检索成绩**。

## 不再拟合的 RAW 预测

显示尺寸、gamma、填充率和模糊仍采用前轮的虚拟设定；RAW 计数映射及非负三色耦合系数原封不动取自**此前 10 个校准来源**。新来源的真实 RAW 只在最后计算误差。每类各 1 个新样本，单位是来源，而不是把一张图内的像素视作独立重复。表中 MAE 是 10 个来源的逐来源平均，单位为 10 位 RAW 计数。

| 固定假设 | 旧开发 10 源 MAE | 新来源 10 源 MAE | 新来源平均 Pearson |
|---|---:|---:|---:|
| RGB 竖条＋RGGB 对角响应 | 46.62 | 54.80 | 0.651 |
| RGB 竖条＋**BGGR 对角响应** | 42.01 | **50.17** | 0.684 |
| RGB 共址＋BGGR 对角响应 | 41.99 | 50.15 | 0.684 |
| 简单数字插值＋BGGR 对角响应 | 42.56 | 50.88 | 0.677 |
| RGB 竖条＋非负三色全耦合 | 41.38 | **49.83** | 0.687 |

新来源上，BGGR 比 RGGB 平均降低 **4.63 计数 MAE**，**10/10 来源改善**；固定的另外两种相位 GRBG／GBRG 均约 56.3，明显更差。逐来源自助抽样 20,000 次所得 BGGR−RGGB 平均差 95% 分位区间为 **−7.13 至 −2.49 计数**。这个区间描述当前十张来源的变异，**不是跨装置泛化的概率保证**。与前轮开发数据相比，新来源的 BGGR 误差从 42.01 上升到 50.17，说明同一装置和同类别内仍有明显内容／采集变化。

| 新来源类别 | 命名原图的 Tag 框同类排名 | RGGB MAE | BGGR MAE |
|---|---:|---:|---:|
| airplane | 6 | 23.2 | 20.9 |
| automobile | 1 | 98.9 | 91.3 |
| bird | 1 | 63.3 | 61.0 |
| cat | 1 | 31.7 | 28.9 |
| deer | 1 | 60.0 | 59.4 |
| dog | 8 | 32.3 | 29.5 |
| frog | 1 | 36.8 | 32.4 |
| horse | 1 | 64.7 | 63.7 |
| ship | 4 | 95.0 | 83.3 |
| truck | 1 | 42.2 | 31.3 |

非负全耦合在新来源上比 BGGR 对角仅再低 **0.34 计数**，改善 6/10，逐来源区间 **−1.12 至 +0.37**，不足以确认额外的交叉响应已被稳定辨认。竖条积分比简单双线性插值平均低 **0.71 计数**，8/10 来源改善；但与同量共址发光的差只有 **+0.02 计数**（竖条略差），因此**没有证据可据此确认真实面板为竖直 RGB 子像素**。这条透视细网格渲染中位仍需约 **8.6 秒／RAW 内容 ROI**，简单双线性对照约 **0.0024 秒**；这里都不含视频解码、几何搜索及移动端计算。

按类别看，新组 automobile 的 BGGR MAE **91.3**、ship **83.3**，远高于 airplane **20.9**；没有人根据这些大误差删除样本或重新拟合。它们限制了“BGGR 修正后模拟已准确”的说法，可能混合内容几何、屏幕实际插值、光照/曝光、焦点以及未知的显示与 CFA 光谱响应。当前测试不能将各来源的误差单独归因。

## 结论边界与下一步

在一个**另选、未参与拟合**的十来源样本上，BGGR 相位带来的改善再次出现；这与[官方软件仓库里出现过的 IMX708 `SBGGR10` 运行日志](https://github.com/raspberrypi/libcamera/issues/62)相容。该日志不是本数据集的设备配置，RAW MKV 是否保持相同裁切原点仍待核实。新样本仍来自同一个实验装置，每类主要对应一个采集日期；不能把此结果推广到手机拍屏、其他面板、RRDataset 的混合再数字化或最终 AI 来源检测任务。

下一道过程证据应是**本数据集的实际 RAW 像素格式及裁切原点**、屏幕输入帧/栅格、颜色与曝光记录，以及在 RGB 与 RAW 之间不依赖目标 RAW 的几何映射。物理设定未锚定前，继续扫描大量 gamma、模糊或子像素样式来追求较低 MAE 容易过拟合现有装置。原先的 10 个保留及 3 个跨日期压力保留继续保存给更成熟的前向模型。

## 复算

```powershell
python -m experiments.data_preparation.raw2event.prepare_registry_phase_confirmation
python -m experiments.data_preparation.raw2event.prepare_download_phase_confirmation
python -m experiments.screen_capture.cfa_phase.raw2event.evaluate_phase_confirmation
python -m experiments.screen_capture.cfa_phase.raw2event.evaluate_summary_phase_confirmation
```

冻结选择和下载验收见 `work/raw2event_phase_confirmation_freeze.json`、`work/raw2event_phase_confirmation_download_audit.json`；每文件指纹在 `work/raw2event_phase_confirmation_downloads/`。新来源每种假设的逐图分数、内容匹配排名、几何诊断、旧拟合文件 SHA256 与代码指纹在 `work/raw2event_phase_confirmation_evaluation.json`；配对差及抽样区间在 `work/raw2event_phase_confirmation_summary.json`。原始数据与派生缓存均在 E 盘。
