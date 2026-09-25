# 已知数字源→真实拍屏 RAW：20 组前向检验与 Bayer 相位诊断

日期：2026-09-25。**结论范围：**这是一台实验室拍屏装置的 CIFAR-10 小图验证，目标是定位模拟链的几何、光谱与采样错误；不是 AI／自然摄影来源判别评测，也不是经过设备标定的完整拍屏模拟器。

## 为什么选这批数据

[Raw2Event 官方数据卡](https://huggingface.co/datasets/raw2event/raw2event/blob/d400d349292a6ae1c8ea967e9ae99624d8b156cd/README.md)说明，CIFAR-10 图像显示在屏幕上，由 Raspberry Pi Camera Module 3 同时录制 Bayer RAW 和 ISP-RGB；[原论文](https://arxiv.org/html/2509.06767)说明相机作圆周运动，自动曝光／增益关闭，自动对焦开启。由此可检查**已知数字输入→显示→光学→RAW**，避免只拿传播后的 JPEG 拟合外观。论文主要研究事件生成；其事件模拟结果不能替代我们对屏摄过程的验证。

上一轮只像素检索核实 2 个来源；本轮把目录中已索引的 5,914 条前缀预先冻结为 10 个校准、10 个开发、10 个保留及 3 个跨日期压力保留，另有 2 个先前看过的探索样本。划分清单 `E:\ai_image_origin_research\data\manifests\raw2event_process_split_v1.csv` 的 SHA256 是 `471fbff8020f88c1b73664e5794285df28da4792e5fb77bdfe682b5ee9bb7a43`。本轮**只下载校准和开发的 20 个来源**，保留组未打开。每类 CIFAR 恰有 1 个校准、1 个开发来源，但类别与采集日期高度相关，因此这不是跨设备或跨日期充分独立的验证。

20 个前缀的 RAW MKV、ISP-RGB MKV、元数据 DAT 共 60 文件、**3,815,794,499 字节**，逐文件与发布仓库固定修订 `9df99d9ed09e5ed705f50cae011e49cb2af620b9` 的 LFS SHA256／Git blob 指纹核对。首帧 RAW 均为 692×520、样本有效位长 10，RGB 均为 692×520×3；相同编码画幅尺寸**不表示**内容已逐像素对齐。官方 CIFAR-10 Python 档案为 170,498,071 字节，MD5 `c58f30108f718f92721af3b95e74349a` 与[发布页](https://www.cs.toronto.edu/~kriz/cifar.html)一致。20 个首帧中，文件名指定的 CIFAR 原图在各自 **6,000 个同类候选**的内容检索里均排名第一。此结果使用同一条 AprilTag 相似变换规则定位每张图的内容框，**未针对各自命名原图调框**；若仅沿用第一张探索图的固定框，只有 16/20 排第一，晚日期的 ship/truck 偏离明显。Tag 方法的选择参考过部分开发样本的相关值，仍不是完全预注册的身份核验。它比文件名／类别一致性更有力，但不能据此外推其余 5,894 个直列前缀，其中另有 2 例在[先前报告](Raw2Event真拍屏RAW-RGB配对实包审计_2026-09-25.md)做过独立核验。

20 个 DAT 的长度都是 38 字节的整数倍，分别对应 316／317／318 条记录；此前两包的结构抽查见[原始实包报告](Raw2Event真拍屏RAW-RGB配对实包审计_2026-09-25.md)。当前没有从这些记录中可靠恢复逐帧曝光、增益或对焦位置，不能因数据卡列出这些元数据字段就当作已取得这些物理参数。

## 前向链与诊断顺序

输入为原始 32×32 CIFAR RGB。当前[拍屏前向渲染器](../../origin_simulation/screen_capture.py)将它按**假定**的编码 sRGB Lanczos 放大到 192×192 屏幕格点，计算发光面积、简化的光学低通、CFA 采样，再与真实首帧 10 位 RAW 比较。显示格点数 192、gamma 2.2、发光填充率 0.85、Gaussian 模糊 σ=0.8 传感器像素都**未经该装置的显示帧或光学参数实测**。模拟到 RAW 计数的系数只用 10 个校准来源拟合，逐来源误差在另 10 个开发来源计算。比较单位是来源，而非把同一图里的几万个相关像素当作独立样本。

先用 AprilTag 得到 RGB 首帧中的内容框及 RGB↔RAW 近似变换。固定 Tag 相似变换下，20 个原图与相机 RGB 的 32×32 灰度相关值最小／中位／最大为 **0.651／0.799／0.919**。检索身份全部正确，但这样的框不能支持像素级 RAW 校验。随后**在看过初轮开发误差之后**，使用已知数字原图与同帧 ISP-RGB 对内容框作受限的平移、缩放和旋转优化；不读取 RAW 目标。20 个相关值达到 **0.981–0.997，中位 0.991**；缩放参数集中在 0.952–0.971，说明最初的近似框系统性偏大。这个步骤是事后诊断，不能被描述成预注册留出性能。

另做一项**仅供定位的上界**：直接用真实 RAW 辅助找原图位置。原图与低通 RAW 的灰度相关中位从 0.662 提高到 0.914，但两例仍低于 0.3。此上界使用了待预测的 RAW，**绝不计入前向模拟的验证分数**。它提示 RGB↔RAW 局部映射与光度响应都可能有误，不能从灰度相关单独定因。

## 量化结果

下表各行均使用上述同一 RGB 内容框，数值是 10 个开发来源的**逐来源平均绝对误差 MAE**，单位为 RAW 的 10 位编码计数，越低越好；“全耦合”是在校准组上对每种 CFA 像素拟合非负的显示 R/G/B 三通道系数，含截距。“对角”仅允许对应颜色通道。下面的 Bayer 相位、耦合形式是**看到初轮开发组结果后探索的**。

| 前向空间处理 | CFA 与显示通道耦合 | 开发 MAE | 开发 Pearson 平均 | 单个 RAW 内容 ROI 渲染中位时间 |
|---|---|---:|---:|---:|
| RGB 竖条发光＋面积积分 | RGGB，对角 | 46.62 | 0.759 | 8.67 s |
| 同量 RGB 共址发光 | RGGB，对角 | 46.58 | 0.759 | 8.60 s |
| 直接数字插值＋相同模糊 | RGGB，对角 | 47.06 | 0.753 | 0.0025 s |
| RGB 竖条发光＋面积积分 | **BGGR，对角** | **42.01** | 0.789 | 8.67 s* |
| 同量 RGB 共址发光 | BGGR，对角 | **41.99** | 0.789 | 8.60 s* |
| 直接数字插值＋相同模糊 | BGGR，对角 | 42.56 | 0.783 | 0.0025 s* |
| RGB 竖条发光＋面积积分 | 三色非负全耦合 | **41.38** | 0.789 | 8.67 s |
| 同量 RGB 共址发光 | 三色非负全耦合 | 41.39 | 0.789 | 8.60 s |
| 直接数字插值＋相同模糊 | 三色非负全耦合 | 41.96 | 0.783 | 0.0025 s |

\* 相位实验复用同一批三通道光场缓存，只改变 CFA 采样位置与校准映射；因此继承光场渲染时间，不是单独完整计时。所有时间都只是本机 RAW 内容 ROI 的模拟核心，**不含视频解码、配准、参数拟合或移动设备端到端推理**。

四种 Bayer 相位在相同“竖条＋对角”条件下的开发 MAE 是 **RGGB 46.62、BGGR 42.01、GRBG 48.21、GBRG 48.20**。BGGR 比 RGGB 低 4.61 计数，10 个来源中 9 个改善。探索性逐来源 bootstrap（20,000 次、种子 20260925）的平均差（BGGR−RGGB）95% 分位区间为 **−7.11 到 −2.59**；相位是在开发结果揭示红蓝交换后提出的，因此这个区间**不是独立确认或正式显著性证据**。[Raspberry Pi 仓库中的一份 IMX708 运行日志](https://github.com/raspberrypi/libcamera/issues/62)显示过 `SBGGR10_1X10` 输出，与这一现象相容；它并非本数据集的采集日志，发布 MKV 经裁剪／重存后的像素原点和 CFA 相位仍无原始配置文件证明。

全耦合拟合的 RGGB 名义矩阵中，代码假定的 R 位置主要接收模拟 B 通道、B 位置主要接收模拟 R 通道，这正是检查相位而非直接宣称“镜头严重串色”的原因。改用 BGGR 后，全耦合相对 BGGR 对角仅再减少 **0.63 计数**；这部分可能是显示光谱与 CFA 的真实交叉响应，也可能吸收了未标定色彩管理、ISP 和配准误差。全耦合参数是**显示原色→RAW 的合成映射**，并不能独立测出传感器光谱曲线。

在 BGGR 对角下，竖条物理积分比直接插值只低 **0.55 计数**，10 个来源有 8 个改善；共址发光的 41.99 又比竖条 42.01 微低。因此这批数据支持首先修正 Bayer 相位，却**不支持已经辨认出真实显示子像素排列**。竖条细网格渲染约 8.6 秒，距离目标的快速大批量模拟仍很远。开发组一张 cat 仍约 **116 计数 MAE**，模型残差没有被这次相位或通道校正解决。

## 代码变更与下一道证据关口

前向模型现在显式接受 `sensor_bayer_pattern` 的 `RGGB/BGGR/GRBG/GBRG` 四值，默认仍为 RGGB；已针对纯红输入检查采样位置，并运行相关测试。**正确的相位应由设备配置／RAW 元数据或独立色块记录锚定**，而非以后只选 MAE 最低的相位。下一步应找真实屏幕的输入缓冲区或显示栅格、相机原始像素格式／裁剪原点、色块和曝光记录；再用这些受约束参数对未开启的保留来源作一次外推检验。现有 10 个保留及 3 个跨日期压力来源继续不用于调参。

这条 Raw2Event 线对模拟过程有效，但 CIFAR-10 没有 AI 生成类别，单台装置也不足以证明对 RRDataset 的四种再数字化路径或手机屏摄普适。来源检测准确率与速度仍须在独立真实传播集上另行评估。

**后续独立来源检验：**另按文件名哈希冻结并取得每类 1 个新来源，[十来源一次外推报告](Raw2Event新来源Bayer相位外推核验_2026-09-25.md)显示 BGGR 对 RGGB 的改善在 10/10 来源复现，但统一 Tag 框的原图检索仅 7/10 排第一，且绝对 RAW 误差仍大。它没有使用上述预留 10 源与跨日期压力保留。

## 复算入口与审计文件

```powershell
python -m work.fetch_raw2event_process_split
python -m work.audit_raw2event_split_first_frames
python -m work.verify_raw2event_split_content --geometry tag_similarity
python -m work.refine_raw2event_content_geometry
python -m work.evaluate_raw2event_source_to_raw_split --geometry source_rgb_refined
python -m work.evaluate_raw2event_spectral_mix
python -m work.probe_raw2event_cfa_phase
python -m work.summarize_raw2event_cfa_phase
python -m pytest tests/test_screen_capture.py tests/test_screen_capture_projective.py tests/test_screen_pipeline.py -q
```

逐文件官方指纹审计：`work/raw2event_process_split_download_audit.json` 及 `work/raw2event_process_split_downloads/*.json`。逐原图同类检索：`work/raw2event_process_split_source_identity_fixed_quad.json`、`work/raw2event_process_split_source_identity_tag_similarity.json`；用各自命名原图优化框得到的 `work/raw2event_process_split_source_identity_source_rgb_refined.json` **不应当作独立身份验证**。几何与 RAW 目标定位诊断：`work/raw2event_process_split_first_frame_audit.json`、`work/raw2event_content_registered_geometry.json`、`work/raw2event_raw_target_registration_diagnostic.json`。逐图模型分数与渲染秒数：`work/raw2event_source_to_raw_split_v1_source_rgb_refined.json`、`work/raw2event_spectral_mix_v1.json`、`work/raw2event_cfa_phase_probe.json`、`work/raw2event_cfa_phase_summary.json`。这些机器文件置于 `work/`，下载原包和派生缓存置于 E 盘；`outputs/` 保存可阅读结论。
