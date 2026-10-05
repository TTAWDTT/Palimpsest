# CompenNet 整包审计与拍投影前向反证

2026-09-25。**结论：**官方真实投影—相机配对已在本机全量验包。125 张均匀色块能标定一个紧凑的有效光度前向模型，但它在未见纹理图上的误差约为逐像素完整 3D LUT 的两倍。只用 13 张纯色标定的通道叠加假设接近紧凑模型，却在未参与标定的混合色块上仍有明显误差。**现阶段发现的是可检验的物理约束与模型缺口，并非完成了真实拍投影过程模拟。**

## 1. 实包与对应关系

数据来自 [CompenNet 作者仓库](https://github.com/BingyaoHuang/CompenNet)所链的公开档案；[CVPR 原论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_End-To-End_Projector_Photometric_Compensation_CVPR_2019_paper.html)描述 24 组真实投影设置、每组 500 张训练纹理输入与 200 张不同测试输入，以及手动固定的相机曝光、对焦、白平衡。作者研究的是投影**逆向补偿**；此处只用其输入—真实相机输出配对检验我们的**正向**模型。论文逆向补偿指标不能直接作为本实验基线。

- E 盘完整档案：`E:\ai_image_origin_research\data\raw\compennet\CompenNetDataset.zip`，**2,282,667,301 字节**，本机 SHA-256 `eac2963fa62f77084e8d6fe8cfed23a80572c8f44a52d7f22b3c315bc0834ef0`。
- **20,709 个非目录成员全部 ZIP CRC 通过**，无重名路径；24 组设置各有完整的 500 个 `cam/warp/train`、200 个 `cam/warp/test`、126 个 `cam/warp/ref` 路径。原输入 `train` 500、`test` 200、`ref` 126 路径也齐全。部分设置另有文本或 Poisson 图，不纳入本次配对。
- 原始色块 `ref/img_0001`–`0125` 是 5×5×5 的均匀 RGB 网格，级别为 0、64、128、191、255；编号按 **R 最快、G 次之、B 最慢**变化。每组相机输出的前 125 个色块逐文件 SHA 均不同。
- 根目录另有 `ref/img_gray.png`，其像素是 RGB(77,77,77)。**不能把每组输出 `cam/warp/ref/img_0126.png` 视作它的一张独立测量：**在 24 组中，该文件分别与已有的色块 0063（15 组）、0094（7 组）、0032（2 组）**逐字节相同**，对应均匀灰 128、191、64。故前向标定只采用已核实的一一对应 125 色块；不为 `img_gray` 臆造对应输出。
- 审计程序 `experiments/data_preparation/compennet/audit_full.py`，机器可读明细 `work/compennet_full_audit.json`。PNG 发布图为 256×256 RGB，其中相机结果路径明确是 `cam/warp`，**不能视作原生相机帧或传感器 RAW**。

## 2. 实验口径

这不是 AI/自然来源判别实验；700 张互联网纹理输入没有可用的原始来源二分类真值。指标为真实相机发布图与前向预测图的**逐像素 RGB 平均绝对误差 MAE**：每通道归一到 0–1，越低越好。每组 200 张测试纹理，24 组共 4,800 条配对；同一张源图跨设置重复，**不能称 4,800 个独立来源**。下表先对每组的 200 张取均值，再对 24 组取等权均值。

模型参数仅来自相应设置的参考色块；平滑尺度候选 0、0.45、0.9、1.35 个**对齐后像素**，用该设置的前 16 张训练纹理选取，测试纹理从未用于拟合或选尺度。这个尺度是发布图的复合有效模糊，不是实测投影镜头 PSF。所有方法预测同一套 200 张测试图，输出裁剪至 [0,1]。

| 正向预测方式 | 色块标定量 | 每设置 float32 参数量 | 24 组测试 MAE |
|---|---:|---:|---:|
| 直接把数字输入当相机结果 | 0 | 0 | **0.20181** |
| 逐像素 RGB 仿射映射 | 125 色 | 3.15 MB | **0.09321** |
| 黑底图 + 表面增益图 × 全局 RGB 3D 曲线 | 125 色 | 1.57 MB | **0.07683** |
| 加性三投影通道 × 空间颜色混合；用 sRGB 逆变换作线性光代理 | **13** 色 | 3.15 MB | **0.07976** |
| 每像素独立 5×5×5 RGB 3D LUT，三线性插值 | 125 色 | **98.30 MB** | **0.03907** |

另外，13 色通道模型若直接在压缩 RGB 码值上相加，测试 MAE **0.08076**；使用 sRGB 逆变换后为 **0.07976**。在完全没有参与其标定的 **112 个混合色块**上，两种口径的 MAE 分别为 **0.07853** 与 **0.06075**。这一改善符合“先在线性光中叠加”的方向，但不证明作者 PNG 的真实相机传递函数就是 sRGB，也不把材料反射率和相机响应单独识别出来。

完整 3D LUT 直接保存每个位置对 125 种均匀输入的响应；在色块节点上可精确重现标定图。它是**容量很高的局部输入—输出对照**，不是物理前向过程；即便如此，对未见纹理仍留下 0.03907 MAE，说明均匀色块测量不足以预测所有空间/上下文变化。

同一台 Ryzen 7 8845H 上，对一张 256×256 输入、参数和像素均已载入内存，预热 8 次后重复 64 次的**单图前向核**时间中位数：13 色通道模型 **6.84 ms**，125 色紧凑模型 **17.33 ms**，125 色完整 LUT **22.98 ms**；同图 PNG 解码另约 **1.55 ms**。这是探索性本机 CPU 测量，不含校准、ZIP 读取、输出编码，也不是手机实时来源判别速度。参数体积与核耗时一起显示为何值得继续研究少量标定且保留物理约束的模型；原始逐次设置见 `work/projector_forward_speed.json`。

为定位这一残差，又固定选取四种设置、各取测试编号 0001–0050：完整 LUT 在训练图选出的模糊尺度下，四设置平均 MAE **0.03725**；不作模糊为 **0.04323**。在每张图内部 224×224 区域，源图梯度最高十分位的预测误差是最低十分位的 **1.64–2.64 倍**；图像边缘 16 像素与内部的平均误差接近。高频内容是明显难点，单靠黑边或饱和区解释不了这些观察。不过对齐/插值、投影和相机光学、相机 ISP 都可能导致边缘残差，**不能把 0.9 像素拟合值直接称作真实光学模糊**。逐图统计见 `work/projector_residual_diagnosis.json`。

## 3. 反证比“赢线性 baseline”更重要

紧凑的黑底×表面增益模型在 **21/24** 组设置中优于逐像素仿射，对全部设置的平均 MAE 从 0.09321 降为 0.07683。但完整 3D LUT 在 **24/24** 组设置中都优于紧凑模型。例如 `light1/pos1/stripes`：紧凑模型 **0.0970**，仿射 **0.0605**，完整 LUT **0.0346**。所以“表面纹理乘上一个全局颜色曲线”遗漏了明显的局部输入颜色作用；条纹设置尤其容易反证它。

13 色模型的校准输入仅为黑、各通道四档强度；使用三个纯色端点建立逐像素 3×3 输出通道混合，另用纯色阶梯估计全局一维通道曲线。它把可归因的**输入通道叠加**与**位置相关的颜色混合**分开，但在 112 种混合色及纹理上仍失败，说明单纯的通道可加性不足。像素饱和、相机非线性、真实光谱互作用、邻域散射、几何配准误差都可能造成差异；**仅凭这些 PNG 无法从中唯一挑出原因**。[原论文对邻域上下文的讨论](https://arxiv.org/html/1904.04335)也警告单像素映射不能解释所有结果。

使用一组设置的全局响应曲线，目标设置只提供黑/白两张空间图，固定 0.9 像素平滑，不重新拟合目标的 125 色曲线：

| 响应曲线 donor → 目标设置 | 直接输入 MAE | 跨设置模拟 MAE | 目标自身 125 色标定 MAE |
|---|---:|---:|---:|
| `light1/pos1/stripes` → `light1/pos2/stripes` | 0.3380 | **0.0850** | **0.0257** |
| `light1/pos1/stripes` → `light4/pos1/stripes` | 0.1486 | **0.0803** | **0.0796** |
| `light2/pos1/squares` → `light2/pos2/squares` | 0.1844 | **0.0947** | **0.0859** |
| `light2/pos2/lavender` → `light3/pos3/lavender` | 0.1611 | **0.0771** | **0.0614** |

四次转移都优于直接输入，但有的远逊于目标自身标定。说明“设备响应跨条件固定、只换表面黑白图”还不能稳定成立；它也可能受光照、位置、饱和与发布配准共同影响。此表是**预选四组探索**，不能推广成所有设置的转移率。

再将目标校准从黑/白两张改成**黑 + 纯 R/G/B 共四张**，以目标逐像素 3×3 颜色混合场、donor 的纯色阶梯曲线预测相同目标测试图；固定 0.9 对齐像素模糊。在线性光代理口径，四组的 MAE 为 **0.0364、0.0651、0.0982、0.1664**；目标自己用 13 张色块标定时分别为 **0.0413、0.0658、0.0841、0.1092**。前两组可转移，后两组退化，尤其最后一组。此处目标 13 色对照的模糊尺度由训练图选择，部分设置并非固定 0.9，故小幅差异不应作强结论。四张纯色已经比黑/白两张更能表达表面与相机的颜色混合，但“曲线跨设置共享”仍是可证伪假设。

## 4. 对真正过程模拟的约束

1. **先校准可分离的阶段。** 纯色端点和阶梯可检验黑电平、有效非线性、通道混合；混合色及纹理分别检查通道互作用与邻域效应。简单颜色 MAE 接近不能代替这些干预检验。
2. **保留未知量。** CompenNet 256 像素发布图已经过单应性配准，原投影像素格、光学 PSF、相机 CFA/ISP、DLP 位平面与逐帧曝光不可从它单独恢复。上述模型参数均为**复合有效参数**，不得写成真实投影机或表面材料常数。
3. **再接原生几何。** 同作者 [CompenNet++ 官方包](https://github.com/BingyaoHuang/CompenNet-plusplus)远程样本确有 640×480 的未配准相机 RGB PNG，可检验画幅与投影边界；其 `cam/raw` 是“未经几何变换”而非 Bayer RAW。当前只核过远程目录与少量成员，约 11 GB 整包尚未下载/全 CRC，不能把它列为已经可用的全量测试。
4. **最终任务另测。** 需要有原始来源标签、真实传播配对及方式标签的数据，才可评估 AI/自然判别准确率与跌幅。RRDataset 的合并 `redigital` 没有逐张四方式标签，不能把本投影实验直接嫁接成 RR 的“拍投影 BA”。

## 5. 复算

在本仓库运行：

```powershell
uv run python -m experiments.data_preparation.compennet.audit_full
uv run python -m experiments.projection_capture.chart_response.compennet.run_chart_model --all --train-count 16 --test-count 200
uv run python -m experiments.projection_capture.chart_response.compennet.run_channel_basis --all
uv run python -m experiments.projection_capture.chart_response.compennet.run_full_lut --all
uv run python -m experiments.projection_capture.chart_response.compennet.run_transfer
uv run python -m experiments.projection_capture.chart_response.compennet.run_channel_transfer
uv run python -m experiments.projection_capture.chart_response.compennet.evaluate_residuals
uv run python -m experiments.projection_capture.chart_response.compennet.benchmark_forward_speed
```

逐设置、逐测试图 MAE、训练尺度选择、档案 SHA 与完整审计分别落在 `work/compennet_full_audit.json`、`work/projector_chart_probe.json`、`work/projector_channel_basis_probe.json`、`work/projector_full_lut_probe.json`、`work/projector_transfer_probe.json`、`work/projector_channel_transfer_probe.json`、`work/projector_residual_diagnosis.json` 和 `work/projector_forward_speed.json`。这些 JSON 是本机中间计算记录；上述程序和 E 盘档案提供可复算入口。已有[物理顺序与证据边界](../../../docs/projector_camera_model_contract.md)进一步区分投影、表面、相机与发布处理。
