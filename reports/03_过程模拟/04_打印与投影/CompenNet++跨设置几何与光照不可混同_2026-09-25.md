# CompenNet++：同名布幕跨设置并非“仅变光照”

2026-09-25。**结论范围：**从作者公开 CompenNet++ 档案又按 HTTP Range 抽出 `light3/pos1/cloud_np` 的 **42 张结构光投影输入与 42 张未几何配准的相机 RGB 图**。所选 84 个 ZIP 成员逐个检查长度、CRC 与本地 SHA；与已核 `light1/pos1/cloud_np` 的 **42 张投影源 SHA 完全相同**。但相同投影像素落在两次相机画幅中的位置差很大，因此不能把路径名里的 `light1/light3` 当成只改灯光的受控干预。**11.48 GB 远程 ZIP 仍未整包下载或全成员审计。**

## 证据链和可重复口径

[作者论文](https://arxiv.org/html/1908.06246)说明跨 setup 会改变全局光照，同时投影机—相机间距在 **500–1000 mm** 范围变化；每个 setup 采集期间手动锁定曝光、对焦与白平衡。[作者数据包 README](https://github.com/BingyaoHuang/CompenNet-plusplus)也把 `light1/light2` 说明为照明级别、`pos1` 说明为**相机、投影机与表面的姿态**，却未保证不同 `light` 目录下同一位置编号代表全局相同物理姿态。我们只把它们视为**同名表面、同一组投影编码图、不同实际 setup**。`cam/raw` 路径仍是未配准的相机 **RGB PNG**，不是传感器 RAW。

两组各自的互补纵横编码均由官方投影图本身建立码表。首组 `light1` 用每一对投影的最弱亮度差 >0.02 筛后有 **75,392/307,200=24.54%** 相机像素可解；第二组 `light3` 为 **59,766/307,200=19.46%**。若阈值 >0.04，则分别 **41,207（13.41%）** 与 **37,964（12.36%）**。白帧减黑帧亮度 >0.12 的画幅占比由 **99.99%** 变为 **67.14%**；两组白帧均值为 **0.806/0.589**、黑帧均值为 **0.033/0.222**（0–1 灰度）。这些是**复合观测**，既受照明/曝光链影响，也受被拍表面、投影覆盖和几何关系影响，不能单独归因于灯光。

更直接地，按解出的 **同一投影机整数像素坐标**把两次相机位置匹配，而非假设相机像素本来对齐：在 >0.04 的严阈值下有 **8,680 个两组共同可见的投影码**；其相机坐标直接位移中位 **112.07 像素**、90 分位 **125.64 像素**。以这些对应关系在 8,680 点上拟合一个相机间单应性，3 相机像素 RANSAC 阈值下内点 **71.42%**，全部点残差中位 **1.11 像素**、90 分位 **23.53 像素**。可见有强烈的大尺度视图变化，同时一个单平面变换也不能解释所有共同码。**它不识别究竟是相机、投影机、布幕还是几者共同移动**；编码误差、高光、遮挡及背景也会贡献残差。

反过来，若错误地把两组相同位置的相机像素逐个相减，在两组都通过 >0.04 的 **10,649 个画幅像素**上，对应投影坐标差的中位竟是 **136.72 投影像素**，严格相同的比例为零。两组严阈值掩码 Jaccard 仅 **0.155**。所以用同一相机像素做光度差，或把可解码率下降写成纯“灯光鲁棒性”，均缺少前置配准与因果依据。白场图用 SIFT 仅有 **16 个**比率筛选候选、**6 个** RANSAC 内点，不能替代结构光几何证据。

## 对前向模拟的反证

同一固定程序、40×40 相机像素交错格留出、最弱编码对比度 >0.04，首组单应性/三次平滑映射的留出投影坐标误差中位分别 **4.48/1.57 像素**；第二组分别 **11.83/2.81 像素**。两组在各自 setup 内三次映射均胜单应性，但第二组缺口更大；这是**两种不同可见区域与设置**下的现象，不能解读为单一光照因子的效应，也不是物理深度标定。

若直接把第一组 10,000 个严阈值点拟合的三次相机→投影映射用于第二组 10,000 个严阈值点，投影坐标误差中位 **132.20 像素**、90 分位 **147.06 像素**。这是一个明确的**旧几何参数不可直接复用**的反例；第二组画幅覆盖也不同，其中有外推误差，不能把 132 像素全部解释为真实物体移动。它和第二组内部重新拟合后的 2.81 像素留出误差不是完全相同的抽样协议，因而只作量级/可用性比较。

因此拍投影模拟必须把**几何条件与照明/表面响应共同作为 setup 条件**，再检验新 setup 的泛化。想单独识别灯光响应，必须有投影机、相机、布幕位置均固定、只改变照明且有曝光记录的受控成对采集；这个公开路径标签不满足条件。此结论同样提醒拍屏和拍打印件研究：目录名是索引，不能代替逐张设备和姿态元数据。

逐成员 SHA/CRC 和全量比较记录：`work/compennetpp_sl_one_setup_manifest.json`、`work/compennetpp_sl_light3_pos1_cloud_manifest.json`、`work/compennetpp_cloud_two_setups_comparison.json`、两组 `work/compennetpp_sl_*decoding.json` 与空间格留出 JSON。原始图仅在 E 盘 `data/derived/compennetpp_sl_*`，报告/Git 不含图像复制件。

复算所选子集和两 setup 比较：

```powershell
uv run python -m experiments.projector.extract_compennetpp_sl_one_setup --setup light3/pos1/cloud_np --folder E:\ai_image_origin_research\data\derived\compennetpp_sl_light3_pos1_cloud --manifest work\compennetpp_sl_light3_pos1_cloud_manifest.json --reuse-source-folder E:\ai_image_origin_research\data\derived\compennetpp_sl_one_setup
uv run --extra analysis python -m experiments.projector.decode_compennetpp_sl_one_setup --manifest work\compennetpp_sl_light3_pos1_cloud_manifest.json --folder E:\ai_image_origin_research\data\derived\compennetpp_sl_light3_pos1_cloud --output-json work\compennetpp_sl_light3_pos1_cloud_decoding.json --preview work\compennetpp_sl_light3_pos1_cloud_preview.png
uv run --extra analysis python -m experiments.projector.evaluate_compennetpp_geometry_surrogates --manifest work\compennetpp_sl_light3_pos1_cloud_manifest.json --folder E:\ai_image_origin_research\data\derived\compennetpp_sl_light3_pos1_cloud --output-json work\compennetpp_sl_light3_pos1_cloud_holdout.json
uv run --extra analysis python -m experiments.projector.compare_compennetpp_light_setups
```
