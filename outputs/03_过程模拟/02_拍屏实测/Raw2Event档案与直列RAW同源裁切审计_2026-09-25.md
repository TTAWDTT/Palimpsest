# Raw2Event：`raw/` 档案与 `frames_raw/` 直列文件是同源裁切视图

**日期：**2026-09-25。**范围：**官方仓库固定修订 `9df99d9ed09e5ed705f50cae011e49cb2af620b9` 的 `raw/raw-0001-of-0250.tar` 中两段录像；不是 250 包或全库审计。

## 为什么核查

[官方数据卡](https://huggingface.co/datasets/raw2event/raw2event/blob/d400d349292a6ae1c8ea967e9ae99624d8b156cd/README.md)把 `frames_raw/<P>.mkv` 描述为拍屏的 10 位 Bayer RAW 流。仓库还保留了单独的 `raw/*.tar`；首包同名来源的 `_raw_10bit.mkv` 大小和画幅与直列文件不同。如果把两个入口误作两次独立真实采集，会使过程模拟的验证样本数和交叉验证失真。

我们只用 HTTP Range 读取 TAR 头及两个指定成员，不下载 17,852,917,760 字节的整包。每次要求 `206 Partial Content`、精确的 `Content-Range` 和 TAR 所列的成员字节数；下载到 E 盘后计算成员 SHA256，解码 FFV1。直列版的下载脚本此前已逐文件核对官方 LFS SHA256。**整份 TAR 的 LFS 指纹没有验证**，不把局部 Range 审计表述成整包验收。

## 逐像素和时间轴结果

两个 TAR 成员都是 614×456、`gray16le` 容器、60 fps；对应直列 `frames_raw` 是 692×520。二者每段均 317 帧，PTS 全部逐帧相同。用模板相关仅在五个预选帧中搜索 TAR 内容在直列画幅中的位置，找到同一个整数平移；随后**不再搜索位置**，在全部帧逐像素检查固定裁切：

\[
\mathrm{RAW}_{tar}(x,y,t)=\mathrm{RAW}_{direct}(x+71,y+15,t),
\quad 0\le x<614,\ 0\le y<456.
\]

| 前缀 | TAR 成员字节 | 固定裁切左上角 | 全帧检查 | 不同像素 | PTS |
|---|---:|---:|---:|---:|---|
| `10000_automobile_5_1087_20251224_105416` | 72,985,743 | `(71,15)` | 317 帧、88,754,928 像素 | **0** | 317/317 相同 |
| `10034_automobile_5_1356_20251224_110057` | 74,779,124 | `(71,15)` | 317 帧、88,754,928 像素 | **0** | 317/317 相同 |

合计 **177,509,856 个 TAR 像素完全相同**。两个来源都属于 automobile、同一采集日；它们是不同图像/录像，但不能据此把整个仓库的裁切规则外推到所有类别、日期或所有 TAR 包。

## 对物理模拟的修正

1. `raw/` 和 `frames_raw/` 中的同前缀成员在已验两例里只是**同一次相机测量的两个画幅**。不可把它们分别放入校准集和测试集，也不可把 614×456 的成员作为额外设备或额外实拍条件计数。
2. 之前基于直列 `frames_raw` 的数字源→RAW 前向 MAE、CFA 相位和积分速度实验，针对的仍是**真实拍屏后发布的 Bayer 像素**，数值结果不因发现裁切而消失。但它们**不能获得第二个独立 RAW 验证来源**；若改用 TAR 坐标比较，同一个传感器到显示平面单应矩阵应先右乘平移矩阵 `[[1,0,71],[0,1,15],[0,0,1]]`。
3. 画幅本身不能回答 692×520 是否传感器原生全视场、是否有读出模式或上游裁剪。两个入口的像素完全相同只证明局部裁切关系，**不能**推断镜头 PSF、屏幕格距、曝光或去马赛克。官方数据卡称 RAW 来自 Pi Camera Module 3；设备原生模式仍待采集代码或可靠元数据核实。
4. 数据去重应采用同一前缀为组单位，而不是文件路径为单位。对于后续不同档案入口，先查时间轴、画幅、固定整数平移及像素哈希，再决定是否是独立观测。现有冻结的 Raw2Event 来源划分以**前缀**为单位，本次没有打开原先保留来源，也没有重跑来源判别 baseline。

### 设备声明不能替代本次实包格式

作者先前的[仓库修订记录](https://huggingface.co/datasets/raw2event/raw2event/commit/5e6960ef9b9baf8868ba33fe430210275b791f89)曾把相机 RAW/RGB 都写作 **1296×972、50 fps**；已下载并解码的两套对应 RAW 实包分别是 **692×520** 和 **614×456**，均为 **60 fps**。这里不是判断作者原采集装置一定没有运行过 1296×972 模式，而是说**现有发布 MKV 不能按该文字直接当作 1296×972 录像**。真实输出之前有无传感器模式裁切、binning、重新马赛克或其他转换，现有 MKV/经验 DAT 都不足以认证。

[Raspberry Pi Camera Module 3 官方规格](https://www.raspberrypi.com/products/camera-module-3/)指出 IMX708 是 Quad Bayer 传感器，列有 2D 缺陷像素校正和 QBC re-mosaic 功能；这些是硬件能力，**不证明**本次 Raw2Event 采集启用了哪项。模拟器里的简单四相 Bayer 网格因此应读作对**已发布 Bayer 样本**的有效输出假设，不能自动上升为每个物理感光单元的排列或电子响应模型。

## 可复算记录

```powershell
python -m experiments.raw2event.fetch_raw2event_first_raw_tar_member
python -m experiments.raw2event.inspect_raw2event_tar_neighbors
python -m experiments.raw2event.fetch_raw2event_second_tar_member
python -m experiments.raw2event.audit_raw2event_tar_vs_direct
python -m experiments.raw2event.audit_raw2event_tar_vs_direct --prefix 10034_automobile_5_1356_20251224_110057 --out work/raw2event_second_tar_vs_direct_audit.json
```

两次成员范围/长度/SHA256 与视频格式在 `work/raw2event_first_raw_tar_member_audit.json`、`work/raw2event_second_raw_tar_member_audit.json`；五帧搜索位置、全部 317 帧像素比较及时间戳在 `work/raw2event_tar_vs_direct_audit.json`、`work/raw2event_second_tar_vs_direct_audit.json`。原始数据在 `E:\ai_image_origin_research\data\raw\raw2event_probe\`。
