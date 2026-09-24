# Raw2Event：真实拍屏 RAW／ISP-RGB 配对的实包审计

**日期：**2026-09-25。**结论范围：**官方数据集中的两个不同类别完整录制前缀；不是全库审计、设备标定或来源判别评测。

## 这批数据能补哪一段证据

[Raw2Event 官方数据卡](https://huggingface.co/datasets/raw2event/raw2event/blob/d400d349292a6ae1c8ea967e9ae99624d8b156cd/README.md)说明，CIFAR-10 图像显示在屏幕上，由 Raspberry Pi Camera Module 3 同时记录 Bayer RAW 与经 ISP 的 RGB 视频，另有事件相机和元数据。官方描述为 59,454 个通过跨模态质检的录制前缀，轨迹为固定圆周运动。它提供 **真实屏幕→光学/传感器→ISP** 的观测机会，但刺激是 CIFAR-10 小图，不是本课题的“自然摄影／AI 生成”整图分类；已审样本中没有可直接使用的逐帧数字显示缓冲区。

本次下载两个前缀的 RAW MKV、RGB MKV 与 DAT，均放在 E 盘 `data/raw/raw2event_probe/`。获取脚本 `work/fetch_raw2event_pair.py` 分别生成 `work/raw2event_probe_download_audit.json` 和 `work/raw2event_probe_airplane_download_audit.json`。

| 前缀 | RAW MKV | RGB MKV | DAT |
|---|---:|---:|---:|
| `10000_automobile_5_1087_20251224_105416` | 96,702,984 B | 88,072,864 B | 12,046 B |
| `1000_airplane_1_9934_20251222_161953` | 98,562,806 B | 89,255,637 B | 12,046 B |

四条视频均通过官方 LFS SHA256 验证；两个 DAT 均通过官方 Git blob SHA1 验证。第一条录制的 RAW/RGB SHA256 分别为 `eb361455828467792a477962ce872f813c1e59b7b121069565eaffaaca4ed863` / `aa952db05ff7daa6343bdf58fa21e5171de250d53596f5e5829476f25d60556a`；第二条分别为 `eac4aac35b8188a874d104a189fd579de5cfd7e45da1595f3e66b8eed0a53f85` / `3421440677ec4c094234de25aa17160fedfa0a59154670f0af624bbac2ed34dc`。全部指纹见两份下载审计。

## 视频、RAW 位深与配准

`ffprobe` 和实际抽帧确认两个前缀的 RAW/RGB 都是 FFV1 无损编码、**692×520、317 帧**，各前缀内两条流的容器 PTS 均为 **317/317 完全相同**，首尾为 0.000 与 5.267 s。RAW 解码为 `gray16le` 容器；每条录制的五个抽样帧都在 10 位范围，按位 OR 为 1023。这支持数据卡的“10-bit Bayer”说明，不能把 16 位容器误称 16 位有效信号。样本内四种奇偶行列位置的均值有稳定差异，两个位置均值接近；这与 Bayer 周期相容，但具体 CFA 相位仍需颜色标靶或稳定配准后验证。

**同帧时间戳不等于同像素坐标。**同一图像中的 AprilTag 36h11、ID 0 在 RAW 和 RGB 上都能检出。五个抽样时刻，Tag 在 RAW 图上的平均边长约为 RGB 的 **1.707–1.737 倍**。以第 0 帧为例，RAW Tag 中心约 `(505, 401)`，RGB 中心约 `(441, 344)`；直接按 `(x,y)` 比较 RAW 与 RGB 会把不同物理位置错配。对每帧分别用 Tag 四角求平面单应矩阵并用公共有效区域评估，7×7 Gaussian 轻度平滑后的亮度 Pearson 由未配准的 **0.370/0.402/0.439/0.400/0.403** 变为 **0.959/0.839/0.843/0.921/0.753**（帧 0/80/160/240/316）。这表明 Tag 配准有实效，也表明不同姿态/运动和 Bayer、ISP 差异仍留下残差；这个 Pearson **不是**模拟保真度或 ISP 校准精度。

第二条 airplane 录制复现了相同结构：五帧 Tag 边长比 **1.708–1.727**，相同亮度对照由未配准的 **0.395–0.430** 升到逐帧 Tag 配准的 **0.816–0.930**。为检验变换是否只是每帧自由拟合，用每条录制**第 0 帧**的 Tag 单应矩阵预测另外四个抽样帧的 RGB Tag 四角，平均误差 automobile 为 **0.82–2.21 像素**、airplane 为 **0.92–2.35 像素**。这支持两条录制中相对稳定的 RAW→RGB 几何关系；五帧抽样仍不足以保证所有录制都固定。

![同一录制首帧的 RAW、ISP-RGB 与基于 AprilTag 的 RAW 配准示意](Raw2Event_RAW_RGB取景对照_2026-09-25.png)

## 元数据声明与实包可见字段

官方数据卡把 `meta_raw/<P>.dat` 描述为含时间戳、传感器序列号、曝光、增益、同步偏移。**两个**实包均为 **12,046 = 317×38** 字节；逐条观察到前 8 字节可按小端 `float64` 读出递增数值（帧间增量中位 16,650），随后 26 字节为可解析的日历时间文本，再随后 4 字节在每条录制的 317 条中全为零。记录数均恰好等于视频帧数。

**在这两个文件内，未识别出可独立验证的曝光、增益或传感器序列号字段。**38 字节结构仅为本次实包的经验拆解，不是作者公开的正式二进制规范；不能据此断言全库或其他元数据包都没有这些字段，也不能把零值尾缀猜作具体物理量。相机曝光/增益目前仍是待证据约束的参数。

## 对过程 simulation 的具体作用和限制

| 可做 | 尚不能做 |
|---|---|
| 在真实同帧数据上核对 RAW 数值范围、CFA 周期、运动时序和 RAW→ISP-RGB 的条件映射；用 Tag 先做显示平面配准。 | 直接按相同像素索引拟合 RAW→RGB；把画面的几何差当镜头变焦或真实显示尺寸。 |
| 把同一设备的部分录制前缀分为校准／整段留出，评估模拟 RAW/ISP 噪声、色彩和局部结构。 | 用这一条录制推断所有手机相机、不同屏幕或距离×角度×曝光干预网格。 |
| 形成“CFA→ISP”机制的独立证据，供 Chimera 拍屏反证后定位缺失环节。 | 用 CIFAR-10 类别声称 AI／自然摄影判别能力，或从这批 RAW 直接解释 Chimera 的 B-Free 分数下降。 |

**下一实验入口：**先确认 CFA 相位和影像配准在不同前缀、不同画面上是否稳定，再在校准前缀上估计黑电平、噪声—信号关系及低维 ISP 映射；以未参与拟合的前缀检验 RAW/RGB 像素统计和时间变化。只有找到真实数字显示帧及可靠物理设置记录，才能进一步约束完整“数字源→显示→镜头→RAW→ISP→发布”链。Chimera 的 840 个 reserved 来源继续不用于参数挑选。

## 复算

```powershell
python work/fetch_raw2event_pair.py
python work/audit_raw2event_probe.py
python work/fetch_raw2event_pair.py --prefix 1000_airplane_1_9934_20251222_161953 --audit work/raw2event_probe_airplane_download_audit.json
python work/audit_raw2event_probe.py --prefix 1000_airplane_1_9934_20251222_161953 --out-dir work/raw2event_probe_airplane --report work/raw2event_probe_airplane_pixel_audit.json
```

抽帧、PTS、Tag 角点、亮度对照和经验元数据结构记录在 `work/raw2event_probe_pixel_audit.json` 与 `work/raw2event_probe_airplane_pixel_audit.json`；示意图源帧位于各自的 `work/raw2event_probe*/`。脚本为只读核查，不执行来源检测器推理。
