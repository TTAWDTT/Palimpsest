# 受控拍屏首轮执行单

状态：物料已生成，**尚未拍摄真实设备照片**。本执行单用于检验屏幕—相机过程模拟，不用于直接训练自然摄影/AI 来源分类器。完整物理问题与可识别性见 [`controlled_screen_capture_protocol.md`](controlled_screen_capture_protocol.md)。

## 已准备的数字物料

代码：`origin_simulation/capture_kit.py`。本机主屏当前显示模式由 Windows 报告为 1920×1080；已经在 `E:\ai_image_origin_research\data\derived\controlled_screen_capture\kit_1920x1080_v2\` 生成 17 张同尺寸、无损 RGB PNG，及 `pattern_manifest.json`、`session_template.json`、`capture_log_template.csv`。图案哈希和测量 ROI 在 manifest 中。**主屏当前模式不证明面板原生分辨率；模版把后者留为 `unknown`。**

| 图案 | 用途 |
|---|---|
| `geometry` | 12×8 棋盘和四角不同颜色标记，确定角点、投影尺度和方向。 |
| `flat_000/016/064/128/192/255` | 重复平场，观察暗区、饱和、亮度曲线及信号相关噪声。 |
| `primary_red/green/blue`、`color_chart` | 锁定曝光和白平衡后的**显示器加相机联合**颜色响应。 |
| `slanted_edges` | 不同画面位置的近 5° 斜边，测联合边缘响应。 |
| `frequency_x_high` | 2、3、4、6、8、12、16、24、32、48、64、80 显示像素周期，沿 x 变化。 |
| `frequency_y_high`、`frequency_x_low` | 未用于第一轮拟合的方向和弱对比响应检验。 |
| `rgb_pixel_stripes`、`gray_ramp` | 未参与拟合的像素相位和连续灰阶检验。 |

图案标记的 `calibration`/`holdout` 是**数字刺激划分**；真实照片还须按距离、角度、设备和拍摄会话另外划分。`frequency_x_high` 的 2 像素周期已查验为逐列 `[218,38,218,38,…]`，避免 Nyquist 探针意外成为纯灰图。

## 显示前检查

1. 核对真实面板原生分辨率、当前运行模式、物理宽高、亮度、刷新率、色彩模式、系统缩放、HDR/夜间色温等设置；记录到会话 JSON。最好在面板原生分辨率呈现图案。若当前模式不同于面板原生分辨率，记下显示链额外缩放，不把图案视作一图案像素对应一面板像素。
2. 使用 `present` 命令在**主屏**不缩放显示；左右方向键切换，`Esc` 退出。按 `s` 保存 OS 合成画面的截图与逐像素比较报告。截图只校验合成画面，不能测量面板发光、PWM 或色彩响应；相机照片仍是必需的物理观测。
3. 屏幕最好完整进入相机视野，至少首次 `geometry` 保留屏幕四角与部分边框。相机固定焦距；曝光、ISO、白平衡、对焦尽量手动锁定。若不能锁定，把自动状态逐张如实记录。保存原生 JPEG；可用时并存 DNG/RAW，不覆盖原件。

```powershell
uv run python -m origin_simulation.capture_kit validate --kit-dir E:\ai_image_origin_research\data\derived\controlled_screen_capture\kit_1920x1080_v2
uv run python -m origin_simulation.capture_kit present --kit-dir E:\ai_image_origin_research\data\derived\controlled_screen_capture\kit_1920x1080_v2 --pattern-id geometry
```

## 第一轮空间响应：小矩阵

相机到屏幕的距离从**相机传感器平面**量到屏幕表面，记录毫米。角度约定相机从正对位置向右偏转为正 yaw；实际姿态与屏幕四角均需记录。先按设备能清晰对焦且屏幕可完整入镜的范围定三个距离，以及左右两个可复位角度，再冻结具体数值。各条件对每个指定图案连续拍三张，保留未裁切原件。

| 条件 ID | 角色 | 操作 | 图案 | 张数 |
|---|---|---|---|---:|
| `C0_start` | 标定 | 正对、中距离、基准曝光 | `geometry`、`flat_128`、`slanted_edges`、`frequency_x_high` | 12 |
| `C_near` | 标定 | 较近距离；重新对焦并记录焦点 | `geometry`、`frequency_x_high` | 6 |
| `C_left` | 标定 | 基准距离、向左倾斜；中心对焦 | `geometry`、`frequency_x_high` | 6 |
| `C_far` | 保留 | 较远距离；重新对焦并记录焦点 | `geometry`、`frequency_x_high`、`frequency_y_high` | 9 |
| `C_right` | 保留 | 基准距离、向右倾斜；中心对焦 | `geometry`、`frequency_x_high`、`frequency_x_low` | 9 |
| `C0_end` | 漂移检查 | 返回基准姿态、距离、曝光 | `geometry`、`flat_128`、`frequency_x_high` | 9 |

总计 **51 张原生拍摄**。`C_near/C_left/C_far/C_right` 的顺序应在拍摄前随机或轮换，并记录实际顺序；`C0_start` 和 `C0_end` 固定在首尾。距离与角度数据若与时间漂移混在一起，需根据首尾基准变化决定是否重采。第一轮只检验几何与频率随控制量的响应，以及弱伪影条件是否覆盖；颜色、对焦和曝光时序的独立干预在第一轮检查通过后再做。

## 会话建立与逐图记录

`pilot_001` 空会话已建立在 `E:\ai_image_origin_research\data\raw\screen_capture_sessions\pilot_001\`，**无需重复初始化**。把相机原件放在其 `camera_originals/` 中，再运行 `record` 自动计算 SHA-256；`capture_log.csv` 每行对应一张 JPEG/RAW 配对。所有未知量保持 `unknown`，不能填猜测值。RAW 的图像内容暂只核字节哈希，JPEG 还会检查可解码。若另开会话，使用 `init-session` 并给新的目录和 ID。

```powershell
uv run python -m origin_simulation.capture_kit record --kit-dir E:\ai_image_origin_research\data\derived\controlled_screen_capture\kit_1920x1080_v2 --session-dir E:\ai_image_origin_research\data\raw\screen_capture_sessions\pilot_001 --capture-id pilot_001_001 --pattern-id geometry --condition-id C0_start --repeat-index 1 --jpeg-path camera_originals\pilot_001_001.jpg --distance-mm 250 --yaw-deg 0 --pitch-deg 0 --exposure-s 0.01 --iso 100
uv run python -m origin_simulation.capture_kit validate --kit-dir E:\ai_image_origin_research\data\derived\controlled_screen_capture\kit_1920x1080_v2 --session-dir E:\ai_image_origin_research\data\raw\screen_capture_sessions\pilot_001
```

上述 `250 mm`、`0.01 s` 等只是**命令语法示例**，不是已拍摄参数。实际采集时应把显示器型号、面板原生分辨率、模式/亮度/刷新率、相机型号、焦距、RAW 状态、白平衡和曝光锁定状态填入 `session.json`；逐图焦点与曝光变动填入记录命令。让 `validate` 的缺失字段告警尽量清零，然后再做模拟器拟合。

## 物理验收边界

先只在 `C0_start/C_near/C_left` 估计几何与合成传递；冻结参数后预测 `C_far/C_right` 和 `frequency_y_high/frequency_x_low`。检查投影棋盘格尺寸/方向、局部频谱峰轨迹与强度、可见/弱伪影的覆盖，以及 `C0_end` 的漂移。需和简单 resize+JPEG、后期贴条纹的对照在相同末次裁切/编码下比较。若持有的仅是截图，结论只能到合成画面；若持有相机 JPEG 而缺 RAW，不能声称 RAW→ISP 子链已验证。

首张真实 `geometry` 照片到手后可先做角点检查。`geometry_measure` 会用四角颜色判定棋盘方向，输出 77 个内部角点、显示帧到相机图的二维单应性、中心附近每个显示像素投影的相机像素尺度及残差；若屏幕四角不可见，它会拒绝给出有方向的结果。它不把二维投影换算成相机距离、镜头参数或真实面板子像素间距。

```powershell
uv run --extra analysis python -m origin_simulation.geometry_measure --kit-dir E:\ai_image_origin_research\data\derived\controlled_screen_capture\kit_1920x1080_v2 --image E:\ai_image_origin_research\data\raw\screen_capture_sessions\pilot_001\camera_originals\pilot_001_001.jpg --output-json E:\ai_image_origin_research\data\derived\controlled_screen_capture\pilot_001_001_geometry.json
```
