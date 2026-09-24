# Simulation 受控采集准备与外部数据可用性审计

2026-09-24。已开始从机理研究转向可执行的测量设计；本轮没有新增物理模拟器或训练模型。审计依据是作者论文/官方仓库及本机已验证的档案。下文的“未发现”只限已查到的公开文件与说明，不等于作者从未保存相关元数据。

## 1. 对现有数据能做什么，不能做什么

| 数据 | 本轮核实的可用性 | 可用于当前 simulation 的问题 | 尚不能证明的事 |
|---|---|---|---|
| RRDataset | 官方测试档案已在 E 盘完整验证；同源原图、平台处理、再数字化图的对应清单已冻结。 | 发布结果分布及固定检测器的同源分数迁移。 | 再数字化的四种物理方式缺逐图标签/设备记录；不能据此校准各路径，更不能把 JPEG 采样格式当方式标签。[论文](https://arxiv.org/html/2509.09172)。 |
| UHDM | [作者项目](https://xinyu-andy.github.io/uhdm-page/)与[官方仓库](https://github.com/CVMI-Lab/UHDM/blob/main/README.md)给出 5,000 组真实屏摄与数字目标，官方仓库可见 Google Drive 下载入口。 | 屏摄后的颜色、残差、频谱及模型任务效应。 | 论文说明为配准后图像，官网未说明提供配准前照片、RAW、EXIF、每张设备/距离/角度记录；这些必须看实包后才可确认。项目页把 500 张称 testing，论文称 validation；沿用具体发布划分时应以文件审计为准。[原论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780634.pdf)。 |
| VD_raw | [官方仓库](https://github.com/tju-chengyijia/VD_raw)提供百度网盘入口，并明确 `moire_raw/moire_rgb` 和伪真值 `gt_raw/gt_rgb`；RAW 为含 black/white level、white balance 的 NPZ。 | 屏摄 RAW 与 ISP 后图的采样/噪声结构，若下载可用。 | `gt_raw` 是算法逆推的伪真值，不是屏前真实无莫尔纹 RAW；官方说明未给出逐张距离/姿态/曝光网格，不能凭它验证干预响应。 |
| DESCAN-18K | 官方数据页和[仓库](https://github.com/mlvc-lab/DESCAN-18K)给出真实印刷扫描/数字页配对、四扫描器。E 盘已下载官方 `Valid.zip` 与 `Test.zip`，各 360 对；档案 SHA-256 及逐图配对审计见本机旧记录。 | 已配准印刷扫描在未见扫描器上的颜色/频谱/纹理分布。 | 官方发布的是 1024×1024 配准裁切 TIFF；无法从裁切后的尺寸推断扫描 dpi 或原始页面几何。见下节本机标签抽查。 |
| VIPPrint | [Zenodo 官方档案](https://zenodo.org/records/4454971)是 14 个 7z 分卷、合计约 146.7 GB；[原论文](https://arxiv.org/html/2102.06792)说明真实自然/GAN 人脸打印扫描。 | 有自然/生成来源标签的真实打印扫描外部任务测试。 | 不下载完整分卷前，无法认证档案里是否同时保存逐张数字原图、稳定配对 ID、扫描元数据；暂不能作为已核实的成像机制配对。 |
| Bricker/BRACE | [论文](https://arxiv.org/html/2606.29845)报告有真实多曝光 RAW 250 训练/40 测试场景，适合研究滚动快门与屏幕调光。 | 论文可指导采集和时间积分模型。 | [作者公开仓库](https://github.com/ZZH-qwq/BRACE)当日只有 README/图片资源，下载与安装仍标 `TODO`；论文“available”不等于目前能获得数据/代码。 |

## 2. DESCAN 本机 TIFF 元数据抽查

在 E 盘官方 `Valid.zip` 与 `Test.zip` 中，已验证分别含 360 组同名 `scan/clean` 配对；每组 1024×1024。既有档案审计保存了以下 SHA-256：

| 档案 | 字节数 | SHA-256 |
|---|---:|---|
| `Valid.zip` | 754,770,440 | `f515fc2723db71e709588b2711577085318611684978fe4ad562042f452d6b2b` |
| `Test.zip` | 767,709,023 | `93c904410f409c1a393e5e79b35bf1748cb5d0ad2d18d1e1958d32213a9d8304` |

本轮不解包重复制图，而是按压缩档内每种扫描器前 20 张 `scan` TIFF 抽查，共 80 张，涵盖 `scanner01`–`scanner04`。全部有 1024×1024、8-bit RGB、LZW 等图像存储标签；在检查的 TIFF 标签中，没有 `XResolution`(282)、`YResolution`(283)、`ResolutionUnit`(296)、`Software`(305)、`DateTime`(306)、`Artist`(315) 和描述字段(270)。因此发布 TIFF 的这批样本**不能提供原扫描 dpi 与时间/设备元数据**；扫描器 ID 来自文件名/官方说明，而不是扫描图里的设备标签。这是 80 张抽查，不是对 720 张逐图元数据的穷尽证明。[官方数据说明](https://github.com/mlvc-lab/DESCAN-18K)。

## 3. 第一轮真正可证伪的拍屏测量

在固定数字源、显示器和相机下，分别改变距离、角度、对焦与曝光。每张保留未裁切相机 JPEG、可用时的 DNG/RAW、屏幕四角、设备设置和 SHA-256；同一条件重复拍摄。预测不是“有莫尔纹”，而是**投影格点尺度与局部峰频/方向联动，视角与颜色传递可能联动，曝光与屏幕调光条带联动**。若联动不成立，就必须调整或否定相关模块；弱/无伪影也是模型应覆盖的结果。[拍屏格点分析](https://www.commsp.ee.ic.ac.uk/~pld/publications/2013_ICASSP_Muammar.pdf)、[屏幕与相机颜色传递](https://arxiv.org/html/1501.01744)、[多曝光 RAW 条带研究](https://arxiv.org/html/2606.29845)。

完整操作矩阵、记录字段、随机化/重复采集和否证规则已写入项目内 `docs/controlled_screen_capture_protocol.md`。受控采集需要可设置曝光并最好能保存 RAW 的相机，以及可固定亮度/分辨率/图片显示矩形的屏幕；若目前无法接触设备，公开集仍可做部分观察性测试，但不能替代干预验证。

## 4. 仓库整理与下一步

目前项目目录原先未独立建 Git 仓库，Git 向上识别到了用户主目录。本轮已在项目内初始化 Git，建立根目录说明与 `.gitignore`，按 `outputs/` 交付、`docs/` 协议、`sources/` 文献目录、`work/` 旧实验工作区、E 盘数据划清位置。旧 `work/` 顶层 Python/PowerShell 脚本会作为历史复算入口纳入版本控制，临时数据、日志、环境和第三方代码排除。由于这些旧脚本仍被报告与断点引用，本轮没有移动或重命名；后续稳定代码先改调用与复算指纹，再退役旧路径。首次提交的结果另行核实记录，不把既有实验追认成事前预注册。

下一项技术工作是审计 UHDM 下载包的目录与 EXIF/原始版情况，优先寻找能补充拍屏几何与时间条件的公开数据。若公开集均缺控制量，采集协议便是进入物理 simulation 实现前所需的最小实测步骤。

## 5. 2026-09-24 仓库状态核实

项目独立 Git 根目录已经建立；首个提交为 `b89ce14 docs(research): establish project repository and capture protocol`。提交包含根目录说明、忽略规则、既有用户可读报告、文献证据目录，以及 `work/` 顶层的历史 Python/PowerShell 复算脚本；大型数据、第三方代码、日志、本地环境没有纳入提交。提交后 `git status --short` 为空。历史报告中少量 Markdown 行尾空格被 `git diff --cached --check` 提示，但属于此前已有内容；本轮保持研究记录原文，不把格式调整伪装成新的科学修改。

## 6. 2026-09-24 UHDM 下载入口的补充审计

继续只读核查[作者下载脚本](https://raw.githubusercontent.com/CVMI-Lab/UHDM/main/scripts/download_data.sh)与[公开 Google Drive 目录](https://drive.google.com/drive/folders/1DyA84UqM7zf3CeoEBNmTi_dJ649x2e7e?usp=sharing)：脚本写有 `train.tar.gz`、`test.tar.gz`、`test_origin.tar.gz` 三个旧文件 ID；当前公开目录可见 `test.tar.gz`（页面元数据约 5,943,548,688 字节）以及 `train_split`、`test_split` 两个文件夹。`train_split` 可见 18 个分卷，`test_split` 可见至少一个分卷。旧的 `train.tar.gz` 与 `test_origin.tar.gz` ID 在本机 HTTP HEAD 测试返回 404，不能把它们当当前可下载档案。

对当前 `test.tar.gz` ID 用固定版本 `gdown 5.2.0` 试行续传，工具返回“Cannot retrieve the public link”，未生成本地文件。公开目录列表能读取**不等于**该 5.9 GB 档案已成功下载，原因可能是大文件确认、共享权限或访问限额，本轮无法区分。没有据此推断文件内部是否含配准前照片或 EXIF。下一步优先找可访问的较小分卷/官方替代入口；如仍无法取得，就按现有公开证据将 UHDM 限于论文级观察，并转向可用配对数据与受控采集。
