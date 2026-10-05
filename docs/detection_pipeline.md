# 区域与原始来源推理接入

[架构](architecture.md) · [仓库首页](../README.md)

## 可调用实现

| 环节 | 实现 | 限制 |
|---|---|---|
| 定位 | OpenCV 四边形提议 | 可见平面边界候选，无语义分类；非四边形可能漏检 |
| 分割 | 官方 SAM 2.1 Tiny 自动掩膜 | 通用候选，可含背景/整帧；不保证找到承载图像的区域 |
| 深度来源 baseline | B-Free、D3 | 本机已有官方权重，保留各自预处理 |
| 传统来源 baseline | Benford 特征与已拟合森林推理 | 旧 RR 运行未保存森林；需要显式已拟合 NPZ，旧分数不能代替模型 |
| 自研算法/模型、训练 | 目录与接口已预留 | 尚未实现，不注册为 CLI 后端 |

SAM 2 使用[官方代码](https://github.com/facebookresearch/sam2)与[Tiny 官方权重](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)。OpenCV 操作见[形状处理文档](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)。候选提议与分割能力不能证明手机来源检测效果。

## 输入、分数与区域

- 非空 `uint8 H×W×3 RGB`，保存像素方向，不自动 EXIF 旋转或色彩管理。
- 原图整数 `x0,y0,x1,y1`，右/下边界不包含，越界拒绝；掩膜是全原图 bool。
- `natural/ai` 描述原始视觉内容；AI 图拍屏后仍标 AI。定位置信度不是 AI 概率。
- 分数更大更偏 AI，严格 `score > threshold` 判 AI。未经标定的 logit/森林分数不填 `ai_probability`。
- 区域模型输入为包围框内原像素，不透视矫正、不掩膜填充，可能含背景。掩膜保留在 Python 结果，JSON 仅写面积摘要。区域分布偏移尚待评测。
- 无候选返回空区域；整图模式显式选择，不作为失败 fallback。

## 依赖、权重与运行

核心检查无需 Torch；`classical` 提供 sklearn 森林导出与表示一致性测试，NPZ 推理本身不需要 sklearn：

```powershell
uv sync --locked --extra dev --extra classical
uv run --locked --extra dev --extra classical pytest -q
uv run --locked palimpsest-detect --help
```

可选 `neural` 提供 Torch/TorchVision/timm 等。GPU CUDA wheel 按 PyTorch 官方说明选择。第三方代码/权重仍在仓库外或被忽略的 `work/vendor`，不提交 Git。默认路径：

```text
repo/work/vendor/bfree/code/
repo/work/vendor/d3/ckpt/classifier.pth
repo/work/vendor/sam2/
models/bfree/BFREE_dino2reg4/     config.yaml 与模型权重
models/d3/ViT-L-14.pt
models/sam2/sam2.1_hiera_tiny.pt
```

SAM 2 本机 revision 固定为 `2b90b9f5ceec907a1c18123530e92e794ad901a4`。推理环境先有 Torch≥2.5.1、TorchVision≥0.20.1、setuptools、numpy、Pillow、tqdm，再安装：

```powershell
git clone https://github.com/facebookresearch/sam2.git work/vendor/sam2
git -C work/vendor/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
uv pip install --python <inference-python> hydra-core==1.3.2 iopath==0.1.10
$env:SAM2_BUILD_CUDA = "0"
uv pip install --python <inference-python> --no-deps --no-build-isolation -e work/vendor/sam2
```

本轮关闭自定义 CUDA 编译、建模后处理和小连通域处理；使用 16×16 提示网格，不采用官方速度表协议。权重 SHA 和本机实际版本记录在接入结果中。B-Free 原非营利许可仍适用于其外部代码/权重；公开本仓库不改变其许可。

在已具备依赖与权重的环境，于仓库根目录执行；`sample.png` 换成图片：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
python -m palimpsest.pipelines.cli sample.png --detector bfree --localizer quadrilateral --output work/new_prediction.json
python -m palimpsest.pipelines.cli sample.png --detector d3 --localizer whole-image
python -m palimpsest.pipelines.cli sample.png --detector bfree --localizer sam2 --max-regions 4
python -m palimpsest.pipelines.cli sample.png --detector benford --weights <fitted_forest.npz> --localizer quadrilateral
```

CLI 拒绝覆盖结果；模型仅显式加载，无权重就报错，不下载或制造预测。

## 保留的 baseline 协议

- B-Free：官方归一化、五裁切包装，>8M 像素用此前审计的 crop-first；可配置投影分块。不会强制改为统一的224输入。
- D3：本地 224×224、batch1、一次打乱+原图，默认初始 seed418；与作者 batch128 协议不同。检测器保存自己的 CPU/CUDA RNG 序列，恢复调用者状态。该隔离针对顺序调用，不覆盖其他线程在预测期间同时进行无关 Torch 抽样。输入顺序属于协议。当前适配器只支持 CUDA。
- Benford：论文启发适配版，中心方形→256 bicubic→内存 DCT→固定 Q95→9 个系数的 JS 与非零率，共18维；不是论文数值复现。`BenfordRFDetector.from_estimator` 导出已拟合森林，原 runner 的 `--save-forest` 为未来显式重跑保存 NPZ，本轮不执行拟合。

旧断点须用原 Git 版本与原环境恢复，不修改指纹绕过检查。新代码的计时包含新增 API/数组转换，历史延迟不改写。

## 评价与计时

`end_to_end` 包含解码、定位、裁切、预处理和同步前向计算，排除模型加载、JSON 序列化和写盘。`model_load` 单列；`cold_call` 包含加载。预热、硬件、图片尺寸、候选数需记录，小规模接入检查不支持实时性能结论。

| 入口 | 范围 |
|---|---|
| `evaluation/localization.py` | 明确标注区域的一对一 Box/Mask IoU、precision、recall |
| `evaluation/detection.py` | AUC、固定阈值 BA、类正确率、同源变化；转为零阈值 margin |
| `evaluation/timing.py` | 延迟分位数，须附样本量和预热条件 |
| `evaluation/channel_fidelity.py` | 固定检测器真实/模拟分数变化与失败来源重合 |

检测器响应一致不等于物理正确或训练收益。训练效用需要冻结来源、训练方案和外部真实传播验证，尚未实施。

小规模接入复算：`python -m experiments.origin_detection.integration.rr.run --output work/new_integration_check.json`。仅比较少数历史分数和人为边界场景中的真实推理，不估计准确率，不重跑全量 RR。
