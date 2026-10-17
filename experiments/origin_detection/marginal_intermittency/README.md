# M 四特征跨尺度白化统计判别器（`M_2a3a3c_zca`）

M 从局部亮度增量的跨距离矩比例、分布形状以及经过最细离散尺度净化后的二进跨尺度自参照差分计算来源证据，不依赖神经编码器、生成模型或 GPU。它完全复用单个 $64\times 4\times 4\times 4\times 3$ 结构函数张量（零额外滤波开销），仅使用 738 个 Chimera 原始来源（`stylegan2_orig`）做有标签稳健标定与 ZCA-CAT 去共线性白化投影；推理没有分类器训练或阈值搜索。

[算法实现](../../../src/palimpsest/detection/algorithms/marginal_intermittency.py) · [冻结参数](../../../src/palimpsest/detection/algorithms/m_calibration.json) · [实验报告与逐图分数](../../../reports/m_intermittency/README.md)

## 安装与单图推理

仓库要求 Python 3.11 以上；以下命令在仓库根目录执行：

```powershell
uv sync --locked
uv run --locked palimpsest-detect example.jpg --detector m --localizer whole-image
uv run --locked palimpsest-detect example.jpg --detector m --localizer whole-image --output work/m-example.json
```

`regions[0].score` 是 M；`origin` 为 `ai` 或 `natural`；`ai_probability` 为空，因为 M 不是概率。`metadata.qc` 和 `metadata.features` 是 JSON 字符串。退化图仍有有限强制分数，并由 QC 标记，不能把常量图的强制判断理解成有效来源证据。

`whole-image` 对应当前验证条件。定位后裁切可调用，但裁切后的准确率没有在本发布中验证。解码保持文件存储朝向，不自动执行 EXIF 旋转、ICC 转换或透视校正。

冻结参数随安装包发布。显式 `--calibration other.json` 可加载另一份 schema 2 参数，但其结果不属于这里报告的冻结 M；推理不会自动拟合参数。

## 批量图片推理

CSV 必需列为 `source_id,category,label,condition,path`，标签 `0` 表示真实、`1` 表示 AI。`source_id` 对同一原始来源的全部视图保持一致。可选 `image_sha256` 校验文件字节；`path` 相对 `--image-root`，省略该参数时相对清单所在目录。

```powershell
uv run --locked python -m experiments.origin_detection.marginal_intermittency.evaluate_manifest --manifest manifest.csv --image-root data --output work/m-fresh --bootstrap 2000
```

示例清单：

```csv
source_id,category,label,condition,path
example-real,example,0,original,real.png
example-ai,example,1,original,ai.png
example-real,example,0,screen,real-screen.png
example-ai,example,1,screen,ai-screen.png
```

输出 `scores.csv` 和 `metrics.json`。来源/条件重复、同来源标签冲突、输入缺失或哈希不符都会停止运行，不静默丢弃图片。输出目录必须尚不存在。

## 不下载图片即可复核公开成绩

发布的 CSV 包含 1155 条 M 分数和同图 B-Free 历史分数，完整保留全部分母。以下命令只读取分数，不运行模型、不下载图片或权重：

```powershell
uv run --locked python -m experiments.origin_detection.marginal_intermittency.evaluate_manifest --scores reports/m_intermittency/development_scores.csv --output work/m-score-check --bootstrap 2000
```

`metrics.M.all_views` 提供普通准确率、BA、pooled AUC 和类别宏 AUC；`paired_M_minus_BFree` 是来源级配对差值及区间。AUC 不等于准确率。重采样按类别×来源标签分层，抽取来源并携带其全部视图，避免把五视图当五个独立样本。

公开 manifest 的相对路径沿用本地数据准备的目录映射；它是输入身份和哈希记录，不是自动下载入口。若自行准备 Chimera，需要将图片映射到这些路径或生成自己的 CSV。数据访问路线见[Chimera 数据准备](../../data_preparation/chimera/README.md)。

## 固定数学定义

RGB/Lanczos 直接缩放到 256×256，除以 255 后以 float64 计算

$$Y=0.299R+0.587G+0.114B.$$

在整图上以 `reflect`、`truncate=4` 做 Gaussian 平滑，$\sigma\in\{1,2,4,8\}$；划成 64 个 32×32 块。在每块内，以 $r\in\{1,2,4,8\}$ 和四偏移 $(r,0),(r,r),(0,r),(-r,r)$ 计算绝对增量矩

$$S_p(b,\sigma,r,\theta)=\operatorname{mean}|Y_\sigma(u+d_{r,\theta})-Y_\sigma(u)|^p,\qquad p=2,3,4.$$

只用两端都在同块内的像素对，斜向距离为 $\sqrt2r$。令相邻距离的 $A=\Delta_r\log S_4$、$B=\Delta_r\log S_3$。相关矩正且有限、$|B|>10^{-4}\max(1,|A|,|B|)$ 时计算

$$f_1=\operatorname{median}_{valid}|A/B-4/3|,$$

$$f_2=\operatorname{median}_{valid}(\log S_4+\log S_2-2\log S_3)-\log(3\pi/8).$$

同时结合三项理论改进（2a + 3a + 3c）：

1. **离散尺度空间截断净化（2a）**：在细尺度 $\sigma=1.0$ 下，单位步长 $r=1$ 处于高离散 Péclet 数（$\mathrm{Pe}=r/\sigma=1$），受像素网格离散化、建筑 1px 人工阶跃边缘与屏幕子像素混叠主导。计算跨尺度轨迹时精确剔除单个高污染网格单元 $(\sigma=1.0, r=1)$，保留全部 $r\ge 2$ 与 $\sigma\ge 2.0$，在每个尺度 $\sigma\in\{1,2,4,8\}$ 上求中位数 $f_1^{\text{san}}(\sigma), f_2^{\text{san}}(\sigma)$。
2. **自参照二进跨尺度差分（3a）**：以粗尺度 $\sigma\in\{4,8\}$ 作为图像内部参考基底，定义细尺度与粗尺度的二进差分：

$$d_1^{\text{san}}=\log\frac{\frac{1}{2}(f_1^{\text{san}}(1)+f_1^{\text{san}}(2))}{\frac{1}{2}(f_1^{\text{san}}(4)+f_1^{\text{san}}(8))},\qquad d_2^{\text{san}}=\frac{f_2^{\text{san}}(1)+f_2^{\text{san}}(2)}{2}-\frac{f_2^{\text{san}}(4)+f_2^{\text{san}}(8)}{2}.$$

其中 $d_2^{\text{san}}$ 在理论上精确抵消屏摄光学 PSF 与传感器底噪注入的跨尺度共模加性对数累积量漂移 $\delta_H$。
3. **ZCA-CAT 相关调整白化判别头（3c）**：在 738 张 calibration 原图上，各轴经中心 $c_i$、尺度 $s_i=1.4826\cdot\operatorname{MAD}$ 与方向 $\operatorname{dir}_i\in\{\pm 1\}$ 标准化为 $z_i=\operatorname{dir}_i(x_i-c_i)/s_i$。用两类中位数中心化残差（Huber 截断于 $[-3,3]$）配合解析 Ledoit–Wolf 收缩估计类内相关矩阵 $R_W$，计算零截距 ZCA-CAT 权重 $w\propto R_W^{-1/2}\Delta_{\text{med}}$（归一化至 $\sum_{i=1}^4 w_i=1$）：

$$M=0.34547305\,z_1+0.21436778\,z_2+0.22461540\,z_3+0.21554377\,z_4.$$

$M\ge0$ 判 AI，$M<0$ 判真实。公共 `Prediction` 使用严格 `score>threshold`，因此适配阈值为零的前一个 float64 值 `-5e-324`，准确保留旧版的零点判 AI 规则。

中心为两类中位数的中点，尺度为合并标定数据的 `1.4826*MAD`（下限 $10^{-12}$），方向取两类中位数之差的符号。它们来自 738 个 calibration 来源的原图，发布时没有重新标定。

## 数学性质与实验范围

对预处理后的浮点亮度，在无裁切、量化和新增噪声的 $Y'=aY+b$、$a\ne0$ 下，$S'_p=|a|^pS_p$。相邻距离的 log 差消去常数增益，矩形状的增益系数 $4+2-2\times3=0$，故全部四项特征对非零线性亮度变换严格不变（误差 $<10^{-10}$），且 $d_2^{\text{san}}$ 对跨尺度共模对数矩偏移严格不变（误差 $<10^{-14}$）。该恒等式不涵盖 JPEG、gamma、缩放或屏摄。

Cauchy–Schwarz 给出 $S_3^2\le S_2S_4$。零均值 Gaussian 绝对矩给出形状参考 $\log(3\pi/8)$，只改变其方差的距离轨迹给出 $4/3$。这些参考对定义特征有用，并没有证明真实照片必须 Gaussian 或 AI 必须违反它们；来源方向仍由标定与实测确认。

ESS 的内部矩坐标借鉴 [Benzi 等的 Extended Self-Similarity](https://doi.org/10.1103/PhysRevE.48.R29)；自然图像尺度混合的解释参考 [Wainwright 与 Simoncelli 的 Gaussian Scale Mixtures](https://papers.neurips.cc/paper_files/paper/1999/hash/6a5dfac4be1502501489fc0f5a24b667-Abstract.html)。该借鉴不把湍流定律当成图像来源定律。

现有来源为 StyleGAN2 与 cat/church/horse 三类内容，覆盖原图和指定设备的普通/对抗屏摄。development 已多轮使用；本发布复算检验实现一致性，没有增加独立泛化证据。实测微信发送/转发/下载和未见生成器仍未验证。
