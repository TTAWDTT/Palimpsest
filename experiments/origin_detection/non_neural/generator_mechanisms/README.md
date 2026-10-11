# 生成机制的准入与反例控制

## 相位矩的机制验证

2026-10-11新增`--suite moments`，不改变旧energy入口或历史收据。[预先写入的控制](../../../../configs/evaluation/polyphase_operator_controls.json)包含7个有限周期算子：覆盖不均、最近邻复制、后接两抽头平均、二阶相位均衡、非零输入均值、相关潜变量和零核。每项均有手写的均值／方差／相邻协方差／差分能量／中心四阶矩真值，每项故意改错方差必须被同一验证路径拒绝。

工具选型：已查BootLoops工具索引及emitall完整指南；emitall负责收据数值引用核对，不提供图像上采样的总体矩评估器。旧`audit_polyphase.py`只覆盖固定信号差分能量，因此扩展其控制入口，并把可复用的线性矩传播置于`evaluation/polyphase.py`，没有复制旧入口或新增兼容转发文件。

使用此套件来检查显式算子模型对像素关系的预测，尤其是单个差分能量量消失时其他关系是否仍存在。输入是周期、离散线性模型和已知潜变量总体，不读取真实图片、训练权重或类别标签。分数并不是AI概率。

输出保留各项精确分数、矩阵路线摘要、数值误差容限、控制分母、耗时和文件SHA。独立实现的有理枚举与NumPy矩阵路线共享算子定义和fixture；它们是计算交叉核对，不是独立科学数据或完整定理证明。正确性须通过预写真值、全均值／协方差矩阵交叉核对及每项损坏双胞胎；参数／协方差非法和分母变动另由测试拒绝。

```powershell
.venv/Scripts/python.exe -m experiments.origin_detection.non_neural.generator_mechanisms.audit_polyphase --suite moments --output <new_receipt.json>
.venv/Scripts/python.exe -m pytest tests/evaluation/test_polyphase_population.py tests/evaluation/test_polyphase_controls.py -q
```

## 原先的单量控制

本目录只检查候选统计量的机制假设，不提供默认AI检测器，不加载模型或神经特征。[证据与下一候选](../../../../reports/04_算法研发/机制研究/生成链候选准入_上采样与解码_2026-10-11.md)说明文献和适用边界。

`audit_polyphase.py`在有限周期信号上检查“奇偶差分能量对比”这一具体量。设 `E0`／`E1`为两个相位组相邻差分平方的平均，`T=(E1-E0)/(E1+E0)`；零能量返回未定义，不强制判自然。

运行前写入的解析真值覆盖最近邻复制、亮度偏移、相位平移、两抽头平均和常量。Fraction逐项计算与NumPy循环差分进行算术核对；错误答案及非法输入必须被拒绝。两条路线共享同一人工fixture与差分定义，这是有限软件核对，不是一般CNN定理或外部数据验证。

控制显示：普通放大能得到与生成器代理完全相同的结果；两抽头滤波把该对比从1变成0，信号均方仍为3/8。它排除了用此量直接证明“AI特有且传播不变”的做法，不否定NPR论文的实际CNN实验或其他局部统计量。

```powershell
.venv/Scripts/python.exe -m pytest tests/evaluation/test_polyphase_controls.py -q
.venv/Scripts/python.exe -m experiments.origin_detection.non_neural.generator_mechanisms.audit_polyphase --output <new_receipt.json>
```

输出拒绝覆盖。所有边界都是周期、偶长度、浮点／有理构造；没有8位取整、JPEG编码、真实照片或拍屏。不要把这个控制包装成完成了真实过程simulation或分类评测。

## 官方LDM单算子核查

`audit_ldm_operator.py`是具名一次性源码对照：读取固定SHA的官方`Upsample`类，单独执行最近邻放大与卷积，权重手动设为单位核和朝右下的2×2均值核，bias为0。它使用Torch作为受控算子的执行工具，不加载作者训练权重，不运行完整解码器，也不是检测器部署组件。

在3×3和4×4潜变量网格的逐基向量输入上，与NumPy周期相关预测比较内部区域；实际Conv2d采用零填充，因此常量输入的边界也单独检查。插入错误内部系数必须触发失败。工具选型沿用上述矩传播评估器，官方类从原始AST抽取并以源码SHA约束，未复制一份自行改写的Upsample实现。

```powershell
# 使用已有带Torch的研究环境；PYTHONPATH指向本仓库src与仓库根目录。
python -m experiments.origin_detection.non_neural.generator_mechanisms.audit_ldm_operator --source <pinned_model.py> --length 4 --output <new_receipt.json>
```

输出记录源码commit／SHA、两条手写核的相位方差、内部系数误差、常量边界差异及错误系数捕获。这里是实际源码类的手工权重控制，不能称作者预训练解码器的实验结果。相信它的前提是源码身份匹配、解析已知值和两种边界的干净／损坏双胞胎通过。
