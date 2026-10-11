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

## 下一组预先登记：滤波、平移与噪声

在第一组代码commit `14bfaf0`之后、谱域控制执行之前登记。只使用上组最近邻复制的4个独立单位方差潜变量、8个周期输出，以及上组二阶平移一致的6输出反例。所有矩都是已知总体量，暂不从单张图片估计。

采用单位化DFT，记C=FΣF*。预期：最近邻复制的C[2,6]=−i；循环平移1格后变成+i，模不变；后接因果[1/2,1/2]平均后为−1/2。另构造保留频率0、1、7的周期低通，预期所有非对角协方差消失，但输出均方非零。

独立白噪声协方差I应只增加频域对角线，C[2,6]不变；该对的谱相干模从1下降为1/2，因此不能把未归一化量的性质移给归一化量。独立但方差按[2,0]交替的噪声不满足平稳假设，预期为该频率对新增实部1，是噪声免疫强说法的负对照。

强制正反控制：空／非有限协方差拒绝；漏掉滤波转移函数的复共轭应与直接矩阵传播冲突；错的平移符号应被已知C值拒绝；错误“低通不能消除线索”应被非零信号对照拒绝；同一个关系对的非对角量与归一化相干须分别检查。预期对输入协方差作1格循环移位而非只移其中一轴。

## 下一组预先登记：有限窗口估计

在首次估计实验之前登记：只考察已知零总体均值的独立±1变量，窗口数M为1、2、4、8、16。最近邻复制4→8的单窗口频率乘积为−iU²，其中U=(z0−z1+z2−z3)/2；EU²=1、EU⁴=5/2。因此独立窗口平均对真实C[2,6]=−i的均方误差是3/(2M)。若所有窗口完全共享同一份z，误差恒为3/2，不随名义窗口数下降。

总体平稳的4个独立符号变量另作无耦合对照：F4的频点1与2的单窗口乘积，均值0、平方模期望1/2；独立平均的均方误差应为1/(2M)。M=1时统计量非零概率1/2，M=2时11/16。不能以有限样本非零作为存在总体耦合的证明。

实验穷举单窗口的16种符号组合，再用有理概率卷积计算M窗口平均的完整分布，不依赖蒙特卡罗种子。需核所有预写均值／均方误差；逐行改错MSE必须触发拒绝；完全共享窗口套用1/M法则和无耦合样本必为零均为强说法负控制。没有真实图片或窗口独立性的经验验证。

## 下一组预先登记：受训上采样层

在读取权重数值或执行受训层响应之前登记。对象是Stability AI发布的ft-MSE原始格式VAE，revision`629b3ad3030ce36e15e70c5db7d91df0d60c627f`，334641190字节，作者SHA`735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75`。它是只微调解码器的变体，不是原始kl-f8训练checkpoint。

仍执行固定CompVis源码中的Upsample类。三个明确层名为`decoder.up.1/2/3.upsample.conv`，参数与bias按原样载入，不训练或改权重。输入网格4×4，目标输出位置为(2+r,2+c)，r／c为0／1。按最近邻索引与卷积权重，把9抽头合并到所依赖的输入位置；预测每个输出的完整局部线性系数，再与实际源码的输入Jacobian比对。

pilot只检查up.1的第0输出通道；正式检查三个层的输出通道0–3。所有系数以及零输入输出必须匹配，容限rtol=atol=1e−12、CPU float64。每层放入一个错误系数必须被拒绝。报告的相位方差只是假设输入各坐标独立、零均值、单位方差时系数平方和；真实解码器的隐藏输入不保证满足此假设，因此不把该方差表当作最终RGB或AI类别的统计量。

这一步验证真实训练权重下的算子响应，没有完整生成器、语义合理样本或来源判别器。与之前手写核的区别须明确；不能因调用Torch实验工具就把它归入最终纯算法部署实现，也不能声称本项目训练了新模型。

## 完整编码／解码链协议与执行

[decoder_chain_probe.json](../../../../configs/evaluation/decoder_chain_probe.json)先提交Git，再以指定commit的blob SHA与working-copy比较，执行前将两者写入新收据。首先用确定性合成chart验证完整组件加载／实际隐藏输入的线性响应路径与成本，再读取已登记4个历史fit原图。选择只使用历史签名metadata中的domain／label／原图SHA，没有从特征数值筛选图片。

这一组检验“子层系数预测能否应用到相关、非高斯的实际隐藏输入”，不假定隐藏协方差是I。预写rtol／atol=1e−9，CPUfloat64；每层损坏输出必须被拒绝。末端RGB的相位表只作描述性记录，不以非零差异证明AI，原输入的真假标签不进入模型／统计计算。

模型参数严格完整载入，输入采用共享bicubic128方形缩放、归一到[-1,1]，编码后取后验mean再解码。该过程不是完整扩散生成，也不是AEROBLADE检测器复现：不调用LPIPS、不拟合阈值、不计算真假准确率。数据已反复曝光，只支持开发期机制观察，不能作为独立验证。

加载协议勘误：第一次chart在forward／像素读取之前，被“未解释参数”闸门拒绝。档案另含`model_ema.decay`与`model_ema.num_updates`两个标量buffer；已查固定版本`LitEma`源码及AE.forward，它们属于EMA记账，不进入此次forward。新协议逐个规定其shape／dtype并计入总数；未知额外键仍拒绝，没有用strict=False绕过，也没有修改数值容限。

chart之后、四张照片之前追加一个明确因果干预：固定编码mean与全部权重，只改decoder.up.1中最近邻为双线性、align_corners=False；两路进入该层的hidden须逐值一致，并核新插值核的局部系数预测。该干预由最近邻／线性插值的机制问题提出，不把新forward当同等训练的另一生成器，不预言最终RGB的统一变化方向。另记原图偶数格采样后最近邻重复的普通处理负对照；不声称其是真实平台或拍屏模拟。

代码独立核对每层输出通道0–3的全部valid内部像素，其余输出通道只参与完整forward。已有四图实验勿重复读取或运行；各收据及结论见[机制报告第10节](../../../../reports/04_算法研发/机制研究/从生成环节推导可观测差异_2026-10-11.md)。

## 四阶循环关系控制

[cyclic_cumulant_controls.json](../../../../configs/evaluation/cyclic_cumulant_controls.json)记录有限总体、独立／信号依赖噪声、低通、相位分组和有限估计对照。先提交本地Git，再以完整revision核working-copy和blob。三个递增协议与三份收据都保留；只验证已知分布，程序不读取图像。

```powershell
$revision = git rev-parse aba4219
python -m experiments.origin_detection.non_neural.generator_mechanisms.audit_cyclic_cumulants `
  --protocol-revision $revision `
  --output work/non_neural/generation_mechanisms/operator_moments/cumulant_recheck.json
```

使用仓库基础环境即可。共享四阶矩／k-statistic计算位于`palimpsest.evaluation.cumulants`；源标签、模型、阈值不进入此程序。有限Gaussian四阶匹配控制、Gaussian八阶求积和真实相机噪声是不同对象。结果见[第11节](../../../../reports/04_算法研发/机制研究/从生成环节推导可观测差异_2026-10-11.md)，不将该模块注册为来源判别器。
