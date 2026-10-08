# 噪声取整探测的公式／单位研究契约（2026-10-08）

本契约核数学与读法，不创建任务目标，不复现分类器，不开独立图像或训练神经网络。Regime1：量化器及边界已定义，允许证明、精确反例或明确未闭合。

## 来源与待核对象

CoDA arXiv2605.24306v1：https://arxiv.org/html/2605.24306v1。
Algorithm1 lines258–277：I∈[0,255]，Y=clip(round(I+N),0,255)，Δ=I−mean(Y)。
Eq12–15 lines197–212：x=I/256，Y=round(x+N)，B=E[Y|x]−x，无clip。
Eq18–20 lines223–231列复系数、正弦级数和首谐波；Eq21–25 lines234–253连接Δ和色分布。
论文VI-B lines393–394声明σ=.10/R50，但没有在已读处统一上述单位。
这些是源中定义，不假定其实际代码按字面实现。外部代码未取得，不作论文实验无效／欺诈结论。

## 预注册数学陈述

A：对任意整数n及关于0对称的可积随机噪声N，若所有半整数位置无概率质量，无clip，round是最近整数，则round(n+N)−n与round(N)同分布，E[round(n+N)−n]=0；R个IID复制的均值残差分布也不随n改变。
B：带clip的A不可全域延伸；仅允许报告精确饱和边界反例。
C：normalized fractional x模型与byte整数n模型不是同一quantization step/noise unit；统一尺度必须明确噪声、步长、clip和重标定。
D：全局signed mean B为0不能推出residual vector norm为0；非均匀颜色分布也不必有非零signed mean B。只找反例，不推分类准确率。
E：核Eq18的复系数与Eq19的实级数是否可同时成立，以及Δ=−B的符号。未获得PDF原tex/作者代码时，数学矛盾只能称当前HTML公式一致性疑点。

## MAY-ASSUME／MUST-NOT-ASSUME

允许：最近整数定义（明确ties convention）、概率对称/可积、线性期望、有限有理数枚举、积分和及Gaussian characteristic function等标准事实但必须明确引用/推导所需条件。
不得假定：A本身、ISP和生成色彩的普遍分离、所有拍摄都仿射、热图残差即真实性、无clip近似可用于饱和区域、代码按HTML或任何一套单位运行、数值小试即完整分类证据。

## 精确T1及许可降级

T1a：N取{-3/4,-1/4,1/4,3/4}各1/4，n={0,64,128,255}；无clip的均值B应全部0，差值law必须逐项相同。
T1b：同T1a带[0,255]clip，n0／255的B应分别+1/4／−1/4；中间n64／128应0。
T1c：N0、x={1/4,3/4}各1/2；B分别−1/4、+1/4，signed mean0而E[B²]=1/16。
T1d：N取±1/10各1/2，x1/4的B=−1/4；整数n64的B0。只演示单位差别，非Gaussian实验替代。
精确负控：把A改成“带clip全域同law／零mean”必须由T1b拒绝；把D改成“signed mean0⇒L2零”必须由T1c拒绝。
若A有符合假设的精确反例，标为REFUTED并保留；若只找到ties/clip越界反例，不移动A而标边界。无tie定义或源公式单位无法确定，标AMBIGUOUS/OPEN，不暗改假设。

## 执行与独立性

Root只抽取契约和组织；先独立X0 adversary，随后两个禁止交叉阅读的不同路线（离散对称／分布积分与Fourier）。各自先写骨架及自身脚本/输出，精确fixture必须实际执行；不得读取其他arm结果。最后未参与撰写者从契约独立枚举并核消费者接口，2–3冷context无反例轮前不叫VERIFIED-CLOSED。缺diversity或fresh verify只叫PROVISIONAL。算术toy不证明CoDA或本项目分类稳定性。
输出只在本任务work及reports/04_算法研发；源码相对工具代码可注册于experiments/origin_detection/quantization_contract。没有detach、自动化、Goal API或外部作者联络；长任务使用受管会话，遵守用户阶段约束。
