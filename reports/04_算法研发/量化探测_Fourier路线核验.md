# 量化探测：Fourier 独立路线核验

日期：2026-10-08。作者身份：Lady 的 Fourier proof arm。**自评 CLOSED-UNVERIFIED / PROVISIONAL**，未声称 VERIFIED-CLOSED。需要未参与撰写的 fresh verifier 独立重建、独立脚本和消费者接口核验，并由总流程记录 2–3 连续冷 context 无反例轮。

## 范围与独立性记录

本路线只先读 `docs/research/quantization_probe_contract.md`，立即建立 `work/robust_statistics/quantization_contract/fourier/journal.md`。没有读取 X0、其他 arm、既有 proof report 或其代码。全局结论来自锯齿 Fourier 系数及卷积推导；18 条有理数断言和 8 个具体 Fourier 积分只是有限检查。本报告不评价论文分类准确率、训练结果或实际作者实现。没有图像数据集、模型、分类器训练、Goal API、自动化、外部联系或 detached helper。

## 已读来源：版本及可见公式

来源为 CoDA: Color Distribution Probing for Efficient and Generalizable AI-Generated Image Detection，Zexi Jia 等，arXiv:2605.24306v1 [cs.CV]，PDF 首页日期 23 May 2026。本地 `source/CoDA_v1.pdf` 的 SHA256 为 `5a21e33267e5162711fa5b4f1f42d7b05354b2bbdb430e51a05844ce502ae332`，与 fetch.json 一致。来源 URL：https://arxiv.org/pdf/2605.24306v1。用自身 pdftoppm 调用重新渲染整页 6–7，并实际看图核对；没有只依赖提取文本或另一 arm 的转述。输出是 `fourier/own_pdf-06.png`、`own_pdf-07.png`，首页身份文本是 `pdf_identity.txt`。

读到的定义与公式（数学转录，未伪装为逐字英文引文）：

| 定位 | PDF 可见内容 |
|---|---|
| p6 Eq11 / p7 Algorithm1 line11 | Δ=I−Î |
| p6 Eq12–15 | x=I/256∈[0,1)，Y=round(x+N)，N~N(0,σ²)，B(x;σ)=E[Y|x]−x；Eq13 用两侧半整数边界 Gaussian CDF |
| p6 Eq16–17 | B=Σ a_m exp(i2πmx)，a_m=∫₀¹B(x)exp(−i2πmx)dx |
| p6 Eq18 | a_m=i(−1)^m/(mπ) exp(−2π²σ²m²)，m≠0 |
| p7 Eq19 | B=Σ_{m≥1}(−1)^m/(mπ) exp(−2π²σ²m²) sin(2πmx) |
| p7 Eq20 | B≈C(σ)sin(2πx)，C(σ)=+(1/π)exp(−2π²σ²) |
| p7 Eq21 | E[Δ_hwc|x_hwc]≈B(x_hwc;σ)≈C(σ)sin(2πx_hwc) |
| p7 Algorithm1 line1/8 | I∈[0,255]，Y_r←clip(round(I+N_r),0,255) |
| p7 Eq25 | ||Δ||≈|C(σ)Σ sin(2πx_hwc)|，该显示公式没有给 norm 的定义 |

契约 E 原定没有原 TeX 或作者代码时只称 HTML 一致性疑点。本路线新增同版 PDF 直读证据，因此可精确报告**所读 v1 PDF 的 Eq18/19/20 内部代数不一致**；原 TeX、作者意图及实际实现仍未核实。没有将此扩大成论文实验无效或欺诈判断。

## 解析推导：假设与 Fourier convention

设最近整数算子 Q，ties 可以任取确定规则；所有实际结论避开半整数的概率质量。Gaussian σ>0 时半整数概率自动为 0。无 clip、步长 1，定义 q(t)=Q(t)−t。除半整数外 q 为周期 1 的有界锯齿，q(t)=−t 在 −1/2<t<1/2，|q|≤1/2。Gaussian N 可积且 EN=0，于是

B(x;σ)=E[Q(x+N)]−x=E[q(x+N)].

这里积分/期望存在，因为 |Q(x+N)|≤|x+N|+1/2。取与源 Eq16–17 完全相同的 convention：q(t)=Σ c_m exp(i2πmt)，c_m=∫₋₁/₂¹/₂q(t)exp(−i2πmt)dt。c₀=0。m∈Z\{0} 时分部积分得

∫₋₁/₂¹/₂(−t)exp(−i2πmt)dt
= [t exp(−i2πmt)/(i2πm)]₋₁/₂¹/₂ − ∫₋₁/₂¹/₂ exp(−i2πmt)/(i2πm)dt
= (−1)^m/(i2πm).

末项积分为 0，边界用 exp(±iπm)=(−1)^m。这是任意非零整数 m 的代数推导，不是由 m=1..8 推广。

Gaussian 平滑后的系数可以直接从积分而非未收敛的点态级数交换得到。令 b_m=∫₀¹E[q(x+N)]exp(−i2πmx)dx。q 有界，Fubini 适用。对每一个 N=t 平移周期积分，内积分为 c_m exp(i2πmt)，从而

b_m=c_m E exp(i2πmN)
=(−1)^m/(i2πm) exp(−2π²σ²m²)，m≠0，b₀=0。

这里唯一用到的概率标准事实是契约许可的 Gaussian characteristic function；可由 Gaussian 密度积分配方或分部积分得到 φ′(u)=−σ²uφ(u)、φ(0)=1，故 φ(u)=exp(−σ²u²/2)。

为避免将 coefficient agreement 当成全局函数等式，还须处理收敛与唯一性：σ>0 时 Σ_{m≠0}|b_m|<∞，因为 e^(−c m²)/|m| 被可求和几何级数控制，故候选级数绝对且一致收敛。B 连续：用卷积表达后，|B(x+h)−B(x)|≤(1/2)||g_σ(·−h)−g_σ||₁，Gaussian 密度平移的 L1 连续性可由其可积导数界 ||g(·−h)−g||₁≤|h|||g′||₁ 得到。

本段公开使用并给出 Fourier 唯一性的基础证明，防止隐性导入：若连续周期函数 f 的所有系数为 0，则其与 Fejér 核 K_M 的卷积恒为 0。K_M≥0，∫₀¹K_M=1，并且在离整数距离≥δ的区域 K_M(t)≤1/((M+1)sin²(πδ))。连续周期函数一致连续，因此将积分拆为 |t|<δ 和其补集，即得 K_M*f→f 一致收敛；于是 f=0。K_M 的公式是 (M+1)⁻¹|Σ_{j=0}^M exp(i2πjt)|²，其非负性、归一化和界分别直接来自平方、有限积分和几何和。将此用于 B 与候选级数之差，获得完整等式。

配对 ±m，得到正确的实级数：

**B(x;σ)=Σ_{m=1}^∞ (−1)^m/(πm) exp(−2π²σ²m²) sin(2πmx)，σ>0。**

## Eq18/19/20 与 Δ 的裁定

1. 源 Eq19 与上面的解析结果一致。
2. 正确复系数是 **a_m=(−1)^m/(2πim)exp(−2π²σ²m²)**。源 Eq18 的 i(−1)^m/(mπ) 是它的 **−2 倍**。按源自己的 exp(+i2πmx) convention，把 Eq18 的 ±m 配对，会给 −2(−1)^m/(πm) 的正弦系数，因此 Eq18 和 Eq19 无法同时成立。不能用未声明的 convention 消除该矛盾，因为 Eq16–17 已声明 convention。
3. Eq19 的 m=1 项为 **−(1/π)exp(−2π²σ²)sin(2πx)**。Eq20 对 B 写了正号，首项符号不一致。
4. 对同一无 clip、同尺度模型，有限 R 条件均值 E[Δ_R|x]=x−E[Y|x]=**−B(x;σ)**，R 只影响随机波动，不改该均值。Eq21 写 EΔ≈B 的第一处等号方向不一致；但 Eq20 的 B 符号也反转，两个错误在最终 +C sin 的 Δ 首项中恰好相消。不能据此把 Eq21 中间的 EΔ≈B 判为正确。

首谐波近似必须带误差条件。设 c=2π²σ²>0，B=−e^(−c)sin(2πx)/π+R₁(x)，则

|R₁(x)|≤(1/π)Σ_{m≥2}e^(−c m²)/m≤e^(−4c)/(2π(1−e^(−5c))).

第二个界由 m≥2 时 1/m≤1/2、相邻指数平方差≥5 得出，适用于全部 x。它是绝对误差界，并不在首项为 0 的 x 处给出相对误差，也不授权任意小 σ 下“首谐波占优”。本路线没有数值截断冒充无穷 Fourier 证明；σ=0 的锯齿只能按非跳点的 Fourier/Fejér 意义处理，不主张绝对或一致收敛。

## A 的全域条件：由 Fourier/Fejér 取得整数处零偏差

Gaussian 结论立即给 B(n;σ)=0，因为全部 sin(2πmn)=0。契约 A 更一般的噪声可用本路线的 Fejér 版本：若 N 关于 0 对称、可积，且所有半整数位置无质量，q 的 Fejér 多项式在 N 的几乎每个实现处趋于 q(N)，且因 K_M 非负归一化，绝对值均≤1/2。每一项是正弦项，E sin(2πmN)=0（有界奇函数在对称分布下积分为 0），故多项式期望均为 0。支配收敛给 Eq(N)=0；对称与可积给 EN=0，于是 E Q(N)=0。

最后用最近整数的点态平移定义：对非 half-tie t，Q(n+t)=n+Q(t)，因到每个候选整数的距离都只是整数平移。故 Q(n+N)−n 与 Q(N) 是同一个随机变量的同一函数，law 不依赖 n。R 个 IID 噪声的向量 law 是同一积 law，其均值及 Δ_R 的 law 也不依赖 n。law 的平移部分来自算子的定义；零均值部分采用 Fourier/Fejér 路线，不假定 A 作为许可引理。可积条件保证 Q(N) 可积。half-tie 排除不可省略为任意 rounding convention 的声明。

## 精确执行 T1a–d、clip 边界与错误主张 plants

一次性脚本：`work/robust_statistics/quantization_contract/fourier/exact_check.py`。脚本只用自身 Fraction 最近整数枚举和 SymPy 小积分，没有其他 arm 或源码实现复用。最近整数在 ties 明确选下侧整数，T1 输入全无 ties，所以 fixture 不依赖这一选择。所有预注册期望来自契约。BootLoops suite 通过 `source /root/.local/share/bootloops-suite/env.sh` 激活并用受管 WSL 前台会话运行。

| 契约 | 精确输出 |
|---|---|
| T1a，n=0,64,128,255，无 clip | 全部 B=0；全部残差 law 为 {−1:1/4,0:1/2,+1:1/4} |
| T1b，带 clip | B(0)=+1/4，B(64)=B(128)=0，B(255)=−1/4；n0 law={0:3/4,+1:1/4}，n255 law={−1:1/4,0:3/4} |
| T1c，N=0，x=1/4,3/4，各1/2 | B=−1/4,+1/4；signed mean=0；E B²=1/16 |
| T1d，N=±1/10 各1/2 | x1/4 的 B=−1/4；整数 n64 的 B=0 |
| exact coefficient m=1..8 | 周期积分=推导系数=Eq19 implied coefficient；Eq18/推导=−2 |

负控逐项实际触发：clip 全域零偏差、clip 全域同 law、signed mean0⇒L2零、正确 coefficient 符号翻转、正确 coefficient 放大 2 倍，共 **5/5 CAUGHT**；正控 exact fixture **18/18 PASS**，exact integral **8/8 PASS**。这些有限计数不是全域分类稳定性的证据。

执行 log 的 5 行原文：

```text
Exact fixture assertions: 18 PASS
Exact Fourier integrals: 8 PASS; Eq18 / derived = -2; Eq19 matches
Wrong-claim/coefficient plants: 5 CAUGHT
Grade: CLOSED-UNVERIFIED / PROVISIONAL
Script SHA256: f9f0987c0575f04bb93107fccec9e9131498842f5da1fbd74ada3220c1e6383e
```

`results.json` 保存逐项有理数、law、plant witness、Python/SymPy 版本、脚本 SHA256 与执行耗时；`run.log` 保存输出。没有运行 Gaussian CDF 数值对照，也没有数字精度认证主张。

## C/D 及消费者接口限制

统一尺度需同时变换量化步长、噪声与 clip。写 Q_h(z)=h round(z/h)。byte 模型步长 h_byte=1，x=I/256 时保持同一算子必须用 h_norm=1/256、N_norm=N_byte/256、clip 上界255/256。源 Eq12 的 round(x+N) 使用 h_norm=1；它折回 byte 单位是步长256，Gaussian normalized σ=.1 折回 byte 为25.6，不能与 byte σ=.1、步长1的模型直接等同。这是字面定义的尺度区别，不是对作者代码的猜测。

T1c 同时是非均匀颜色分布却 signed bias mean 为0的精确反例。它也给 residual vector B=(-1/4,+1/4)，||B||₂²=1/8>0；平均每项平方为1/16。因而 zero signed mean 不推出 residual vector norm 为0。源 Eq25 若消费者将 ||Δ|| 解释为通常 L2（或 L1、L∞），signed sum 的绝对值不能等同 vector norm；若解释为 |ΣΔ|，则是不同统计量，必须显式定义。Gaussian 噪声下有限 R 的随机 norm 还不能等同 mean vector 的 norm。本报告只给数学接口条件和反例，不把它外推到分类准确率。

clip 的 T1b 是边界反例，不推翻无 clip 且无 half-tie 的 A。对连续 Gaussian 严格来说所有有限 x 都有越过有限 clip 端点的正概率，所以 away-from-saturation 的 unclip 只是待量化近似；本路线没有对该近似作全域等式声明。当前既未核作者源码，也未闭合 clip、scale、finite-R、norm 到最终检测性能的接口。

## 待 fresh verification 的清单

- 从契约独立重写全部 T1 和 plants，不复用此脚本。
- 检查任意 m 的分部积分边界符号、Fubini、Gaussian characteristic function、Fejér 唯一性和对一般对称噪声的 dominated-convergence 步骤。
- 用自身 PDF 渲染核 Eq18–21、Algorithm1 的 units/clip/Δ 以及 Eq25 的实际消费者定义。
- 区分 Gaussian σ>0 的 absolute/uniform convergence 与 σ=0 的跳点极限；区分 analytic deductions 与18+8个有限检查。
- 保留 CLOSED-UNVERIFIED / PROVISIONAL，直至总流程的 fresh verification 与 consumer interface trace 完成。
