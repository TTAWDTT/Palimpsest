# 第60轮：紧凑判别方向上的有限视图间隔，事前协议

第59轮流程对齐后仍下降4.44pp。本轮检验有限训练视图的共同间隔，而非让所有远离边界的分数也一致。复用第26轮的最大hinge LP；它在原CLIP表示上失败，本轮更换为当前三个拟合内判别方向，不称新损失原则或不变性证明。

只读取已签名的第56轮2576维数字缓存，不推理图像、不训练神经参数、不打开RR reserved或独立外部像素。反复曝光的RR／Chimera仍只是开发集。冻结来源1260 fit、420 threshold、420 selection；fit与threshold只用raw／Q90，selection四编码保持现有定义。

三个基与末级共享训练行，各折全部重拟合，不使用旧全fit模型，也不称OOF stacking。每个准确训练数组建立一次基组。现有基提供器需要一个零强度logistic头初始化；共12个基组、36次基拟合和12次辅助logistic末级拟合，辅助头不用于最终判别。新末级40次CV及4次最终LP另计。

对标准化三分数z、来源s、标签y=±1：最小化加权来源slack加L1系数乘|w|和，各来源每个视图满足y(w·z+b)≥1-slack_s。固定L1网格(.0001,.001,.01,.1)，fit内部采用第59轮三折拟合／一折class-threshold标定／一折评价，选择规则原样复用。真实内部标签均为真标签；null内部全部为伪标签，最终threshold仍真标签，不是部署路径的置换p检验。内部拟合规模与全fit不同。

四最终头：zero为固定L1 .001的平均逐视图hinge，selected为内CV选择的最大来源hinge，错源只改变末级来源slack分组、基保持真fit来源，null则全部基与头重拟合伪标签。若selected惩罚不为.001，zero对照同时改变聚合与惩罚，不能单独归因于最大聚合。wrong按域／场景／类别／条件／编码分层置乱，保留同类及面板计数。

LP复用SciPy HiGHS-IPM、120秒求解限时，重算原始／对偶可行性及gap≤1e-6。真实fit前运行解析对称间隔、来源复制等权、不可行／假对偶、标签冲突、零信息hinge=1、基缓存隔离／类别反转／保存加载／单条批量及三角色分离控制。故障停止、存失败而不作分类结论。

先测一个真实三折训练冷基＋最大hinge（.0001）和同基暖LP（.1），包含标定／held分数；10冷＋30暖只作条件代理，排除输入、全fit及外层评价；任次超过120秒先停止重估。完整CV输出以每个强度的五折OOF记录为进展单位。最后四头各60指标／130配对，整数核算和错误BA负控；内OOF30／35重选及角色记录核对，emitall和错误转录孪生。软件核对不独立认证真实图像标签、优化器训练成员或泛化。

验收仍是BA≥80%、最大同源退化≤2pp以及独立真实数据验证；既有开发集达标也不单独完成验收。有限训练特征凸包的共同margin不覆盖未知真实处理，class-threshold标定后还会移动边界。使用单图分数，不用查询来源、处理标签或配对原图。

工具选择：BootLoops索引与emitall完整GUIDE已读；emitall只核转录，不拟合分类。项目已有source_hinge、ScoreSubspaceFitter、calibrated_crossfit及source_consistency_campaign覆盖求解／基缓存／角色／最终评价，本轮仅增加组合适配，保留旧执行源码与收据。运行入口`python -m experiments.origin_detection.source_score_margin.run_iteration --pilot`，成功后去掉`--pilot`；本地记录`work/robust_statistics/source_score_margin`拒绝覆盖。

求解界面依据[SciPy 1.15.3官方文档](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs.html)的自由变量、退出状态、限时及marginals符号。有限问题依据既有[机理阅读](../../../reports/04_算法研发/有限视图鲁棒间隔_优化先行工作与适用边界_2026-10-06.md)；本轮没有新增全文阅读或把文献扰动集合当物理摄像头集合。
