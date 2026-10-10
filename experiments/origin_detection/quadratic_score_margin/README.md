# 第61轮：三个分数的二次软间隔，事前协议

当前反复曝光开发结果第60轮selected为85%／3.33pp，独立真实验收未完成。本轮仅检验低阶弯曲边界：三个拟合内监督分数标准化后扩为九项，再用既有平均／来源最大软hinge LP。已知多项式特征原则，不称新核、新损失或摄像头不变性。增加表达能力也可能加重开发过拟合。

基仍从第56轮2576维签名数字缓存读取输入，各拟合折重新训练三个监督方向。3分数的均值／尺度只用该拟合折加权估计，floor .001；输出z三项以及z_i*z_j六项，i≤j，非对角乘sqrt(2)。未做末级标准化前，其内积为z·u+(z·u)^2；LP另有fit-only逐项尺度和L1，因此不声称实际正则化等同某标准核SVM。原生权重、编码参数、来源角色不变，无新图像推理、神经训练或simulation；RR reserved及独立像素封存。

其余与第60轮相同：四L1(.0001,.001,.01,.1)；五source折中三拟合／一class-threshold标定／一评价；内部全部来自1260原fit来源，每折756／252／252来源、4536／1512／1512行。真实内部真标签；null内部全伪标签、最终threshold真标签，非部署置换检验。基与头同训练行，非OOF stacking。弱raw／Q90用于fit及threshold；selection四编码。参数选择只看当前fit OOF，不用当前selection挑λ。

四最终头：zero平均hinge .001、selected最大hinge所选λ、wrong保持真基／映射但置乱末级同类来源组、null全部基／图映射／头拟合伪标签。zero与selected若λ不同则聚合与惩罚同时改变，报告实际λ。单图查询只读表示，没有条件／来源标签或对应原图。规则保存所有参数，逐条／批量、映射／原生／保存加载一致是软件控制。

新增人工答案：二次内积的手算值与缺失交叉项负控，XOR标签在线性平方组合上恢复，非法维数／无限／溢出拒绝；全基到头类别分离、标签反转隔离、地图矩不随查询变化及冲突来源拒绝。复用解析hinge／复制等权／假原始或对偶／标签冲突及三角色控制，全部先于真实fit。SciPy HiGHS-IPM限时120秒，原始／对偶残差及gap≤1e-6；失败封存，不解释为算法性能。

先测三折训练冷基与九项LP .0001，再暖复用LP .1，含标定／held；10冷＋30暖为条件代理，排除输入、全fit、外层且其他λ可能有不同迭代代价。完整主运行12基组／36基／12辅助logistic／12矩映射、40内CV LP＋4最终LP，另行pilot与人工控制不计其中。每候选五折OOF文件是进展单位。所有BA、真假率、AUC、配对跌幅与区间沿用现有评价；末尾整数／角色／错误分数及emitall负控。第60轮审计器复用，诊断记录不独立重求LP。

有限margin只覆盖映射后的已观测特征凸包，软slack可允许不正margin；未知物理传播及重新标定边界稳定性均需实际配对评价。BA≥80%／退化≤2pp及独立真实验证要求不变，开发成功也不单独完成验收。

工具复用：BootLoops emitall仅核转录；source_hinge覆盖求解，现有基／角色／最终评价组件继续复用。新增共享`calibrated_readout_campaign`抽取控制、小试及完整标定运行次序，避免每个后续实验复制整套入口；旧第59／60轮源码、协议、收据保持原样。二次map不同于既有高维随机Fourier图，不修改已有图的接口。

[scikit-learn官方特征说明](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)用于核二次交互和过拟合边界；[官方核说明](https://scikit-learn.org/1.7/modules/kernel_approximation.html#polynomial-kernel-approximation-via-tensor-sketch)用于定位惯常原则，只定向读这些文档段落，未新增论文全文阅读，也不运行TensorSketch。低维直接展开而不引入近似核随机误差。

先`python -m experiments.origin_detection.quadratic_score_margin.run_iteration --record-controls`，后`--pilot`，成功后无参数完整运行。仅本地`work/robust_statistics/quadratic_score_margin`收据，拒绝覆盖；未上传的收据并不等于外部读者可以复算全部图像评价。
