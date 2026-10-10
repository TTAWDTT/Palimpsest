# 第64轮：四分数的L2平滑来源一致性，事前协议

第63轮raw方向和二次项全部为零、分数与62相同，85.56%／5pp未达标。只保持四监督方向、14项、较强面板及数据角色，替换末级L1软hinge为既有L2平滑source logistic及来源分数方差。当前未用新增项不能证明L1导致失败；同时改变损失、正则形式和强度选择，结果不能单独归因于稀疏性。既有原理在其他表示上已试失败，新组合仍需实测，不称新损失或物理不变性。

标准化14项后，末级用T=.1 source-view logistic、ridge .01、scale_floor .001、λ*加权source score方差，固定λ(0,.1,1,10)。SourcePanel保留raw／Q90／Q60的fit11340／threshold3780、selection5040；内部6804／2268／2268、cal36、45指标／75配对，各最终头60／130。四score矩只在fit估计；真／伪标签的全基与图都重拟合，所有查询只读单图特征。Q60已见，Q70图片曝光，RR reserved和外部像素封存，无新图像推理／simulation／神经训练。

四最终头：zero λ0、selected当前fit OOF选择λ、wrong保持真基和图但置乱末级方差分组、null全部基／图／头伪标签。分类风险仍用真source，错组只改方差。内部null cal／held伪标签而最终threshold真标签，非部署置换检验。基与末级共享拟合行，非OOF stacking。T=.1及λ常数只定义有限经验目标，不承诺分类跌幅。

复用SemanticMarginFitter公开构造路径建bank／图，因此每个全数组额外产生一个不用于最终判别的平均LP头。若12不同旧子数组，主运行48基、12旧辅助logistic、12图矩、12初始化LP及40CV＋4最终source-consistency模型拟合入口调用。非零λ的既有fitter还先求baseline source loss再求惩罚目标，44入口调用不等于44优化器求解次数。初始化LP不算当前末级LP，报告按实账分开计。规则沿用semantic_score_margin参数布局，kind识别布局而非拟合损失；fit诊断明确当前L2目标、.01 ridge是真L2。

真实拟合前，保留第63轮原Git和收据，修复成功辅助head计数并补共享旧bank路径2／3断言。panel最终ledger改为通用cv_readout_fits／final_readout_fits，按参数与fold数导出，不给非LP贴40LP标签；数值拟合／标定／评分体未改，修订记录在sources。旧源审计恢复Git f026530；不改旧pins或数字收据，不引入兼容路径文件。

预先控制：抽象raw信号恢复／擦除、平方内积及错误／非法字段、完整bank-map-head保存加载／单条批量、cal回调值及标签一致、类别反转、共享计数与三编码来源角色；复用source consistency数值梯度、λ0精确路径、非法强度等已有控制。它们不认证全driver＋bootstrap／文件或真实token可实现性。模型solver保2000迭代、梯度≤1e-5且成功退出，否则拒绝、保存失败，不当检测器成绩。

先测真实三fold λ10冷建和λ1暖复用，包含cal／held，10冷＋30暖条件代理；输入、全fit、外层排除，其他λ可有不同代价，任项>120秒先诊断。完整选择先于四固定头；最后保存分数整数／Fraction、45／75、recorded roles及λ重选，错误BA／score／cal折和emitall错误孪生。LP初始化有自己的原始／对偶残差；当前平滑头按梯度收敛核，不用LP状态伪装审计。

source variance小、非零新系数、人工答案或梯度合格都不是验收。BA≥80%、最坏同源退化≤2pp及独立真实验证不变，最坏130项事后CI不作总体保证；RR方式标签／近重复及预训练重叠问题保留。入口`python -m experiments.origin_detection.semantic_score_consistency.run_iteration --record-controls`、`--pilot`、无参数，输出本地`work/robust_statistics/semantic_score_consistency`拒绝覆盖。
