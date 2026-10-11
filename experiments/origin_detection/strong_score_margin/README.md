# 第62轮：较强编码覆盖，分类事前协议

第61轮真实标签二次项为零、判决与第60轮相同，开发85%／3.33pp未达标。只改变处理覆盖：沿用冻结CuRe、2576维输入、三个拟合内方向、九项图、L1网格(.0001,.001,.01,.1)、平均及来源最大软hinge与class-threshold政策，fit和threshold增加Q60。新编码提取另见[数据准备协议](../strong_token_views/README.md)，不新建simulation或训练神经参数。

原6300图、2100来源与原始角色保持：fit1260来源三编码raw／Q90／Q60，共11340条；threshold420来源三编码3780；selection420来源仍四编码5040。新增5040条来自原fit／threshold图片。合并仅用只读数组视图，不重写第56轮缓存或整套复制大矩阵；签名、唯一身份、角色及覆盖全部先核。Q60变为已见编码，不称Q60编码外推；Q70没有用于fit／threshold，但selection图片已反复曝光，仍只是开发对照。

SourcePanel明确每来源3条件×3编码＝9视图。五折source中三fit、一cal、一held；每次756／252／252来源和6804／2268／2268行，cal36组。内部每强度45指标／75配对，与旧30／35不同；保留旧6视图门控，用新显式panel校验，缺少强视图就拒绝。折按原真标签／场景固定、seed20261007，全部来源在原fit角色；原登记不改。真实内部真标签，null内部全伪标签、最终threshold仍真标签，不是部署路径置换检验。

四最终头：zero平均hinge固定L1 .001；selected来源最大按当前fit OOF选择L1；wrong真基／映射但末级来源按域／场景／类／条件／编码置乱；null全部基／图／头用既有同源配平伪标签重拟合。两个标签目标的各五个fit数组与各一个全fit数组，共12基组、36基、12辅助logistic、12图矩、40CV＋4最终LP，不计单独人工与pilot。基与末级共享fit行，非OOF stacking。查询只用单图特征，没有来源／条件／配对原图标签。

预先控制包括三编码完整来源角色plant：11340条、各训练6804、cal／held2268，已知真假特征恢复BA1／跌幅0，45／75计数；缺少一个强视图／非法fold拒绝、错来源保面板和同类。复用二次项／XOR／完整基头、解析LP／假原始或对偶／冲突类和保存加载控制。它们是组件与完整基头软件控制，未通过完整共享driver＋所有bootstrap／文件写入的构造科学任务。真实接入另外核signed数据和运行结果，不能把tests通过称独立真实性能。

先真实三fit折冷建.0001、暖复用.1，并做相同cal／held，10冷＋30暖为条件代理；排除输入／全fit／外层，其他参数成本可能不同，任项>120秒先停诊断。LP限时120秒，原始／对偶可行性及gap≤1e-6；异常封存且不评价。完整内CV按每候选五折OOF写盘推进；真实选择冻结后才运行四最终头、36threshold视图以及旧四编码selection。末尾四头各60／130整数核算，内45／75、角色及实际参数重选核对，并运行错误BA／分数／折和emitall负控。

所有BA、真假率、AUC、同源配对下降与区间用现有生产评价；区间仍2000次类别内来源重采样，seed20260924，最坏130项事后选择未多重校正。与第61轮的对比同时改变基、矩、末级训练分布及标定覆盖，不能把变化单独归因于某个基方向或class-threshold。BA≥80%、下降≤2pp及独立真实验证要求不变；RR reserved／外部像素封存。软slack及映射后的有限凸包不能保证未知物理传播，预训练／近重复与RR逐图方式标签缺口保留。

旧source_training、source_folds、feature_views、算法、整数／Fraction及评价复用。新增SourcePanel及panel driver是显式9视图流程，不对旧控制或源码作兼容绕过；其后可注册别的面板，当前声明只覆盖本协议的RR／Chimera形状。共享控制记录及方法绑定沿用现有接口，不复制求解器。

`python -m experiments.origin_detection.strong_score_margin.run_iteration --record-controls`，数据准入后`--pilot`，成功后无参数运行。输出仅本地`work/robust_statistics/strong_score_margin`，拒绝覆盖。不得据selection结果选择本轮L1或把新编码称新独立来源。
