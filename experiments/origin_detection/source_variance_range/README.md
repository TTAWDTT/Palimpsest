# 第65轮：fit内部扩展方差强度范围，事前协议

第64轮内部λ10的最低BA87.41%、最大跌幅2.92pp，处于原网格边界且有80%余量；以此fit OOF为扩展动机，另签λ(10,30,100,300)。不使用第64轮selection选当前λ，不推断较大λ必然有效。末级、四方向14项、数据、权重、fold及标定全部保持。全部是反复曝光RR／Chimera开发，独立真实验收未完成。

新网格所有λ正，zero最终控制仍额外拟合λ0的T=.1风险；真实／null各当前五fold三fit／一cal／一held选择，不能复用过去OOF成绩作为当前跨参数输出。每source9视图，fit11340、threshold3780、selection5040；内6804／2268／2268、cal36，各参数45／75，最终60／130。sourcefold seed20261007、真fold划分不随null变；null内cal／held伪标签、最终threshold真标签，不是部署置换检验。

源风险T=.1、L2 .01、scale_floor .001、2000迭代／梯度≤1e-5及source variance定义不变；wrong仅方差组错置，风险仍按真source。新选中和wrong有正λ，若未来选择0须限定错组无作用。元训练与基共享fit行，非OOF stacking，单图查询无来源／处理标签或原图，Q60已见、Q70图片曝光，RR reserved与外部像素封存。无新推理／神经训练／simulation。

复用第64轮的7项构造／组件控制重新签当前pins，不设计会被当前真实输出影响的新答案。它们不认证全driver、bootstrap、文件或可实现真实token。先测λ300冷建与λ30暖复用含cal／held，10冷＋30暖条件代理，排除输入、全fit及外层且参数代价可能不同；任项>120秒先诊断。LP初始化与平滑当前头分开计，主运行若12旧数组不同：48基、12aux、12maps、12初始化LP、40CV＋4最终平滑拟合入口；正λ另baseline＋目标求解，不把44入口当44优化器。

先冻结当前fit选择，再评价zero／selected／wrong／null、计算BA、真假率、AUC及同源下降／翻转。最终整数／Fraction和内角色、45／75、λ重选核，错误BA／score／cal折、codec和emitall错误孪生。扩展auditor仅新增显式四参数选项，默认旧四值不变，整数recounter与数值生产体不改；旧64源审计恢复Git9475475，原记录不回填。

λ大导致分数趋零、方差低或CV跌幅下降均不等于实际分类保持；必须完整配对评价及独立真实验证。验收仍BA≥80%、同源下降≤2pp和独立真实数据；保留多重选择、RR方式标签、近重复／预训练缺口。既有scorevariance原理，不声明新方法或全物理不变性。

入口`python -m experiments.origin_detection.source_variance_range.run_iteration --record-controls`，再`--pilot`，成功后无参数；本地`work/robust_statistics/source_variance_range`拒绝覆盖。完成后审计命令`python tools/audit_smooth_source_panel_cv.py --directory work/robust_statistics/source_variance_range --parameters 10 30 100 300`。
