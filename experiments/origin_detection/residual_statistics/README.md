# 第六轮：局部残差共生与原图／处理联合标定

2026-10-06，真实图像提取前登记。第五轮证明顺序表示的固定非线性读出可恢复部分拍屏关联，但原图不进入fit且原图表现弱。本轮固定长边256、加入原图fit，并以同样标定下的ordinal256作消融，检验新增残差统计的增量；不把跨轮点值改变全归因于新特征。

## 算子与两个候选

详见[算子定义](../../../src/palimpsest/detection/algorithms/residual_statistics/README.md)。379维为顺序28＋RGB三个尺度各39残差共生轨道；另一个候选仅ordinal256的28维。无剪裁搜索、无特征筛选、无频带／阈值追加调参。思想来源[SRM](https://ws.binghamton.edu/fridrich/Research/TIFS2012-SRM.pdf)的残差、量化、邻接共生，阅读§II-A及二阶差分段；该文任务为隐写分析。本轮局部归一化、三阶／三尺度与顺序组合是适配，不宣称理论鲁棒性或新颖性。

两个候选各固定森林128树、深度6、叶≥8、sqrt特征、无bootstrap、单线程seed20261006。拟合十二视图（两域原图、RR两处理及Chimera三场景×两拍屏），各视图总权重相等，fit标签沿用来源角色。threshold来源十二视图最大化最差BA，单一严格score>threshold；不向运行时提供条件或源元数据。联合fit从1260来源获得3780图，并非3780独立来源。

额外仅一次hybrid来源级标签置乱：同域同场景内fit源标签排列，三版本一致；不置乱threshold和selection。不扫seed，null通过全部门槛时拒绝解释。一次null不能证明无泄漏。

## 固定准入与停止

沿用八处理视图AUC来源bootstrap2000次的95%下界>0.5、真假正确率≥0.55、两原图聚合真假正确率≥0.55。统一resize256→JPEG90/4:4:4对照的四处理聚合BA≥0.55且下降≤5个百分点；这个对照不消除历史编码，不能认证无捷径。不同于以前resize512版本，名称显式为jpeg90_444_after_resize256。

准确度未过停止当前候选，不用selection追加调参。速度仍测两个候选以量化新表示代价：同一60文件×3轮、CPU/OpenCV单线程，≤2MP核心P95≤10ms、含解码P95≤50ms；大图单列不隐去。两个候选都过时按最差处理视图AUC挑选。通过仍只属开发筛选，需后续未接触真实条件验证。

## 数据与接触账本

第三轮签名来源清单直接继承，2100来源／6300原图及真实处理图；两版本12600特征条目。所有RR／Chimera已被多轮设计接触，来源角色不会变回盲测。RR reserved未打开。RR再数字化缺逐图物理方式；全面近重复、生成模型及设备覆盖仍有限。原baseline签名缓存复用，不推理B-Free、D3、Benford。

## 先行控制

1. 独立枚举125种量化三元组及符号／反转等价类，39轨道；逐像素Python循环计数核对，不由生产lookup生成答案。符号反转、序列反转及方形转置／旋转一致；乱序必须改变邻接统计。
2. 常数、线性斜面二阶差分精确零；增益／偏置边界与固定+1底噪不变性限制。量化边界、NaN、小图、非法类型；直方图归一化与每通道块顺序核对。
3. 手写树已知切分方向、阈值等号、浮点输入32位转换；节点环路、共享子树、维度与概率坏值拒绝。独立sklearn分类器及边界输入对照≤1e-12，保存加载一致。这验证树导出／推理，不认证学习科学性。
4. 独立XOR CSV全管线fit／threshold／selection，AUC与两类正确率为1；零特征AUC精确0.5且拒绝门槛。缓存缺行、重复、NaN、来源角色／标签破坏、实际文件SHA／尺寸变化必须拒绝，干净双胞胎通过。
5. 小试与正式运行都是相同图像提取入口。先随机60图测时，依实测决定全量；进度单位是features.csv图数，不以进程活跃替代产物增长。

BootLoops planted-truth／independence-bookkeeping／timing-discipline及emitall沿用。检索工具索引后，图像统计与森林由本项目和sklearn承担；符号积分工具不适用。公共portable森林是可复用推理工具，新增参数模式、控制与目录索引；不修改冻结第三至第五轮代码或提供中转兼容文件。

## 运行

```powershell
python -m experiments.origin_detection.residual_statistics.run_iteration --pilot 60
python -m experiments.origin_detection.residual_statistics.run_iteration --prepare
python -m experiments.origin_detection.residual_statistics.run_iteration
python -m experiments.origin_detection.residual_statistics.run_iteration --benchmark
```

输出`work/robust_statistics/joint_residual_statistics`，存在则拒绝覆盖。代码／配置、父清单、特征与森林JSON有SHA；载入先审计，不更新指纹绕过错误。原始文件按存储方向解码，无自动EXIF旋转。公共推理仅NumPy／OpenCV，不需要sklearn或joblib；学习留在实验侧。
