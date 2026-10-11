# 第七轮：固定表示，训练权重×JPEG增强

[第六轮](../residual_statistics/README.md) · [研发流程](../../../docs/robust_ai_detection_plan.md)

## 计算前登记

本轮不改第六轮379维算子、源码、清单、来源角色或推理协议；不读取RR reserved。
四个固定候选为等视图／两域等权×仅raw／raw加JPEG90/4:4:4。RF仍128树、depth6、sqrt、无bootstrap、单线程、seed20261006。仅训练决策边界，不做新表示或分类器声明。

每个raw视图样本权重为1/N；两域等权时RR每视图变为3/N，Chimera不变，从25/75变为50/50域总权重。增强版本两个编码视图各除以2；3780训练记录变7560，min_samples_leaf从8变16，使名义最小叶记录比例相同，但不保证相同独立来源数。source、condition与编码副本均不拆角色；元数据只用于fit权重和审计，不喂给像素分类器。

**四个候选阈值均只用相同十二个raw threshold视图**，保留第六轮maximin BA与tie约定。equal_view_raw必须逐字段复现第六轮hybrid JSON，否则拒绝继续。其他三个候选不以selection调整阈值或森林参数。

原raw十二视图、JPEG90选择指标复用有签名缓存。JPEG90对增强候选属于已见训练分布，不能称独立编码稳健性验证。另对固定selection的1260图做resize256→JPEG70/4:2:0，**不进入fit或threshold**。原文件SHA、尺寸、派生JPEG字节SHA与直方图覆盖均核对。不物化中转JPEG文件，不访问其他来源。

沿用第六轮五门槛，额外要求JPEG70的四处理域聚合BA≥55%、相对raw跌幅≤5pp。八raw处理视图AUC95%下界>0.5、各真假正确率≥55%、两域raw原图真假≥55%。一次固定equal_domain_augmented来源内配对标签置乱，fit内按域／场景置乱，所有版本同标签；如通过，解释拒绝。只从全通过者按最弱raw处理AUC选择；无通过则chosen=null。

先做独立人工控制，再60selection文件试算计时，再1260编码提取和四模型fit。保存新代码、父收据、清单与CSV指纹；拒绝覆盖。单行／批量判决必须一致。准确度全部失败时不重复相同表示的速度测量；若通过再按第六轮相同60文件benchmark。BootLoops emitall核验报告数字，不能替代科学证据。

Chimera永久属于已接触开发数据；新编码也只是开发来源上的留出处理，不能称独立数据集测试。RR再数字化没有四方式逐图标签，来源预训练重叠及完整近重复仍未知。

```powershell
python -m experiments.origin_detection.residual_training.run_iteration --pilot 60
python -m experiments.origin_detection.residual_training.run_iteration --prepare
python -m experiments.origin_detection.residual_training.run_iteration
```
