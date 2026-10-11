# 固定阈值失败的定位

[第九轮](../threshold_calibration/README.md)没有可通过全部要求的单阈值。只读其父签名规则和**threshold角色**，给每视图求两类都≥55%时的精确有限样本区间，再计算交集。区别同一视图本身排序不足，与各可行视图要求的阈值相互冲突。不是新训练、阈值搜索、独立测试或总体不可能性证明。

AI iff score>t。natural分数第ceil(.55*N)小值是闭下界；AI分数第N−ceil(.55*N)+1小值是开上界。全局闭下界最大／开上界最小给交集；记录界限来源，不去修改失败参数。

先已知区间、等号和两个各自可行但交集为空的反例，再读取真实缓存。禁止用selection或JPEG70定位之后回调参数。

```powershell
python -m experiments.origin_detection.threshold_diagnostics.audit_thresholds
```
