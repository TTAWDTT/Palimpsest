# 训练行metadata接口（2026-10-08）

## 做什么

共享source-panel CV与最终fit可以显式给算法当前训练行的处理／来源metadata，构造有向处理边。普通数值fitter保持原调用参数；预测只收图像特征，不接收原图或处理标签。

## 何时用

处理顺序／配对图不能从数值嵌入猜测时使用`RecordAwareCampaignMethod`。相邻的普通`CampaignMethod`用于不需要metadata的来源风险、方差等方法。record-aware目前针对source-panel driver，不支持旧six-view driver。

## 输出和边界

返回仍为rule／diagnostics，`fit_training_rows`拒绝行数不一致或非fit角色；CV把metadata和数组用同一train mask筛出。仅检查分发／记录对齐，不独立认证优化器实际读了哪些成员。字段仍由冻结清单提供，不把metadata变成判别特征。

## 准入测试及版本

已知完美面板核五fold的训练／标定／held互斥、逐行view索引对应、45／75计数；旧numeric签名和非fit角色拒绝也通过。新solver另核有向损失／梯度／零强度及极小损失仍翻转的反例。当前真实效果要靠独立评测，不由这些软件测试推出。

[版本修订记录](../../sources/2026-10-08_training_records_interface_revision.json)保留变更前SHA及Git3a22a93，旧64–67收据不重写；复算源审计先恢复receipt对应版本。原特征提取实现未变，所以新算法复用已签发缓存。通用typed audit已验证旧14项记录，专用旧auditor删除，没有代码转发stub；参见[工具导航](../../tools/README.md)。
