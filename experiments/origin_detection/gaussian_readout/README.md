# Gaussian readout — prospectively fixed twelfth development screen

检验均值线性读出遗漏的方差/协方差判别信息；不是物理不变性声明。
参数在真实计算前固定：旧379维、组合499维各比较pooled共享协方差、diagonal高斯朴素贝叶斯、class_full收缩QDA。
fit仅1260来源的raw与额外Q90，各类先验固定1/2；域权重相等。
所有头用fit全局标准化，协方差ridge0.1、全矩阵向对角收缩0.5、尺度下限0.001。
同一24阈值视图逐类最差校准，不根据selection改参数。

复用第六/七/十一轮签名缓存，无新增像素读取、RR保留来源读取或最终神经训练。
先检查同均值不同方差人工控件，再用`--pilot`验证共享协方差组合头成本（非调参）。
正式运行固定六候选及hybrid/class_full来源标签置乱阴性对照；单张/批量严格一致。
比例边界追加用整数计数审计；准确度达标后才进行端到端速度基准和独立真实数据验证。

```powershell
.venv/Scripts/python.exe -m experiments.origin_detection.gaussian_readout.run_iteration --pilot
.venv/Scripts/python.exe -m experiments.origin_detection.gaussian_readout.run_iteration
```

结果在`work/robust_statistics/gaussian_readout`，拒绝覆盖。经典Gaussian分类器有先验工作，本轮没有声称头结构新颖。
[固定编码器Gaussian判别研究](https://arxiv.org/abs/2608.18523)使用神经特征；这里使用固定像素统计，不能照搬它的成绩或界。
