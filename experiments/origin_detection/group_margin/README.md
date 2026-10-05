# Group margin — thirteenth development protocol

固定第十一轮499维组合；比较ERM、配对惩罚、熵加权组风险、两者结合四项。
不是新表示、精确max-group DRO、CoRe定理实例或物理不变性保证。
固定ridge0.01、pair系数0.1、熵温度0.1；拟合前全局标准化，参数仅由fit学得。
组是fit的域×内容场景×处理条件×编码版本×真假类别，共48组；组先验由两域等权样本权重确定。
熵风险为`temperature*log(sum(q_group*exp(group_mean_logloss/temperature)))`；与最大组损失有偏差，不能称逐组下限证明。
配对为原图raw到同源其余五图的均方分数变化，域等权；只训练使用对应关系。

反事实对照：结合目标用同类、同域、同场景、同条件的错源循环配对，保留组像素边际但全部来源不相等。
另有来源真假标签置乱的ERM阴性读出。
同一1260fit/420threshold/420selection来源及24视图阈值校准；旧JPEG70已暴露，不称盲测试。
不打开RR reserved，不训练最终神经模型，无新像素提取。

先有限差分检查梯度、常量log2对照、人工少数组捷径反转控件；然后ERM完整fit计时小试。
L-BFGS-B最多500步，要求最大梯度≤1e-5且目标不比初始常量差；失败就记录，不能静默使用未收敛规则。
准确度、两类正确率、整数比例边界、单张/批量一致性分别检查；软件过关不替代科学目标。

```powershell
.venv/Scripts/python.exe -m experiments.origin_detection.group_margin.run_iteration --pilot
.venv/Scripts/python.exe -m experiments.origin_detection.group_margin.run_iteration
```

导出复用`StableRule`的显式单图分数格式；拟合目标由本轮协议及收据区分。
记录在`work/robust_statistics/group_margin`，拒绝覆盖。组件先行工作见[机制调研](../../../reports/04_算法研发/鲁棒判别跨学科机制补充调研_2026-10-06.md)。
