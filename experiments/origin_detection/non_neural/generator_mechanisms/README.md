# 生成机制的准入与反例控制

本目录只检查候选统计量的机制假设，不提供默认AI检测器，不加载模型或神经特征。[证据与下一候选](../../../../reports/04_算法研发/机制研究/生成链候选准入_上采样与解码_2026-10-11.md)说明文献和适用边界。

`audit_polyphase.py`在有限周期信号上检查“奇偶差分能量对比”这一具体量。设 `E0`／`E1`为两个相位组相邻差分平方的平均，`T=(E1-E0)/(E1+E0)`；零能量返回未定义，不强制判自然。

运行前写入的解析真值覆盖最近邻复制、亮度偏移、相位平移、两抽头平均和常量。Fraction逐项计算与NumPy循环差分进行算术核对；错误答案及非法输入必须被拒绝。两条路线共享同一人工fixture与差分定义，这是有限软件核对，不是一般CNN定理或外部数据验证。

控制显示：普通放大能得到与生成器代理完全相同的结果；两抽头滤波把该对比从1变成0，信号均方仍为3/8。它排除了用此量直接证明“AI特有且传播不变”的做法，不否定NPR论文的实际CNN实验或其他局部统计量。

```powershell
.venv/Scripts/python.exe -m pytest tests/evaluation/test_polyphase_controls.py -q
.venv/Scripts/python.exe -m experiments.origin_detection.non_neural.generator_mechanisms.audit_polyphase --output <new_receipt.json>
```

输出拒绝覆盖。所有边界都是周期、偶长度、浮点／有理构造；没有8位取整、JPEG编码、真实照片或拍屏。不要把这个控制包装成完成了真实过程simulation或分类评测。
