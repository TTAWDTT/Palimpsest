# 第八轮：配对分数稳定约束

[第七轮](../residual_training/README.md) · [公共算子](../../../src/palimpsest/detection/algorithms/paired_stability.py)

## 计算前登记

第七轮域权重与JPEG增强未解决留出处理。固定同一379维表示与来源，检验把“同源处理分数应接近”写入目标能否改善。不是DICA复现或新颖性证明：相关研究[DICA](https://proceedings.mlr.press/v28/muandet13.pdf)§2.1/2.2同时控制域差异与类别关系；[CTC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5321138/)Abstract/§2提醒边缘对齐未保证条件关系。这里只阅读这些相关段，不声称完成全文或相关文献覆盖。

fit含raw及JPEG90，同第七轮两域等权。fit-only加权均值与RMS标准化，scale_floor=.001。设y=±1，求唯一凸二次解：

`J(w,b)=E_fit[(y−b−wᵀz)²]+λ E_pairs[(wᵀδ)²]+0.1||w||²`。

十二fit视图×两编码，权重归一化，总RR／Chimera各半。每fit source六个条件／编码，以original/raw为锚，另五个delta，**不减delta均值**。pair权重每域半、域内来源等权、来源内五变化等权。bias为fit加权标签均值；w解379阶正定方程。像素推理只需特征和379维内积，未知来源／设备／处理不作为输入。

固定λ=0、1、10，共三候选，ridge与scale下限不搜索。λ=0是未加稳定约束的同学习器对照，不与RF变化混称配对收益。阈值仍只用同十二raw threshold maximin BA。λ=1一次固定source-shuffled null，来源版本保留同标签；null通过则拒绝解释。

有限性质：常数预测器可行，所以fit目标≤fit标签方差≤1；λ>0时fit配对漂移≤1/λ。这是训练对上的均方性质，**不能推出未见JPEG、屏摄或逐图稳健性**。若真实处理侵蚀判别信号，惩罚可能抹去真信号，必须同时看绝对AUC／BA、真假类别和未加约束对照，不以低漂移宣布成功。

复用第六／七轮签名缓存，不解码新图、不碰RR reserved。沿用第七轮七项门槛及JPEG70留出处理；阈值不在selection回调。另报告raw/JPEG90/JPEG70同源分数漂移，按域半权重；用raw selection分数加权方差归一化漂移，以识别单纯缩小全部分数造成的假改善，方差≤1e-12时记null。raw是每来源两个delta，另编码是每来源五个delta，跨列不当成相同处理组成。三状态单图／批量判决一致；线性规则显式JSON，不需sklearn/pickle推理。

先独立标量闭式答案、正交信号、信号与变化同轴反例、零标签信号、身份损坏与序列化控制；再小维实际缓存试算记录计时，后完整计算。全部准确度失败则不另做速度benchmark，也不能称目标完成。

```powershell
python -m experiments.origin_detection.paired_stability.run_iteration --pilot
python -m experiments.origin_detection.paired_stability.run_iteration
```
