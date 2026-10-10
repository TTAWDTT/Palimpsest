# 真实处理后的局部统计候选

[研发流程](../../../../docs/robust_ai_detection_plan.md) · [注册配置](../../../../configs/evaluation/rr_robust_statistics.toml)

本问题实现可被真实传播样本否定的统计假设。当前先用旧 simulation 的 development 来源；不接触 reserved 图像来调参，不重跑深度 baseline。来源、特征、规则和选择记录在同一个结果目录。

## 2026-10-05：首轮注册

注册于候选真实图像统计运行前。最多900来源（真假各450），按整个精确重复连通组划分：fit各270、selection各90、threshold各90。三个真实条件一起跟随来源。development 不足时拒绝，不从 reserved 补齐。重复组含多个来源时只取一个代表；共享原图/处理图SHA的组不得跨角色；已知 train/val 重叠排除。近重复尚未全面核查。

### 假设卡

| 候选 | 定义与解释 | 主要反例/淘汰条件 |
|---|---|---|
| H1 local | 局部亮度均值/方差归一化后的绝对矩、二阶矩、峰度和四向邻域相关 | 归一化统计也响应 JPEG、模糊和噪声；若只看原图有效或编码统一后消失，不通过 |
| H2 scale | 相同内容区域的两个尺度上，梯度能量比、梯度幅度相关、归一化矩变化 | 缩放核/内容可能主导；若传播与再数字化验证中不能同时区分，淘汰 |
| H3 color | 局部 RGB 残差的通道相关、色度/亮度梯度比 | ISP/采样改变通道关系；识别最终相机不等于识别屏幕内来源 |

H1借鉴 [Mittal等 BRISQUE](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) §III-A的局部归一化与邻域关系。原文相关页已读；全文逐段阅读未完成。本构造用直接矩与相关，**不是 BRISQUE 复现**；质量检测有效性不能推出AI检测有效性。短句“each distortion modifies the statistics in its own characteristic way”（§III-A，印刷页4698）作为反证风险记录。

H2/H3是本项目的探索假设，没有普适不变性证明。手工描述子已有工作，如 [Nirob等](https://arxiv.org/html/2601.19262) §IV、VI；其CIFAKE实验不能直接支持真实拍屏稳健性。本轮核对方法和限制段，未完成新颖性综述。

### 冻结的计算/选择方式

- 同一特征实现，长边最多512或1024的两种预处理；等比例 AREA 缩小、不放大。每图取去重的四角小块，边长最多256；第二尺度在块内AREA减半。输入尺寸/文件名/元数据不进入特征。
- 共17个统计量，三个族分别组合；最多6个族×预处理候选。每图先做块特征中位数，再按fit条件等权样本的中位数/MAD标定；不是原计划示意中的先块评分后聚合，差异预先登记。
- 特征方向仅由fit传播和再数字化的合并AUC决定；保留在fit两个处理条件下方向一致且各AUC≥0.55的特征，等权平均clip至±5的标准分数。没有合格特征时输出恒零作为失败候选，不暗改方向。
- selection上两个处理条件AUC的来源级分层bootstrap下界均>0.5，才允许进入后续原型。此区间用于筛选，存在多次选择，不是最终确证。通过候选按较差条件AUC选择，固定名称打破并列。
- 仅选中候选使用threshold角色选择全局阈值，以较差处理条件BA为目标；最终内部留出尚不运行。编码统一JPEG90/4:4:4仅在开发角色作诊断，不用它的结果重选本轮候选。

### 先运行的已知答案与破坏控制

1. 数据清单干净孪生通过；重复文件、错误标签、缺条件和跨角色重复被拒绝；合成SHA共享组保持同角色，train/val重叠整组排除。
2. 常量图各残差/梯度量为零、分数有限；明确方向的条纹输入产生可预测邻域方向关系；小图和坏dtype有明确行为。
3. AUC完美排序=1、倒序=0、恒零=0.5；阈值严格大于语义；错位/漏行缓存拒绝。这些是软件控制，不是AI证据。
4. 规则参数保存/加载与文件推理完整路径一致；像素相同但文件路径不同结果相同。
5. Emitall 的已知答案自测及数字故意篡改必须失败。共享结果文件的两种核验不称为独立科学验证。

### 工具选择与复用

Lady 已读 BootLoops `tools/README.md`、`emitall/GUIDE.md`、`gatekeeper/GUIDE.md` 和本机验证说明：emitall适用报告数字核对，先运行本机自测再用于本项目；gatekeeper面向AMFlow数值与PSLQ留出，不接受图像来源清单，故不套用。既有`evaluation`负责分类和同源变化，`io.hashing`负责指纹；新增的是本问题的数据角色审计与候选统计，不重复这些工具，也不修改安装目录。

## 执行入口

```powershell
uv run --locked python -m experiments.origin_detection.robust_statistics.rr.audit_sources
uv run --locked python -m experiments.origin_detection.robust_statistics.rr.run_features --limit 60
uv run --locked python -m experiments.origin_detection.robust_statistics.rr.run_features
uv run --locked python -m experiments.origin_detection.robust_statistics.rr.evaluate_features
```

`--limit`只产生带独立名称的计时pilot；完整特征文件必须覆盖签发清单才能用于筛选。结果位于`work/robust_statistics/rr_first_iteration/`。命令不自动运行，模块导入不读数据。
