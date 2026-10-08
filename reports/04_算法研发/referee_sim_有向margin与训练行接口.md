# 有向 margin 与训练行接口审查

审查日期：2026-10-08。Lady 在独立于实现者的新代理上下文中按 `bootloops-research:referee-sim` 审查事前方法及接口，并按 `bootloops-research:prose-lint` 核对本报告的数字、范围与表述。审查只读取源码、事前文稿、既有软件控制和 cost pilot；未读取随后分类 CV 的成绩，未增加真实数据拟合、像素提取或模型推理。

有向损失的数学条件和已声明的图构造与实现一致。审查发现一个 MEDIUM 标签类型缺陷、一个 MEDIUM 真实零参考验收缺口；未发现足以推翻当前 int64 事前运行的 HIGH 问题。软件检查支持接口和公式的有限结论。BA≥80%、最坏跌幅≤2个百分点及独立真实数据要求仍需各自的科学证据。

## 读者与问题

读者包括凸优化与统计学习研究者、pairwise ranking 方法研究者、AI 图像检测和传播处理研究者、实验统计与基线方法使用者、软件接口与复现维护者，以及会依赖检测结果的实践者。排序文献为方法借用，旧第67轮为比较基线；这些读者均分别接受正确性和新颖性审查。

| 读者 | 其标准异议 | 轴 | 答案所在位置 | 严重性 | 修订情况 |
|---|---|---|---|---|---|
| 凸优化与统计学习 | 正部平方是否凸？共同 bias、标准化和来源加权是否改变导数？ | 正确性 | processing_order_margin.md 第三段；directed_margin.py 的 penalty/objective；本报告有限梯度核对 | LOW | 无需改公式；固定映射下凸，完整监督 bank 的训练不是联合凸问题 |
| 凸优化与统计学习 | 原理是否超出已有凸惩罚？ | 新颖性 | README 首段明确 task-specific adaptation；不能由凸性主张新算法原理 | LOW | 保留局部任务适配定位 |
| Pairwise ranking | Joachims 的单位排序间隔和线性 slack 与本项是否相同？ | 正确性 | README 写 mean squared positive；公式写零目标正部平方；原论文§4.1式12–15 | LOW | 精确定位为借用 pairwise difference；未将原 Ranking SVM 的结论移植为图像稳定性证明 |
| Pairwise ranking | 这是否重新命名既有排序／hinge 惩罚？ | 新颖性 | README 首段已写不是新的 ranking principle | LOW | 原理新颖性不作主张；本次阅读只核原论文对应公式，非完整先行工作综述 |
| 图像检测与传播处理 | 给定边上的零损失能否保证未知物理通道仍正确？ | 正确性 | README Pair graph and scope 和 processing_order_margin.md 第二段 | LOW | 已说明只覆盖已知评分边；有限惩罚和曝光来源不建立未知通道结论 |
| 图像检测与传播处理 | 同一五方向和20项下，相比对称 variance 有什么真实收益？ | 新颖性 | README 首段和 Procedure；当前材料只有待比较设计 | LOW | 机制只区别改善／退化的符号；不写经验改善结论 |
| 实验统计与第67轮基线 | 是否已经证明真实零强度与执行过的67完全相同，而非仅本轮可序列化？ | 正确性 | README 要求零参考；driver 的 zero_reference_gate 仅本轮保存／加载／映射比较 | MEDIUM | 见 M2；真实匹配尚缺专门收据，报告没有修饰成已经匹配 |
| 实验统计与第67轮基线 | 错源负控是否只改附加项？强度0时能否支持来源配对效应？ | 新颖性 | README wrong-source段；DirectedScoreFitter 中风险／bank仍用真实 sources | LOW | 已核代码；若选中0，错源与正确配对相同是设计结果，不能作为非零配对收益证据 |
| 软件接口维护者 | 合法0/1无符号标签是否贯穿整个包装器，而非只在最终优化器修正？ | 正确性 | 最终优化器有 int64 转換，bank provider 没有；本报告人工复现 | MEDIUM | 见 M1；未编辑生产源码 |
| 软件接口维护者 | 增加 record-aware 分发是否真保留旧方法签名及可复现版本？ | 新颖性 | training_records_interface.md；共享driver差异及修订ledger | LOW | 明确属于训练接口扩展；不将接口工作宣传成检测原理贡献 |
| 实践者与一般读者 | “零退化”是否意味着80%准确或实测最多跌2pp？REAL阈值平局怎样处理？ | 正确性 | README条件段与中文研究说明，保留绝对BA和独立要求 | LOW | 数学命题只约束正确到错误翻转，初始正确率可任意低 |
| 实践者与一般读者 | 可加载的新规则是否构成更好的检测器证据？ | 新颖性 | README Procedure最后段；序列化kind范围和pilot范围 | LOW | 可加载和cost仅为软件／时间信息，收益需完成事前比较 |

## 已核条件

以 `FAKE iff score>τ`、`REAL iff score≤τ` 为规则。同标签边若 `t(s_before−s_after)≤0`，FAKE after分数不低于before，REAL after分数不高于before。before正确即可推出after正确，等于阈值的REAL也包括在内。有限条边、每条权重严格为正时，正部平方和恰为0等价于逐边满足条件。命题还依赖共同阈值，不能套用分别改变阈值的两个判决。

独立检查对标签、before、after、阈值的半整数格点枚举了810组满足条件的配置，无正确到错误翻转。该枚举仅检查实现约定，任意实数情形由上述单调不等式直接证明。`ε=10⁻⁸` 的FAKE反例给before=ε、after=−ε、τ=0；损失为 `1/2500000000000000`，判决仍翻转。既有10项软件控制中的相应反例和REAL tie控制与这一范围一致。

固定数值映射 z 下，边向量为 `d=t(z_before−z_after)`，损失为 `Σ a_e[max(0,d_e·w)]²`。梯度为 `2Σ a_e max(0,d_e·w)d_e`，bias导数为0；正部平方在0点也可微。独立有限差分最大误差为 `1.3729106740356656×10⁻¹⁰`，100次随机凸组合检查通过。凸性来自凸函数与线性映射的复合，随机检查不替代这个证明。source logistic的熵平滑与ridge保持固定映射读出的凸性，bank本身仍由同批监督训练来源拟合。

人工4来源、每来源9节点构造出60边，每来源15边。用不等节点权重得到来源边总质量0.1、0.2、0.3、0.4，权重和为1。图包括每variant的original到两种处理条件，以及每条件raw→Q90、raw→Q60、Q90→Q60；三variants的顺序来自显式配置。真实756/1260来源的11340/18900边数与这一公式相符，未读取真实边实例重新计算。

图验证role为fit、节点identity唯一、三条件及完整variant组合、组内domain／scene／当前训练标签一致。真实组还要求同一个domain/source。错误配对采用虚拟group且允许跨source；在人工同domain／scene／真类分层乱配下得到60边，其中24条跨真实source。风险、bank、映射继续使用真实sources。独立检查把提供的训练labels反转而保留metadata真标签，所有边符号按提供labels反转；图没有暗中恢复真标签。

共享CV用同一train mask截取数组和records，source组重新编号后仅传6804训练节点。标定和held每个2268节点；既有记录对齐测试用行号、类别和来源核对五折训练／标定／held互斥，旧数值调用参数及非fit拒绝通过。`fit_training_rows`只核长度和role，逐行数值对应依赖调用者同mask与已签发行顺序，接口说明已经写明这一边界。DirectedScoreFitter的返回规则只保留数值bank／map／head，预测接口没有records参数。

既有人工包装器测试核对相同bank路径的零强度分数与QuantileScoreFitter严格相等，并核保存后重新加载分数相等。当前序列化kind为 `quantile_score_consistency`，表示共享数值布局；它不编码本次训练目标。diagnostics另含 `readout_method`、`ordering_strength`、有向损失和边计数，README已在Procedure注明两者区别。审查不把kind解释为旧对称variance目标证据。

## M1：包装器无符号标签

`DirectedScoreFitter.fit`先用labels生成正确的±1边符号，再将原labels传给provider.fit。最终 `fit_directed_source_risk` 内部转int64发生得较晚，不能保护bank的旧source风险拟合。`source_view_risk.py:51` 的 `2*y−1` 对uint8 REAL得到255；校验仅检查值集合0/1，接受这一dtype。

独立人工36行fixture中，int64包装器成功拟合；新包装器改为uint8则在bank构建时抛出 `ValueError`，优化梯度约64。先用int64填充同一包装器缓存、再调用uint8，则分数严格相等。现有无符号测试覆盖graph和最后readout，未覆盖完整包装器。缓存key将labels统一成int64，数值相同的两种dtype得到同一个key，实际provider计算却不同，形成已复现的缓存先后顺序相关行为。

修复建议：在包装器入口验证一维0/1标签后统一成int64，再用于pair、key、provider和最终fit；添加完整包装器uint8与int64相等的人工检查。当前真实training和source-null均构造int64，本问题不否定其已启动事前运行。若修复生产源码，应保留已运行收据的旧pins并追加修订和控制记录，不能覆盖旧收据伪装成修复后的执行。

## M2：真实零参考缺收据

README要求新零head与执行67的零分数一致。共享driver `zero_reference_gate` 实际核的是当前rule原生／映射／保存加载分数，未读取第67轮的零规则、bank或分数；人工包装器相等只证明该人工配置下相等。将该字段名称或软件测试读成已经匹配真实67，会超出已核证据。

修复建议：当前运行完成后，对已保存的新零与历史67零逐行核identity、bank/map/head、阈值和margin，单独写比较收据并链接到结果说明；若不等，先定位差异，再解释目标比较的范围。该核对可使用已有保存记录完成，不要求新增真实拟合；当前审查未查看随后结果，也未执行这一比较。

## 版本、时间与冷读

接口修订ledger保留历史Git `3a22a932a50d3a140199a83f2011afe7a50d5d4c`、共享源码before/after SHA、旧64–67收据不重写的声明及恢复对应执行版本的说明。当前driver的差异是显式record-aware分发；新directed runner pins包含training_fit和共享driver。特征实现不在这些修改中，真实缓存入口仍逐一核特征提取pins和父前缀。历史67的复算源审计需恢复其执行Git，当前源码不能直接代替历史源码。

已保存的软件控制为10项通过；cost pilot冷15.825726100010797秒、暖2.7124792999966303秒，投影40CV为239.63164000000688秒。时间只涵盖该pilot train/cal/held调用，投影依赖10冷+30暖及参数成本变化，排除输入、full fit和outer。pilot有向惩罚非零，符合文稿“有限强度不保证零违反”的限制。这些数字没有给出真实检测性能、独立性或未知物理通道结论。

README没有独立abstract。作为abstract-only读者冷读标题、首段和条件段，标题保留prospective comparison，首段把新增项称作任务适配；条件段紧邻graph限定已知边、非零损失、小损失反例和独立验收。中文说明的第二段在命题后立即写未知通道和绝对BA边界。未发现需要靠正文深处挽回的HIGH范围误导。对外若压缩为摘要，应同时保留共同阈值、已知同标签边、有限惩罚无保证及尚需真实独立验证。

## 审查依据

- [方法说明](../../docs/research/processing_order_margin.md)、[事前README](../../experiments/origin_detection/directed_score_margin/README.md)、[训练接口说明](../../docs/maintenance/training_records_interface.md)、[版本ledger](../../sources/2026-10-08_training_records_interface_revision.json)。
- 独立人工检查源码 `work/robust_statistics/directed_score_margin/referee/audit.py` 及检查输出 `work/robust_statistics/directed_score_margin/referee/audit.json`；源码SHA `35de2290f6032441df84e6d42f32f6c5517649b7ec9f2df4b8afcd0847c7a87b`。
- 既有软件控制 `work/robust_statistics/directed_score_margin/software_controls.json`、既有cost pilot `work/robust_statistics/directed_score_margin/pilot.json`。
- [Joachims 2002 原始论文](https://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf)，仅定向读取摘要和§4.1式12–15及§4.2开头；其排序pairwise差分支持方法来源定位，图像处理收益仍由本任务实验决定。

生产文稿和源码未由本审查代理修改。报告的数字／必要条件核对与语言复读已完成；M1、M2需要实现者追加修订或结果比较记录后再核。

实现者已确认后续安排：保留当前真实运行的执行源码和pins；运行结束后修复public-wrapper uint8入口并补人工测试。历史67的零参考比较使用保存数组、身份、完整规则和margin，发现差异先分析。以上为修订计划，尚不代表两项发现已修复或验收通过。

## 2026-10-08 修订复核

M1、M2在以下限定范围内关闭，前文保留首次审查时的发现与证据。真实主运行完成后才修订入口；其执行源码保存在Git `fd061849e05c0bf9cf65821826583dd1f0bf5c04`，当前修订由[标签类型修订记录](../../sources/2026-10-08_directed_label_dtype_revision.json)追踪。原人工失败复现、真实运行收据及旧controls继续保留，当前修订不改写历史执行。

M1修复已独立用人工fixture复核。包装器在生成pair、key和bank前核labels为一维0/1，并统一转为int64；当前源码SHA `ce7b327134f75e30fdd45f855f88fd0043506ea75b242a49fc5704808c25ce2e` 与修订记录after一致。独立36行、不同处理／编码数值及不等来源权重的人工fixture，在强度0和1下分别从两个冷包装器拟合uint8和int64，完整rule、评分和array key均严格相等。二维标签、非二元值和非有限值共3种非法输入，均在拟合前拒绝。M1因冷入口修复及数值相等检查而关闭；当前真实training／null原本使用int64，本复核未重新拟合真实数据。

M2关闭依据为已有比较收据 `work/robust_statistics/directed_score_margin/historical_zero_reference.json`。收据记载5040行身份相等，bank、center、scale、readout及margins严格相等，最大margin差0、改变判决数0。阈值包含在完整readout字段内；收据未另设threshold布尔项。历史scores SHA为 `708ecf54cf411261cbec854e899b991364f26e7e3ed899120ba3453b02871747`，当前scores SHA为 `4652f925bf1ae5f2285da5a416b13ee09b0f40d93bbaebfcf0544761d738deb0`；两个文件SHA不相等，收据中的逐行零margin比较相等，文件内容整体一致不在本项结论中。本代理核读收据的字段一致性，没有再比较真实保存数组或调用真实规则计算评分。收据范围为已曝光数据上的匹配零目标，独立真实性要求不由这项比较满足。

修订复核源码保存在 `work/robust_statistics/directed_score_margin/referee/revision_review.py`，输出为同目录 `revision_review.json`；复核源码SHA `43eee1aaf835aa2c5e149fa3f3a03893b66f5e039725e81b1f1625fb3ac72dbe`，所读历史零比较收据SHA `e58f621171230c13683abbfce5dc7cd90a31e8c4cc16c1cb0b2ba9a56adee17c`。本次只追加审查记录并运行人工fixture，未新增像素、模型推理、真实拟合或主运行成绩计算。
