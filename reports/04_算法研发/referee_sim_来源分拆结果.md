# referee-sim：第七十一轮来源分拆结果

审查日期：2026-10-08。对象：`reports/04_算法研发/第七十一轮来源分拆_罚项激活但判别力下降_2026-10-08.md`。按 BootLoops `referee-sim` 在未参与本文撰写的新 agent context 中审查；本次仅读取已保存 margin、成员身份、诊断、协议、源码和 Git blob，执行整数算术及字节比较。未调用 producer evaluator、真实 fit、新像素或 Goal API，未修改 producer、pins 或数值收据。此审查文件是唯一新增产物。

**结论：没有发现 HIGH 或成绩矛盾；有两项 MEDIUM 表述问题，均可用一条范围句修补。** 本轮所能建立的是反复曝光开发数据上的保存结果与规定记录相符；selected 最低域 BA 为 73/90，最大同源下降为 1/20，2pp 门槛失败，独立真实验证仍缺失。四头完整计数和 OOF 选值经独立整数复核通过。结果不建立来源近重复排除、真实传播不变性、因果解释或优化器全局最优。

## HIGH／MEDIUM 发现

1. **MEDIUM：源码恢复与原字节 pin 的行尾约定未写明。** 本文“源码可在 Git72f4b40 恢复”在语义层面成立，但数值收据含 107 个源码 pin，Git `72f4b407503357b7d955d2437aacf4d31ebdb761` 的原始 blob 只有 103 个逐字节匹配。另四个路径是 `src/palimpsest/detection/algorithms/source_view_risk.py`、`src/palimpsest/evaluation/classification.py`、`src/palimpsest/evaluation/pairing.py`、`src/palimpsest/evaluation/robust_views.py`；它们 Git blob 的 LF 转 CRLF 后均命中原 pin，当前工作文件也命中原 pin。当前属性指定 `eol=lf`，普通 checkout 不能自动解释为恢复了全部原字节。建议把该句写为：“源码语义可由 Git72f4b40 恢复；107 个原 pin 中四个旧文件须按原 CRLF 行尾还原才逐字节匹配，原 pin 保留。”不改 pins；不把行尾差异解释为算法变更。

2. **MEDIUM：stationarity 的边界只在借用的协议／源码中明确，结果摘要与“梯度审计通过”附近未直接交代。** 数值优化读者会先问何种收敛；当前结果正文记录了严格梯度门槛和失败，但未直接说单起点非凸求解不认证全局最优。协议确有这句，源码 scope 也写明；需要沿摘要读者路径出现。建议在摘要或“完整重跑”审计句后加：“梯度通过仅指单起点非凸求解的保存 stationarity 记录满足 1e-5 门槛，不认证全局最优或硬约束可行。”这是范围补充，不削弱已核对的有限开发成绩。

本审查没有修改被审正文。以上是建议的最小防御性编辑，由主任务在正文中落实；没有为下一轮固定平均建立有效性论证。

## 独立计数与全部并列核对

输入为 `work/robust_statistics/split_rate_score/selection_scores.json`，判定仅用保存 margin 的 `score > 0`；BA 用 REAL／FAKE 正确数各除以本类来源数后取平均，以 Fraction 保持精确。逐来源重建各配对的共同成员和正确数差；四头各 60 指标／130 配对全部匹配 `iteration.json`。没有调用项目分类／配对计算函数，也未以 `score_counts_audit.json` 的 passed 值代替本次算术。

| 头 | 最低域 BA | 最低场景 BA | 最大下降 | 正 margin／负 margin／恰零 |
|---|---:|---:|---:|---:|
| zero | 73/90 = 81.111…% | 73/80 = 91.25% | 1/16 = 6.25pp | 2598／2442／0 |
| selected | 73/90 = 81.111…% | 37/40 = 92.5% | 1/20 = 5pp | 2581／2459／0 |
| selected_wrong_source | 73/90 | 37/40 | 1/20 | 2581／2459／0 |
| selected_null | 121/240 = 50.4166…% | 9/20 = 45% | 1/10 = 10pp | 4409／631／0 |

四头最低域／场景分别都只有一个最小项，没有未报告的并列：

- zero 最低域：`rr/all/redigital/jpeg60_420_after_resize256`，REAL 73/90、FAKE 73/90。最低场景：`chimera/horse/mac_iphone/jpeg60_420_after_resize256`，REAL 33/40、FAKE 40/40。
- selected 与 wrong 最低域：`rr/all/redigital/jpeg70_420_after_resize256`，REAL 73/90、FAKE 73/90。最低场景：`chimera/horse/mac_iphone/jpeg60_420_after_resize256`，REAL 34/40、FAKE 40/40。
- null 最低域：`chimera/all/lg_blackfly/jpeg70_420_after_resize256`，REAL 15/120、FAKE 106/120。最低场景：`chimera/horse/lg_blackfly/jpeg70_420_after_resize256`，REAL 4/40、FAKE 32/40。

全部最大下降项如下；各头恰有两项并列，报告未遗漏：

| 头／配对 | REAL 正确数前→后 | FAKE 正确数前→后 | 净 BA 下降 | 决策翻转来源数 |
|---|---:|---:|---:|---:|
| zero：horse original→mac_iphone，同 Q60 | 38→33 /40 | 40→40 /40 | 1/16 | 5 |
| zero：horse original→lg_blackfly，同 raw | 39→34 /40 | 40→40 /40 | 1/16 | 5 |
| selected／wrong：church original→mac_iphone，同 Q90 | 40→37 /40 | 40→39 /40 | 1/20 | 4 |
| selected／wrong：church mac_iphone raw→Q90 | 40→37 /40 | 40→39 /40 | 1/20 | 4 |
| null：cat original→mac_iphone，同 Q90 | 10→1 /40 | 37→38 /40 | 1/10 | 14 |
| null：church original→mac_iphone，同 Q70 | 11→8 /40 | 40→35 /40 | 1/10 | 12 |

selected 与 wrong 的全部 5040 个保存 row（含身份、标签、margin）逐项相同。这支持所规定的聚合率结构身份控制，不能充当成功的错配敏感性消融。null 同时存在正负 margin，实际为 4409 FAKE／631 REAL 预测，明确不是 allREAL 常量；最终 true-label cal 91/180、下降 3/40、BA 无可行也在记录和正文中标成失败。

## OOF 选值、成员与数值失败

从 `crossfit.json` 保存 OOF margin 和 supplied labels 分别重算 truth／null 四档、每档 45 个 BA／75 个配对下降；全部精确匹配。truth 四档最低 BA 为 77/90、77/90、23/27、457/540，最大下降 1/15、1/15、7/108、1/20，四档皆过 80% 开发 BA，按最小下降选 1000。null 为最低 BA 29/60、29/60、61/120、61/120，最大下降 1/240、1/180、1/180、1/135；没有档过 80%，按最高最低 BA 后较小强度回退选 100。null 回退不被当成成功信号。

独立重建成员没有导入 producer splitter：读取保存 OOF 的来源、scene、supplied label 和 fold，排除 held/cal fold；在 `(domain, scene, supplied label)` 分层内对 compact UTF8 JSON `[20261008,domain,src]` 做 SHA256 排序，按协议奇数层交替 floor／ceil。全部 40 CV 诊断和四最终头的 basis/readout 集合精确匹配，均互斥且合并为其 input context。null 用保存 pseudo label，未偷换成真标签。

- 每个 CV：head 3402 行／378 来源，basis 3402／378，input 6804／756；cal 和 held 各 252 来源。
- 每最终头：head 5670／630，basis 5670／630，input 11340／1260。
- 44 个诊断均保存 20 映射项、5 方向、45 训练率组、75 比较，`optimizer_ftol=0`；最大 main 梯度 9.930725885023972e-7，直接 basis 诊断最大梯度 9.648363821920195e-7。这里只核保存诊断，不重算优化器梯度，不独立认证真实执行成员或全部隐藏优化步骤。

`split_rate_score_numerical_v1/failure_receipt.json` 为明确的 passed=false，relative-function convergence 时梯度 1.0219490180263247e-5 大于 1e-5。`sources/2026-10-08_split_solver_stop_revision.json` 保留失败和七份完成 CV，旧 null1000 没有完整结果。独立读取旧七份和新 OOF，每份 11340 个保存 row 逐项相同；没有仅相信 `numerical_retry_comparison.json` 的 exact 字段。

源码检查表明 rate solver 默认 `optimizer_ftol=1e-14` 保留；本轮 wrapper 显式传 0，非零路径使用它，zero 路径返回已解 baseline。gradient gate 仍比较 `residual > tolerance`，保留 finite、success、目标不恶化检查，最大迭代 2000、接受梯度 1e-5。ftol 调整是停止准则重试，不是改标签、网格或降低门槛；上述七份相同也不证明失败 null1000 的旧结果会相同。

`bank_cache_audit.json` 的账面 12 bank/map、12 stage模板、48 base入口、12 aux、12 shape、12 LP、12 初始化 source 读出和 44 最终读出入口与 wrapper/调用布局一致。非零头先 baseline 后 penalty，44 入口不能被叫作总优化次数。训练 soft/hard 数值是读出半上的零阈值量，不能转成 cal／outer 证据。完整 CV 214.049 秒和末阶段 100.191 秒不同，后者不是端到端速度；旧失败完整总耗时未保存，正文未冒充已知。

## 受众、标准反对意见与 verdict

受众来源包括 AI 图像检测应用、真实传播实验、统计评测、数值优化、借用 Cotter 约束学习的社群、软件复现者，以及会按摘要做决定的一般读者。每个社群按正确性与新颖性分开审查；“已回答”只代表当前有限主张的标准反对意见被明确限定，不构成该算法成功或新颖性的认证。

| 受众 | 最强标准反对意见 | 轴 | 回答所在位置／缺口 | 严重程度 | 最小修补／已施行 |
|---|---|---|---|---|---|
| 检测工程实践者 | 在多轮曝光数据上得到 BA，不证明部署泛化；最差通道退化仍大于 2pp | 正确性 | 摘要首段明确未达标及开发曝光；全部最大下降项明确界限 | LOW | 已回答；不编辑 |
| 检测工程实践者 | 有什么胜过现有检测器的证据，是否旧 zero 对比失真 | 新颖性 | 第一段明确常规分拆；罚项预算段明确 zero 不再是旧 full-source zero；未宣称新方法或胜过 incumbent | LOW | 已回答；不编辑 |
| 真实传播实验者 | 标准 JPEG／设备标签是否等于真实 RR 逐图传播物理过程 | 正确性 | 首段否认真实传播不变性保证，成本记录明确 RR 物理方式和新像素缺口 | LOW | 已回答；不编辑 |
| 真实传播实验者 | 软件 split/control 是否构成新的物理实验验证 | 新颖性 | 研究范围和成本记录明确无外部新像素，修订段只声称控制和记录核对 | LOW | 已回答；不编辑 |
| 统计评测者 | CV/cal/selection 的来源角色是否分开；反复选择和130项最大值不等于总体上界 | 正确性 | 摘要曝光范围；预算段、最大项段和重跑段直接区分角色／单项区间／总体范围 | LOW | 已回答；不编辑 |
| 统计评测者 | wrong identity 和一个随机标签负控是否建立新的统计或因果结果 | 新颖性 | 表后明确 wrong 是结构身份；预算段否认单一因果归因；null仅失败门槛 | LOW | 已回答；不编辑 |
| 数值优化者 | 我们需要的收敛究竟是stationarity还是全局最优／硬可行；梯度通过是哪一层 | 正确性 | 失败门槛在重跑段，stationarity限定在链接协议和源码，正文摘要缺直接边界 | MEDIUM | 建议在摘要／审计句旁增单起点范围句；未改正文 |
| 数值优化者 | ftol重试是否新算法或目标改变；是否挑停止准则迎合结果 | 新颖性 | 重跑段有失败保留、改动和重跑比较；源码确认目标和梯度门槛保留 | LOW | 已回答；不编辑 |
| Cotter／约束学习 incumbent | 样本互斥是否满足两玩家理论及其条件 | 正确性 | 开头直接说不是Cotter两玩家算法；协议记录强凸／Lipschitz等理论条件未引入 | LOW | 已回答；不编辑 |
| Cotter／约束学习 incumbent | 常规 sample split 有什么新颖性；完整文献定位是否完成 | 新颖性 | 开头明确常规适配，协议说 full proof／novelty review尚欠；本文没有新颖性声称 | LOW | 已回答；不编辑 |
| 软件复现者 | Git恢复是否能逐字节得到原107pins；行尾和数值收据能否对上 | 正确性 | 正文只给Git72f恢复，缺四文件LF/CRLF约定 | MEDIUM | 建议明确107中4需原行尾还原；未改pins或正文 |
| 软件复现者 | 16软件控制、独立成员重建是否新增科学独立证据 | 新颖性 | 重跑段直接写记录核对非真实成员／近重复独立认证；私有审计scope亦一致 | LOW | 已回答；不编辑 |
| 一般／摘要读者 | “激活罚项”是否意味着目标完成或后续fixedmean已经有效 | 正确性 | 标题写仍未达标，摘要明确81.11/5且无独立验证；接续写尚无组合结果 | LOW | 已回答；不编辑 |
| 一般／摘要读者 | 本轮观察是否证明减少数据或修复过拟合这一机制 | 新颖性 | 预算段明确无法单因果归因，bank/映射/读出一起改变 | LOW | 已回答；不编辑 |

## 摘要冷读与文句检查

最后单独冷读标题和首段：未形成已达标、独立真实验证完成、sample split 即理论保证或后续平均已有结果的印象；失败摘要的曝光范围和目标缺口可见。训练下降、四头表和下轮方案亦未冒充因果发现。唯一需提升到读者路径前端的数值范围是 MEDIUM 第2项。

本审查正文及建议句已手工做事实／文句检查：所有数字来自本次保存数据核对，未加入性能、因果、新颖性或方法有效性新主张；整数算术独立与科学证据独立分别写明。无需文档编译。两项 MEDIUM 仍应由主任务在正文落实，随后可仅复核改句，无须新增真实拟合。

## 2026-10-08：两项 MEDIUM 的文字修订关闭

本节是追加关闭记录；上文的原审查与原 verdict table 保留为历史。**当前第71轮结果文档的未关闭发现为 0 HIGH、0 MEDIUM。** 这只关闭文字范围问题，不改变本轮 81.11%／5pp 开发结果未达到完整目标的判定。

已重新阅读修订后的整篇正文及 `sources/2026-10-08_split_source_byte_restore.json`：

- 原 MEDIUM 第2项已关闭。结果表之前、开头方法段之后新增“非零率罚目标是非凸的，单起点小梯度仅检查局部一阶数值状态，不能称全局最优或硬约束可行证书。”该句出现在数值读者形成成绩印象的位置，明确目标与梯度检查的边界，不再需要进入链接协议才获得这一限制。
- 原 MEDIUM 第1项已关闭。“成本与记录”首段明确 107 pins 中 103 与 Git blob 逐字节相同，另四个 Python 源文件需恢复原 CRLF，且直接说明普通 checkout 不能称全部 107 个 byte pins 恢复。新增说明记录使用完整 Git72f commit、107／103 计数、四个准确路径及转换范围，与本审查此前独立 blob 比较一致。原 pins 保留，未以改 pin 消除差异。

受众正确性／新颖性边界仍保持：报告未把来源互斥称科学独立或因果结果，未把训练 soft/hard 下降称 outer 成绩，未把失败运行片段当完整结果，也未把后续两互补模型固定平均当已有成果。“接续”仍明确该组合尚无数值结果、需要完整 CV／cal／outer，模型分数平均不等于准确率平均。本关闭记录不审查或认证新增 Mean72 library payload／schema 的实现或有效性；后续源文件变动不被用来替代第71轮的 Git72f 和原字节 pins。

本次只核两处范围修订及说明记录，无新增数值计算、拟合或像素读取，无数值收据改动。建议句和当前新增句已手工复核事实与表达，没有增加科学主张，无需重跑已通过的有限整数核对。

关闭时文档 SHA256：`e285ea04bf544d10dc122f4a80a0de2b6abf06b4bbaf9b7869e3b8339797b081`；字节恢复说明 SHA256：`28ddf5e4e716d8808c6a12d0e642f63129254856012c0859228a49f9984a0f9b`。这些是本次读到的修订版本身份，后续文字编辑需另留追加记录。
