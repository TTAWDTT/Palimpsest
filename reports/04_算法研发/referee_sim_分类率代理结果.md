# 第70轮分类率代理结果：独立 referee-sim

日期：2026-10-08。依 `bootloops-research:referee-sim` 在未参与写作的 fresh agent context 独立审查。对象是《第七十轮分类率代理_训练罚项不激活而留出仍退化_2026-10-08.md》及已保存协议、producer、数值与角色审计。

结论：未发现 HIGH 级矛盾；保存结果支持“真实标签训练罚项不激活，留出仍退化，无新达标方法”。有两项 MEDIUM 级呈现修订：表格旁应立即标明 null 全判 REAL、判别失败；“唯一最坏”应限定为最大下降，补全最低域/场景 BA 的并列项。以下修订建议未代写进主报告，也未改 producer 或任何数值收据。

## 受众与双轴审查

| 受众及其权威来源 | 最强标准异议 | 轴 | 回答所在位置或缺口 | 严重度 | 修订建议/状态 |
|---|---|---|---|---|---|
| 图像来源检测实践者：据 BA/drop 决定采用 | null 下降为0是否意味着稳健？全部判一类也能达到该值 | correctness | 表格仅写50/50/0；“判别失败”到训练项章节才出现 | MEDIUM | 表格旁加“null全部判REAL，REAL100%、FAKE0%；下降0来自退化解，判别失败” |
| 图像来源检测实践者及现有第69轮候选 | 这次是否改善已有最佳方法？ | novelty | 首段写最佳仍为69轮85%/2.5pp，训练章节明说zero相同 | LOW | 已回答；无需新增优越性声明 |
| 数值统计与配对评价：采用BA/同源比较 | “唯一最坏”是否隐去其他指标的并列失败？事后最大是否有同时统计保证？ | correctness | 最大下降唯一属实，事后最大边界已写；域/场景最低的全部并列未列 | MEDIUM | 限定“最大同源下降唯一最坏”；补下文全部并列最低项 |
| 数值统计与配对评价 | 收据核表、软件反例能否算新增独立科学证据？ | novelty | 准入段及身份核对段明确区分软件、收据算术、来源独立性 | LOW | 已回答；本审查也只核保存数值 |
| 约束优化/率代理社区：借用sigmoid与罚函数及Cotter文献 | soft可行能否推出硬BA/drop约束？求解是否全局、是否继承两玩家定理？ | correctness | 协议明确反例、非凸/非全局；结果开头明说未达标，不声称证书 | LOW | 已回答；罚项为0只限保存训练诊断 |
| 约束优化/率代理社区 | 是否重现Cotter或提出新优化原理？ | novelty | 主报告第二段与接续检验均明确常规适配及无定理继承 | LOW | 已回答；没有新颖性结论 |
| 表示学习/来源CV社区：五监督方向加读出 | 训练与留出差异是否已经证明同源监督重复造成过拟合？ | correctness | 训练项章节将其称为待检验解释并明确不能单独证明原因 | LOW | 已回答；只能提示同源学习过拟合可能性 |
| 表示学习/来源CV社区 | source split是否已经实现，并证明了改进？常规分拆的研究增量是什么？ | novelty | “接续检验”使用未来行动语气，未称已得到结果，但状态可再明示 | LOW | 加“下一轮拟检验；source split尚未实现，无结果”；目前不构成实证进展 |
| RR/Chimera数据与物理成像社区：借用数据及处理方式 | 是否覆盖独立真实图像、保留集、全部物理方式与预训练重叠？ | correctness | 首段开发数据限制；准入与末段记缺口、未用新像素/reserved | LOW | 已回答；算术一致不能填补图像独立性缺口 |
| RR/Chimera数据与物理成像社区 | 是否新增物理机制/不变性原理？ | novelty | 第二段明确不是物理不变性原理，保存缓存复用 | LOW | 已回答 |
| 通用读者与成本使用者 | 44入口及126.166秒是否是总优化次数或图像端到端成本？ | correctness | 成本段明确排除项、初始化、非零头baseline加惩罚两解 | LOW | 已回答；下文静态核对支持该界限 |
| 通用读者与成本使用者 | 本轮是否提出更快方法？ | novelty | 主报告未声称加速；实际CV大于条件代理 | LOW | 已回答；pilot不能用于速度优越性结论 |

## 保存margin直接重计

审查使用单独的Python标准库 `json`、`collections`、`fractions.Fraction`，不导入生产拟合器或evaluator，不读图像。每行 `score>0` 判FAKE，否则REAL；按domain/scene/condition/variant/source重建类正确整数和配对差，Chimera汇总all另计。四头每头60指标、130配对全部与保存分数/精确分数收据一致。

| 头 | 强度 | 最低域BA | 最低场景BA | 最大同源下降 | 最大下降并列数 |
|---|---:|---:|---:|---:|---:|
| zero | 0 | 17/20=85% | 39/40=97.5% | 11/180=6.111…pp | 1 |
| selected | 0 | 17/20=85% | 39/40=97.5% | 11/180=6.111…pp | 1 |
| selected_wrong_source | 0 | 17/20=85% | 39/40=97.5% | 11/180=6.111…pp | 1 |
| selected_null | 100 | 1/2 | 1/2 | 0 | 130 |

真实三头的全部最低域BA并列项：`rr/all/transfer/jpeg60_420_after_resize256`、`rr/all/redigital/jpeg90_444_after_resize256`。全部最低场景BA并列项：`chimera/cat/mac_iphone/jpeg60_420_after_resize256`、`chimera/cat/lg_blackfly/jpeg60_420_after_resize256`、`chimera/church/lg_blackfly/jpeg60_420_after_resize256`、`chimera/horse/lg_blackfly/jpeg90_444_after_resize256`、`chimera/horse/lg_blackfly/jpeg60_420_after_resize256`。

最大下降唯一为 `rr/all/original>redigital/jpeg90_444_after_resize256`。180来源，每类90；REAL正确数下降5、FAKE下降6，因此类准确率变化分别−1/18、−1/15，BA下降11/180。未把其他BA并列最低项排除出报告范围。

null共5040行，没有一行判FAKE。全部24域汇总项及36场景项都是BA1/2；全部130个保存配对都是下降0、类别准确率变化0，全部并列最坏。24域项是RR/all与Chimera/all各自3条件×4变体；36场景项是Chimera cat/church/horse各3条件×4变体。这穷尽60指标及130配对，不能挑一个worst_pair当唯一最坏。全判REAL是判别失败，未据此主张约束有效。

## OOF、罚项与两轮身份

分别读取四个 `cv_truth_*.json` 与 `crossfit.json`：四档每档11340条OOF记录（含身份、标签、fold/calibration_fold与分数）逐值相同；每档45个BA/75个drop的直接重计为最低97/108、最大5/108。真实标签20个训练折诊断rate penalty全为0；最低soft BA范围0.9974543791047765–0.9988911262906168，最大soft drop0.002524779899427705，均与独立诊断收据一致。最终真实三头5040记录全部逐值相同；完整zero保存规则与5040记录均与69轮 `paired_ba_calibration` zero相同。这是已保存对象身份，不是独立优化重放。

null四档OOF直接重计依次为最低19/40、19/40、29/60、29/60，最大下降均1/240；100与1000同分按小强度选100。训练诊断的null罚项非零，不能把“罚项不激活”泛化至null。最终真实头cal为53/60与3/80、仅BA可行回退；null为1/2与0、无BA可行回退。未重跑标定切点搜索。

OOF保存fold均与1260个source fold映射一致，cal fold=(held+1)%5，诊断每折756训练/252标定/252留出来源；保存OOF来源与最终selection来源无交集。已有角色审计passed，但此检查只认证保存身份/角色的一致性，不认证近重复或预训练重叠独立性。Q70只在查询，cal支持raw/Q90/Q60。

## 执行成本及控制版本

静态检查producer与实际audit：12完整bank/map，旧base48、aux12、shape12、LP12；`RateScoreFitter`先为12个bank建立模板，其12次初始化source读出没有作为最终rate头使用；rate入口44=40CV+4最终。非零rate入口内部先解baseline再解惩罚目标，不能把44当总优化次数。

新版pilot cold14.9570612秒、warm1.7656186秒，10cold+30warm条件代理202.53917秒；实测CV301.6381787秒。最终阶段126.1659173秒从四头循环前计时，到构造iteration时取elapsed；不含CV、输入/pilot及后续null区间/iteration与bank audit写盘。它不是图像端到端时延。

旧controls保存10项通过，旧pilot20.4636014/2.6036798秒；新controls11项通过，新pilot14.9570612/1.7656186秒。旧/新producer与算法hash一致，测试和README hash改变；版本文件记录旧Git013b0a4。旧三个收据SHA均仍与版本记录匹配；当前六个rate相关代码/协议/测试pin与iteration一致。新增零罚硬BA70%、逐行tie REAL、有害方向、改善无罚的软件断言确在新版测试中，但未再次执行这些生产拟合测试。

审查证据目录为 `work/robust_statistics/rate_score_readout`；旧控制为 `work/robust_statistics/rate_score_readout_precheck_v1`。保存分数SHA `28c46f6a7fe9286faa0129b356197879db3b3b6b4db1f2aaecb0db74836b351c` 与iteration和inactive诊断吻合。未改producer/收据，未启动训练或使用Goal API。

## 最后冷读首段

题名限定训练罚项；首段已有未达验收、开发数据、最佳仍69轮三项关键信息，支持失败实验的正确框架。表格独读仍可能把null的0下降理解为一项成功，故保留MEDIUM要求在表格处显式标明退化解。首段没有因果结论，主体也没有把待检验解释写成原因。下一轮source split尚无实现/结果；它是常规样本分拆的拟议适配，不继承两玩家约束泛化定理。

## 文字修订关闭记录 — 2026-10-08

重新只读核对第70轮主报告文字后，两项MEDIUM及一项LOW已关闭：表格紧后明确null所有查询判REAL、FAKE正确率0%、判别失败；最坏项段将“唯一”限定为最大下降，并完整列出2个域最低BA及5个场景最低BA并列项；接续检验明确“下一项计划，尚无该流程的数值结果”。修订对应原审查要求，没有扩大因果解释或方法成功的结论。

source split代码可进入实施阶段，但本次关闭只确认主报告尚无该流程数值结果的文字边界；本审查未核代码实现、未认证source split效果。原审查“尚无实现”的句子记录的是初审时点，不是对后续开发状态的持续断言。

本次未重计、重跑或改写任何数值收据，未调用拟合/evaluator。原数值核对结论保留。针对本轮结果文档的HIGH/MEDIUM未关闭项为0；科学验收仍未达到80%／2pp／独立真实验证。
