# 分类率代理事前协议：fresh-context referee-sim

以下原稿保留首次评审时的信息；文末追加记录补正首版pilot时序，并确认两项MEDIUM已关闭。新版成本小试随后完成，不能把原稿“尚未运行”读成当前状态。

日期：2026-10-08。对象为分类率代理 README、run_iteration.py、rate_penalty.py、rate_score_readout.py 与 test_rate_penalty.py，并追读它们调用的训练记录、source-risk、五方向规则、配对标定和 panel driver。评审在未撰写上述方案的独立代理上下文进行，应用 `bootloops-research:referee-sim`。

当前结论：未发现类平衡、下降方向或 bias 梯度的源码错误；存在两项 MEDIUM 事前控制／呈现缺口，建议在全量前补齐。当前仅有人工软件检查，未运行本轮真实 fit、成本 pilot、OOF 或外层评价。源码与人工检查通过不能证明 BA≥80%、所有同源下降≤2pp，也不构成独立真实验证。此评审没有改 producer、pins 或数值收据。

## 受众及最强常规异议

| 受众 | 获得审计行的原因 |
|---|---|
| 图像来源检测研究者及既有方法社区 | 判别 AI／自然内容，继承既有表示与基线 |
| 约束优化／分类率学习研究者 | 借用平滑分类率及罚函数概念 |
| 数值优化研究者 | 非凸 L-BFGS-B 与 stationarity 门控 |
| 实验与统计评审 | 使用来源折、训练 null、反复曝光数据与多候选选择 |
| 软件／可复算评审 | 依赖记录角色、缓存、保存规则与代码指纹 |
| 实践者及摘要读者 | 可能把代理可行、软件 passed 或内部 BA 误认成用户标准达成 |

| 受众 | 最强常规异议 | 轴 | 回答所在／未回答 | 严重度 | 本次处理 |
|---|---|---|---|---|---|
| 检测研究者 | 旧开发样本上拟合更高 BA，如何证明未知来源和真实处理仍达80%／2pp？ | 正确性 | README没有首段明确反复曝光／独立验证未进行；driver iteration scope 才明确重复开发 | MEDIUM | 记录呈现缺口；建议在README首段写明开发性质与独立验收仍欠，不改结果 |
| 检测既有方法社区 | 五方向与二次读出已存在，贡献到底是什么，和当前检测方法比较了吗？ | 新颖性 | README明确仅改训练目标，不声称新优化原理或文献全面性 | LOW | 不扩大贡献；本次未做文献全面性认证 |
| 分类率学习研究者 | sigmoid proxy E≤.2不能保证硬错误≤.2；你验证的是同一个指标吗？ | 正确性 | README明确 proxy不证明hard；但现有测试只覆盖hard完美／soft差的反向例，未覆盖proxy零罚／hard失败 | MEDIUM | 下文给出人工反例；建议加入预检后再全量 |
| 分类率学习研究者 | 有限罚函数与单目标优化不是Cotter两玩家算法，何以继承其保证？ | 新颖性 | README明确不是proxy-Lagrangian复现，不继承独立数据约束泛化定理；完整P3阅读欠项显式列出 | LOW | 维持方法归属边界，不声称新理论／继承保证 |
| 数值优化研究者 | 目标非凸；单起点成功／小梯度只说明局部一阶数值状态，凭什么叫可行或最优？ | 正确性 | README首个方法段；fit_rate_source_risk检查success、finite value／residual、梯度≤1e-5和objective不增，scope明确单起点 | LOW | 不把门控称全局最优、约束证书或性能结果 |
| 数值优化研究者 | 这与普通平滑罚函数有何新优化方法区别？ | 新颖性 | README conventional rate proxies/penalty methods，与首个方法段一致 | LOW | 不生成新的方法优先权声明 |
| 实验／统计评审 | 训练null是否偷偷用真label，cal／held是否渗入训练，wrong-source是否真正干预成对因果关系？ | 正确性 | supplied labels驱动basis／head／panel；fit-only记录拒绝；CV cal采用其mode标签，final cal仍真标签；README明确wrong身份控制 | LOW | 认可代码语义边界；final null只能按声明的控制角色解释 |
| 实验／统计评审 | 反复尝试众多读出后的改善能算独立新证据吗？ | 新颖性 | driver最终scope明确Repeated development；README尚未首段突出 | MEDIUM | 与首行呈现缺口合并处理，不重复计算一项问题 |
| 软件／可复算评审 | 保存kind与strength能认证训练loss吗？只比较同一规则序列化能证明旧bank原样复用吗？ | 正确性 | save仍quantile_score_consistency；零路径baseline对象；人工旧fitter精确parity；真实同rule portable gate仅软件身份 | LOW | 下文限定保存字段与cache ledger能证明的范围；真实旧bank／原输入证据仍待主流程验证 |
| 软件／可复算评审 | 这是否新增了可独立审计的算法对象，还是旧规则表示承载新训练器？ | 新颖性 | RateScoreFitter改变训练器，输出仍QuantileScoreRule | LOW | 清楚称训练目标候选，不借rule kind宣称新算法类别 |
| 实践者／摘要读者 | 看到rate-aware及passed能否依赖它满足用户BA／下降标准？ | 正确性 | README已经写proxy不证明hard，但未第一段集中给出本轮未运行和独立验收状态 | MEDIUM | 建议首段状态句；本报告首段已经放置边界 |
| 实践者／摘要读者 | 其相对现有系统已证实的实用改进是什么？ | 新颖性 | 尚无本轮真实结果，README是事前注册 | LOW | 不声称有效、实用改进或达标 |

## 数学与软件核对

`training_rate_panel`在每个(domain, scene, condition, variant)及scene=all汇总组中，对各类独立归一化给半质量。当前protocol的来源权重在每个domain内各来源、各视图相等，因此这些组的hard error与约定BA互补；此结论不自动适用于外部任意不等权来源。`processing_pairs`先检查完整fit-only来源板、唯一身份、同来源单类与场景，传入的labels决定类别，不读取records的真标签代替训练null。

下降方向正确：BA_before−BA_after = E_after−E_before。处理条件比较是original→处理；编码比较按注册variants组合次序。`rate_penalty`对正的E_after−E_before−.02施罚，反传在after加、before减；bias梯度为所有行系数之和。源码及已有有限差分测试包括bias，没有复用会消掉bias的成对差分梯度。

本评审仅运行了纯人工完整板的三个即时检查，输出留在工具对话，未生成项目数值收据：每类10个来源、每个来源3条件×3编码，各类3/10来源signed margin=−.001，其余7/10=3；smoothing=.25。所有组一致时，proxy penalty=0，最低soft BA约0.8496956994777778，但最低hard BA=0.7，hard drop=0。它直接显示软可行不能certify80%硬指标；这不是图像性能实验。

零参数时每行score=0，strict score>0判FAKE使全部预测REAL；REAL行正确、FAKE行错误。类平衡总体hard error和proxy error都约.5，所以只断言总体数字会漏掉逐类tie差异。再将condition=a每类额外一来源置为signed margin=−.001，所得最大正向soft drop约0.05009938544920667，penalty约0.00018124892167182565，符合有向罚项。本次未声称这是完整人工控制集。

**MEDIUM控制缺口：**现有四tests没有直接断言上述proxy可行／hard失败、逐行strict tie、纯改善不受下降罚或null supplied labels独立于records.label。README的预检清单已承诺soft/hard tie mismatch及目标语义；建议将关键边界写进人工预检，再记录software_controls。此意见要求加强控制，不要求调整强度或看真实结果后改协议。

## 记录、缓存与规则边界

rate训练器复用QuantileScoreFitter产生的五方向bank与二十项映射；template缓存键包含输入数组、supplied labels、weights、true sources。wrong-source不参与键、basis、source-risk或aggregate rate项，因此本方案wrong-source严格身份相等是结构事实，不能当作有效因果成对干预或不变性证据。训练records未进入键，但panel每次由明确records重建；缓存仅复用不依赖processing records的bank／map。

protocol从签名parent和Q60 subcache以`mmap_mode='r'`读入并检查旧prefix、identity和probability；随后训练数组与视图会物化。源码说明支持只读接入，不能证明本轮真实cache已检查通过，也不能解释为全训练过程零复制。人工zero与旧QuantileScoreFitter的精确parity覆盖小fixture；最终portable gate只比较同一规则的native／mapped／loaded路径，并非对历史真实bank的独立核验。

保存仍使用`kind='quantile_score_consistency'`，零强度返回原baseline对象。这种表示能够承载相同分类函数；**实际saved零规则的kind或strength字段都不certify其训练损失**。若后续检查actual saved规则，应结合对应运行代码pins、诊断与数组／来源provenance；本评审未读取或认证本轮actual saved规则。bank audit记成功入口及共享缓存数，不是独立optimizer调用认证，更不是新科学证据。

## 冷读首段与结论

README冷读的主要问题是状态和开发集性质不够靠前；proxy／hard、非凸局部状态、wrong-source身份及文献欠项本身已明确。建议在首段直接写“反复曝光开发缓存上的事前候选；本轮真实fit／pilot／OOF／外层尚未运行，独立真实验收仍欠”，并在新增检查后更新本报告的审计状态。未来进度变化应追加记录，不把这份事前评审改写为事后性能证书。

本轮没有HIGH源码反驳项。上述MEDIUM缺口应在全量前补齐；之后仍需原协议software_controls、实测成本pilot、锁定强度OOF／外层及独立整数核算。用户最终标准保持BA≥80%、所有同源下降≤2pp和独立真实验证。已声明的P3完整论文／附录阅读及文献全面性欠项继续欠，不允许据此称新理论或继承Cotter保证。

## 追加：人工控制与首段修订复核

2026-10-08，复核新增人工测试、README修订及两个私有目录的短收据。原评审文字保留为首次审计记录；其中把“未运行成本pilot”作为全流程状态的表述不准确：评审当时没有读取root已经完成的首版pilot。本次只读补正该时序，不将首版成本结果解释为分类全量或独立验证。

| 原问题 | 修订及核对 | 追加状态 |
|---|---|---|
| MEDIUM：关键人工边界未进入预检 | 新增20来源完整板测试，直接断言proxy零罚／soft BA>.8而hard BA=.7、逐行零分判REAL、处理恶化产生正向drop和罚、纯改善零罚 | 已关闭此控制缺口 |
| MEDIUM：首段未突出开发集／运行状态 | README首段已明确RR／Chimera反复曝光、独立验证未通过、首个pilot只测成本且完整campaign未运行；末段另述旧收据保留与新版本须重测 | 已关闭此呈现缺口 |

本次读取的`work/robust_statistics/rate_score_readout/software_controls.json`记录11 passed，其中本模块五tests；收据的全部code pins与读取时源码匹配。比较Git `013b0a4`与当前版本，rate_penalty.py、rate_score_readout.py及run_iteration.py三个生产文件没有diff。上述新增测试和记录控制属于软件检查，不能把人工70%反例或software passed当作真实图像性能。

首版收据保存在`work/robust_statistics/rate_score_readout_precheck_v1/`，其controls记录10 passed；pilot记录cold=20.46360139999888秒、warm=2.6036797999986447秒、条件40CV投影=282.74640799994813秒。原先所谓“四test”是首版rate模块的四tests，完整首版控制组实际为10项。该pilot测一个train／cal／held折，单CPU线程、先strength100再10；包括训练、cal标定、held分数核查，条件投影按10冷＋30暖计算，排除输入读取、全fit、外层评价，其他参数耗时仍可能不同。它用真实缓存拟合小试头，但没有执行40CV／四最终头的分类全量，也不产生独立验收结论。

本次读取时，新版目录已有匹配当前pins的controls，尚无pilot.json；旧版与新版目录均无crossfit.json或iteration.json。因此两项MEDIUM修复不追认旧pilot拥有新增测试，新版仍须保存自己的成本pilot再按协议进入全量。首版数值、规则和pins保持原目录与Git版本的对应关系，未修改数值收据。

README仍明确完整59页论文／附录、两玩家定理、作者新近工作及引用链欠读。本次只核对修订和记录范围，没有补做完整文献综述；候选不获得新颖性／新理论声明，也不继承Cotter的优化或约束泛化定理。保存的零规则kind／strength依然只描述规则表示及字段，不能认证训练损失。下一阶段的硬指标与独立真实验收要求保持原样。
