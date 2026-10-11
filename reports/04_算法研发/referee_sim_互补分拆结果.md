# 互补分拆结果：独立 referee-sim

Lady 在未参与报告编写的 agent context 中审查第七十二轮结果。保存分数支持最低域 BA 84.44%、最大同源下降 5.56pp，联合验收仍未达到；本次未发现开放的 HIGH 或 MEDIUM。审查确认保存分数的算术和报告范围，真实优化器未重放，独立真实验证仍缺失。

对象为 `reports/04_算法研发/第七十二轮互补平均_判别力恢复但退化仍明显_2026-10-09.md`、`experiments/origin_detection/frozen_features/cure/complementary_split/README.md`、互补及 split 源码、typed 审计收据与缓存边界记录。数值输入均来自 ignored 私有目录：`work/robust_statistics/complementary_split/iteration.json`、`crossfit.json`、`selection_scores.json`、`bank_cache_audit.json`、`calibrated_oof_audit.json`、`software_controls.json`、`pilot.json`；旧拒绝记录位于 `work/robust_statistics/complementary_split_precheck_v1/`。这些路径按仓库相对路径书写，以便本地复核；本审查没有把 ignored 文件称为 Git 已保存数据。

本审查应用 BootLoops referee-sim，并按 prose-lint 分别核对新文字的数字、归属、范围与逐句含义。检查只读取现有 JSON、源码和文件哈希，使用标准库整数及有理数重算已保存 margin 的分类结果；没有调用真实 fit、evaluator、encoder、新像素或 Goal API。未修改 producer、数值收据及原报告。

## 审查受众

| 受众 | 纳入理由 |
|---|---|
| 图像来源取证及物理传播研究者 | RR/Chimera、再数字化及 JPEG 条件构成研究对象。 |
| 样本分拆与集成学习研究者 | 方法使用互补来源半集、固定 raw logit 均分及整体阈值标定。 |
| 约束学习与非凸优化研究者 | 方法使用 sigmoid 率罚、组件梯度和 BA/drop 约束。 |
| 实验统计研究者 | 结果使用 OOF 选择、null、配对下降和区间。 |
| 数值及性能工程研究者 | 报告给出 bank、组件入口、缓存与计时。 |
| 软件复现研究者 | 结果依赖 typed schema、嵌套 payload、代码 pins 和历史拒绝。 |
| 现有检测方法使用者及一般读者 | 标题和首段会影响是否部署、是否达到用户标准的判断。 |

## 两轴判定

| 受众 | 最强常规异议 | 轴 | 答复所在位置 | 严重度 | 处置 |
|---|---|---|---|---|---|
| 图像来源取证 | 反复曝光的开发集及未补预训练、近重复、RR 物理方式缺口，不能支撑新来源泛化。 | 正确性 | 首段明确开发与未达独立验证；成本节末尾列缺口。 | LOW | 保留边界；未出具独立验证结论。 |
| 图像来源取证 | 固定组合对现行检测器有什么新优势？缺外部比较时不能声称新检测范式。 | 新颖性 | 协议第二段明确 conventional sample splitting/equal score averaging；结果只比较第71轮且保留第69轮最佳。 | LOW | 正文可加一句“本轮检验常规互补分拆与等权分数平均的组合，没有外部方法优势结论”；未改原文。 |
| 样本分拆与集成 | 每个成员内部来源互斥并不意味着整个 ensemble 或真实数据独立。 | 正确性 | 方法首段直接写跨整个方法的两阶段参与和组件依赖。 | LOW | 保存组件集合检查支持记录互斥；科学独立性未认证。 |
| 样本分拆与集成 | raw logits 的平均不是平均正确率；规模或尺度变化可抵消正确决策。 | 新颖性 | 选值节给各75%、平均50%的人工反例，并否定更多成员必然更稳。 | LOW | 源码固定 .5/.5、成员threshold0；只支持当前经验比较。 |
| 约束学习与优化 | 最大组件梯度不是对整体硬错误率求导，单起点非凸停止不证明可行或全局最优。 | 正确性 | 选值节及成员审计节均在说明梯度时写明边界；源码gradient_scope一致。 | LOW | 核对保存梯度取组件最大值；没有优化器重放。 |
| 约束学习与优化 | 能否借约束学习理论给当前组合或未来硬错误率优化提供新保证？ | 新颖性 | 协议排除新定理/Cotter算法身份，结果“后续范围”仅提出研究计划。 | LOW | hard-error exact optimization尚未实现或注册，不属于本次认证对象。 |
| 实验统计 | null小下降会不会仅来自常量预测？wrong身份控制是否被写成因果或独立控制？ | 正确性 | 表格后立即写null5039FAKE/1REAL、BA失败；wrong为聚合率结构身份。 | LOW | 独立重算四头全部分数确认；不从小下降推稳健判别。 |
| 实验统计 | 130项事后最大、单项bootstrap及重复开发选择能否替代全体稳定性上界或独立确认？ | 新颖性 | 最坏配对段明确不能提供全体上界，首段明确开发曝光。 | LOW | 未计算新的区间、显著性或全体保证。 |
| 数值及性能 | 24数值bank、80CV+8final组件入口和一个编码特征是否混为物理bank、神经前向或全部优化调用？ | 正确性 | 成本节分别给两成员各12、40ensemble/80component、4/8，并列初始化与baseline。 | LOW | 预算收据和源码计数一致；两个成员共用一个特征输入，没有实测整体图片速度。 |
| 数值及性能 | 复用特征或pilot代理能否支撑速度创新？代理已低估实际CV。 | 新颖性 | 成本节写230.634秒代理低于304.983秒CV，146.821秒的排除项及图片速度未测。 | LOW | 不从这些计时推断图像完整链条速度优势。 |
| 软件复现 | 初版cache拒绝是否绕过pins？新schema是否仍容许缺方法身份？保存记录能否证明优化器实际使用的样本？ | 正确性 | 修订节写拒绝先于真实fit、恢复原字节及新19控制；审计明确实际成员未重放。 | LOW | 核对拒绝档案哈希、恢复模块和112当前pins；typed收据缺身份计数为0、禁止缺basis，原事前MEDIUM已有修订记录。 |
| 软件复现 | 软件controls、codec和缓存通过究竟增加什么科学证据？ | 新颖性 | 修订节明确软件身份不认证科学独立性；协议仅描述接口维护。 | LOW | 不把19测试或112pins计入泛化证据。 |
| 使用者及一般读者 | “恢复判别力”是否暗示已完成研发或可按80%/2pp部署？ | 正确性 | 标题写仍未达标，首段列84.44/5.56与独立验证缺失，表格后解释null失败。 | LOW | 开头冷读可见主要失败条件；未发现需要移到首段的隐藏排除项。 |
| 使用者及一般读者 | BA略升而最坏下降变大，具体改进到哪里？ | 新颖性 | 对第71轮81.11%/5pp的比较明确判别力回升但下降更大；最佳第69轮85%/2.5pp仍未达标。 | LOW | 仅解释当前开发对比，没有领先现有方法或完成标准的结论。 |

## 保存分数复算

分类约定按保存 margin 的 `score>=0` 判 FAKE、`score<0` 判 REAL。对每个视图分别用两类整数正确数计算 BA，再按源码声明的域/all 与具体场景聚合；所有同源比较都按 `src` 对齐。四头各5040行唯一分数，对各60指标及130配对的逐项 BA/drop 重算全部一致；最大值按有理数比较，保留全部并列。

| 头 | 最低域BA | 最低场景BA | 最大下降 | 最坏项数／翻转来源 |
|---|---|---|---|---|
| zero | 149/180 | 77/80 | 13/180 | 唯一，17来源 |
| selected | 38/45 | 19/20 | 1/18 | 唯一，14来源 |
| selected_wrong_source | 38/45 | 19/20 | 1/18 | 唯一，14来源 |
| selected_null | 1/2 | 1/2 | 1/180 | 两项并列，各1来源 |

zero、selected与wrong的最低域BA均唯一在RR/redigital/Q90。最坏下降均唯一在RR原图→redigital同Q90，180来源、REAL/FAKE各90；null最大下降并列为RR/redigital/Q90→Q70及Q90→Q60。wrong与selected的5040整行保存记录完全相同，包含分数；null判5039个FAKE、1个REAL，最低域BA的并列为除RR/redigital/Q90外的23域条件，最低场景BA在全部36个Chimera具体场景条件并列。

最低场景的并列也已检查。zero为church/lg_blackfly/Q60及horse/lg_blackfly/raw、Q90、Q60四项；selected/wrong为horse/mac_iphone/raw、Q60及horse/lg_blackfly/raw、Q70四项。报告未逐列这些最低场景并列，但没有声称最低场景唯一，因此不构成数字错误。

truth/null各四强度的OOF保存分数都含11340行；独立逐项重算各45指标、75配对，全部与crossfit收据一致。truth按0/10/100/1000的最低BA依次77/90、77/90、467/540、47/54，最大下降2/27、13/180、7/108、8/135；四档BA均过80%，最小最大下降选1000。null最低BA依次29/60、19/40、13/30、19/40，全部未过80%；按最高最低BA回退选0，不能把回退理解为目标通过。

selected最终标定收据为最低BA53/60、最大下降3/80、最坏类别正确率17/20，joint_calibration_feasible=False、calibration_ba_feasible=True、fallback=ba_feasible_minimum_drop。null为1/2、0、0，两个可行字段均False，fallback=no_ba_feasible_maximum_ba。标定真实标签的null控制范围见既有协议和事前审查；本次只核对收据及保存分数，没有再产生标定或区间。

## 单位与版本

40个CV ensemble入口对应80个组件head入口；四个最终ensemble入口对应8个组件head入口。两个成员各44入口，共88；每成员12个数值bank/map，合计24。收据的legacy base_fits各48合计96、auxiliary各12合计24、shape各12合计24、LP各12合计24、初始化source读出各12合计24，与报告一致；非零组件另外先解baseline，入口数不能当全部求解次数。

全部40个CV与4个final诊断的保存来源集合互斥且互补，ftol=0、组件最大梯度小于1e-5。CV顶层6804行/756来源是两个readout集合之和，每个组件3402行/378来源；final对应11340/1260、每组件5670/630。顶层mapped_dimensions=2对应两个raw scores，basis_count=10对应两成员各5方向；源码给每成员20多项式项。保存组件集合与梯度记录核对不等于真实求解器样本重放。

分数与crossfit文件SHA256分别匹配`a5446acc6649a3ebca6e0a5611f3bf8399a7e4a88390231b09860f32cdb7a12c`、`dda1f21dbaefda98167507bad8b4aa68746b5d1c72ee7a9a2dc7104e7c56e597`，iteration声明的112文件pins与当前字节全部相符。旧controls及拒绝档案SHA匹配`sources/2026-10-09_complementary_cache_boundary_revision.json`，恢复模块匹配其`a8985f6fe77cbc530a669608a044715a0c27527f6a77be947cfe98a7c1719a5d`；源码在inputs/training完成后才建立fitter，支持拒绝先于真实拟合的说明。新controls收据保存19 passed，新pilot保存cold15.3052361秒、warm2.5860429秒；本审查没有重跑测试或pilot。

## 首段冷读与审查范围

报告没有独立abstract，故最后单独冷读标题、首段、主表和结尾。首段同时给绝对BA、最坏下降、验收失败和重复开发范围；主表后立即揭示null退化，方法定义附近解释整体依赖，结尾仍把硬错误率优化作为后续研究。读者无需先读数值附录才知道当前方案未成功。

原报告主要结论可按重复开发的失败试验对外解释；本审查没有认证新来源泛化、物理传播不变性、全体置信上界、全局优化或下一方向。唯一文字建议为LOW的新颖性定位句，已在判定表具体给出；原报告无需HIGH/MEDIUM级修订。Lady 对新报告作两遍数字与范围阅读、逐句含义与表格结构检查，未给出渲染器执行或第三方最终审阅的声明。
