# 互补分拆事前协议：fresh-context referee-sim

第 72 轮提出两个相反来源分拆成员的固定分数均值，尚无本轮真实 controls、pilot 或完整结果。Lady 在独立于编写者的 agent context 中审查协议、实现、共享接口和审计程序；发现一项 MEDIUM：新协议要求显式方法身份，但 typed auditor 接受缺失 `readout_method`。未发现需要否定固定分拆定义的 HIGH；这项判定只覆盖所读代码和人工软件检查，未认证科学收益。

审查对象为 `experiments/origin_detection/complementary_split/{README.md,run_iteration.py}`、`src/palimpsest/detection/algorithms/complementary_split.py`、共享 QuantileScoreRule payload API、SplitRateScoreFitter complement 参数、CampaignMethod 预算字段、两个共享 driver，以及 `tools/audit_typed_source_panel_cv.py` 和对应测试。应用 BootLoops referee-sim；新文字按 prose-lint 的证据、数字、首段范围和逐句清晰度要求检查。未修改 producer、源码 pins、旧数值收据；未运行真实 fit、图像推理、GPU、像素提取或 Goal API。

## 审查受众及理由

| 受众 | 必须审查的理由 |
|---|---|
| 图像来源取证研究者 | 使用 RR/Chimera 及物理传播条件作为研究范围；会评估一般化和已有方法比较。 |
| 样本分拆与集成学习研究者 | 使用相反半集成员和等权 score ensemble；会区分成员隔离、整体独立性及分数尺度。 |
| 非凸优化及约束学习研究者 | 使用来源风险、sigmoid 率罚、梯度停止标准；协议提及 Cotter 方法的边界。 |
| 实验与统计评估研究者 | 使用五折 fit/cal/held、supplied null、配对 BA/drop 和开发选择。 |
| 数值与性能工程研究者 | 使用缓存 bank、组件入口、优化器和 pilot 代理计时。 |
| 软件复现与独立审计研究者 | 依赖 payload、source pins、typed stage proof 和历史版本恢复。 |
| 现有检测方法使用者及一般读者 | 会据摘要判断方法是否更好、更快和达到 80%/2pp 联合标准。 |

## 两轴异议与判定

| 受众 | 最强常规异议 | 轴 | 答复所在位置或缺口 | 严重度 | 处置 |
|---|---|---|---|---|---|
| 图像来源取证 | RR/Chimera 重复开发且物理方式、近重复和预训练重叠缺口仍在；即使达到门槛也不足以支持新来源泛化。 | 正确性 | README 首段写 repeated development / no independent validation，末段排除新像素和 RR reserved；输入沿用既有签名缓存。 | LOW | 后续结果摘要保留开发范围和独立缺口，不能把缓存复核等同独立科学验证。 |
| 图像来源取证 | 改两个传统读出并复用同一冻结表示，对现行检测器的具体优势尚未证明。 | 新颖性 | README 第二段明确常规样本分拆加分数平均，无新优化定理。 | LOW | 方法定位为组合假设检验；本轮尚无收益或现行外部比较。 |
| 样本分拆与集成 | A-basis/B-head 与 B-basis/A-head 可分别隔离，但所有来源仍跨整个 ensemble 的两个角色；成员依赖，不能使用独立 bagging 解释。 | 正确性 | README 第二段在平均定义前明确整体非训练来源独立、数据和标签依赖；fitter 验证两套来源键互补。 | LOW | 保留成员内隔离与整个 ensemble 的不同范围。 |
| 样本分拆与集成 | 等权 raw logits 不意味着等权统计贡献；平均分数也不等于平均 BA，抵消可丢失信息。 | 新颖性 | README 第三段固定 raw mean，注明平均可损害正确判决；人工测试给每成员 BA=.75、均值 BA=.5 及互为负数时全部判 REAL。 | LOW | 仅称固定分数均值检验，不声称稳健性继承或无损恢复。 |
| 非凸优化与约束学习 | 1e-5 最大组件梯度是两个独立非凸目标的数值停止记录，不能成为 ensemble hard-rate 约束证书。 | 正确性 | README 第二/四段及 fitter `gradient_scope` 明确；每成员 head 调用率罚目标，ftol=0，整体只校阈值。 | LOW | 后续结果并列报告组件诊断、整体硬判率及标定状态，不能把三者混成优化保证。 |
| 非凸优化与约束学习 | 互补分拆没有实现两玩家随机分类器，不能借已有约束泛化定理作为创新或保证。 | 新颖性 | README 第二段显式限定 conventional sample splitting / equal score averaging / 非 Cotter 算法。 | LOW | 保持当前明确边界；没有推导新的定理。 |
| 实验与统计评估 | null 的成员分区、内部 cal 必须使用 supplied 标签，最终 wholecal 用真标签是另一个已声明控制范围；metadata truth 不得偷偷影响分拆。 | 正确性 | source_half_split 使用 supplied y；CV 保存 supplied 标签，内部 cal 使用 y；panel final views 使用原 threshold 真标签。人工 stub 在 truth metadata 不变时反转 y，分区和审计均相应改变。 | LOW | 结果必须区分 null 内部标定与最终真标签 wholecal，保留当前 null 控制的条件。 |
| 实验与统计评估 | 反复选择开发上的改善并不能构成独立确认，也不能将后验 ensemble 权重搜索藏进固定平均。 | 新颖性 | README 首/三段：尚无结果，固定 .5/.5，不能 outer 后选 weights 或 subset；网格仍 0/10/100/1000。 | LOW | 后续选择仍限 fit-only；本次没有新的科学成绩。 |
| 数值与性能工程 | 两成员只有同一 3920 数值输入，但 numeric readout 成本增大；40 ensemble 入口不是 40 组件头或 40 优化求解。 | 正确性 | README 第三/四/五段区分冻结提取和数值读出，列 80 CV 组件头及 8 最终入口，非零组件先解 baseline。共享预算默认 12/1，新方法 24/2。 | LOW | 完整验收同时检查 bank audit 与最终收据；pilot 120s 每次和 40 调用代理均为有条件数值成本。 |
| 数值与性能工程 | 两倍 bank 不代表新的神经网络或速度创新；fit 时间没有覆盖图像完整链条。 | 新颖性 | README 第三段只复用一个 image encoder/feature extraction，拒绝从 fit 时间推断图像完整速度。 | LOW | 两个 5 方向/20 项成员和顶层 2 raw scores 分别计数；整体 basis_count=10。 |
| 软件复现与审计 | 协议要求 typed audit 显式 newmethod tag，当前程序却对缺 `readout_method` 仅累加 `method_tags_absent` 后继续；`--allow-missing-basis-count` 还可让新顶层 basis_count 缺失被接受。 | 正确性 | auditor 的 `method is None` 分支以及 basis 缺失分支；成员 count、ftol、梯度、complement 和 source sets 检查仍在。实际 AST 局部执行证实 None 被接受，未运行完整 main。 | MEDIUM | 建议为 `--complementary-split` 强制顶层方法身份及 basis_count，保持旧 schema 的原兼容语义。审计修订后应独立记录新版本，当前报告不代改 producer/pins。 |
| 软件复现与审计 | 嵌套 payload、默认预算和 complement API 是软件组合维护；恢复旧第 71 轮不能据当前源码重新判其结果无效，也不能把普通 checkout 声称为全部 byte pins 恢复。 | 新颖性 | Quantile payload schema 1/kind/rule 未改；旧 defaults 为 complement=False、12/1；revision 记录 actual old rule byte roundtrip。第 71 轮 Git+CRLF 恢复记录另存。 | LOW | 描述为序列化/预算接口修订；旧结果来源版本用历史 Git 与四文件 CRLF 恢复，不回写旧数值收据。 |
| 使用者及一般读者 | 标题容易被读成已恢复性能，实际目前只有协议和人工软件检查，没有整体分类收益或速度结果。 | 正确性 | README 标题 protocol before fitting，首段 no ensemble result；本报告首段明确真实 controls/pilot/full 均待运行。 | LOW | 结果产生前只报事前协议审查状态；不声称联合标准已通过。 |
| 使用者及一般读者 | 单轮互补组合尝试与真正领先现有方法的距离是什么？ | 新颖性 | README 第二/三段定位传统分拆均值且没有稳健性继承；尚无本轮 external comparison。 | LOW | 后续若收益出现，先作为重复开发中的经验改进，再完成独立真实验证。 |

## 人工软件检查及证据边界

人工检查使用 6 个来源、每来源 9 个视图、同一 8 维假输入，把方向和 head 数值求解全部替换为 stub。原 supplied 标签为 0,0,0,1,1,1，forward basis 为 {0,4,5}；保留 metadata truth 并反转 supplied 标签后为 {0,1,4}。两个成员的 typed partition 重建在原标签和反转标签条件均通过，故这个已构造样例支持 supplied 标签路径正确；它没有认证真实数值优化器或现实来源独立性。

缓存检查依次输入原数组、反转标签、重命名 metadata、原数组、原数组配不同 strength。两个成员各仅调用 basis stub 三次；最后两次原 `stage_template_sha256` 均与首次一致，反转标签不同。改变 metadata 后有新的模板签名；数值 child bank 可去重、metadata stage template 不可混同。该检查未读取真实 feature cache，未执行真实 fit。

typed auditor 的缺方法身份分支用实际文件解析出的 AST 独立执行，`method=None` 时没有 ValueError，计数为 1。这是局部判定反例；不把局部接受夸大成伪造完整 11340 行 CV 已通过，也没有写新的 `calibrated_oof_audit.json`。现有成员证明仍会重建 deterministic SHA 排序、supplied class/scene 层、相反半集和 solver 行数；证明对象是保存记录，不是重新运行或独立认证优化器的实际数据成员。

当前 code_pins 返回 112 项；所查互补算法、Quantile payload、split 参数、source partition、共享 budget/driver、training_fit、panel crossfit 和 typed auditor 九个关键源文件均被包含。主任务告知六个 targeted 测试文件已通过和旧第 71 轮 selected rule 实际 codec roundtrip 文件相同；本审查读取 `sources/2026-10-08_complementary_composition_revision.json` 的原/roundtrip SHA256 均为 `94af3a5ff6f84053a663a4c7eae309b3d62fb53a202d370efd1521371d034e92`，没有重复运行这些测试或重新认证旧成绩。`work/robust_statistics/complementary_split` 未见本轮运行产物，真实控制记录、pilot 和完整结果仍待完成。

## 数字、来源与首段冷读

内层每成员 basis/head 各 378 来源、3402 行；组件 head 合计 756 来源、6804 行。最终每成员各阶段 630 来源、5670 行；两个 head 合计 1260 来源、11340 行。一个成员是 5 方向及 20 多项式项，整体 mapped_dimensions=2 是两条 raw logits，basis_count=10 是组件方向总数；没有把 40 多项式项作为联合拟合头。整体梯度记录取两组件最大值，仍是分开目标。

完整计划为 truth/null 各 5 折、4 strength，合计 40 ensemble fit 入口、80 组件 head；最终四候选合计 4 ensemble 入口、8 组件 head。24 numeric bank/map 是两成员各 12，pilot 为 2；非零 head 另外解 baseline，完整模板初始化还会执行已有辅助拟合，故这些入口数没有被转写成总优化器调用数。120s 是 pilot 两次分别限时，40-CV 代理按 10 cold+30 warm；尚无本轮实测时长，不能产生完整运行 ETA 或图像速度结论。

第 71 轮原执行代码可用 `72f4b407503357b7d955d2437aacf4d31ebdb761` 恢复逻辑，107 原 pins 中 103 与 Git blob 相同。另四个文件 `src/palimpsest/detection/algorithms/source_view_risk.py`、`src/palimpsest/evaluation/classification.py`、`src/palimpsest/evaluation/pairing.py`、`src/palimpsest/evaluation/robust_views.py` 需要恢复原 CRLF 字节，范围见 `sources/2026-10-08_split_source_byte_restore.json`。共享源码修订使当前字节不同，不能据此重新否定旧第 71 轮结果；旧 pins、旧 receipts 和历史恢复范围各保留原含义。

README 首段将第 71 轮 81.11% 最低域 BA 与 5pp 下降作为开发动机，并写尚无本轮结果；它没有暗示互补均值已达 80%/2pp。整体训练来源不独立、分数平均可损害 BA、成员目标分开和未知实际开销均出现在方法定义附近，外部读者无需拼合远处限定才能知道当前边界。后续结果摘要须保留这些限制，并明确 typed audit 只是保存分数算术、角色和成员记录的核对。

最小修复建议只针对新 typed schema 的身份必需项，不改变数值方法或过去的收据。报告全文按证据归属、单位、完成状态和首句独立可读性顺读；未新增科学结果、定理或实际运行认证。

## 软件修订复审追加

原表的 MEDIUM 对应审查时 typed auditor SHA256 `a7455dd160da32d6f5a314f0ddec54de5abcfa8ab68a8ce4b95b67525e723254`，保留为修订前快照。根任务随后修改新 complementary schema 的必需身份检查；复审所读 auditor SHA256 为 `94d17b1cb342d0cf62e5abbbdd17068ace73869b15c274135bd1cb216f62df83`，原字段缺口在以下已检查局部源分支中关闭。

Lady 从修订文件独立提取并执行实际 AST：complementary=True 时缺 `readout_method` 被拒绝，False 时仍按旧 schema 记录缺失；新 flag 拒绝 `--allow-missing-basis-count`，要求 mapped_dimensions=2、basis_count=10，合法 2/10 组合通过 schema 判定。成员 solver 分支另增加准确 rate readout 方法字符串核对，已经读源，但此次没有执行完整 main。该复审只确认具体分支的拒绝条件和旧兼容行为，未出具完整 CV 审计通过收据或本轮科学结果；原表的 MEDIUM 不再是修订源码上的开放缺口。

## 缓存依赖边界修订与新 pilot 复审追加

初版 19 项软件 controls 通过之后，首次 pilot 在数据输入的旧 Q60 cache strictpin 校验处被拒绝，错误为 `Parent code changed: src\palimpsest\evaluation\calibrated_readout_campaign.py`。panel driver 在 `data.inputs()`、`data.training()` 返回之后才建立 fitter，因此这个缓存拒绝发生在真实 fit 之前；旧 controls 和 failure 仍保存在 `work/robust_statistics/complementary_split_precheck_v1`，两个文件 SHA256 与 `sources/2026-10-09_complementary_cache_boundary_revision.json` 一致。没有把初版 controls 通过写成真实 pilot 已开始或已完成。

当前 `calibrated_readout_campaign.py` 的 SHA256 为 `a8985f6fe77cbc530a669608a044715a0c27527f6a77be947cfe98a7c1719a5d`，已恢复原模块完整字节，CampaignMethod 保留原 parameters/pilot_parameters/tests/pins/fitter/calibrate/load 七字段。预算现在由 `training_fit.py` 的 `bank_budgets` 和 BudgetedRecordAwareCampaignMethod 提供，panel driver 读取旧默认 12/1 或新方法显式 24/2；预算类型拒绝 bool、非整数、非正值及 full 小于 pilot。RecordAwareCampaignMethod 和 BudgetedRecordAwareCampaignMethod 都继承 `record_aware=True`，新 runner 确实使用后者；人工 stub 已确认 fit 记录按 tuple 传入，没有退回遗失 metadata 的 legacy 数值调用。

Lady 对 `strong_token_views`、`cure_token_covariance`、`cure_token_quantiles_full` 三份真实缓存收据的全部声明 code pins 作只读文件哈希比较，三者差异数均为 0。修订后协议明确旧 cache writer 模块不变，输入校验仍保留严格 pins；没有增加忽略路径、重写 cache、放宽 pin 或重跑 encoder 的兼容例外。初版 `sources/2026-10-08_complementary_composition_revision.json` 的 shared CampaignMethod 字段改动及 after hash 描述其历史源码快照；当前模块边界应依本追加和 2026-10-09 cache boundary revision 说明，不把初版 after 值当作当前恢复模块的 hash，也不更改历史记录。

新 `work/robust_statistics/complementary_split/software_controls.json` 保存 `19 passed in 2.07s`；初版档案为 `19 passed in 3.24s`。本复审读取两份 stdout 并核对哈希，没有重复运行测试。新 controls 与 pilot 的 code_pins 都与当前 runner 返回 pins 完全一致，pilot 的 feature receipt SHA256 和保存规则 SHA256 也与实际文件一致；这支持运行版本边界一致，不认证真实图片标签或科学泛化。

新 pilot 已保存 passed=True，cold strength100 为 15.3052361 秒，warm strength10 为 2.5860429 秒。其 40 个 ensemble CV 入口代理为 230.633648 秒，条件仍是 10 cold+30 warm，其他 strength 的实际成本可变，输入/fullfit/outer 均排除；120 秒标准分别作用于两次已保存计时。两次诊断的整体 head 计数均为 6804 行/756 来源，mapped_dimensions=2、basis_count=10；各成员 basis/head 为 3402 行/378 来源，来源键互斥且相反互补，complement 标记为 False/True。组件最大梯度分别为 `4.1236551289900933e-07` 和 `9.966791930269767e-07`，整体 inner calibration 各含 45 组/75 比较，均采用 `ba_feasible_minimum_drop` fallback；这不是 80%/2pp 联合可行或外层收益证明。

复审快照中完整 `iteration.json` 尚不存在，完整数值账、外层成绩、方法收益与实际完整耗时仍未知。预算和缓存边界修订没有改变两成员分开拟合、固定 raw logit .5/.5、成员阈值零及仅整体 cal 的定义；未发现本次边界修订引入新的 HIGH 或 MEDIUM。Lady 本次只读源码/收据并运行预算及调用边界 stub，没有真实 fit、encoder、新像素或 Goal API 调用；原报告的“controls/pilot 待运行”句保留为此前快照，此追加更新为新 controls 与 pilot 已完成、完整结果待运行。
