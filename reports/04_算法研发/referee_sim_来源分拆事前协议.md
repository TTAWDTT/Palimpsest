# 来源分拆事前协议审查

Lady 的独立 fresh-context 审查，2026-10-08。按 `bootloops-research:referee-sim` 核对第71轮 README、运行入口、来源分拆器、split fitter、typed auditor 与新增测试，并顺读 rate、quantile、record-aware、panel CV 依赖。作者提供的执行状态为新增4个人工测试通过，真实 controls、pilot、full 尚未运行；本审查没有重跑那些测试，没有调用真实模型拟合、图像推理、encoder、新像素或 Goal API。

## 结论

发现2项 MEDIUM 软件问题，建议在记录真实 controls 和 pilot 前修复。缓存反例可令实际 basis 与新 head 重叠，而保存成员名单仍显得互斥；训练记录传递模块缺少本轮源码 pin，变更该边界不能被现有 pins 检查察觉。两项都不能据此断言固定的真实面板已产生污染，但必须先排除它们，才可依赖本轮执行记录解释结果。没有发现 README 将来源分拆写成 Cotter 保证、独立验证或源复用致败的因果证明。

审查员只新增此报告，未修改 producer、pins、科学收据或旧第70轮材料。这里提出的修改对应已经声明的来源分拆与执行追溯要求，不新增人工批准流程。

## M1：缓存缺少分拆身份

`src/palimpsest/detection/models/frozen_features/cure/split_rate_score.py:31–39` 先用 records 的 domain/src/scene 生成 mask，再以整组 x/y/w/s 的摘要取模板。摘要不含 records 身份或实际 basis mask。数值数组相同而来源身份变化，模板缓存命中，但分拆可能改变；随后成员诊断由新 records 重新生成，不能证明命中模板实际使用了这些成员。

纯人工反例使用现有4来源、每来源9视图 fixture，x 的首列为数值来源组。basis 与 head 拟合函数均被替换成只记录传入数组的 stub，没有运行优化器。第一次 src 为 `0,1,2,3`；第二次改为 `0_0,0_1,0_2,0_3`，x/y/w/s 完全相同。实际输出：

```text
original_basis_numeric_sources [0, 2]
renamed_expected_basis_numeric_sources [1, 3]
basis_stub_calls [[0.0, 2.0]]
same_training_arrays_key True
stale_basis_rows_now_in_head [0, 2]
diagnostic_records_new_basis [('d', '0_1'), ('d', '0_3')]
```

第二次的模板仍由数值来源0和2产生，新 head 又接收0和2；分拆声明失效。新诊断却声明 basis 为新身份1和3。最小修复是让模板键依赖实际 basis 输入及必要的分拆身份，或把 records/分拆摘要纳入现有键，并增加这个冷/暖缓存反例；保存诊断应对应模板创建时的成员。也应检查同一 basis、不同 head 数值时是否错误增加 wrapper 模板数并导致其 audit 与 provider bank 数不符。

固定数据内各折的来源身份没有变化，本审查未证明真实运行会触发此反例。严重度为 MEDIUM，依据是 README 宣称 bank/map 只看规定 basis，现实现对合法的缓存重复调用不能维持这一约束。

## M2：record-aware 边界未 pin

`experiments/origin_detection/frozen_features/cure/split_rate_score/run_iteration.py:26–33` 继承 pins 并加入新模块和测试，但缺少 `src/palimpsest/evaluation/training_fit.py`。这不是仅供文档参考的依赖：内层 CV、pilot 和 final 均通过其中 `fit_training_rows` 将 records 交给 split fitter，`RecordAwareCampaignMethod` 也定义于该模块。

仅 import 第71轮的 `code_pins()` 并规范化路径分隔符，得到：

```text
src/palimpsest/evaluation/training_fit.py False
src/palimpsest/evaluation/source_panel_crossfit.py True
src/palimpsest/evaluation/panel_readout_campaign.py True
tools/audit_typed_source_panel_cv.py True
src/palimpsest/detection/algorithms/source_partition.py True
```

最小修复是在本轮 pins 中加入该运行依赖，并使 controls/pilot 按修复后源码重新记录。此意见只针对将要产生的第71轮记录；旧收据的 pins 保持其执行时含义。严重度为 MEDIUM，因为记录传递逻辑的变更不能触发当前源码一致性拒绝。

## 审稿人异议表

| 受众与纳入依据 | 最强常规异议 | 轴 | 回答位置或缺口 | 严重度 | 本审查处理 |
|---|---|---|---|---|---|
| 统计学习：借用了 sample splitting | 一旦重复调用，表示拟合是否仍严格使用规定的独立来源半集？ | correctness | split fitter 31–39存在M1缓存反例；README开头与stage段宣称分拆 | MEDIUM | 给出人工反例，要求缓存身份修复 |
| 统计学习：借用了 sample splitting | 拆半是已有方法，所谓进步是理论、估计效率还是本任务上的经验结果？ | novelty | README首段明确 conventional heuristic，当前无性能结果；拆半减少样本与两部分模型同时变化也明示 | LOW | 保持此限定，不能据新轮变化唯一归因为过拟合 |
| 受约束学习/Cotter：引用其权威 | 两个独立来源集是否真的满足两玩家、凸性和约束泛化假设？ | correctness | README首段和引用段明确无players/硬约束与保证，有限罚项和驻点也明示 | LOW | 没有新增理论声明；未代替原文/证明阅读审计 |
| 受约束学习/Cotter | 相较既有sample split和实用代理约束方法，新意具体在哪里？ | novelty | README引用段写明文献阅读未完成、未完成novelty review | LOW | 可开展重复开发诊断，尚不能提出完成的新意结论 |
| 数值优化：使用stationary solver | 3402/378是head求解器还是完整训练？驻点是否被当成全局最优或硬BA保证？ | correctness | fitter的fit_records_scope；README预算/soft-hard段；auditor显式source-split分支 | LOW | 单位与对象一致；没有优化器重解认证 |
| 数值优化 | 新损失或求解法的提升是否只是半样本与表示变更？ | novelty | README说明same rate loss、single-start，且zero不同于旧full-source zero | LOW | 没有新优化器或无条件提升声明 |
| 图像取证/RR、Chimera数据社区：沿用开发面板 | 新轮能否说明未见处理、新来源或新设备的泛化？ | correctness | README开头已明示repeatedly exposed，query Q70不在fit/cal支持，无独立验证 | LOW | 不把人工软件控制记为图像性能证据 |
| 图像取证/RR、Chimera数据社区 | 与当前 incumbent 检测器的进步在哪里，是否只在自家多轮开发上选方案？ | novelty | README无此比较或领先声明，实验目标是source-disjoint阶段诊断 | LOW | 未完成外部比较，不能主张领域性能进步 |
| 软件复现/独立审计：使用pins、缓存与auditor | 如何知道执行的记录边界与保存成员确实对应所写源码和实际优化器？ | correctness | M2缺pin；auditor输出scope明示仅保存记录/算术，README也限定成员记录语义 | MEDIUM | 要求补执行依赖pin；成员名单不升级为实际优化器认证 |
| 软件复现/独立审计 | 相比已有saved-role审计，新审计增加什么可核对象，能否推广为训练过程认证？ | novelty | 新增SHA阶段成员重建和3402/378预算；auditor scope明确仍只核保存记录 | LOW | 增量限于阶段成员记录；下述同源真值问题另记为控制缺口 |
| 实务使用者：可能采用输出分类器 | null真标签最终标定和allREAL退化是否可能被隐藏成成功？ | correctness | README结果段明确必须作为失败显示；final沿用真标签views，driver逐candidate保存成绩 | LOW | 真实运行尚不存在，要求结果时显示这一已声明信息 |
| 实务使用者 | 为什么采用新zero，而非旧zero或已有部署方案？ | novelty | README直接标明新sample-split baseline，与旧full-source零不同 | LOW | 仅可做该轮内部参数基线；未声称部署优势 |
| 通用科学读者：先读标题与开头 | 来源分开是否意味着所有统计信息都独立，或实验已成功？ | correctness | README标题写before real fitting；开头写no new performance/independent validation | LOW | 开头限定足够；本报告也单列未运行状态 |
| 通用科学读者 | 第71轮是否已构成一项新发现？ | novelty | README首段写test heuristic，作者状态仅4个人工测试 | LOW | 不将代码与人工测试解释成科学结果 |

M1、M2是启动前的软件修复意见。审稿表中的独立人工真值缺口为控制证据的范围限制：现测试确认两个实现对同一producer预期的一致性，尚未提供独立已知成员的植入真值。补充控制可在固定的纯人工面板上完成，无需任何新图像或真实模型运行。

## 已核的语义

SHA排序使用 compact UTF8 JSON `[20261008,domain,src]`，每个stratum取前floor，排序后的奇数strata交替floor/ceil，偶数总来源下总半数相等。分层依据传入y，未读取metadata truth label；同来源的scene和 supplied label 冲突会拒绝。wrapper先执行完整fit-only processing_pairs验证，每来源全部9视图一起进入一个阶段。basis方向与bank moments接收basis切片，最后head接收other切片；M1命中旧缓存是已发现的例外。

null的OOF保存行把 supplied pseudo label 写为 `label`，所以新auditor按这些保存标签重建分拆，未用真实metadata标签替换。training_rate_panel对每个处理组按各类总权重分别归一化为1/2；即使两半各类来源数量不同，也不把BA代理变成总体准确率。既有独立rate测试覆盖权重失衡；本轮新增fixture的实际head仍为各类1来源，没有单独覆盖odd strata分拆后的不等类别head。

显式 `--source-split` 令auditor检查 solver3402行/378来源、basis3402行/378来源、完整context6804行/756来源；默认仍检查旧solver6804行/756来源，没有因新开关放宽旧默认。auditor按保存members、OOF身份与fold角色重建集合，但不重放optimizer。外层630来源/5670行每阶段是README预算，本auditor主要审核内层CV，没有以这个脚本声称认证所有final阶段成员。

shape方向只读取legacy前缀之后的特征，原quantile bank的legacy covariance-aware子模块仍只读取旧前缀。新wrapper继承数值数组键但未处理分拆metadata，构成M1。wrong-source数组不进入目标函数，故只能作结构identity控制；保存的QuantileScoreRule kind继续描述数值布局。zero与旧full-source zero没有相等承诺，soft penalty零与hard BA/drop不等价的继承控制也保留。

## 冷读开头与待执行状态

README开头、标题、预算段和引用段足以让只读摘要位置的读者看到：来源分拆尚待真实拟合、面板已反复曝光、既无新性能结果也无独立验证。Cotter无保证的限定在开头出现，拆半损失样本预算与两部分模型同时变化也直写在zero段。这些边界不需要额外批准或弱化，结果发布时需要原样保留。

真实controls、pilot和full尚未获得本次审查的执行证据。M1/M2修复后按既定顺序运行软件controls、同源码单线程cold100/warm10 pilot及每call120秒预算检查，再决定是否满足已有条件；本审查不提供真实运行许可或成功结论。第71轮输出目录为 `work/robust_statistics/split_rate_score`，此ignored私有路径只以inline code记录。

## 被审版本

| 文件 | SHA256 |
|---|---|
| source_partition.py | eb7459f09f6792210132d6e2714d14d0385f6d89d7e060d3929be1e0e5b88306 |
| split_rate_score.py | c1e811e4458ac8cfdb78665d5bb2d8f37e61d71725f46927b740b6bef8cd8a20 |
| audit_typed_source_panel_cv.py | 5049a60184f06237c188a5ea8af9be45dd10bfed0ce9b7eda5344d81244fa14d |
| split_rate_score/run_iteration.py | 029f823cff1a300f4fe995f68ea1a77b412cc22ed6cfcdd796f0ce7a9feaa363 |
| split_rate_score/README.md | 9f3fee964c4b47e8d2ad13d66386cd0e82ffe12e6d23451bfdc018f6a0975d0d |
| test_source_partition.py | c507a42b47d9b77f51c259c6815344f6b7d8c82461279d89c13706e76f3067ed |
| test_split_readout_audit.py | 0e00a0b2891ff88951713190058524e919b8fa163561e31334dc04909049c689 |

按 `bootloops-research:prose-lint` 分别核对了本报告的状态与数字、措辞与段落开头。没有把人工stub记录说成实际优化器证据，没有复述尚未重新核验的Cotter证明细节。后续修复需要在本报告之后追加其具体检查结果，保留上述被审版本和原始反例。

## 修订复核与关闭记录

Lady 在同一独立审查上下文核对修订，保留上文原始发现与SHA。缓存键已分成 `training_arrays_sha256` 和 `stage_template_sha256`；后者包含完整数值数组摘要及逐行有序 `(domain,scene,src,condition,variant,basis_mask)`。来源身份或SHA阶段归属变化会产生新的stage template，数值相同的basis则允许child bank按其原有数值键去重。audit分别记录stage_templates和numeric banks，避免把成员身份变化误作数值bank数必须增加。

本次复核只重跑stub缓存边界、pin成员检查及saved-stage auditor人工测试，共5项通过，用时1.60秒。随后用.NET SHA256对六个literal compact JSON分别求摘要，独立确认秩 `0<1<2` 和 `4<5<3`。新版完整controls收据记录16项通过，用时1.98秒，其code_pins与当前第71轮 `code_pins()` 完全相同。审查员没有重跑完整16项，也没有执行或审查真实pilot/full；其中数值child-bank去重的2个stage templates/1个bank断言由新版完整controls覆盖。

| 原问题 | 修订证据 | 复核结论 |
|---|---|---|
| M1 缓存缺少分拆身份 | stage_template_sha包含有序记录与mask；相同数值、改src的stub测试确认basis变为1/3、head为0/2，numeric key相同而stage key不同；child-bank与stage-template分账 | 已关闭；原反例不再复用错误阶段模板 |
| M2 record-aware边界未pin | 本轮运行入口显式加入training_fit.py；targeted pin测试通过，完整controls pins与当前一致 | 已关闭；执行边界源码已纳入本轮追溯 |
| 同源expected控制缺口 | auditor测试不再调用producer分拆器；硬编码4来源basis0/2、6来源odd strata basis0/4/5，supplied labels反转而truth_label保持则basis0/1/4；.NET独立秩重算吻合 | 已关闭到人工已知成员控制范围；仍仅核保存成员语义 |

.NET实际摘要如下，输入依次为 `[20261008,"d","0"]` 至 `[20261008,"d","5"]`：

```text
0 5b9d434a492fb49dc26856182022610dfa072db561c20a85c4dec34955c33cac
1 6f52761f2d87971de8ba02a1d8878ce7e5430afbe69046ad3c15d3ce22147bf9
2 d448c38eaa9e16fcbd3a800ec4cf8e57a758472e90c991d25888e878da7cff50
3 fa7abd788915a3e9a4a55c019c6ca98cc1096fa1181af2640ed01e9ac4b36c13
4 e3eb2258ed9b9c01627931f1b0dad0ab060f575b1a96808b1b7147ec536880a5
5 ef6c2c9b5ad7eb5461ec056c0c52dd1d26494b3f493743e9a6f96d015c6752a1
```

新版software controls收据为 `work/robust_statistics/split_rate_score/software_controls.json`，SHA256 `6d3ee1189313af4096967ce1449d3d3fa47a744f0a239b2025b2738f8d1f2e13`。作者另报原13项controls已保存于 `work/robust_statistics/split_rate_score_precheck_v1`、原源版本Git `1c21889`；本次没有重新检查旧私有收据，故其状态仍保留作者提供的来源说明。

| 复核文件 | 修订SHA256 |
|---|---|
| split_rate_score.py | cff3669aa3289da9107b3319f08092ea7a6286db4901157f550153b58faeb1e4 |
| split_rate_score/run_iteration.py | 582f18e19f786f61e3f71e93abe08ba746f175c8a3c7ca46e0ac97ee4df8f897 |
| test_source_partition.py | f2073df7c2a279c7693b21d109f9e964aafba9b2d5c5eb6dbb229695f6501bd7 |
| test_split_readout_audit.py | 1736b33293053f9cf70aac19a1aed932ee5d28a3e29a2ece96ac63342652ba6c |

修订复核未发现新的HIGH或MEDIUM软件问题需要阻止full。成本pilot、完整真实运行、保存算术与结果失败解释仍按既定要求分别提供证据。来源分拆的数值cache正确性和保存成员检查不证明来源独立性，不提供Cotter泛化保证，也不把本轮实验升级为性能结果。新增关闭文字已再次核对数字、状态来源和措辞；此追加没有修改producer、pins或科学收据。
