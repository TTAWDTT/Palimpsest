# Simulation 第二阶段：平台裁切机制与独立扫描配对数据

2026-09-24。研究目标是模拟传播及再数字化对图像来源判别的影响。本报告记录一个**模拟质量内部试验**，尚不是新判别算法的性能结果。

## 1. 发现：RRDataset 的部分平台传播图经过中心裁切

对 RR 测试集中的低相关原图/传播图逐对检查：仅将整幅原图缩放到传播图尺寸时，四组例子的 128×128 灰度相关度为约 0.20–0.33；先按传播图宽高中心裁切原图，再比较，相关度均达到 0.998–1.000。说明这些困难样本的几何机制是裁切，而单次整图缩放模拟会制造错误的视野。这个结论来自可见配对与图像相关性诊断，尚不表示所有平台都遵循同一规则。

由此实现一个类别盲的非 Oracle 传播模拟器：从校准来源组提取输出尺寸比、JPEG 量化表及采样设置；比较每个校准对的整图缩放与中心裁切拟合度，以决定几何模式；给未见过的原图按相近尺寸随机选择一个校准 donor。**生成验证图像时没有读取验证图的处理参数，也没有用真实/AI 标签拟合。**

## 2. RR 内部留出验证

- 用固定 SHA-256 规则按**来源组**分割 RR 测试数据：校准 3,328 组，其余 13,658 组可作内部验证；排除此前发现的 14 个与 RR train/val 逐字节重叠来源组。
- 每种条件从校准组抽取 1,000 个 donor。验证从每类选 500 组，共 1,000 个来源；每组同时评价平台传播和再数字化。筛选要求原图及对应两种处理图每张不超过 800 万像素，故这一子集不代表所有大图。
- 基线为 donor 尺寸比 + 整图缩放 + 单次 JPEG；改进版仅对校准判断为中心裁切的 donor 改用中心裁切，其余协议相同。两个版本的 donor ID、验证来源 ID 完全一致。校准 donor 中 347/1,000 判为中心裁切。
- 对真实与模拟处理图，计算相对同源原图的 128×128 灰度相关度、平均 RGB 绝对变化、粗尺度梯度能量比、宽高比例。表中 W1 是两组等量样本的一维 Wasserstein 距离，越低表示该项**边际分布**越接近。

| 平台传播，1,000 对 | 仅缩放+JPEG | 裁切感知+JPEG |
|---|---:|---:|
| 灰度相关度 W1 | 0.13654 | **0.01087** |
| RGB 平均变化 W1 | 0.00608 | **0.00060** |
| 梯度能量比 W1 | 0.01456 | **0.00187** |
| 宽度比 W1 | 0.00352 | 0.00352 |
| 高度比 W1 | 0.00353 | 0.00353 |
| JPEG 亮度量化表指纹总变差 | 0.229 | 0.229 |

另用来源组隔离的 50/50 划分训练随机森林“两样本评论者”，让它区分真实处理图与模拟图。每一来源的真实/模拟样本固定落在同一 train 或 test；模型和特征字典仅在 train 拟合。下表 AUC 越接近 **0.5** 越难区分；95% 区间对 test 来源组配对重采样 1,000 次，固定已训练模型。

| 平台传播两样本区分 AUC，test 506 个来源 | 仅缩放+JPEG | 裁切感知+JPEG |
|---|---:|---:|
| 五项数值特征 | 0.799 [0.772, 0.823] | **0.537 [0.504, 0.569]** |
| 五项特征 + JPEG 表指纹 | 0.834 [0.810, 0.856] | **0.606 [0.572, 0.639]** |

**解释边界：**0.537 不证明模拟图在像素或检测器特征层面逼真；评论者仅看到五项低频/几何统计量。量化表指纹仍可区分真实与模拟，说明 donor 的 JPEG 分布及其与其他参数的联合关系还有偏差。整个试验从 RR *test* 数据内部拆分，用于发现机制，**不能作为 RR 测试集上无泄漏的新方法泛化证据**。正式模型/算法评估须在独立训练/校准集合设计模拟器，并封存测试集。

## 3. 再数字化分支仍未解决

同一 1,000 来源的 RR 再数字化图上，当前单次缩放+JPEG 模拟与真实图的五特征两样本 AUC 仍为 **1.000**。此前 300 来源试验的真实/模拟灰度相关度中位数约 0.788/1.000、RGB 平均变化约 0.070/0.0001。可见物理成像引入的视野、色彩、纹理等变化不能由 JPEG 表复刻。平台裁切模块并未用于再数字化分支。

## 4. 外部真实扫描配对数据：DESCAN-18K

作者发布的 [DESCAN-18K](https://github.com/jhcha08/Descanning) 提供 `clean/scan` 同名配对；论文说明先由数字杂志页打印并用真实设备扫描，然后对齐、裁成 1024×1024 补丁。它可用于**打印→扫描分支**的模拟校准与异设备检查，但内容是杂志页、没有 AI 来源标签，不能直接计算真实/AI 判别准确率；也不能代表拍屏幕、拍打印件或拍投影。作者说明页面来源涉及 CC BY-NC-SA 3.0，当前只在本机保留原档与派生统计量，不再分发图像。[数据说明](https://huggingface.co/datasets/ENCLab/DESCAN-18K)

- `Valid.zip`：754,770,440 字节，SHA-256 `f515fc2723db71e709588b2711577085318611684978fe4ad562042f452d6b2b`；360 对，scanner03/04 各 180。
- `Test.zip`：767,709,023 字节，SHA-256 `93c904410f409c1a393e5e79b35bf1748cb5d0ad2d18d1e1958d32213a9d8304`；360 对，scanner01/02 各 180。
- 两档案都经大小、SHA-256、ZIP CRC、路径和配对清单检查；720 对全部可解码，均为 1024×1024。同名 `scan/clean` 配对与档案 `metadata.jsonl` 全部吻合。
- Valid 与 Test 的 clean 图像字节 SHA 交集、解码 RGB 像素 SHA 交集均为 0；**近重复或同一杂志页的不同裁块尚未排除**。
- 补做 360×360 跨划分候选筛查：63 位感知哈希距离 ≤8 的图对为 0，最小距离 12；前几组感知哈希或缩小 RGB 余弦最相近候选经可视检查没有发现相同裁块。不过文本页面之间会产生感知特征假相似，这个筛查不能排除内容相同但裁块位置不同的来源重叠。机器记录为 `work/descan_near_duplicate_screen.json`。

在 128×128 成对统计中，四设备组的梯度能量比中位数：scanner03 **1.024**、scanner04 **0.777**、scanner01 **1.004**、scanner02 **1.060**；RGB 平均变化中位数分别为 **0.0228、0.0257、0.0154、0.0222**（像素值归一化到 0–1）。各设备组的描述统计明显不同，且 Test 设备不在 Valid 中；由于未确认相同 clean 内容跨设备精确配对，这些差别也可能含有页面内容与裁块组成差异，不能直接解释为设备因果效应。数据仍适合**尝试**跨设备迁移试验，但须同时控制视觉来源重叠与内容分布。

## 5. 与现有检测 baseline 的关系

对同一 1,000 个 RR 验证来源事后估计传播几何：中心裁切 353、其余 647。B-Free 固定零阈值的原图→传播图 BA 跌幅：裁切组 **9.23** 个百分点、其余组 **6.89**；D3 分别为 **5.16** 与 **6.26** 个百分点，方向并不一致。分组依据处理后图像，且内容、尺寸、压缩等因素可能同时变化；这些只是分层相关性，**不能推断裁切导致准确率下降**，也没有证明模拟器可提高判别模型性能。

## 6. 下一步的实验门槛

1. 对平台传播补齐 JPEG 指纹与参数联合分布、分平台多轮编码，增加频谱/块边界/特征层评论者；重新量化真实与模拟差距。
2. 对打印扫描使用 DESCAN Valid 校准，在未见设备的 Test 上报告同源及分布指标；补近重复审计。对拍屏幕、拍打印件、拍投影另找带物理方式标签的真实配对，不能混入扫描结果。
3. 将模拟器固定于真正的训练/校准集合后，再训练快速判别算法与模型；在封存的真实传播测试集比较 BA/AUC 的配对跌幅和含解码预处理端到端时延。当前 RR test 内部机制试验不得用于这一最终声明。

## 复算入口

工作目录为本任务根目录。解释器：`E:\ai_image_origin_research\envs\bfree\Scripts\python.exe`（模拟与配对审计）、`E:\ai_image_origin_research\envs\classical\Scripts\python.exe`（随机森林评论者）。

```powershell
& 'E:\ai_image_origin_research\envs\bfree\Scripts\python.exe' -m experiments.rr.rr_simulator_v0 --donors-per-condition 1000 --validation-per-class 500 --neighbors 30 --geometry resize --output work\rr_simulator_resize_1000_evaluation.json
& 'E:\ai_image_origin_research\envs\bfree\Scripts\python.exe' -m experiments.rr.rr_simulator_v0 --donors-per-condition 1000 --validation-per-class 500 --neighbors 30 --geometry crop-aware --output work\rr_simulator_crop_1000_evaluation.json
& 'E:\ai_image_origin_research\envs\classical\Scripts\python.exe' -m experiments.rr.evaluate_rr_simulator_critic --input work\rr_simulator_crop_1000_evaluation.json --output work\rr_simulator_crop_1000_critic.json
& 'E:\ai_image_origin_research\envs\bfree\Scripts\python.exe' -m experiments.print_scan.download_descan_calibration
& 'E:\ai_image_origin_research\envs\bfree\Scripts\python.exe' -m experiments.print_scan.audit_descan_pairs
& 'E:\ai_image_origin_research\envs\classical\Scripts\python.exe' -m experiments.print_scan.audit_descan_near_duplicates
& 'E:\ai_image_origin_research\envs\bfree\Scripts\python.exe' -m experiments.rr.analyze_rr_transfer_geometry_effect
```

主要机器可读产物：`work/rr_simulator_resize_1000_evaluation.json`、`work/rr_simulator_crop_1000_evaluation.json`、对应 `*_critic.json`、`work/descan_archive_audit.json`、`work/descan_pair_audit.json`、`work/rr_transfer_geometry_effect.json`。模拟器与评论者脚本 SHA-256 分别为 `2015a4bfbf9fbf7c59f09d5e1a03cb4f649d1e9d858e7c63ee648d5c2d83f62c`、`b7ca390785d8fdaf9e00c6fa1e96574068c2ca25fa1faa34e5b793a3404ccd33`。

## 2026-09-24 来源口径补充

本报告中 RR 的“真实传播图/真实再数字化图”应读作**作者发布的处理图**；论文虽描述实际平台发送及四种物理方式，但公开数据没有逐图操作记录。仓库存在纯数字 PDF 渲染流水线，且发布再数字化图不能全部归因于该代码未经后处理的直出。由此，前文的两样本 AUC 衡量的是**模拟图与作者发布图**在所列特征上的接近程度，并非已经认证的真实设备采集模拟精度。具体证据与调整后的评测协议见《RRDataset处理来源核查与Simulation评测协议》。
