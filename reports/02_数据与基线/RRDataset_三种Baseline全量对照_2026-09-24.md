# RRDataset 原始视觉来源判别：三种 baseline 全量对照

日期：2026-09-24。目标是对单张图像判断其**原始视觉内容**来自自然摄影（真实类）还是 AI 生成（AI 类），即使图像后来经过平台传播或再数字化。这是现有方法的基线评测，**尚不是自研模拟算法、快速算法或新模型的结果**。

## 数据与共同口径

使用 RRDataset 官方测试包。50,999 张文件经过档案长度、MD5、逐图解码与哈希审计。发现测试原图有 14 个来源组与作者 train/val 原图逐字节重叠，因此下表在所有条件同步排除这些来源组：**原图 16,986、平台传播 16,986、再数字化 16,985，共 50,957 张**。再数字化的真实类原包少一张。三种方法的逐图输出已经按文件名、来源 ID 和标签核对，覆盖完全一致，分数和耗时均为有限值。

表中 **BA** 为固定判别阈值下真实类正确率与 AI 类正确率的平均；**AUC** 衡量分数排序，和固定阈值的 BA 是不同问题；**P50/P95** 为单张图像包含文件读取、解码、预处理及判别的端到端耗时中位数与第 95 百分位数。正分判 AI，各方法未在 RR 测试集上调整阈值。百分点跌幅在同一来源的原图与处理图上配对计算；再数字化只配对实际存在的 16,985 个来源。

## 全量结果

| 方法 | 原图 BA / AUC | 平台传播 BA / AUC | 再数字化 BA / AUC | 原图→传播 BA 跌幅 | 原图→再数字化 BA 跌幅 | 全部图端到端 P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| B-Free，官方权重，GPU | **81.60% / 87.56%** | **73.84% / 74.34%** | 61.53% / 61.06% | 7.77 点 | 20.07 点 | 323 / 403 ms |
| D3，官方权重，GPU | 79.17% / 87.18% | 73.27% / **82.14%** | **62.04% / 64.20%** | 5.90 点 | 17.13 点 | 104 / 181 ms |
| Benford-RF，传统算法适配版，CPU | 68.47% / 74.28% | 65.26% / 70.78% | 56.26% / 58.43% | 3.21 点 | 12.21 点 | 36 / 122 ms |

D3 的原图 BA 比 B-Free 低 **2.43 点**。传播图像上两者 BA 仅差 **0.57 点**；按同一批来源分真假类别配对 bootstrap 2,000 次，D3 减 B-Free 的 95% 区间为 **−1.44 至 +0.30 点**。再数字化 BA 差 **+0.51 点**，区间 **−0.18 至 +1.20 点**。因此这两种处理条件下，当前数据不足以支持 D3 固定阈值 BA 明确高于 B-Free 的结论。D3 的传播后 AUC 则高 **7.80 点**；这表示它的排序能力更强，但固定零阈值没有转化为相应的 BA 优势。

再数字化后，三种方法的 AI 类正确率分别为 **27.35%（B-Free）、39.24%（D3）、35.54%（Benford-RF）**，真实类正确率分别为 **95.71%、84.84%、76.98%**。这解释了为什么 D3 在 AI 类上明显优于 B-Free，合并 BA 却仅高 0.51 点。三者都距离“真实传播后精度下跌极小”的目标很远。Benford-RF 虽然绝对跌幅较小，但各条件的绝对 BA 更低，不能据跌幅认定其整体更稳健。

D3 与 B-Free 都在本机 RTX 4060 Laptop GPU 上以单图 batch=1 计时；D3 的整体端到端 P50 约为 B-Free 的 **1/3.11**。这只是当前模型、输入协议、Python 实现、硬件和磁盘下的测量，**不是手机实时速度**。Benford-RF 在 CPU 上运行，与两款 GPU 模型的毫秒数不是同硬件速度排名。D3 各条件端到端 P50/P95 为原图 **111/198 ms**、传播 **97/167 ms**、再数字化 **104/174 ms**。

## 方法边界与研究含义

- **B-Free**：直接评测官方权重及官方五块输入，单图 score > 0 判 AI。超大图采用已对照验证的先裁五块执行路径。该基线在再数字化时 AI 类分数明显朝真实侧移动。
- **D3**：使用官方 CLIP ViT-L/14 与官方分类头，单图随机 patch 置乱，固定全局种子 418，raw logit > 0 判 AI。这里的 **batch=1 随机置乱**与作者验证脚本的 batch=128 分数协议不同，因此是官方权重的本机单图评测，不能写成作者原论文指标的逐字复现。其预训练数据与 RR 图像是否重叠尚未知。
- **Benford-RF**：受 Bonettini 等关于 DCT 系数首位数字的论文启发，使用固定量化与理论 Benford 偏离特征、100 棵随机森林，在 RR train 原图 2,500 张上训练；**不是原作者代码或论文方法的忠实复现**。RR 原图中 AI 多为 PNG、真实多为 JPEG，编码历史可能成为它的捷径。格式规则“PNG 判 AI、JPEG 判真实”在原图的 BA 为 94.93%，处理条件均为 50.00%，证明该数据偏差确实存在。

目前只排除了已知的**逐字节** train/val 重叠。测试包内部还有 39 组跨路径逐字节重复，跨划分重编码、裁剪及语义近重复尚未审计；置信区间可能偏窄，不能称为彻底无泄漏测试。作者提供的四种再数字化方式没有逐图方法标签，本表只能报告合并条件，**不能单独声称拍屏准确率**。三个现成方法的表现为下一步研发提供参照：先补足真实传播过程标签和更严格的内容去重，再设计更准确的模拟及快速判别方法，并在统一的单图端到端协议下验证精度跌幅与速度。

## 复算记录

- 逐图结果：`work/d3_rr_full.csv`、`work/rr_bfree_complete.csv`、`work/benford_rr_full.csv`；机器汇总：`work/d3_rr_full.json`、`work/rr_bfree_evaluation.json`、`work/benford_rr_full.json`。
- 同源逐图对齐、独立测试清单覆盖审计、耗时分位数和方法间配对 BA 差脚本：`experiments/origin_detection/baselines/evaluate_rr_full_baselines.py`；输出：`work/rr_full_baseline_comparison.json`。调用：`E:\ai_image_origin_research\envs\bfree\Scripts\python.exe -m experiments.origin_detection.baselines.evaluate_rr_full_baselines`。配对区间分别对真假类别来源独立有放回抽样 2,000 次、固定随机种子。官方测试清单 SHA-256 为 `1b5ca7632034680bbffcb8f74524ee4225ab5bb11aeead79f7fce244928501e5`。
- CSV SHA-256：D3 `706fe0ee5c8a0af1cc89643ec1d1901fe5070e2b85f41703415cc613f3520455`；B-Free `0e4b82ea9a3134ea464cc89575303a76de4d9420056f34bb2601731351570a9b`；Benford-RF `adebcfed28465c013358a3c75331443dd29201806ec242b6bab1c381707e535d`。
- D3 入口 `experiments/origin_detection/baselines/run_d3_rr.py --output-prefix work/d3_rr_full`，脚本 SHA-256 `979019dd22a12594ea969c8ab391e77cf0c564ca239592d7d75742319a712b08`；官方代码提交 `14f21ad3797ef1e42f2f6090aa8ad4fabf07896c`，CLIP 权重 SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`，分类头 SHA-256 `c0bb66e0de538b0330613da1865da025756c1dc7a7290990d5343c35290ab63b`。日志 `work/d3_rr_full.stdout.log` 和 `work/d3_rr_full.stderr.log`，后者只有超大图的 Pillow 解压尺寸警告，实际 50,957 张成功完成。
- 详细方法及数据审计分别见本目录《RRDataset_B-Free真实传播Baseline》《RRDataset_Benford传统算法全量Baseline》及《数据集整理与Baseline准入审计》。原始数据清单位于 `E:\ai_image_origin_research\data\manifests\rr_test_files.csv` 和 `rr_trainval_files.csv`。

文献与代码：[RRDataset](https://arxiv.org/html/2509.09172)、[B-Free](https://github.com/grip-unina/B-Free)、[D3](https://github.com/BigAandSmallq/D3)、[Bonettini 等 Benford 检测](https://arxiv.org/abs/2004.07682)。
