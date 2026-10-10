# CuRe方法归属迁移 — 2026-10-11

## 分支关系

本次迁移直接提交到 `ttawdtt/robust-statistics`，归入已有PR #7，base为 `main`。工作前fetch核对：`origin/main...HEAD`为0／90，研究分支包含当前main，不另建重复或依赖未合并研究代码的新PR。本次没有合并PR或改写分支历史。后续非神经研发应在本PR合并后从更新的main另开专用分支，接续第15轮并继续递增迭代编号。

## 归属与范围

- CuRe三／四／五方向与相关读出变体迁入 [models/frozen_features/cure](../../src/palimpsest/detection/models/frozen_features/cure/README.md)。完整方法含神经编码器。
- 第67—72轮的六组实验迁入 [experiments/origin_detection/frozen_features/cure](../../experiments/origin_detection/frozen_features/cure/README.md)。历史更早冻结表示实验仍按原问题组织，本次不声称已重整全部实验。
- 通用风险、间隔、阈值标定与规则载体进入 [algorithms/readouts](../../src/palimpsest/detection/algorithms/readouts/README.md)，保留纯像素算法的复用能力。
- `StableDetector`保留在像素算法中，`StableRule`与配对ridge数值拟合抽出；不是旧路径转发。
- 新增文件检测器显式组合冻结特征与保存规则。保持CuRe文件后缀预处理，未接区域RGB／CLI。

[机器迁移记录](cure_method_migration.json)列35个直接移动文件及共用规则抽取。更新全部现有调用、代码pin路径和文档链接；旧入口不保留兼容模块。没有新拟合、编码器训练、图像推理或阈值选择。

## 检查

本机完整 `pytest -q`：412通过；ruff与布局检查通过，新实验 `--help` 可导入运行。9项核心规则与文件接口检查覆盖完整schema、保存规则、阈值等号、文件后缀、单次提取和非法描述符拒绝。

`tools/check_cure_migration.py`与原执行提交比较：15个迁移数值模块除导入外AST相同；抽出的3个定义AST相同；旧转发文件0。使用已签名3920维开发缓存重放第69轮5040查询：最大分数差0、改变判决0，规则JSON往返精确。它是软件迁移核对，不是新的分类研究或独立验证。

```powershell
.venv/Scripts/python.exe tools/check_cure_migration.py
.venv/Scripts/python.exe tools/check_cure_migration.py --cache work/robust_statistics/cure_token_quantiles_full --rules work/robust_statistics/paired_ba_calibration
```

完整源码AST检查需Git含原执行提交。缓存重放需本机私有规则与缓存，公开checkout只提供脚本；不链接或公开私有结果。

保存分数SHA256：`81bbed3304449f33dfa559faa5139c0cfa401a908126822311ae42df094c77fd`。第69轮规则文件SHA256：`675de87bca95ba909e33e2296fad9a733d3d69ffecf08f97b6b04344e63429f6`。

## 历史边界

历史数据、参数与收据保持不变，代码pin对应 `d279b7e965e2ac8ad2f63b4110cd46cace55addb`。新源码可能触发旧收据的来源拒绝；不修改旧SHA来让它通过。当前候选仍是最低域汇总BA85%／最大下降2.5个百分点，尚未满足2个百分点及独立真实验证门槛。
