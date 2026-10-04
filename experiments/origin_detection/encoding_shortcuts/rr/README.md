# 编码捷径与对照 / rr

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

RR 发布包内部的历史统计试验；缺逐图物理方式标签，内部留出不是独立实拍验证。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成数据准备与已知精确重叠排除，核对冻结来源角色；拟合仅用校准部分，物化与评价使用同源对照。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [audit_transformation_provenance.py](audit_transformation_provenance.py) | 核查结构、配对或假设 | Audit observable encoding provenance of RR transformed test images. |
| [run_format_shortcut.py](run_format_shortcut.py) | 执行明确的实验或对照 | Quantify the RRDataset train/val label shortcut from decoded file format. |
| [run_oracle_codec_control.py](run_oracle_codec_control.py) | 执行明确的实验或对照 | Test how much of RR processing one resize/JPEG pass can explain. |
| [evaluate_format_control.py](evaluate_format_control.py) | 评测、比较或汇总 | Apply the train/val format-only PNG→AI, JPEG→real rule to RR test. |
| [evaluate_metadata.py](evaluate_metadata.py) | 评测、比较或汇总 | Measure how much ReWIND labels leak through format and image dimensions. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `MANIFEST = DATA_ROOT / 'manifests/rr_test_files.csv'`
- `MANIFEST = DATA_ROOT / 'manifests/rr_trainval_files.csv'`
- `ROOT = DATA_ROOT / 'derived/rr_test'`
