# 打印数值积分 / synthetic

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

虚拟输入上的结构、收敛或速度诊断；没有真实设备标定，不能作为实拍保真度证据。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先审阅虚拟参数和积分/采样约定，再运行对应数值对照；收敛和速度是独立验证问题。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [audit_print_scan_convergence.py](audit_print_scan_convergence.py) | 核查结构、配对或假设 | Check the prototype's internal fine-grid convergence on a virtual patch. |
