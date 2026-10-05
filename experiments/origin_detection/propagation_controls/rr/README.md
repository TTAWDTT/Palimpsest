# RR 同源传播对照

[上级入口](../README.md) · [完整协议与验收](../../../../docs/propagation_evaluation.md)

仅使用冻结的1,000开发来源（AI/自然摄影各500），不读取 reserved 图像。新实验固定参数、保留历史图像和分数，不做判别算法研发或模型训练。

| 入口 | 作用 |
|---|---|
| [protocol.py](protocol.py) | 冻结输入、来源角色、两种数字处理和预期图像清单 |
| [prepare_controls.py](prepare_controls.py) | 校验原图/传播图/当前模拟的SHA；物化对照、索引及五特征诊断 |
| [run_controls.py](run_controls.py) | 仅对新对照做官方 B-Free 推理；前台顺序运行子进程，保存配置/代码/权重/分数指纹 |
| [统一汇总入口](../../baselines/evaluate_rr_suite.py) | 原 CSV 直接进入共享分类/配对/计时与模拟响应评价 |

按顺序显式执行：

```powershell
uv run --locked python -m experiments.origin_detection.propagation_controls.rr.prepare_controls
# 需要已有 B-Free 独立环境、官方代码/权重和 CUDA：
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& ../envs/bfree/Scripts/python.exe -m experiments.origin_detection.propagation_controls.rr.run_controls
uv run --locked python -m experiments.origin_detection.baselines.evaluate_rr_suite --scope all
```

输入/输出路径来自 `palimpsest.paths`。官方代码在 `work/vendor/bfree/code`、权重在同级 `models/bfree`；数据在同级 `data/derived/rr_propagation_controls`。新索引、manifest、分数、summary、运行签名和诊断置于 `work/rr_propagation_controls`；默认统一汇总为 `work/rr_evaluation_suite.json/.md`。

物化仅接受新文件或已有字节一致文件；推理跳过输入/判别实现签名与输出SHA一致的完成项。明确记录为 failed 的运行可加 `--resume`：核对原 CSV/summary SHA、判别代码/权重/vendor/清单、图像SHA和成功前缀，记录旧运行签名后才恢复。running 或无运行记录的残留输出不能自动恢复。

复用 `platform_statistics/rr/protocol.py` 的已有 JPEG 读取与渲染以保持配对消融的编码一致；通用文件读取和指标来自公共库。该依赖不是旧路径兼容入口。

本结果只约束数字处理与一个固定检测器的反应，不能证明物理过程正确、最终训练有效或业界领先。
