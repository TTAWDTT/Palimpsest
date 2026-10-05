# 固定 RR 规则的真实拍屏诊断

[算法流程](../../../../docs/robust_ai_detection_plan.md) · [规则与选择协议](../rr/README.md)

2026-10-05：RR筛选结束后，冻结其选中参数，不用Chimera标定或选择本轮规则。完整1,200来源×原图/MacBook→iPhone/LG→Blackfly用于检查这条颜色线索的真实拍屏边界。项目此前研究过Chimera，因此称外部数据集诊断，不能称全新独立盲测。StyleGAN2、三个主题与两个设备组合限制结论范围。

`run_algorithm.py --limit 60`只做同配置计时pilot，完整运行不加limit。先核验发布图审计与清单、参数指纹；每图核实SHA后调用共用文件推理。完整结果复用统一分类/同源评价，与缓存B-Free同清单比较。单一阈值、分数方向保持RR冻结值，不根据拍屏条件修正。运行进度是已完成CSV行数。

```powershell
uv run --locked python -m experiments.origin_detection.robust_statistics.chimera.run_algorithm --limit 60
uv run --locked python -m experiments.origin_detection.robust_statistics.chimera.run_algorithm
```

结果位于`work/robust_statistics/chimera_first_iteration/`。RR的reserved仍不读取；本诊断无论成功失败都不反向修改该冻结规则。

## 2026-10-05：第二轮规则复用此入口

增加显式`--projection`选项，只有第二轮RR注册门槛通过且代码/参数指纹一致时运行。结果写入`work/robust_statistics/chimera_paired_projection/`，旧颜色结果保持原位置且不覆盖。两种规则复用同一解码、逐图SHA、覆盖审计与缓存baseline评测，不复制另一个推理脚本。

```powershell
uv run --locked python -m experiments.origin_detection.robust_statistics.chimera.run_algorithm --projection --limit 60
uv run --locked python -m experiments.origin_detection.robust_statistics.chimera.run_algorithm --projection
```

新增统计权重只来自RR fit，阈值只来自RR threshold；Chimera旧结果已暴露，只作固定参数诊断。pilot与完整结果分别保存，避免把计时pilot当完整准确率。
