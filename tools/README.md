# 研究辅助工具

工具核计算／记录，不认证数据独立性或真实物理过程。历史执行用对应Git版本及收据，不能用当前源码替换旧pins。

## 来源面板审计

`audit_typed_source_panel_cv.py`核已保存OOF的整数BA、配对跌幅、fit-CV选值、录下的来源／标定折、维度与有限梯度。适用于1260来源、三条件、raw/Q90/Q60的45指标／75配对面板；不拟合模型或独立重解优化器。方向／项数必须显式给出：当前形状读出是`--mapped-dimensions 20 --basis-count 5`。旧记录缺basis count时只能显式`--allow-missing-basis-count`，收据报告缺失数量，不回推该字段。

输出`calibrated_oof_audit.json`的`passed`仅表示保存记录核对通过；method tag缺失、放宽的basis字段均单列。**异常或非零退出不是审计通过；已有输出拒绝覆盖。** 相信输出前需已知完美面板／错误分数、标定折负控通过，并核实际生产源pins。已核当前20项5方向记录及历史第65轮14项4方向（后者40个basis字段缺失显式记账）。

邻近工具：`audit_source_panel_cv.py`核旧LP面板与state；`audit_source_crossfit.py`／`audit_calibrated_source_cv.py`提供整数及来源／标定折独立核对。14项专用旧auditor属于历史版本，当前应使用显式schema，避免把20项误作14项。

## 其他入口

| 工具 | 用途与边界 |
|---|---|
| `check_layout.py` | 文件角色与Markdown本地链接，非科学结论检查 |
| `audit_query_shape_rankings.py` | 已保存fit-only排序／tie及fold计数，含各折AUC；非单一部署模型成绩 |
| `diagnose_saved_thresholds.py` | 已曝光固定分数的共同阈值可行性与最坏配对翻转；不采用oracle阈值 |
| `audit_calibrated_margin_cv.py` | 旧间隔／来源校准面板的保存记录，非优化器独立重放 |
| `benchmark_strong_view_stream.py` | 实际旧工程查询提取的预算测量，非分类性能 |
| `ResearchPaths.ps1` | 配置解析与任务路径 |

图像特征提取／分类执行入口见[实验导航](../experiments/origin_detection/README.md)，字段含义与结果见[研究导航](../reports/README.md)。

## 第69轮后的固定分数诊断登记

扫描前限定第68轮和第69轮`selected`，复用不改的整数 tied-event sweep；第67轮selected与69只差共同平移，不重复扫描。父收据SHA验证保存分数字节，60组／130配对和零阈值计数先审；旧producer pins只作为历史记录复制，不把当前源码冒充旧执行。穷举所有不同预测状态，报告80%条件下最小最大下降、2pp条件下最大最低BA及当前阈值所有最坏并列的逐来源正确→错误／错误→正确；同分判REAL。人工控制为原Fraction brute sweep／80%与2pp边界／坏覆盖／NaN及新的REAL等号翻转测试，先于真实扫描运行。

诊断仅排除这两个固定分数在已曝光面板上的共同阈值族；不采用扫描阈值，不重新拟合、bootstrap或读图。从仓库根目录按模块启动：`.venv/Scripts/python.exe -m tools.diagnose_saved_thresholds --directory work/robust_statistics/paired_ba_calibration --output work/robust_statistics/paired_ba_calibration/posthoc_threshold_diagnostic.json`；有向轮只更换目录。直接按文件启动会缺仓库模块路径，首个错误调用未产生收据；已修正调用方式，不添加路径兼容层。已有输出拒绝覆盖，收据保存实际计时与本次诊断源pins。文件复用相邻实验的已核算术，未新建第二份扫描实现。
