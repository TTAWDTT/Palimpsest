# 实验命名与依赖约定

[实验工作流](../experiments/README.md) · [公共模块](architecture.md) · [维护清单](maintenance/experiment_refactor_checklist.md)

## 目录和文件名

```text
experiments/<研究领域>/<研究问题>/<数据集或 synthetic>/
  README.md                 问题、状态、前置条件、入口和依赖
  protocol.py               这个问题共享的准备、设置及计算
  prepare_<对象>.py         获取输入、冻结划分或物化对照
  audit_<对象>.py           完整性、配对、数值或假设核查
  fit_<模型或响应>.py       按既定校准划分拟合
  run_<假设或对照>.py       实验执行
  evaluate_<对照>.py        指标、比较及汇总
  plot_<诊断>.py            图表
  benchmark_<方法>.py       速度测量
```

数据准备采用 `data_preparation/<数据集>/`；固定 baseline 汇集于 `origin_detection/baselines/`，文件名保留必要的数据集限定。只创建实际需要的角色文件，不为每个问题生成空模板。

同一个问题的变体用具体含义区分，例如 `evaluate_cross_spi_four_sources.py` 与 `evaluate_cross_spi_twelve_sources.py`。数量、裁切、插值、来源角色和相位假设属于协议，不能只改文件名后混为一项实验。已有入口尚未统一配置化，新增可配置流程应先明确参数与指纹契约。

不再新增泛称 `probe`、时间戳或含糊的 `v2_final` 文件。版本记录在配置、结果元数据和 Git 中；确有不同科学假设时，用假设名区分。

## 模块边界

- 多问题通用的读取、校验、数值算法和评价进入 `palimpsest.data`、`io`、`simulation`、`evaluation`。
- 某问题共享的来源准备与冻结设置进入其 `protocol.py`。入口负责参数、调用和输出，不承担其他问题的通用工具职责。
- 公共库不导入 `experiments`。跨问题依赖必须有具体理由并在问题 README 中列出；后续遇到真正通用能力，再提取到公共库。
- 模块导入不打开研究图像、下载数据、拟合参数、启动推理或写结果。研究步骤放入 `main()` 并由显式入口调用。

现存少数跨问题入口依赖仍在目录表中明确保留，并非本轮已建立完全独立的工作流调度器。不要仅为消除一条依赖而复制实现或更改协议。

## 结果与复算

输入和结果位置继续由 `palimpsest.paths` 控制。目录迁移不更改已签发清单、历史分数、阈值、种子或产物名称。代码提取会改变代码指纹，旧断点不能当作当前代码的运行结果，也不能通过手动修改指纹续跑。

源到 RAW 与频谱缓存现在包含提取后的协议和解码依赖；旧缓存保留原值。要精确复算已有结果，恢复报告记录的历史 Git 版本与原环境。

## 维护检查

```powershell
uv run --locked ruff check src experiments tests tools
uv run --locked python tools/check_layout.py
uv run --locked pytest -q
```

检查覆盖命名角色、内部导入、库依赖边界、显式顶层研究操作、错误的路径方法接收者、迁移目标和 Markdown 链接。AST 检查不能证明任意动态代码没有副作用；此前有顶层操作的模块另用无数据目录的独立进程验证。检查不下载数据或运行全量 baseline。
