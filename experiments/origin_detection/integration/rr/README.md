# 接口与真实权重接入核对

`run.py` 比较 B-Free 3 张、D3 顺序前缀 2 张历史分数，检查四边形/SAM 2→包围框→B-Free 的真实调用；验证 D3 的随机序列不受无关 Torch 抽样干扰。

固定输入来自已有 RR 结果；人为场景仅增加一个边界。它不是新准确率、真实拍屏或定位质量评测，不用分数调参，不执行全量推理或拟合。数据/权重不存在时明确失败。

运行：`python -m experiments.origin_detection.integration.rr.run --output work/new_integration_check.json`，使用已安装相应依赖的推理环境。新结果保留数值、环境和本地权重指纹，拒绝覆盖历史产物。
