# 组合推理

`ImageOriginPipeline` 直接调用定位接口与来源接口：文件解码 → 定位 → 原像素包围框裁切 → 来源推理。

Python 输入是 RGB；`predict_file` 包含解码时间；`predict` 的 decode 为零。来源 baseline 使用未透视校正的包围框，可能含背景；掩膜保留供定位评测使用。没有自动背景填充或多区域投票。

`locator=None` 明确表示整图模式。定位没有候选时不执行来源模型。CLI 为 `palimpsest-detect`；未来视频/手机实时链路在此扩展，目前没有演示应用。
