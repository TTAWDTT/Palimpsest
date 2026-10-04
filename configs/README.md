# 实验配置

[仓库首页](../README.md) · [代码结构](../docs/architecture.md)

- [workspace.example.toml](workspace.example.toml)：本机数据、权重、环境、缓存和结果目录的配置模板。
- [screen/virtual_screen.example.json](screen/virtual_screen.example.json)：未标定的虚拟拍屏参数示例；不是已校准设备。

默认数据和权重位于仓库同级的 `data/`、`models/`；结果默认仍在仓库的 `work/`。配置读取不创建目录，也不启动任何实验。

需要修改本机路径时，把模板复制为 `workspace.local.toml`。该文件不提交。配置路径相对于仓库根目录解析，不受命令执行目录影响。

优先级：环境变量 → 本机 TOML → 默认路径。常用覆盖：

```powershell
$env:PALIMPSEST_DATA_ROOT = 'F:\research-data'
$env:PALIMPSEST_WORK_ROOT = 'F:\research-results'
```

也可以用 `PALIMPSEST_CONFIG` 指向另一个配置文件。显式配置不存在、配置键拼错时会报错；不会悄悄回退到默认路径。

历史冻结清单中的路径和指纹保留原值。更改路径配置不意味着旧清单已经迁移，也不能绕过历史实验的完整性检查。
