# 本机资料区

这里的数据与当前桌面绑定，不能充当其他人的配置。

- `plane_calibrations/`：原标定备份、候选、拟合和独立验证。
- `height_diagnostics/`：高度、时间波动、覆盖率诊断。
- `height_backgrounds/`：明确空桌采集的局部高度参考。
- `outputs/`：v05 的 PNG / JSON。
- `screenshots/`：原根目录历史截图；`reference_inputs/` 为复盘所用外部原图副本。
- `documentation/`：含现场图的本地复盘源稿。
- `CURRENT_SETUP.md`：当前机位的启动命令。
- `migration_manifest_20261101.json`：旧/新路径、校验和及保护文件清单。

根目录旧数据名指向这里，旧命令继续可用。原 JSON/NPZ/PNG 未改写。
本目录除本说明外默认忽略，不自动纳入 GitHub 发布内容。
