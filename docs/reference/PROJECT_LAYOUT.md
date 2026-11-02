# 目录职责、迁移与还原

整理日期：2026-11-01。分开启动说明、实现代码、旧实验、私有采集和阅读材料。
没有改变检测、阈值、PCA 或评分，没有删除原始数据。

## 目录职责

| 位置 | 内容 | 发布建议 |
| --- | --- | --- |
| 根目录 | README、当前脚本、依赖、pytest 配置 | calibration.json 不公开 |
| `experiments/` | 相机、点云、RANSAC、自动 ROI 旧实验 | 标记实验用途 |
| `docs/guides/` | 分阶段使用指南 | 可公开 |
| `docs/reference/` | 知识手册、目录说明 | 可公开 |
| `docs/reports/` | 软件测试记录 | 不等同现场验收 |
| `docs/checklists/` | 验收与发布清单 | 可公开 |
| `docs/archive/` | 整理前 README | 追溯用，不是启动入口 |
| `tests/` | 离线自动测试 | 可公开 |
| `data/` | 采集、候选、参考、截图、迁移清单 | 除 README 外默认不公开 |
| `output/pdf/` | 复盘、知识 PDF | 默认不公开；复盘含实拍 |
| `tools/` | 文档生成 | 可公开，复盘需要本地证据 |

## 移动清单

- 五份指南移入 `docs/guides/`，原 READ_ME 正文改名 LEGACY_WORKFLOW。
- 三份测试记录移入 `docs/reports/`；两份验收/发布清单移入 `docs/checklists/`。
- 根目录 READ_ME 只指向统一 README，不维护第二套流程。
- 四个旧实验移入 `experiments/`，字节不变；原实验启动命令需增加此前缀。
- 十张根目录 PNG 移入 `data/screenshots/`，未改写。旧根目录图片链接需按清单查找。
- 四组采集目录移入 data，原名保留相对符号链接，之前分享的采集路径仍可访问。

| 旧目录（兼容入口） | 实际存放 |
| --- | --- |
| `plane_calibrations/` | `data/plane_calibrations/` |
| `height_diagnostics/` | `data/height_diagnostics/` |
| `height_backgrounds/` | `data/height_backgrounds/` |
| `outputs/` | `data/outputs/` |

迁移时核对了 56 个文件；8 份现用 Python 和 calibration.json 保持原样。
移走的 10 份 Markdown 机械修复了相对链接，旧实验指南的命令增加 experiments/。
校验值见本地 `data/migration_manifest_20261101.json`。
原 JSON 内的历史路径不重写，保留证据；标定/参考/NPZ/PNG 内容未改。

## 为什么现用 Python 仍在根目录

它们互相导入，并基于脚本位置定位标定和输出。搬动全部实现会引入路径与启动回归风险，
不符合本次“资料整理”的范围。v03 也有相对脚本的标定路径，暂留作对照。

公开克隆没有私有子目录和兼容链接时，程序仍能创建原默认目录；
希望统一保存到 data 时显式用 `--output-dir data/目录名`。
Windows 不需要建立这些链接才能使用程序；链接只服务当前 Linux 历史路径。

## 还原方法（没有执行）

1. 退出相机程序，先备份当前项目和迁移清单。
2. 依据清单确认旧名是链接、新名是真实目录；不要递归删除目录。
3. 仅移除兼容链接本身，再移回真实目录；若旧名已是普通目录，先比较，不覆盖。
4. 旧实验/截图按清单反向移动；README 与 archive 原文比较恢复，不覆盖新增文档。
5. 修复恢复布局后的文档链接，重跑测试并核对标定和参考 SHA-256。

没有初始化 Git。兼容链接与 .gitignore 不替代备份，发布前仍检查待提交内容。
