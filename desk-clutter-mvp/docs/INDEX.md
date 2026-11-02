# 文档导航

从根目录的 [README](../README.md) 开始。这里按用途分组，避免混淆不同阶段。

发布准备：[发布指南](RELEASING.md)、[发布验证记录](reports/RELEASE_READINESS.md)、[发布说明](../RELEASE_NOTES.md)。

## 使用指南

- [照片对照与四角标定](guides/LEGACY_WORKFLOW.md)：第一次标定，旧实验历史。
- [v04 稳定网格、内缩和 PCA](guides/V04_GUIDE.md)：核心检测的详细解释。
- [高度诊断](guides/HEIGHT_DIAGNOSTIC_GUIDE.md)：偏差、时间波动、缺测。
- [多帧平面细化](guides/MULTIFRAME_CALIBRATION_GUIDE.md)：C 拟合、V 独立验证。
- [局部空桌参考](guides/BACKGROUND_HEIGHT_GUIDE.md)：薄电脑漏检的可选模式。

## 原理与结构

- [知识点手册](reference/KNOWLEDGE_HANDBOOK.md)：链路、公式、参数、实现与练习。
- [目录与迁移](reference/PROJECT_LAYOUT.md)：文件用途、兼容路径、还原说明。
- 本地复盘源稿：`data/documentation/PROJECT_HISTORY.md`，含实拍，不随公共仓库提交。
- 两份阅读 PDF：`output/pdf/`，用 `tools/build_project_pdfs.py` 从源稿生成。

## 检查与验证

- [现场验收](checklists/V05_ACCEPTANCE.md)
- [发布前清单](checklists/RELEASE_CHECKLIST.md)
- [v04 验证](reports/V04_TEST_REPORT.md)
- [v05 初版验证](reports/V05_TEST_REPORT.md)
- [局部模式追加验证](reports/BACKGROUND_HEIGHT_TEST_REPORT.md)
- [本次资料整理验证](reports/ORGANIZATION_TEST_REPORT.md)
- [PCA 方向门槛对照试验](reports/PCA_GATE_TRIAL.md)：2026-11-02，默认仍为 2.5。
- [PCA 方向面积门槛试验](reports/PCA_AREA_TRIAL.md)：可选 50cm²，默认关闭。

历史指南中的“本次”“当前版本”指写作时的阶段，不覆盖最新 README。
整理前 README 原文在 `archive/`，仅供追溯；里面相对路径属于原布局。
