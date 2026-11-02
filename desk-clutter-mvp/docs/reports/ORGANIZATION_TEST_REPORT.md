# 文档与目录整理验证记录

整理：2026-11-01 至 2026-11-02。范围仅为文档、归档、兼容路径和文档工具。
没有更改现用检测算法、阈值、PCA、评分或原始标定，没有启动真实相机。

## 已执行的软件验证

以下命令均在项目根目录执行。

| 检查 | 结果 | 不代表什么 |
| --- | --- | --- |
| `.venv/bin/python -m unittest discover -s tests -q` | 164 项通过 | 不是真实桌面精度测试 |
| `python3 -m unittest discover -s tests -q` | 另一环境 164 项通过 | 不是全平台兼容认证 |
| v05 默认合成模式、headless 导出 | 成功生成 PNG + JSON | 不是相机采集 |
| v05 局部合成模式、headless 导出 | 成功生成 PNG + JSON | 不证明真实薄电脑稳定性 |
| 当前候选标定与空桌参考只读加载 | 成功，参考可用率约 99.27% | 可用率不是检测准确率 |
| 两份 PDF 文本、边界、书签检查 | 复盘 13 页、知识手册 16 页通过 | 自动检查不能替代目视检查 |
| PDF 逐页渲染审阅 | 29 页均已查看，修复末页溢出后复查 | 未把截图效果当长期性能 |

本次测试环境：项目虚拟环境 Python 3.14 / NumPy 2.5.3 / OpenCV 5；
系统环境 Python 3.14 / NumPy 2.3.5 / OpenCV 4.10。
PDF 使用独立的 Python 3.12 文档运行时，没有向检测环境安装新包。

```bash
.venv/bin/python desk_status_v05.py --demo --headless --output-dir /tmp/desk-organized-demo
.venv/bin/python desk_status_v05.py --demo --demo-background --headless --output-dir /tmp/desk-organized-local-demo
.venv/bin/python tools/check_project_docs.py
```

最终文档/迁移检查通过：45 个本地 Markdown 链接、56 个迁移文件的存在性及非 Markdown 原件哈希、
9 个保护文件哈希、4 个兼容链接，以及 15 份 Python 文件的语法。工具只读，不启动相机。

## 数据保护与目录核对

- 迁移清单记录 56 个文件，迁移时逐个核对 SHA-256；10 张根目录截图归档。
- 8 份现用 Python 与 calibration.json 共 9 个保护文件保持原字节。
- 4 个旧数据目录保留相对符号链接，已有历史采集路径仍可访问。
- 原始 JSON、NPZ、PNG 和旧实验代码不改写；移动后的指南允许修复链接和命令。
- 根 README 已更新，整理前原文另存于 `docs/archive/`。
- 文档工具重新检查文件存在性、非 Markdown 原件哈希、保护文件哈希、兼容链接及 Python 语法。

候选标定与空桌参考的 SHA-256 已核对未变，具体值和启动命令见本地 `data/CURRENT_SETUP.md`。
完整迁移清单在本地 `data/migration_manifest_20261101.json`。
迁移/采集证据默认不提交公共仓库；公共检出缺少迁移清单时，工具会明确跳过该项。

## 文档交付与可重复生成

- [README](../../README.md)：演示、日常启动、空桌参考、首次标定、按键与限制。
- [知识手册源稿](../reference/KNOWLEDGE_HANDBOOK.md)：原理、公式、调参与验证。
- 本地 `data/documentation/PROJECT_HISTORY.md`：阶段变更、原因、效果、证据和未解决问题。
- `output/pdf/`：两份有章节书签的中文 PDF；复盘含 7 张真实图像。
- 本地 `data/documentation/pdf_qa_20261102.json`：各页文本量、图片数和书签检查记录。
- [文档工具说明](../../tools/README.md)：生成、检查、渲染的方法与字体要求。

PDF 按 PDF 技能的渲染检查流程制作；这使图文复盘与知识手册均经过逐页排版核对。
渲染预览仅为临时文件，完成检查后清理；源稿、原图、最终 PDF 和检查记录保留。

## 仍需现场/发布验证

没有新增“全部问题已解决”的结论。真实空桌 → 放电脑 → 旋转 → 拿走的连续验收仍待完成；
尤其要检查空桌持续误报、电脑轮廓跳动和移除恢复。PCA N/A 仍按现有门槛保留。
参考采集不是独立验收，单张截图也不是稳定性统计。

尚未做干净环境安装、跨设备验证、许可证选择或 GitHub 发布。
复盘包含真实桌面照片/布局，分享前应先检查隐私；没有初始化、提交或推送 Git。
