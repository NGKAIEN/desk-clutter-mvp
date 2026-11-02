# 文档工具

这些工具不导入或启动相机，不参与实时检测。命令从项目根目录执行。

## 检查文档与目录

```bash
.venv/bin/python tools/check_project_docs.py
```

检查 README 和 docs 的本地链接、Python 语法，以及本地迁移清单中的文件/哈希/兼容链接。
历史 README 原文不检查旧布局链接；移动后的 Markdown 可维护，但必须存在。
公共检出没有私有迁移清单时明确跳过哈希核对，不要求复制个人数据。
默认允许 Python 实现继续演进并打印历史差异提示；原始迁移数据和标定仍核对。
需要严格回查整理当时的源码字节时加 `--verify-migration-snapshot`，后续 PCA 改动会使该历史检查失败。

## 生成公开副本

```bash
python3 tools/prepare_release.py
```

按 release_files.txt 逐文件导出，拒绝私有目录和符号链接，保存 ZIP 与哈希清单。
有 LICENSE 时一并复制；缺少时标记为草稿，不代选许可证或上传。
操作和干净安装步骤见 [发布指南](../docs/RELEASING.md)。

## 生成 PDF

建议使用独立文档环境，避免改变检测环境：

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-docs.txt
# 公共检出不含私有复盘源稿和实拍，只生成知识手册
.venv-docs/bin/python tools/build_project_pdfs.py --only knowledge
# 本机有完整 data/documentation 与图片证据时生成两份
.venv-docs/bin/python tools/build_project_pdfs.py
```

源稿为 `docs/reference/KNOWLEDGE_HANDBOOK.md` 和本地 `data/documentation/PROJECT_HISTORY.md`。
输出在 `output/pdf/`，重新生成会覆盖同名 PDF；需要旧版时先另存。
源稿用 `<!-- pagebreak -->` 显式分章，生成器仅支持项目使用的 Markdown 子集，并非通用转换器。

Linux 默认查找以下字体。其他系统用 `--font-dir /实际字体目录`，目录内需要四个同名 TTF：

- DejaVuSans.ttf
- DejaVuSans-Bold.ttf
- DejaVuSansMono.ttf
- DroidSansFallbackFull.ttf

字体用于中英文混排并嵌入 PDF；缺少字体或字形时会报错，不用方块替代正文。
字体依赖独立于相机 SDK。PDF 和包含实拍的复盘源稿默认不提交 Git。

## 渲染和核对

安装系统 Poppler 后，先生成两份 PDF，再执行：

```bash
mkdir -p tmp/pdfs
pdftoppm -r 100 -png output/pdf/desk-clutter-change-review.pdf tmp/pdfs/history
pdftoppm -r 100 -png output/pdf/desk-clutter-knowledge-handbook.pdf tmp/pdfs/knowledge
.venv-docs/bin/python tools/check_project_pdfs.py --contact-sheets
```

检查器按本次版本预期 13 / 16 页和相同数量的章节书签，检测空页、异常字符、文字越界，
输出本地 `data/documentation/pdf_qa_20261102.json`，可选生成四页拼图。
它目前要求两份 PDF 都存在；只构建知识手册时请直接目视检查，或以后扩展检查器。
页数/章节数因正文调整而合理改变时，应同步更新检查器的版本预期和记录名。

必须查看每页渲染，检查图片、表格、公式和分页；自动检查并不证明排版完美。
确认后只清理本次生成的 tmp/pdfs 预览 PNG，不删除源稿、原图或最终 PDF。
