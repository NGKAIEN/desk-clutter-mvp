# 发布准备与手动上传

## 1. 先确认范围

本版发布的是固定机位 MVP，而不是全场景准确率保证。默认参数保持不变。
当前现场试验可以显式使用 `--pca-ratio 1.5 --min-pca-area-cm2 50`；
不把个人 `calibration.json` 或 `background.json` 当作公共默认配置。

阅读 [发布说明](../RELEASE_NOTES.md)、[验证记录](reports/RELEASE_READINESS.md)
及 [第三方依赖](../THIRD_PARTY.md)。作者已选择 [Apache-2.0](../LICENSE) 与 [NOTICE](../NOTICE)，
公开署名为 MilkSpread (NGKAIEN)。仓库名称/可见性和实际发布操作仍需确认；没有创建远程仓库。

## 2. 生成干净副本

```bash
python3 tools/prepare_release.py
```

每次新建 `releases/desk-clutter-mvp-v0.5-时间戳/` 对应目录（实际名以终端为准），里面包含：

- `source/`：允许公开的文件副本。
- `desk-clutter-mvp-v0.5-source.zip`：同一份源代码压缩包。
- `manifest.json`：文件清单、SHA-256、压缩包哈希和 LICENSE / NOTICE 是否存在。

导出依据 [明确文件清单](../tools/release_files.txt)，不是递归打包工作目录。
程序拒绝符号链接、采集目录、个人标定、非白名单图片和明显凭据/个人路径。
文本扫描是防误上传措施，不是完整安全审计；仍需人工审查代码和合成示例图。
不要为了让导出通过而直接放开整个 data 或截图目录。

修改 LICENSE、NOTICE 或源文件后，应重新导出并重新核对；旧 ZIP 不会自动更新。
早先无许可证的 57 文件草稿保留用于追溯，不作为当前发布包。
程序不运行 git、不创建远程仓库、不上传文件。

## 3. 独立安装检查

在导出的 `source/` 目录运行，使用新虚拟环境，不能借用原项目的 .venv：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python tools/check_project_docs.py
.venv/bin/python desk_status_v05.py --demo --headless --output-dir tmp/default
.venv/bin/python desk_status_v05.py --demo --demo-background --headless \
  --pca-ratio 1.5 --min-pca-area-cm2 50 --output-dir tmp/local
.venv/bin/python desk_status_v05.py --demo --demo-scene missing --headless --output-dir tmp/missing
```

如需复用此次 Linux/Python 3.12 验证的依赖版本，安装时加
`-c constraints-demo-tested.txt`；该约束不是跨系统通用锁文件。
Windows 使用 `.venv\Scripts\python.exe`，但本轮尚未在 Windows 实测。
若终端预设了 PYTHONPATH（例如 ROS），虚拟环境仍可能看到外部包；
可仅在验证命令前使用 `env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1`，
不要更改其他项目的配置。新虚拟环境仍共享本机系统库，不等于全新机器。

无 GUI 服务器可在单独环境使用 opencv-python-headless 替代 GUI 包，不能混装；
本轮干净安装验证的是 requirements.txt 中的 GUI wheel 的 headless 运行路径。
实时硬件和 Open3D 安装属于另外的验证范围。

## 4. 作者确认后再建立 GitHub 仓库

先审查导出的 source，不要在含个人数据的原始工作目录直接 `git add .`。
确定许可证、版权署名和远程仓库后，在干净 source 内手动执行：

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
# 确认待提交内容没有标定/采集/虚拟环境后才继续
git commit -m "Prepare desk clutter MVP v0.5"
```

随后由作者在 GitHub 创建所选账户、名称、可见性的空仓库，按 GitHub 提供的实际地址
添加 origin 并推送。这里不预填账户或地址，也没有代执行以上步骤。
推送后查看 Actions；工作流配置好不代表 GitHub 已运行或通过。

CI 使用官方 [Python 工作流指南](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
中的 checkout/setup-python 方式，执行 Ubuntu 上的多 Python 版本离线检查，无上传步骤、无相机。
当前配置的其他 Python 版本须等实际 CI 执行后才算验证过。
