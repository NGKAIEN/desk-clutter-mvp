# 第三方依赖与许可检查入口

此文件是依赖索引，不替代依赖自带的 LICENSE/NOTICE，也不是全部许可合规的认证。
当前发布包只含本项目源码和合成示例，不内置第三方 wheel、SDK、字体或系统库。
项目自身许可证不能覆盖这些第三方组件；最终发布前仍需作者确认项目代码的授权来源。

| 用途 | 依赖 | 官方来源 / 许可材料入口 |
| --- | --- | --- |
| 数组与 PCA | NumPy | [项目](https://github.com/numpy/numpy)，使用安装版本附带的许可文件 |
| 图像、连通区域、GUI | opencv-python / OpenCV | [wheel 项目](https://github.com/opencv/opencv-python)，同时检查 wheel 的第三方许可说明 |
| 实时深度相机 | pyrealsense2 / librealsense | [官方仓库](https://github.com/realsenseai/librealsense) |
| 旧四角标定、点云实验 | Open3D | [官方仓库](https://github.com/isl-org/Open3D) |
| 可选 PDF 生成与检查 | ReportLab、pypdf、pdfplumber | [ReportLab](https://www.reportlab.com/)、[pypdf](https://github.com/py-pdf/pypdf)、[pdfplumber](https://github.com/jsvine/pdfplumber) |
| 可选 PDF 字体和渲染 | DejaVu、Droid、Poppler | 从使用的系统软件包获取字体/工具许可；本源码包不附带字体或工具 |

注意 OpenCV wheel、NumPy wheel 可能携带第三方二进制组件，不能只看顶层包名的许可分类。
如果将来改成打包可执行程序、分发 wheel/字体或内置 SDK，需要重新检查该分发方式的要求。

当前干净安装使用的两个演示包版本记录于 `constraints-demo-tested.txt`。
安装到其他平台、改变版本或切换 headless 包后，应重新记录依赖及其许可材料。
