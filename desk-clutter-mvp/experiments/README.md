# 早期实验，不是当前 MVP 入口

从项目根目录按需运行，不要批量导入；这些脚本可能立即打开相机或窗口。

| 文件 | 用途 |
| --- | --- |
| `test_camera.py` | 手动 RGB/深度连通性检查 |
| `pointcloud_live.py` | 实时点云可视化 |
| `ransac_test.py` | 单帧最大平面 RANSAC |
| `desk_occupancy.py` | 自动 ROI 占用率实验 |

例如 `.venv/bin/python experiments/test_camera.py`。需要相机依赖，点云类还需 Open3D。
只移动了这四份脚本，没有改写内容。自动测试只发现 `tests/`。
