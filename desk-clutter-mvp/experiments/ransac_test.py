import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    640, 480,
    rs.format.z16,
    30
)

pipeline.start(config)

try:
    # 等相机稳定
    for _ in range(30):
        frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()

    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)

    vertices = (
        np.asarray(points.get_vertices())
        .view(np.float32)
        .reshape(-1, 3)
    )

    # -----------------------
    # 基础过滤
    # -----------------------
    valid = (
        np.isfinite(vertices).all(axis=1)
        & (vertices[:, 2] > 0.2)
        & (vertices[:, 2] < 2.0)
    )

    vertices = vertices[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # 降采样，RANSAC 会快很多
    pcd = pcd.voxel_down_sample(
        voxel_size=0.008
    )

    print("Total points:", len(pcd.points))

    # -----------------------
    # RANSAC 找最大平面
    # -----------------------
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,   # 1 cm
        ransac_n=3,
        num_iterations=1000
    )

    a, b, c, d = plane_model

    print()
    print("Plane equation:")
    print(
        f"{a:.3f} x + "
        f"{b:.3f} y + "
        f"{c:.3f} z + "
        f"{d:.3f} = 0"
    )

    print(
        "Plane points:",
        len(inliers)
    )

    print(
        "Plane ratio:",
        f"{len(inliers) / len(pcd.points) * 100:.1f}%"
    )

    # -----------------------
    # 分开显示
    # -----------------------
    plane = pcd.select_by_index(inliers)

    objects = pcd.select_by_index(
        inliers,
        invert=True
    )

    # 桌面候选：绿色
    plane.paint_uniform_color(
        [0.1, 0.9, 0.1]
    )

    # 其他物体：红色
    objects.paint_uniform_color(
        [0.9, 0.1, 0.1]
    )

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2
    )

    o3d.visualization.draw_geometries(
        [plane, objects, frame],
        window_name="RANSAC Plane Detection",
        width=1100,
        height=800
    )

finally:
    pipeline.stop()
