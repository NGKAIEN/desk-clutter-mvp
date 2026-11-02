import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# -------------------------
# RealSense
# -------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    640, 480,
    rs.format.z16,
    30
)

profile = pipeline.start(config)

# RealSense point cloud generator
rs_pc = rs.pointcloud()

# -------------------------
# Open3D
# -------------------------
pcd = o3d.geometry.PointCloud()

vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name="Live Depth Point Cloud",
    width=1000,
    height=700
)

geometry_added = False

try:

    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Depth -> Point Cloud
        points = rs_pc.calculate(depth_frame)

        vertices = np.asarray(
            points.get_vertices()
        ).view(np.float32).reshape(-1, 3)

        # Remove invalid points
        valid = (
            np.isfinite(vertices).all(axis=1)
            & (vertices[:, 2] > 0)
            & (vertices[:, 2] < 3.0)
        )

        vertices = vertices[valid]

        # Optional downsample
        vertices = vertices[::2]

        pcd.points = o3d.utility.Vector3dVector(vertices)

        if not geometry_added:

            vis.add_geometry(pcd)

            render = vis.get_render_option()
            render.point_size = 2.0

            geometry_added = True

        else:

            vis.update_geometry(pcd)

        if not vis.poll_events():
            break

        vis.update_renderer()

finally:

    pipeline.stop()
    vis.destroy_window()
