import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2


# =========================
# PARAMETERS
# =========================

MIN_DEPTH = 0.20
MAX_DEPTH = 2.00

PLANE_THRESHOLD = 0.01     # 1 cm

MIN_OBJECT_HEIGHT = 0.015  # 1.5 cm
MAX_OBJECT_HEIGHT = 0.40   # 40 cm

GRID_SIZE = 0.01           # 1 cm x 1 cm


# =========================
# REALSENSE
# =========================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

pipeline.start(config)

pc = rs.pointcloud()


def get_vertices():

    frames = pipeline.wait_for_frames()

    depth = frames.get_depth_frame()

    points = pc.calculate(depth)

    vertices = (
        np.asarray(points.get_vertices())
        .view(np.float32)
        .reshape(-1, 3)
    )

    valid = (
        np.isfinite(vertices).all(axis=1)
        & (vertices[:, 2] > MIN_DEPTH)
        & (vertices[:, 2] < MAX_DEPTH)
    )

    return vertices[valid]


# =========================
# CALIBRATION
# =========================

print("Waiting for camera...")

for _ in range(30):
    pipeline.wait_for_frames()

print("Calibrating table...")


vertices = get_vertices()

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    vertices
)

pcd = pcd.voxel_down_sample(
    voxel_size=0.008
)


plane_model, inliers = pcd.segment_plane(
    distance_threshold=PLANE_THRESHOLD,
    ransac_n=3,
    num_iterations=1500
)


a, b, c, d = plane_model

normal = np.array(
    [a, b, c],
    dtype=np.float64
)

normal /= np.linalg.norm(normal)

d /= np.linalg.norm(
    [a, b, c]
)


# -------------------------
# Make normal point toward camera
# -------------------------

# Camera origin = [0, 0, 0]
# We want objects ABOVE table
# (toward camera) to have positive height

if d < 0:

    normal = -normal
    d = -d


plane_cloud = pcd.select_by_index(
    inliers
)

plane_points = np.asarray(
    plane_cloud.points
)


# =========================
# CREATE TABLE COORDINATE
# =========================

origin = plane_points.mean(axis=0)


# Pick an axis that is not parallel to normal
reference = np.array(
    [1.0, 0.0, 0.0]
)

if abs(np.dot(reference, normal)) > 0.9:

    reference = np.array(
        [0.0, 1.0, 0.0]
    )


# Table X axis
u_axis = (
    reference
    - np.dot(reference, normal) * normal
)

u_axis /= np.linalg.norm(u_axis)


# Table Y axis
v_axis = np.cross(
    normal,
    u_axis
)

v_axis /= np.linalg.norm(v_axis)


# =========================
# FIND TABLE ROI
# =========================

relative = (
    plane_points
    - origin
)

u = relative @ u_axis
v = relative @ v_axis


# Ignore tiny RANSAC outliers
u_min, u_max = np.percentile(
    u,
    [2, 98]
)

v_min, v_max = np.percentile(
    v,
    [2, 98]
)


width = u_max - u_min
height = v_max - v_min


print()
print("===== TABLE CALIBRATION =====")

print(
    f"Plane normal: {normal}"
)

print(
    f"Table width estimate: "
    f"{width:.3f} m"
)

print(
    f"Table depth estimate: "
    f"{height:.3f} m"
)

print(
    f"ROI area: "
    f"{width * height:.3f} m²"
)

print("=============================")
print()


# =========================
# GRID
# =========================

cols = int(
    np.ceil(
        width / GRID_SIZE
    )
)

rows = int(
    np.ceil(
        height / GRID_SIZE
    )
)


print(
    "Occupancy grid:",
    cols,
    "x",
    rows
)


# =========================
# LIVE LOOP
# =========================

try:

    while True:

        vertices = get_vertices()

        relative = (
            vertices
            - origin
        )


        # -------------------------
        # Table coordinates
        # -------------------------

        u = relative @ u_axis

        v = relative @ v_axis


        # -------------------------
        # Distance ABOVE table
        # -------------------------

        object_height = (
            vertices @ normal
            + d
        )


        # -------------------------
        # Only points inside table
        # -------------------------

        inside_roi = (
            (u >= u_min)
            & (u <= u_max)
            & (v >= v_min)
            & (v <= v_max)
        )


        # -------------------------
        # Points ABOVE tabletop
        # -------------------------

        object_mask = (
            inside_roi
            & (object_height > MIN_OBJECT_HEIGHT)
            & (object_height < MAX_OBJECT_HEIGHT)
        )


        obj_u = u[object_mask]

        obj_v = v[object_mask]


        # =========================
        # OCCUPANCY GRID
        # =========================

        grid = np.zeros(
            (rows, cols),
            dtype=np.uint8
        )


        ix = (
            (obj_u - u_min)
            / GRID_SIZE
        ).astype(int)

        iy = (
            (obj_v - v_min)
            / GRID_SIZE
        ).astype(int)


        valid_grid = (
            (ix >= 0)
            & (ix < cols)
            & (iy >= 0)
            & (iy < rows)
        )


        grid[
            iy[valid_grid],
            ix[valid_grid]
        ] = 255


        occupied_cells = np.count_nonzero(
            grid
        )

        total_cells = (
            rows * cols
        )


        occupancy = (
            occupied_cells
            / total_cells
            * 100
        )


        print(
            f"\rVisible Occupancy: "
            f"{occupancy:5.1f} %",
            end=""
        )


        # =========================
        # DISPLAY
        # =========================

        display = cv2.resize(
            grid,
            (
                cols * 6,
                rows * 6
            ),
            interpolation=cv2.INTER_NEAREST
        )


        display = cv2.cvtColor(
            display,
            cv2.COLOR_GRAY2BGR
        )


        cv2.putText(
            display,
            f"Occupancy: {occupancy:.1f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            "Desk Occupancy BEV",
            display
        )


        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break


finally:

    pipeline.stop()

    cv2.destroyAllWindows()
