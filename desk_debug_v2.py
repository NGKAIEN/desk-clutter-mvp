import os
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from desk_compare_view import (
    CYAN, camera_overlay, color_pixel_to_table, comparison_view, draw_corners,
    draw_linked_cursor, draw_view_footprint, ordered_table_corners, project_roi,
    rectified_plane_photo,
)


CALIB_FILE = "calibration.json"

MIN_DEPTH = 0.20
MAX_DEPTH = 2.00

PLANE_THRESHOLD = 0.01

MIN_OBJECT_HEIGHT = 0.003
MAX_OBJECT_HEIGHT = 0.40

GRID_SIZE = 0.01


# ==========================================
# RealSense
# ==========================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

rs_pc = rs.pointcloud()


def get_scene():
    """Keep depth coordinates unchanged; map their pixels into the RGB sensor."""
    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth and color_frame:
            break

    rs_pc.map_to(color_frame)
    points = rs_pc.calculate(depth)
    vertices = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    texture_uv = np.asarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    valid = (
        np.isfinite(vertices).all(axis=1)
        & (vertices[:, 2] > MIN_DEPTH)
        & (vertices[:, 2] < MAX_DEPTH)
    )
    camera = {
        "intrinsics": color_frame.profile.as_video_stream_profile().get_intrinsics(),
        "extrinsics": depth.profile.get_extrinsics_to(color_frame.profile),
    }
    # Copies own their memory after the SDK frames/points leave this function.
    return (vertices[valid].copy(),
            np.asanyarray(color_frame.get_data()).copy(),
            texture_uv[valid].copy(), camera)


def save_comparison(image, phase):
    filename = Path(__file__).resolve().parent / (
        f"desk_compare_{phase}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    )
    if cv2.imwrite(str(filename), image):
        print(f"Comparison saved: {filename}")
    else:
        print(f"Could not save comparison: {filename}")


# ==========================================
# Coordinate conversion
# ==========================================

def project_to_table(vertices, origin, u_axis, v_axis, normal, d):

    relative = vertices - origin

    u = relative @ u_axis
    v = relative @ v_axis

    # signed distance to table
    h = vertices @ normal + d

    return u, v, h


# ==========================================
# RANSAC
# ==========================================

def detect_plane(vertices):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

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

    norm = np.linalg.norm(normal)

    normal /= norm
    d /= norm

    # Camera origin = [0,0,0]
    # Make "toward camera" positive
    if d < 0:
        normal *= -1
        d *= -1

    plane_cloud = pcd.select_by_index(inliers)

    plane_points = np.asarray(
        plane_cloud.points
    )

    origin = plane_points.mean(axis=0)

    # Build table coordinate axes
    reference = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    u_axis = (
        reference
        - np.dot(reference, normal) * normal
    )

    u_axis /= np.linalg.norm(u_axis)

    v_axis = np.cross(
        normal,
        u_axis
    )

    v_axis /= np.linalg.norm(v_axis)

    return (
        normal,
        d,
        origin,
        u_axis,
        v_axis,
        plane_points
    )


# ==========================================
# Calibration BEV UI
# ==========================================

def calibrate(vertices, color, texture_uv, camera):

    print("\nRunning RANSAC...")

    (
        normal,
        d,
        origin,
        u_axis,
        v_axis,
        plane_points
    ) = detect_plane(vertices)

    print("Plane normal:", normal)
    print("Click four desk corners directly in the LEFT camera image.")
    print("Enter = save, U = undo, R = reset corners, F = retake photo and refit plane.")
    print("Right = the same photo flattened onto the table. Hover either view to match locations.")
    print("V = photo/points view; O = highlight the detected plane in the camera photo.")
    overlay_enabled = False
    plane_candidates = np.abs(vertices @ normal + d) < PLANE_THRESHOLD

    # Project all scene points
    u, v, h = project_to_table(
        vertices,
        origin,
        u_axis,
        v_axis,
        normal,
        d
    )

    # Only show things reasonably close to table height
    nearby = (
        (h > -0.10)
        & (h < 0.50)
    )

    u_show = u[nearby]
    v_show = v[nearby]
    h_show = h[nearby]

    # Visualization bounds
    u_min = np.percentile(u_show, 1)
    u_max = np.percentile(u_show, 99)

    v_min = np.percentile(v_show, 1)
    v_max = np.percentile(v_show, 99)

    W = 1000
    H = 700
    PAD = 40

    range_u = u_max - u_min
    range_v = v_max - v_min

    scale = min(
        (W - 2 * PAD) / range_u,
        (H - 2 * PAD) / range_v
    )

    offset_x = (
        W - range_u * scale
    ) / 2

    offset_y = (
        H - range_v * scale
    ) / 2


    def uv_to_pixel(uu, vv):

        x = (
            (uu - u_min) * scale
            + offset_x
        )

        y = (
            H
            - (
                (vv - v_min) * scale
                + offset_y
            )
        )

        return (
            x.astype(int),
            y.astype(int)
        )


    def pixel_to_uv(x, y):

        uu = (
            (x - offset_x) / scale
            + u_min
        )

        vv = (
            ((H - y) - offset_y) / scale
            + v_min
        )

        return uu, vv


    # Cache both right-hand views from the same frozen RGB/depth pair.
    photo_view, _ = rectified_plane_photo(
        rs, color, pixel_to_uv, origin, u_axis, v_axis, normal, d, camera, W, H)
    footprint_u, footprint_v = pixel_to_uv(
        np.array([0, W - 1, W - 1, 0]), np.array([0, 0, H - 1, H - 1]))
    footprint_rgb = project_roi(
        rs, np.column_stack((footprint_u, footprint_v)),
        origin, u_axis, v_axis, normal, d, camera)
    show_photo = True
    hovered_uv = None
    selected_uv = []
    placements = {}
    message = "Hover either image: the cyan link points to the SAME SPOT. Click desk corners on the left."

    def select_corner(event, x, y, flags, param):
        nonlocal message, hovered_uv
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN) or not placements:
            return
        hovered_uv = None
        rgb_pixel = placements["rgb"].to_source(x, y)
        if rgb_pixel is not None:
            uv = color_pixel_to_table(
                rs, rgb_pixel, origin, u_axis, v_axis, normal, d, camera,
                depth_limits=(MIN_DEPTH, MAX_DEPTH),
            )
            if uv is None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    message = "This pixel cannot reach the detected plane. Try a desk corner or press F to retake."
                return
        else:
            bev_pixel = placements["bev"].to_source(x, y)
            if bev_pixel is None:
                return
            uv = np.asarray(pixel_to_uv(*bev_pixel))
        hovered_uv = uv
        if event != cv2.EVENT_LBUTTONDOWN or len(selected_uv) >= 4:
            return
        selected_uv.append(uv)
        message = ("Four corners selected. Check BOTH outlines, then press Enter to save."
                   if len(selected_uv) == 4 else
                   "Click the next desk corner in the LEFT photo. U = undo.")

    cv2.namedWindow("TABLE CALIBRATION")
    cv2.setMouseCallback("TABLE CALIBRATION", select_corner)


    points_view = np.zeros(
        (H, W, 3),
        dtype=np.uint8
    )

    px, py = uv_to_pixel(
        u_show,
        v_show
    )

    valid_px = (
        (px >= 0)
        & (px < W)
        & (py >= 0)
        & (py < H)
    )

    px = px[valid_px]
    py = py[valid_px]

    heights = h_show[valid_px]

    # Plane-ish points = green
    plane_mask = (
        np.abs(heights)
        < PLANE_THRESHOLD
    )

    points_view[
        py[plane_mask],
        px[plane_mask]
    ] = (0, 180, 0)

    # Objects above plane = red
    object_mask = (
        (heights > MIN_OBJECT_HEIGHT)
        & (heights < MAX_OBJECT_HEIGHT)
    )

    points_view[
        py[object_mask],
        px[object_mask]
    ] = (0, 0, 255)

    while True:
        canvas = photo_view.copy() if show_photo else points_view.copy()
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), CYAN, 2)

        selected = np.asarray(selected_uv, dtype=np.float64).reshape(-1, 2)
        corner_x, corner_y = uv_to_pixel(selected[:, 0], selected[:, 1])
        draw_corners(canvas, np.column_stack((corner_x, corner_y)), closed=len(selected_uv) == 4)
        rgb_corners = project_roi(
            rs, selected_uv, origin, u_axis, v_axis, normal, d, camera)
        rgb = camera_overlay(
            color, texture_uv, plane_candidates, rgb_corners, overlay_enabled)
        draw_view_footprint(rgb, footprint_rgb)
        # Keep selected ROI corners above the context outline.
        draw_corners(rgb, rgb_corners, closed=len(selected_uv) == 4)
        display, placements = comparison_view(
            rgb, canvas,
            f"RECALIBRATE / match the SAME DESK in both photos ({len(selected_uv)}/4 corners)",
            message,
            "[Enter] Save  [U] Undo  [R] Reset  [F] Retake  [V] Photo/points  [O] Plane  [S] Image  [Q] Quit",
            calibration=True,
            right_title=("SAME PHOTO / FLATTENED TO TABLE" if show_photo else
                         "PLANE / OBJECT POINTS (V = PHOTO)"),
            legend=("Cyan border = right-view area | Hover = same spot | Photo assumes a flat table; tall items may stretch."
                    if show_photo else
                    "Green = plane | Red = objects | Cyan cursor = corresponding point ON THE TABLE PLANE"),
        )
        if hovered_uv is not None:
            cursor_rgb = project_roi(
                rs, [hovered_uv], origin, u_axis, v_axis, normal, d, camera)[0]
            cursor_x, cursor_y = uv_to_pixel(
                np.asarray(hovered_uv[0]), np.asarray(hovered_uv[1]))
            draw_linked_cursor(display, placements, cursor_rgb, (cursor_x, cursor_y))
        cv2.imshow("TABLE CALIBRATION", display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            selected_uv.clear()
            message = "Corners cleared. Click four corners in the LEFT photo."

        elif key in (ord("u"), 8, 127):
            if selected_uv:
                selected_uv.pop()
            message = "Click a desk corner in the LEFT photo. Enter saves four valid corners."

        elif key == ord("f"):
            cv2.destroyWindow("TABLE CALIBRATION")
            return None  # The caller captures a new RGB/depth pair and refits the plane.

        elif key in (ord("q"), 27):

            raise KeyboardInterrupt

        elif key == ord("o"):
            overlay_enabled = not overlay_enabled

        elif key == ord("v"):
            show_photo = not show_photo

        elif key == ord("s"):
            save_comparison(display, "calibration")

        elif key in [13, 10]:
            try:
                polygon_uv = ordered_table_corners(selected_uv, min_area=GRID_SIZE ** 2)
            except ValueError as error:
                message = str(error)
            else:
                break

    cv2.destroyWindow("TABLE CALIBRATION")

    calibration = {

        "normal": normal.tolist(),
        "d": float(d),

        "origin": origin.tolist(),

        "u_axis": u_axis.tolist(),
        "v_axis": v_axis.tolist(),

        "polygon_uv": polygon_uv.tolist(),

        "grid_size": GRID_SIZE
    }


    with open(
        CALIB_FILE,
        "w"
    ) as f:

        json.dump(
            calibration,
            f,
            indent=2
        )


    print()
    print("Calibration saved:")
    print(CALIB_FILE)

    return calibration


# ==========================================
# Load calibration
# ==========================================

def load_calibration():

    with open(CALIB_FILE) as f:
        data = json.load(f)

    data["normal"] = np.array(
        data["normal"]
    )

    data["origin"] = np.array(
        data["origin"]
    )

    data["u_axis"] = np.array(
        data["u_axis"]
    )

    data["v_axis"] = np.array(
        data["v_axis"]
    )

    data["polygon_uv"] = np.array(
        data["polygon_uv"]
    )

    return data


# ==========================================
# Main
# ==========================================

try:

    print("Warming up camera...")

    for _ in range(30):
        pipeline.wait_for_frames()


    # -----------------------
    # Calibration decision
    # -----------------------

    if os.path.exists(CALIB_FILE):

        answer = input(
            "\nCalibration found. "
            "[L]oad / [R]ecalibrate: "
        ).strip().lower()

    else:

        answer = "r"


    if answer.startswith("l"):

        calibration = load_calibration()

        print("Loaded saved calibration.")

    else:

        calibration = None
        while calibration is None:
            vertices, color, texture_uv, camera = get_scene()
            calibration = calibrate(vertices, color, texture_uv, camera)


    normal = np.array(
        calibration["normal"]
    )

    d = calibration["d"]

    origin = np.array(
        calibration["origin"]
    )

    u_axis = np.array(
        calibration["u_axis"]
    )

    v_axis = np.array(
        calibration["v_axis"]
    )

    polygon_uv = np.array(
        calibration["polygon_uv"]
    )


    # ======================================
    # Build fixed table mask
    # ======================================

    u_min = polygon_uv[:, 0].min()
    u_max = polygon_uv[:, 0].max()

    v_min = polygon_uv[:, 1].min()
    v_max = polygon_uv[:, 1].max()


    cols = int(
        np.ceil(
            (u_max - u_min)
            / GRID_SIZE
        )
    ) + 1

    rows = int(
        np.ceil(
            (v_max - v_min)
            / GRID_SIZE
        )
    ) + 1


    table_mask = np.zeros(
        (rows, cols),
        dtype=np.uint8
    )


    poly_grid = np.column_stack([

        (
            (polygon_uv[:, 0] - u_min)
            / GRID_SIZE
        ).astype(int),

        (
            (polygon_uv[:, 1] - v_min)
            / GRID_SIZE
        ).astype(int)

    ])


    cv2.fillPoly(
        table_mask,
        [poly_grid],
        255
    )


    total_table_cells = np.count_nonzero(
        table_mask
    )


    baseline = calibration.get(
        "baseline",
        None
    )

    if baseline is not None:
        print(
            f"Loaded baseline: "
            f"{baseline:.1f}%"
        )
    else:
        print(
            "No baseline saved yet."
        )


    print()
    print("============================")
    print("LIVE MODE")
    print("B = set clean baseline")
    print("O = toggle RGB plane overlay")
    print("S = save comparison image")
    print("Q = quit")
    print("============================")
    print()


    overlay_enabled = True

    while True:

        vertices, color, texture_uv, camera = get_scene()

        u, v, h = project_to_table(
            vertices,
            origin,
            u_axis,
            v_axis,
            normal,
            d
        )


        ix = (
            (u - u_min)
            / GRID_SIZE
        ).astype(int)

        iy = (
            (v - v_min)
            / GRID_SIZE
        ).astype(int)


        valid_grid = (
            (ix >= 0)
            & (ix < cols)
            & (iy >= 0)
            & (iy < rows)
        )


        # ------------------------
        # Visible plane cells
        # ------------------------

        plane_grid = np.zeros(
            (rows, cols),
            dtype=np.uint8
        )

        plane_points = (
            valid_grid
            & (np.abs(h) < PLANE_THRESHOLD)
        )

        plane_ix = ix[plane_points]
        plane_iy = iy[plane_points]

        inside = (
            table_mask[
                plane_iy,
                plane_ix
            ] > 0
        )

        plane_grid[
            plane_iy[inside],
            plane_ix[inside]
        ] = 255


        # ------------------------
        # Object cells
        # ------------------------

        object_grid = np.zeros(
            (rows, cols),
            dtype=np.uint8
        )

        object_points = (
            valid_grid
            & (h > MIN_OBJECT_HEIGHT)
            & (h < MAX_OBJECT_HEIGHT)
        )

        obj_ix = ix[object_points]
        obj_iy = iy[object_points]

        inside = (
            table_mask[
                obj_iy,
                obj_ix
            ] > 0
        )

        obj_ix = obj_ix[inside]
        obj_iy = obj_iy[inside]

        object_grid[
            obj_iy,
            obj_ix
        ] = 255


        occupied_cells = np.count_nonzero(
            object_grid
        )

        occupancy = (
            occupied_cells
            / total_table_cells
            * 100
        )


        # ==================================
        # Debug visualization
        # ==================================

        debug = np.zeros(
            (rows, cols, 3),
            dtype=np.uint8
        )

        # ROI background
        debug[
            table_mask > 0
        ] = (40, 40, 40)

        # visible table
        debug[
            plane_grid > 0
        ] = (0, 180, 0)

        # objects
        debug[
            object_grid > 0
        ] = (0, 0, 255)


        cv2.polylines(
            debug,
            [poly_grid],
            True,
            (0, 255, 255),
            1
        )


        # Tint only the detected table points inside the selected ROI.
        plane_in_roi = np.zeros(len(vertices), dtype=bool)
        plane_indices = np.flatnonzero(plane_points)
        plane_in_roi[plane_indices] = table_mask[plane_iy, plane_ix] > 0
        rgb_corners = project_roi(
            rs, polygon_uv, origin, u_axis, v_axis, normal, d, camera)
        rgb = camera_overlay(
            color, texture_uv, plane_in_roi, rgb_corners, overlay_enabled)

        # Flip so BEV feels more natural
        debug = cv2.flip(
            debug,
            0
        )


        scale = max(
            2,
            min(
                6,
                int(
                    800
                    / max(rows, cols)
                )
            )
        )


        display = cv2.resize(
            debug,
            (
                cols * scale,
                rows * scale
            ),
            interpolation=cv2.INTER_NEAREST
        )


        if baseline is not None:

            relative = (
                occupancy
                - baseline
            )

            status_text = (
                f"Current: {occupancy:.1f}%  "
                f"Baseline: {baseline:.1f}%  "
                f"Delta: {relative:+.1f} pp"
            )

        else:

            status_text = (
                f"Current: {occupancy:.1f}%  "
                f"Baseline: NOT SET"
            )


        # Match the corner IDs after the vertical flip and nearest-neighbor resize.
        bev_corners = np.column_stack((
            (poly_grid[:, 0] + .5) * scale,
            (rows - 1 - poly_grid[:, 1] + .5) * scale,
        ))
        draw_corners(display, bev_corners)
        display, _ = comparison_view(
            rgb, display,
            "DESK COMPARISON / matching corner IDs in both views",
            status_text,
            "[B] Set baseline  [O] RGB overlay  [S] Save image  [Q] Quit",
        )
        cv2.imshow("Desk Debug BEV", display)

        key = cv2.waitKey(1) & 0xFF


        if key == ord("b"):

            baseline = occupancy

            # Save baseline into calibration
            calibration["baseline"] = float(
                baseline
            )

            # Convert NumPy arrays back to normal lists
            # so json.dump() can serialize them.
            save_data = calibration.copy()

            save_data["normal"] = (
                np.asarray(
                    calibration["normal"]
                ).tolist()
            )

            save_data["origin"] = (
                np.asarray(
                    calibration["origin"]
                ).tolist()
            )

            save_data["u_axis"] = (
                np.asarray(
                    calibration["u_axis"]
                ).tolist()
            )

            save_data["v_axis"] = (
                np.asarray(
                    calibration["v_axis"]
                ).tolist()
            )

            save_data["polygon_uv"] = (
                np.asarray(
                    calibration["polygon_uv"]
                ).tolist()
            )

            with open(
                CALIB_FILE,
                "w"
            ) as f:

                json.dump(
                    save_data,
                    f,
                    indent=2
                )

            print(
                "\nBaseline SAVED:",
                f"{baseline:.1f}%"
            )

        elif key == ord("o"):
            overlay_enabled = not overlay_enabled

        elif key == ord("s"):
            save_comparison(display, "live")

        elif key in (ord("q"), 27):

            break


finally:

    pipeline.stop()
    cv2.destroyAllWindows()
    
