"""Desk clutter MVP v0.3 — diagnostic object regions and orientation.

Reads the existing calibration.json created by desk_debug_v2.py.
Does NOT modify the calibration or the saved baseline.

Controls: Q quit | M toggle morphology | S save screenshot
"""

import json
from pathlib import Path

import cv2
import numpy as np


CALIB_FILE = Path(__file__).resolve().parent / "calibration.json"
GRID_DEFAULT = 0.01
MIN_DEPTH, MAX_DEPTH = 0.15, 2.00
PLANE_THRESHOLD = 0.01
MIN_OBJECT_HEIGHT, MAX_OBJECT_HEIGHT = 0.010, 0.40
MIN_REGION_CELLS = 8        # Reject very small islands (8 cells = 8 cm^2 with 1 cm grid)
MIN_ASPECT_RATIO = 1.6     # Near-square/circular blobs have unreliable orientation
MIN_LONG_EDGE_CELLS = 5   # Direction of tiny blobs is unreliable
MAX_DISPLAY_REGIONS = 16

COLORS = [
    (240, 170, 70), (115, 210, 115), (215, 135, 220),
    (85, 220, 220), (225, 180, 115), (165, 155, 240),
    (135, 225, 175), (195, 125, 90),
]


def load_table(path=CALIB_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run desk_debug_v2.py and save the table ROI first."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    origin = np.asarray(data["origin"], dtype=np.float64)
    u_axis = np.asarray(data["u_axis"], dtype=np.float64)
    v_axis = np.asarray(data["v_axis"], dtype=np.float64)
    normal = np.asarray(data["normal"], dtype=np.float64)
    polygon = np.asarray(data["polygon_uv"], dtype=np.float64)
    grid_size = float(data.get("grid_size", GRID_DEFAULT))
    if grid_size <= 0 or polygon.shape != (4, 2):
        raise ValueError("Invalid grid_size or polygon_uv in calibration.json")

    u_min, v_min = polygon.min(axis=0)
    u_max, v_max = polygon.max(axis=0)
    cols = int(np.ceil((u_max - u_min) / grid_size)) + 1
    rows = int(np.ceil((v_max - v_min) / grid_size)) + 1

    poly_grid = np.column_stack([
        ((polygon[:, 0] - u_min) / grid_size).astype(np.int32),
        ((polygon[:, 1] - v_min) / grid_size).astype(np.int32),
    ])
    table_mask = np.zeros((rows, cols), np.uint8)
    cv2.fillPoly(table_mask, [poly_grid], 255)
    if cv2.countNonZero(table_mask) == 0:
        raise ValueError("Table ROI has zero area")
    return {
        "origin": origin, "u_axis": u_axis, "v_axis": v_axis,
        "normal": normal, "d": float(data["d"]),
        "grid_size": grid_size, "u_min": u_min, "v_min": v_min,
        "rows": rows, "cols": cols,
        "table_mask": table_mask, "poly_grid": poly_grid,
        "baseline": data.get("baseline"),
    }


def depth_to_vertices(rs, rs_pc, depth_frame):
    points = rs_pc.calculate(depth_frame)
    vertices = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    good = (
        np.isfinite(vertices).all(axis=1)
        & (vertices[:, 2] > MIN_DEPTH)
        & (vertices[:, 2] < MAX_DEPTH)
    )
    return vertices[good]


def make_grids(vertices, table):
    rows, cols = table["rows"], table["cols"]
    plane_grid = np.zeros((rows, cols), dtype=np.uint8)
    object_grid = np.zeros_like(plane_grid)
    if len(vertices) == 0:
        return plane_grid, object_grid

    relative = vertices - table["origin"]
    u = relative @ table["u_axis"]
    v = relative @ table["v_axis"]
    h = vertices @ table["normal"] + table["d"]
    # Floor is important for negative coordinates (unlike plain astype(int)).
    ix = np.floor((u - table["u_min"]) / table["grid_size"]).astype(np.int32)
    iy = np.floor((v - table["v_min"]) / table["grid_size"]).astype(np.int32)
    inside_bounds = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    ix, iy, h = ix[inside_bounds], iy[inside_bounds], h[inside_bounds]
    inside_roi = table["table_mask"][iy, ix] > 0
    ix, iy, h = ix[inside_roi], iy[inside_roi], h[inside_roi]

    on_table = np.abs(h) < PLANE_THRESHOLD
    on_object = (h > MIN_OBJECT_HEIGHT) & (h < MAX_OBJECT_HEIGHT)
    plane_grid[iy[on_table], ix[on_table]] = 255
    object_grid[iy[on_object], ix[on_object]] = 255
    return plane_grid, object_grid


def rectangle_direction(box):
    """Long-edge orientation in degrees [0, 180), or None for unreliable shapes."""
    e0 = box[1] - box[0]
    e1 = box[2] - box[1]
    len0, len1 = np.linalg.norm(e0), np.linalg.norm(e1)
    longest, shortest = max(len0, len1), min(len0, len1)
    if (longest < MIN_LONG_EDGE_CELLS or shortest < 1e-3
            or longest / shortest < MIN_ASPECT_RATIO):
        return None, longest / max(shortest, 1e-6)
    vec = e0 if len0 >= len1 else e1
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])) % 180)
    return angle, longest / shortest


def extract_regions(object_grid, table_mask, morphology=True):
    mask = cv2.bitwise_and(object_grid, table_mask)
    if morphology:
        # Join 1-cell gaps in visible surfaces, remove isolated noise.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.bitwise_and(mask, table_mask)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = sorted(
        (i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= MIN_REGION_CELLS),
        key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True,
    )[:MAX_DISPLAY_REGIONS]

    regions = []
    for label in candidates:
        y, x = np.nonzero(labels == label)
        points = np.column_stack((x, y)).astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        angle, aspect = rectangle_direction(box)
        regions.append({
            "label": label, "area_cells": int(stats[label, cv2.CC_STAT_AREA]),
            "center": (float(centroids[label, 0]), float(centroids[label, 1])),
            "box": box, "angle": angle, "aspect": aspect,
        })
    return mask, labels, regions


def angle_gap(a, b):
    delta = abs(a - b) % 180
    return min(delta, 180 - delta)


def mean_pairwise_gap(regions):
    angles = [r["angle"] for r in regions if r["angle"] is not None]
    if len(angles) < 2:
        return None, len(angles)
    gaps = [angle_gap(a, b) for i, a in enumerate(angles) for b in angles[i + 1:]]
    return float(np.mean(gaps)), len(angles)


def render(table, plane_grid, object_grid, labels, regions, morphology):
    rows, cols = table["rows"], table["cols"]
    scale = max(2, min(6, int(750 / max(rows, cols))))
    margin = 12
    top = 119
    panel_width = cols * scale
    width = panel_width * 2 + margin * 3
    height = rows * scale + top + margin
    canvas = np.full((height, width, 3), (23, 24, 26), np.uint8)

    raw = np.zeros((rows, cols, 3), dtype=np.uint8)
    raw[table["table_mask"] > 0] = (55, 55, 55)
    raw[plane_grid > 0] = (25, 150, 35)
    raw[object_grid > 0] = (40, 60, 230)

    processed = np.zeros((rows, cols, 3), dtype=np.uint8)
    processed[table["table_mask"] > 0] = (55, 55, 55)
    for i, region in enumerate(regions):
        processed[labels == region["label"]] = COLORS[i % len(COLORS)]

    poly = table["poly_grid"]
    cv2.polylines(raw, [poly], True, (0, 255, 255), 1)
    cv2.polylines(processed, [poly], True, (0, 255, 255), 1)
    left = cv2.resize(cv2.flip(raw, 0), (panel_width, rows * scale), interpolation=cv2.INTER_NEAREST)
    right = cv2.resize(cv2.flip(processed, 0), (panel_width, rows * scale), interpolation=cv2.INTER_NEAREST)
    x1, x2 = margin, panel_width + margin * 2
    canvas[top:top + rows * scale, x1:x1 + panel_width] = left
    canvas[top:top + rows * scale, x2:x2 + panel_width] = right

    occupancy = 100 * cv2.countNonZero(object_grid) / cv2.countNonZero(table["table_mask"])
    baseline = table["baseline"]
    baseline_label = f" | Baseline {baseline:.1f}% | Delta {occupancy - baseline:+.1f}pp" if baseline is not None else ""
    gap, valid_count = mean_pairwise_gap(regions)
    gap_text = f"{gap:.1f} deg" if gap is not None else "N/A"

    cv2.putText(canvas, f"Raw occupancy: {occupancy:.1f}%{baseline_label}",
                (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, .57, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Regions: {len(regions)} | Orientations: {valid_count} | Pairwise angle gap: {gap_text}",
                (margin, 52), cv2.FONT_HERSHEY_SIMPLEX, .57, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"[Q] Quit   [M] Morphology: {'ON' if morphology else 'OFF'}   [S] Save screenshot",
                (margin, 77), cv2.FONT_HERSHEY_SIMPLEX, .54, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "LEFT: plane(green) + detected points(red)      RIGHT: candidate object regions", 
                (margin, 101), cv2.FONT_HERSHEY_SIMPLEX, .48, (190, 190, 190), 1, cv2.LINE_AA)

    for i, region in enumerate(regions):
        def to_display(xy):
            x = x2 + int((xy[0] + .5) * scale)
            y = top + int((rows - 1 - xy[1] + .5) * scale)
            return x, y

        box = np.asarray([to_display(p) for p in region["box"]], dtype=np.int32)
        cv2.polylines(canvas, [box], True, (245, 245, 245), 1, cv2.LINE_AA)
        center = to_display(region["center"])
        annotation = f"{i + 1}: {region['angle']:.0f}deg" if region["angle"] is not None else f"{i + 1}: N/A"
        cv2.putText(canvas, annotation,
                    (min(center[0] + 4, width - 88), max(top + 15, center[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    import pyrealsense2 as rs

    table = load_table()
    print("Loaded calibration.json (READ ONLY). Baseline will not change.")
    print(f"Table occupancy grid: {table['cols']} x {table['rows']}, cell {table['grid_size'] * 100:.0f} cm")
    print("Try 3 separate rectangular objects with clear gaps, then rotate them.")
    print("Q = quit, M = morphology on/off, S = screenshot")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    rs_pc = rs.pointcloud()
    morphology = True
    try:
        for _ in range(20):
            pipeline.wait_for_frames()
        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth:
                continue
            vertices = depth_to_vertices(rs, rs_pc, depth)
            plane_grid, object_grid = make_grids(vertices, table)
            _, labels, regions = extract_regions(object_grid, table["table_mask"], morphology)
            image = render(table, plane_grid, object_grid, labels, regions, morphology)
            cv2.imshow("Desk v0.3: segmentation / orientation diagnostics", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("m"):
                morphology = not morphology
            if key == ord("s"):
                filename = Path(__file__).resolve().parent / "v03_debug.png"
                cv2.imwrite(str(filename), image)
                print(f"Saved screenshot: {filename}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
