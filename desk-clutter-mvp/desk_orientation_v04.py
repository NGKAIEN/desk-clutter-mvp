"""Read-only desk calibration + three-state temporal grids + 2D PCA.

This diagnostic version does not calculate a tidiness score or track identities.
Run --help, or --demo for a synthetic example without a RealSense camera.
"""

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np


UNKNOWN, FREE, OCCUPIED = 0, 1, 2
ROOT = Path(__file__).resolve().parent
COLORS = [(220, 170, 70), (110, 210, 140), (190, 120, 220),
          (100, 220, 220), (220, 160, 180), (200, 190, 100)]


@dataclass(frozen=True)
class Settings:
    grid_m: float = 0.010
    edge_margin_m: float = 0.020
    min_height_m: float = 0.010
    max_height_m: float = 0.40
    plane_tolerance_m: float = 0.010
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0
    min_cell_points: int = 3
    support_fraction: float = 0.60
    window: int = 5
    votes: int = 3
    hold_seconds: float = 0.50
    min_region_cm2: float = 8.0
    min_axis_span_m: float = 0.05
    pca_ratio: float = 2.5
    max_display_regions: int = 16
    min_pca_area_cm2: float = 0.0  # Direction-only gate; zero preserves legacy behavior.

    def validate(self):
        numeric = [self.grid_m, self.edge_margin_m, self.min_height_m, self.max_height_m,
                   self.plane_tolerance_m, self.min_depth_m, self.max_depth_m,
                   self.support_fraction, self.hold_seconds, self.min_region_cm2,
                   self.min_axis_span_m, self.pca_ratio, self.min_pca_area_cm2]
        if not np.isfinite(numeric).all():
            raise ValueError("Settings must be finite")
        if not (0 < self.grid_m and 0 < self.min_height_m < self.max_height_m):
            raise ValueError("Require grid > 0 and 0 < minimum height < maximum height")
        if self.edge_margin_m < 0:
            raise ValueError("Edge margin cannot be negative")
        if not (0 < self.min_depth_m < self.max_depth_m and self.plane_tolerance_m > 0):
            raise ValueError("Invalid depth limits or plane tolerance")
        if self.min_cell_points < 1 or not 0.5 < self.support_fraction <= 1:
            raise ValueError("Require at least one point and 0.5 < support fraction <= 1")
        if self.window < 1 or not self.window / 2 < self.votes <= self.window:
            raise ValueError("Votes must be a strict majority of the window")
        if min(self.hold_seconds, self.min_region_cm2, self.min_axis_span_m) <= 0:
            raise ValueError("Hold time, minimum area and axis span must be positive")
        if self.pca_ratio <= 1 or self.max_display_regions < 1:
            raise ValueError("PCA ratio must exceed 1; display limit must be positive")
        if self.min_pca_area_cm2 < 0:
            raise ValueError("PCA minimum area must be non-negative (0 disables)")


def inset_mask(polygon, table_mask, low, grid_m, margin_m):
    """Keep entire cells inside a metric inset of a convex calibrated polygon.

    Each edge contributes an inward half-plane. The extra half-cell projection
    keeps a retained cell's corners out of the excluded strip, including on
    diagonal edges. Zero margin deliberately restores the original raster ROI.
    """
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("Edge margin must be finite and non-negative")
    if margin_m == 0:
        return table_mask.copy()
    following = np.roll(polygon, -1, axis=0)
    edges = following - polygon
    twice_area = np.sum(polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0])
    lengths = np.linalg.norm(edges, axis=1)
    if abs(twice_area) < 1e-12 or np.any(lengths < 1e-12):
        raise ValueError("Cannot inset a degenerate ROI")
    winding = np.sign(twice_area)
    y, x = np.indices(table_mask.shape)
    u = low[0] + (x + .5) * grid_m
    v = low[1] + (y + .5) * grid_m
    keep = table_mask > 0
    for start, edge, length in zip(polygon, edges, lengths):
        inward = winding * np.array([-edge[1], edge[0]]) / length
        distance = (u - start[0]) * inward[0] + (v - start[1]) * inward[1]
        half_cell = .5 * grid_m * np.abs(inward).sum()
        keep &= distance >= margin_m + half_cell - 1e-10
    if not np.any(keep):
        raise ValueError("Edge margin leaves no active ROI; reduce --edge-margin-mm")
    return np.uint8(keep) * 255


def table_from_data(data, grid_m, edge_margin_m=0.020):
    """Build an independent grid; never update calibration.json or its baseline."""
    table = {key: np.asarray(data[key], dtype=np.float64)
             for key in ("origin", "normal", "u_axis", "v_axis", "polygon_uv")}
    table["d"] = float(data["d"])
    for key in ("origin", "normal", "u_axis", "v_axis"):
        if table[key].shape != (3,) or not np.isfinite(table[key]).all():
            raise ValueError(f"Invalid calibration field: {key}")
    basis = np.stack([table["u_axis"], table["v_axis"], table["normal"]])
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-3):
        raise ValueError("Calibration axes must be orthonormal")
    poly = table["polygon_uv"]
    if (poly.shape != (4, 2) or not np.isfinite(poly).all()
            or not np.isfinite(table["d"])
            or not cv2.isContourConvex(poly.astype(np.float32))):
        raise ValueError("Calibration needs a finite, convex four-corner ROI")
    if not np.isfinite(grid_m) or grid_m <= 0:
        raise ValueError("Grid size must be positive and finite")
    low, high = poly.min(axis=0), poly.max(axis=0)
    size = np.ceil((high - low) / grid_m).astype(int) + 1
    cols, rows = map(int, size)
    if min(rows, cols) < 2 or rows * cols > 1_000_000:
        raise ValueError("ROI/grid combination is empty or too large")
    poly_grid = np.floor((poly - low) / grid_m).astype(np.int32)
    table_mask = np.zeros((rows, cols), np.uint8)
    cv2.fillPoly(table_mask, [poly_grid], 255)
    active_mask = inset_mask(poly, table_mask, low, grid_m, edge_margin_m)
    excluded_mask = (table_mask > 0) & (active_mask == 0)
    active_contours, _ = cv2.findContours(active_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = np.roll(poly, -1, axis=0) - poly
    edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    table.update(grid_m=grid_m, u_min=low[0], v_min=low[1], rows=rows,
                 cols=cols, table_mask=table_mask, poly_grid=poly_grid,
                 active_mask=active_mask, excluded_mask=excluded_mask,
                 active_contours=active_contours, edge_margin_m=edge_margin_m,
                 reference_deg=float(np.degrees(np.arctan2(edge[1], edge[0])) % 180))
    return table


def load_table(path, grid_m, edge_margin_m=0.020):
    with Path(path).open(encoding="utf-8") as source:
        return table_from_data(json.load(source), grid_m, edge_margin_m)


@dataclass
class Observation:
    raw: np.ndarray
    evidence: np.ndarray
    point_counts: np.ndarray


def make_observation(vertices, table, cfg):
    """Classify point support, keeping missing/conflicting data UNKNOWN.

    raw retains the original ROI for excluded-edge diagnostics. evidence also
    requires point support and an active-ROI cell; it never fills excluded cells.
    """
    shape = (table["rows"], table["cols"])
    raw = np.zeros(shape, np.uint8)
    evidence = np.zeros(shape, np.uint8)
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    good = (np.isfinite(vertices).all(axis=1)
            & (vertices[:, 2] > cfg.min_depth_m)
            & (vertices[:, 2] < cfg.max_depth_m))
    vertices = vertices[good]
    relative = vertices - table["origin"]
    u, v = relative @ table["u_axis"], relative @ table["v_axis"]
    h = vertices @ table["normal"] + table["d"]
    ix = np.floor((u - table["u_min"]) / table["grid_m"]).astype(np.int64)
    iy = np.floor((v - table["v_min"]) / table["grid_m"]).astype(np.int64)
    inside = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    ix, iy, h = ix[inside], iy[inside], h[inside]
    inside = table["table_mask"][iy, ix] > 0
    ix, iy, h = ix[inside], iy[inside], h[inside]
    flat = iy * shape[1] + ix
    on_object = (h > cfg.min_height_m) & (h < cfg.max_height_m)
    # Mutually exclusive votes, including when the user tries a 3 mm threshold.
    on_free = (np.abs(h) < cfg.plane_tolerance_m) & ~on_object
    total = np.bincount(flat, minlength=raw.size).reshape(shape)
    objects = np.bincount(flat[on_object], minlength=raw.size).reshape(shape)
    free = np.bincount(flat[on_free], minlength=raw.size).reshape(shape)
    raw[free > 0] = FREE
    raw[objects > 0] = OCCUPIED
    denominator = np.maximum(total, 1)
    evidence[(free >= cfg.min_cell_points)
             & (free / denominator >= cfg.support_fraction)] = FREE
    evidence[(objects >= cfg.min_cell_points)
             & (objects / denominator >= cfg.support_fraction)] = OCCUPIED
    evidence[table["active_mask"] == 0] = UNKNOWN
    return Observation(raw, evidence, total)


class TemporalGrid:
    """Majority of recent *frames*, not of only the valid observations.

    UNKNOWN casts neither a free nor an occupied vote. The last state may be
    held for hold_seconds after the last supporting observation, then expires.
    Old votes also expire by wall time, so a stalled stream cannot retain them.
    """

    def __init__(self, shape, cfg):
        cfg.validate()
        self.cfg = cfg
        self.history = deque(maxlen=cfg.window)
        self.state = np.zeros(shape, np.uint8)
        self.last_seen = np.full(shape, -np.inf)
        self.last_time = None

    def reset(self):
        self.history.clear()
        self.state.fill(UNKNOWN)
        self.last_seen.fill(-np.inf)
        self.last_time = None

    def update(self, evidence, now):
        if evidence.shape != self.state.shape or not np.isin(evidence, [0, 1, 2]).all():
            raise ValueError("Invalid temporal observation")
        if not np.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError("Temporal timestamps must be finite and monotonic")
        self.last_time = now
        self.history.append((now, evidence.copy()))
        while self.history and now - self.history[0][0] > self.cfg.hold_seconds:
            self.history.popleft()
        self.state[now - self.last_seen > self.cfg.hold_seconds] = UNKNOWN
        frames = np.stack([item[1] for item in self.history])
        self.state[np.sum(frames == FREE, axis=0) >= self.cfg.votes] = FREE
        self.state[np.sum(frames == OCCUPIED, axis=0) >= self.cfg.votes] = OCCUPIED
        supported = (self.state != UNKNOWN) & (self.state == evidence)
        self.last_seen[supported] = now
        self.state[now - self.last_seen > self.cfg.hold_seconds] = UNKNOWN
        held = (self.state != UNKNOWN) & (self.state != evidence)
        return self.state.copy(), held


def pca_direction(points_uv, cfg):
    points = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or not np.isfinite(points).all():
        return None, None, 0.0, 0.0, "too few points"
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    minor, major = np.maximum(values, 0)
    axis = vectors[:, 1]
    ratio = float(major / max(minor, 1e-12))
    span = float(np.ptp(centered @ axis))
    if span < cfg.min_axis_span_m:
        return None, axis, ratio, span, "too small"
    if ratio < cfg.pca_ratio:
        return None, axis, ratio, span, "no dominant axis"
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180)
    return angle, axis, ratio, span, "ok"


def extract_regions(state, table, cfg):
    # Mask again here so even stale/injected border states cannot reach PCA.
    mask = np.uint8((state == OCCUPIED) & (table["active_mask"] > 0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_cells = max(3, int(np.ceil(cfg.min_region_cm2 / (table["grid_m"] * 100) ** 2)))
    regions = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_cells:
            continue
        y, x = np.nonzero(labels == label)
        centers = np.column_stack([x + 0.5, y + 0.5])
        points_uv = centers * table["grid_m"] + [table["u_min"], table["v_min"]]
        absolute, axis, ratio, span, reason = pca_direction(points_uv, cfg)
        area_cm2 = area * (table["grid_m"] * 100) ** 2
        # Suppress direction only, never the region, its cells, or occupancy.
        if area_cm2 < cfg.min_pca_area_cm2:
            absolute, reason = None, "area below PCA minimum"
        angle = None if absolute is None else (absolute - table["reference_deg"]) % 180
        regions.append(dict(label=label, area_cells=area,
                            area_cm2=area_cm2,
                            center=centers.mean(axis=0), axis=axis, angle=angle,
                            ratio=ratio, span_m=span, reason=reason,
                            box=cv2.boxPoints(cv2.minAreaRect(centers.astype(np.float32)))))
    regions.sort(key=lambda item: item["area_cells"], reverse=True)
    return labels, regions


def active_percent(mask, table):
    """All displayed occupancy/coverage percentages use only the active ROI."""
    active = table["active_mask"] > 0
    return 100 * np.count_nonzero(mask & active) / np.count_nonzero(active)


def state_image(state, table):
    image = np.zeros((*state.shape, 3), np.uint8)
    active = table["active_mask"] > 0
    image[active] = (48, 48, 48)
    image[(state == FREE) & active] = (35, 155, 45)
    image[(state == OCCUPIED) & active] = (45, 60, 230)
    # Exclusion is not free space or missing depth. Keep its display static.
    y, x = np.indices(state.shape)
    image[table["excluded_mask"]] = (45, 32, 48)
    image[table["excluded_mask"] & ((x + y) % 6 < 2)] = (95, 65, 95)
    # Held states retain their normal color; only the Held counter changes.
    return image


def render(table, observation, stable, held, labels, regions, cfg,
           temporal=True, mode="LIVE", paused=False):
    margin, panel_w, panel_h, top = 12, 360, 400, 156
    width, height = panel_w * 3 + margin * 4, top + panel_h + 118
    canvas = np.full((height, width, 3), (24, 24, 24), np.uint8)

    def text(line, y, scale=0.51, color=(225, 225, 225), x=margin):
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    roi = table["active_mask"] > 0
    total = int(roi.sum())
    original_total = np.count_nonzero(table["table_mask"])
    percent = lambda mask: active_percent(mask, table)
    valid_axes = sum(r["angle"] is not None for r in regions)
    title = f"Desk v0.4 | {mode}{' | PAUSED' if paused else ''}"
    title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, .60, 1)[0][0]
    text(title, 25, .60 * min(1, (width - 335) / max(title_width, 1)))
    badge_x = width - 300
    cv2.rectangle(canvas, (badge_x, 5), (width - margin, 33),
                  (35, 95, 35) if temporal else (30, 85, 165), cv2.FILLED)
    text(f"Temporal {'ON' if temporal else 'OFF'} | T -> {'OFF' if temporal else 'ON'}",
         25, .53, x=badge_x + 9)
    text(f"Grid {cfg.grid_m * 1000:g}mm | Height {cfg.min_height_m * 1000:g}-{cfg.max_height_m * 1000:g}mm"
         f" | Cell support >= {cfg.min_cell_points} points, {cfg.support_fraction:.0%}"
         f" | Time votes: {cfg.votes}/{cfg.window}, hold {cfg.hold_seconds:g}s", 49)
    edge_hits = np.count_nonzero((observation.raw == OCCUPIED) & table["excluded_mask"])
    text(f"Edge inset: {table['edge_margin_m'] * 1000:g}mm | Active: {total}/{original_total} cells"
         f" ({100 * total / original_total:.1f}%) | Raw object cells in excluded band: {edge_hits} (ignored)", 73)
    text(f"Classified now: {percent(observation.evidence != UNKNOWN):.1f}%"
         f" | Stable known: {percent(stable != UNKNOWN):.1f}%"
         f" | Held: {percent(held):.1f}% | Regions: {len(regions)} | Valid PCA axes: {valid_axes}", 97)
    if not np.any(observation.evidence[roi]):
        text("No current classified evidence in ACTIVE ROI - check depth, lighting and calibration.", 121,
             color=(100, 180, 255))
    else:
        text(f"Occupied / ACTIVE ROI: raw {percent(observation.raw == OCCUPIED):.1f}%"
             f" | stable {percent(stable == OCCUPIED):.1f}%  (diagnostic only; NOT a tidiness score)", 121)

    panels = [state_image(observation.raw, table), state_image(stable, table),
              state_image(stable, table)]
    panels[2][(stable == OCCUPIED) & roi] = (90, 90, 90)
    shown = regions[:cfg.max_display_regions]
    for index, region in enumerate(shown):
        panels[2][labels == region["label"]] = COLORS[index % len(COLORS)]
    titles = ["RAW: height / active ROI", "STABLE: support + time votes",
              "REGIONS: PCA axes / table edge"]
    if not temporal:
        titles[1] = "FRAME: support filter only"
    scale = min((panel_w - 12) / table["cols"], (panel_h - 12) / table["rows"])
    dw, dh = max(1, round(table["cols"] * scale)), max(1, round(table["rows"] * scale))
    offsets = []
    for index, panel in enumerate(panels):
        x0 = margin + index * (panel_w + margin) + (panel_w - dw) // 2
        y0 = top + (panel_h - dh) // 2
        offsets.append((x0, y0))
        text(titles[index], top - 15, .49, x=margin + index * (panel_w + margin))
        cv2.polylines(panel, [table["poly_grid"]], True, (0, 225, 245), 1)
        if table["edge_margin_m"] > 0:
            cv2.drawContours(panel, table["active_contours"], -1, (235, 220, 20), 1)
        canvas[y0:y0 + dh, x0:x0 + dw] = cv2.resize(cv2.flip(panel, 0), (dw, dh),
                                                        interpolation=cv2.INTER_NEAREST)

    x0, y0 = offsets[2]

    def pixel(point):
        return (x0 + int(point[0] * dw / table["cols"]),
                y0 + int((table["rows"] - point[1]) * dh / table["rows"]))

    for index, region in enumerate(shown):
        box = np.asarray([pixel(p) for p in region["box"]], np.int32)
        cv2.polylines(canvas, [box], True, (210, 210, 210), 1)
        center = region["center"]
        if region["angle"] is not None:
            half = region["axis"] * region["span_m"] / (2 * table["grid_m"])
            cv2.line(canvas, pixel(center - half), pixel(center + half), (255, 255, 255), 2)
        angle = "N/A" if region["angle"] is None else f"{region['angle']:.0f}deg"
        ratio = f"{min(region['ratio'], 999):.1f}"
        label = f"{index + 1}: {angle} r={ratio}"
        position = pixel(center)
        label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .40, 1)[0][0]
        position = (max(x0, min(position[0] + 5, x0 + dw - label_w)),
                    max(y0 + 14, min(position[1] - 7, y0 + dh - 4)))
        cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    .40, (255, 255, 255), 1, cv2.LINE_AA)
    footer = top + panel_h
    text("Green: desk | Red: object | Gray: unknown | Purple hatch: excluded (NOT empty desk)", footer + 23, .49)
    text("Yellow: original ROI | Cyan: active ROI | Excluded band is omitted from statistics AND PCA.", footer + 45, .47)
    text("PCA: 0..180deg / desk edge; r is shape, not accuracy. IDs: frame-local. Held: counter only, no dimming.", footer + 67, .47)
    text("[T] temporal ON/OFF  [R] reset history  [SPACE] pause  [S] screenshot  [N] next demo angle  [Q] quit", footer + 94, .47)
    return canvas


def demo_table(cfg):
    return table_from_data(dict(origin=[-.6, .4, 1], normal=[0, 0, -1], d=1,
                                u_axis=[1, 0, 0], v_axis=[0, -1, 0],
                                polygon_uv=[[0, 0], [1.2, 0], [1.2, .8], [0, .8]]),
                           cfg.grid_m, cfg.edge_margin_m)


def demo_vertices(table, cfg, rng, angle=30):
    """Synthetic rectangle + square with dropout/outliers, NOT recorded sensor data."""
    rows, cols, g = table["rows"], table["cols"], table["grid_m"]
    mask = np.zeros((rows, cols), np.uint8)
    for center, size, theta in [((.38 / g, .4 / g), (.30 / g, .10 / g), angle),
                                ((.95 / g, .4 / g), (.15 / g, .15 / g), 0)]:
        cv2.fillConvexPoly(mask, np.rint(cv2.boxPoints((center, size, theta))).astype(np.int32), 1)
    y, x = np.nonzero(table["table_mask"])
    keep = rng.random(len(x)) > .10
    x, y = x[keep], y[keep]
    x, y = np.repeat(x, 5), np.repeat(y, 5)
    uv = (np.column_stack([x, y]) + .5 + rng.uniform(-.3, .3, (len(x), 2))) * g
    h = np.where(mask[y, x] > 0, .045, 0.0) + rng.normal(0, .003, len(x))
    h[rng.random(len(x)) < .02] = .025
    return (table["origin"] + uv[:, :1] * table["u_axis"]
            + uv[:, 1:] * table["v_axis"] + h[:, None] * table["normal"])


def save_image(path, image):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Will not overwrite existing image: {path}")
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")
    print(f"Saved: {path.resolve()}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    parser.add_argument("--grid-mm", type=float, default=10)
    parser.add_argument("--edge-margin-mm", type=float, default=20,
                        help="Exclude this width along all calibrated edges; 0 disables (default: 20)")
    parser.add_argument("--min-height-mm", type=float, default=10)
    parser.add_argument("--max-height-mm", type=float, default=400)
    parser.add_argument("--min-cell-points", type=int, default=3)
    parser.add_argument("--support-fraction", type=float, default=.60)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--hold-seconds", type=float, default=.50)
    parser.add_argument("--min-region-cm2", type=float, default=8)
    parser.add_argument("--pca-ratio", type=float, default=2.5)
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Synthetic data; no camera/calibration needed")
    parser.add_argument("--demo-angle", type=int, choices=[0, 30, 90], default=30)
    parser.add_argument("--snapshot", type=Path, help="With --demo: save one warmed-up image, no GUI")
    args = parser.parse_args(argv)
    if args.snapshot and not args.demo:
        parser.error("--snapshot requires --demo; live screenshots use S")
    cfg = Settings(grid_m=args.grid_mm / 1000, edge_margin_m=args.edge_margin_mm / 1000,
                   min_height_m=args.min_height_mm / 1000,
                   max_height_m=args.max_height_mm / 1000, min_cell_points=args.min_cell_points,
                   support_fraction=args.support_fraction, window=args.window, votes=args.votes,
                   hold_seconds=args.hold_seconds, min_region_cm2=args.min_region_cm2,
                   pca_ratio=args.pca_ratio)
    try:
        cfg.validate()
    except ValueError as error:
        parser.error(str(error))
    return args, cfg


def run(args, cfg):
    table = demo_table(cfg) if args.demo else load_table(args.calibration, cfg.grid_m, cfg.edge_margin_m)
    temporal = not args.no_temporal
    stabilizer = TemporalGrid((table["rows"], table["cols"]), cfg)
    rng = np.random.default_rng(4)
    pipeline = None
    gui_open = False
    paused = False
    frame_index = 0
    angle = args.demo_angle
    mode = "SYNTHETIC DEMO (not sensor validation)" if args.demo else "LIVE"
    print(f"{mode}\nSettings: {cfg}")
    print(f"Grid: {table['cols']} x {table['rows']} | reference: longest calibrated desk edge")
    print(f"Edge inset: {cfg.edge_margin_m * 1000:g}mm | active cells: "
          f"{np.count_nonzero(table['active_mask'])}/{np.count_nonzero(table['table_mask'])}")
    if not args.demo:
        print(f"Read-only calibration: {args.calibration.resolve()} (grid uses --grid-mm)")
    print("All panels/statistics/PCA use active ROI only. Purple edge band is EXCLUDED, not free space.")
    print("Raw=any point; middle=point support + optional time votes; right=PCA. No tidiness score.")
    print(f"Temporal {'ON' if temporal else 'OFF'}; T will turn it {'OFF' if temporal else 'ON'}.")
    print("T temporal | R reset | SPACE pause | S screenshot | N demo angle | Q quit")
    try:
        if not args.demo:
            try:
                import pyrealsense2 as rs
            except ImportError as error:
                raise RuntimeError("Install pyrealsense2 in the project environment, or use --demo") from error
            pipeline = rs.pipeline()
            stream_config = rs.config()
            stream_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # Assign only after start succeeds, for reliable cleanup on failure.
            active_pipeline = pipeline
            pipeline = None
            active_pipeline.start(stream_config)
            pipeline = active_pipeline
            pointcloud = rs.pointcloud()
            for _ in range(20):
                pipeline.wait_for_frames()
        while True:
            if not paused:
                if args.demo:
                    vertices = demo_vertices(table, cfg, rng, angle)
                    now = frame_index / 30 if args.snapshot else time.monotonic()
                else:
                    depth = pipeline.wait_for_frames().get_depth_frame()
                    if depth:
                        points = pointcloud.calculate(depth)
                        vertices = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3).copy()
                    else:
                        vertices = np.empty((0, 3), dtype=np.float32)
                    now = time.monotonic()
                observation = make_observation(vertices, table, cfg)
                if temporal:
                    stable, held = stabilizer.update(observation.evidence, now)
                else:
                    stable = observation.evidence.copy()
                    held = np.zeros(stable.shape, bool)
                labels, regions = extract_regions(stable, table, cfg)
                frame_index += 1
            image = render(table, observation, stable, held, labels, regions, cfg,
                           temporal, mode + (f" | {angle}deg" if args.demo else ""), paused)
            if args.snapshot:
                if frame_index >= max(18, cfg.window):
                    save_image(args.snapshot, image)
                    return
                continue
            cv2.imshow("Desk v0.4 - stable grid / PCA", image)
            gui_open = True
            key = cv2.waitKey(30 if args.demo or paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("Desk v0.4 - stable grid / PCA", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                paused = not paused
                if not paused:
                    stabilizer.reset()
            elif key == ord("t"):
                temporal = not temporal
                stabilizer.reset()
                paused = False
            elif key == ord("r"):
                stabilizer.reset()
                paused = False
            elif key == ord("n") and args.demo:
                angle = {0: 30, 30: 90, 90: 0}[angle]
                stabilizer.reset()
                paused = False
            elif key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_image(ROOT / f"v04_{stamp}.png", image)
    finally:
        if pipeline is not None:
            pipeline.stop()
        if gui_open:
            cv2.destroyAllWindows()


def main(argv=None):
    args, cfg = parse_args(argv)
    try:
        run(args, cfg)
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}")
        if not args.demo:
            print("Check calibration, camera permissions/USB and whether another program owns the camera.")
        print("Headless check: python desk_orientation_v04.py --demo --snapshot /tmp/desk_v04_demo.png")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
