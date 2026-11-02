"""Measure signed desk height without changing calibration or v04 detection.

Live: clear the desk, fix the camera/lighting, then press C to capture 5 seconds.
--demo saves a synthetic, headless report; it is NOT a sensor measurement.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

import desk_orientation_v04 as v04


ROOT = Path(__file__).resolve().parent
WINDOW = "Desk height diagnostic - C capture, Q quit"
MIN_FRAMES, MIN_COVERAGE = 10, .50
TOP, PANEL_W, PANEL_H, GAP = 150, 360, 380, 12
MAX_CELL_SAMPLES = 25_000_000


@dataclass
class FrameGrid:
    median_mm: np.ndarray
    spatial_span_mm: np.ndarray
    point_counts: np.ndarray
    object_support: np.ndarray


def frame_grid(vertices, table, cfg):
    """Full original ROI, no height clipping, no temporal filter, no hole filling.

    Each sufficiently sampled cell contributes one median per frame. Also retain
    its within-frame P90-P10 and v04-style object support BEFORE edge exclusion.
    Negative heights are important evidence, not invalid samples.
    """
    shape = (table["rows"], table["cols"])
    size = shape[0] * shape[1]
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    valid = (np.isfinite(vertices).all(axis=1)
             & (vertices[:, 2] > cfg.min_depth_m) & (vertices[:, 2] < cfg.max_depth_m))
    vertices = vertices[valid]
    relative = vertices - table["origin"]
    u, v = relative @ table["u_axis"], relative @ table["v_axis"]
    h = (vertices @ table["normal"] + table["d"]) * 1000
    x = np.floor((u - table["u_min"]) / table["grid_m"]).astype(np.int64)
    y = np.floor((v - table["v_min"]) / table["grid_m"]).astype(np.int64)
    inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    x, y, h = x[inside], y[inside], h[inside]
    inside = table["table_mask"][y, x] > 0
    flat, h = (y[inside] * shape[1] + x[inside]), h[inside]
    counts = np.bincount(flat, minlength=size)
    on_object = (h > cfg.min_height_m * 1000) & (h < cfg.max_height_m * 1000)
    objects = np.bincount(flat[on_object], minlength=size)
    support = ((objects >= cfg.min_cell_points)
               & (objects / np.maximum(counts, 1) >= cfg.support_fraction))
    sampled = np.flatnonzero(counts >= cfg.min_cell_points)
    quantiles = np.full((3, size), np.nan, dtype=np.float32)
    if len(sampled):
        ordered = h[np.lexsort((h, flat))]
        starts = np.cumsum(counts) - counts
        for i, q in enumerate([.1, .5, .9]):
            positions = starts[sampled] + (counts[sampled] - 1) * q
            lo, hi = np.floor(positions).astype(int), np.ceil(positions).astype(int)
            quantiles[i, sampled] = ordered[lo] + (ordered[hi] - ordered[lo]) * (positions - lo)
    return FrameGrid(quantiles[1].reshape(shape),
                     (quantiles[2] - quantiles[0]).reshape(shape),
                     counts.astype(np.uint32).reshape(shape), support.reshape(shape))


class Capture:
    def __init__(self, table):
        self.shape = table["table_mask"].shape
        self.frames = []
        self.times = []

    def add(self, frame, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0 or (self.times and elapsed < self.times[-1]):
            raise ValueError("Sample times must be finite, non-negative and monotonic")
        if any(getattr(frame, name).shape != self.shape for name in FrameGrid.__annotations__):
            raise ValueError("Frame shape does not match calibration")
        if (len(self.frames) + 1) * np.prod(self.shape) > MAX_CELL_SAMPLES:
            raise ValueError("Capture is too large; use a coarser grid or shorter duration")
        self.frames.append(frame)
        self.times.append(float(elapsed))

    def arrays(self):
        if not self.frames:
            raise ValueError("No captured frames")
        return {name: np.stack([getattr(frame, name) for frame in self.frames])
                for name in FrameGrid.__annotations__}

    def summarize(self, table, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0 or (self.times and elapsed < self.times[-1]):
            raise ValueError("Invalid capture duration")
        arrays = self.arrays()
        heights = arrays["median_mm"]
        valid_frames = np.isfinite(heights).sum(axis=0)
        n = len(self.frames)
        required = max(MIN_FRAMES, int(np.ceil(n * MIN_COVERAGE)))
        reliable = (valid_frames >= required) & (table["table_mask"] > 0)
        median, temporal, spatial, hits = [np.full(self.shape, np.nan, np.float32) for _ in range(4)]
        if np.any(reliable):
            low, mid, high = np.nanquantile(heights[:, reliable], [.1, .5, .9], axis=0)
            median[reliable], temporal[reliable] = mid, high - low
            spatial[reliable] = np.nanmedian(arrays["spatial_span_mm"][:, reliable], axis=0)
            hits[reliable] = (100 * arrays["object_support"][:, reliable].sum(axis=0)
                              / valid_frames[reliable])
        return dict(median_mm=median, temporal_span_mm=temporal, spatial_span_mm=spatial,
                    object_support_pct=hits, valid_frames=valid_frames,
                    coverage_pct=(100 * valid_frames / n).astype(np.float32),
                    reliable=reliable, n_frames=n, elapsed_seconds=float(elapsed),
                    required_frames=required)


def preview_result(frame):
    valid = np.isfinite(frame.median_mm)
    return dict(median_mm=frame.median_mm, temporal_span_mm=np.full(valid.shape, np.nan),
                spatial_span_mm=frame.spatial_span_mm,
                object_support_pct=np.where(valid, frame.object_support * 100., np.nan),
                coverage_pct=valid * 100., valid_frames=valid.astype(int), reliable=valid,
                n_frames=0, elapsed_seconds=0., required_frames=MIN_FRAMES, preview=True)


def panel_layout(table):
    scale = min((PANEL_W - 12) / table["cols"], (PANEL_H - 12) / table["rows"])
    dw, dh = round(table["cols"] * scale), round(table["rows"] * scale)
    return [(GAP + i * (PANEL_W + GAP) + (PANEL_W - dw) // 2,
             TOP + (PANEL_H - dh) // 2, dw, dh) for i in range(3)]


def clicked_cell(px, py, table):
    for x0, y0, dw, dh in panel_layout(table):
        if x0 <= px < x0 + dw and y0 <= py < y0 + dh:
            x = int((px - x0) * table["cols"] / dw)
            y = table["rows"] - 1 - int((py - y0) * table["rows"] / dh)
            if table["table_mask"][y, x]:
                return y, x
    return None


def colorize(values, low, high, palette):
    fraction = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1)
    colors = np.asarray(palette, dtype=float)
    stops = np.linspace(0, 1, len(colors))
    image = np.stack([np.interp(fraction, stops, colors[:, c]) for c in range(3)], axis=-1).astype(np.uint8)
    image[~np.isfinite(values)] = (65, 65, 65)
    return image


def cell_info(result, table, selected):
    if selected is None:
        return None
    y, x = selected
    info = dict(row=int(y), col=int(x), u_m=float(table["u_min"] + (x + .5) * table["grid_m"]),
                v_m=float(table["v_min"] + (y + .5) * table["grid_m"]),
                in_v04_active_roi=bool(table["active_mask"][y, x]),
                reliable=bool(result["reliable"][y, x]), valid_frames=int(result["valid_frames"][y, x]))
    for key in ["median_mm", "temporal_span_mm", "spatial_span_mm", "coverage_pct", "object_support_pct"]:
        value = float(result[key][y, x])
        info[key] = value if np.isfinite(value) else None
    return info


def render(table, result, cfg, status, selected=None, mode="LIVE"):
    canvas = np.full((740, 1128, 3), 24, np.uint8)
    preview = result.get("preview", False)

    def text(line, y, x=12, scale=.49, color=(225, 225, 225)):
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    text(f"Desk height diagnostic | {mode} | {status}", 26, scale=.58)
    text(f"Grid {cfg.grid_m * 1000:g}mm | >= {cfg.min_cell_points} points/cell/frame"
         f" | Object support reference: {cfg.min_height_m * 1000:g}-{cfg.max_height_m * 1000:g}mm, {cfg.support_fraction:.0%}", 51)
    text(f"Captured: {result['n_frames']} frames / {result['elapsed_seconds']:.2f}s"
         f" | Reliable: >= {MIN_FRAMES} frames AND >= {MIN_COVERAGE:.0%} coverage", 75)
    text("Keep desk EMPTY and camera/light FIXED. Full original ROI measured, INCLUDING the v04 excluded band.", 99)
    text("C: capture / recapture  |  Q: quit  |  After capture: click a cell, S: save selected result", 122)
    specs = [
        ("CURRENT height median (mm)" if preview else "Height median (mm)", result["median_mm"], -30, 30,
         [(220, 90, 40), (240, 240, 240), (30, 45, 225)], ["<= -30", "0", ">= +30 mm"]),
        ("Temporal spread (after capture)" if preview else "Temporal spread: P90-P10 (mm)", result["temporal_span_mm"], 0, 20,
         [(240, 240, 240), (35, 200, 245), (20, 35, 220)], ["0", "10", ">= 20 mm"]),
        ("CURRENT sampled / missing cells" if preview else "Observed frames / captured frames", result["coverage_pct"], 0, 100,
         [(90, 35, 90), (100, 160, 100), (45, 205, 65)], ["0%", "50%", "100%"]),
    ]
    for i, ((title, values, low, high, palette, ticks), (x0, y0, dw, dh)) in enumerate(zip(specs, panel_layout(table))):
        left = GAP + i * (PANEL_W + GAP)
        text(title, TOP - 12, x=left)
        panel = colorize(values, low, high, palette)
        panel[table["table_mask"] == 0] = 0
        cv2.polylines(panel, [table["poly_grid"]], True, (0, 225, 245), 1)
        if table["edge_margin_m"]:
            cv2.drawContours(panel, table["active_contours"], -1, (235, 220, 20), 1)
        canvas[y0:y0 + dh, x0:x0 + dw] = cv2.resize(cv2.flip(panel, 0), (dw, dh), interpolation=cv2.INTER_NEAREST)
        if selected is not None:
            y, x = selected
            center = (x0 + int((x + .5) * dw / table["cols"]),
                      y0 + int((table["rows"] - y - .5) * dh / table["rows"]))
            cv2.drawMarker(canvas, center, (0, 0, 0), cv2.MARKER_CROSS, 15, 3)
            cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 15, 1)
        bar = colorize(np.linspace(low, high, 330)[None, :], low, high, palette)
        canvas[540:552, left:left + 330] = np.repeat(bar, 12, axis=0)
        for tick, offset in zip(ticks, [0, 150, 245]):
            text(tick, 572, x=left + offset, scale=.43)
    info = cell_info(result, table, selected)
    if info:
        def fmt(key):
            return "N/A" if info[key] is None else f"{info[key]:.1f}"
        text(f"Cell row={info['row']}, col={info['col']} | {'ACTIVE' if info['in_v04_active_roi'] else 'EXCLUDED by v04'}"
             f" | Median {fmt('median_mm')}mm | Time P90-P10 {fmt('temporal_span_mm')}mm"
             f" | Observed {fmt('coverage_pct')}% ({info['valid_frames']} frames)", 601)
        quality = ("CURRENT frame only" if preview else
                   "Enough samples (not proof of accuracy)" if info["reliable"] else "INSUFFICIENT SAMPLES")
        text(f"Within-frame P90-P10: {fmt('spatial_span_mm')}mm"
             f" | Object-supported / observed frames: {fmt('object_support_pct')}%"
             f" | {quality}", 626, scale=.46)
    else:
        text("Click the far-corner patch in ANY panel to inspect the same cell across all three maps.", 601)
        text("Measurements alone cannot distinguish calibration error, depth bias and real objects.", 626)
    text("Gray: insufficient samples (NOT zero). Height: blue below / white near plane / red above.", 653)
    text("Yellow: original ROI | Cyan: v04 active ROI. Both sides of cyan are measured here. Display colors are clipped only.", 678, scale=.46)
    text("Time spread is variation of CELL MEDIANS, not sensor accuracy; also inspect coverage and within-frame spread.", 703, scale=.46)
    text("Preview shows ONE frame only; time statistics appear when capture finishes. No calibration writes."
         if preview else "DONE: report saved automatically. C starts a separate capture. No calibration writes or automatic correction.",
         728, scale=.46)
    return canvas


def json_grid(array):
    if np.issubdtype(array.dtype, np.floating):
        return np.where(np.isfinite(array), array, None).tolist()
    return array.tolist()


def save_report(base, table, cfg, capture, result, mode, source, selected=None):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    directory = base / ("height_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir()  # Never reuse/overwrite an existing result directory.
    summary = dict(mode=mode, calibration=source,
                   frame_count=result["n_frames"], elapsed_seconds=result["elapsed_seconds"],
                   minimum_observed_frames=result["required_frames"],
                   grid_mm=cfg.grid_m * 1000, edge_margin_mm=cfg.edge_margin_m * 1000,
                   min_cell_points=cfg.min_cell_points, support_fraction=cfg.support_fraction,
                   object_height_range_mm=[cfg.min_height_m * 1000, cfg.max_height_m * 1000],
                   depth_range_m=[cfg.min_depth_m, cfg.max_depth_m],
                   geometry={key: table[key].tolist() for key in ["origin", "normal", "u_axis", "v_axis", "polygon_uv"]},
                   plane_d=table["d"], u_min=table["u_min"], v_min=table["v_min"],
                   original_mask=json_grid(table["table_mask"] > 0), active_mask=json_grid(table["active_mask"] > 0),
                   maps={key: json_grid(value) for key, value in result.items() if isinstance(value, np.ndarray)},
                   selected_cell=cell_info(result, table, selected),
                   limitations=["Empty desk must be ensured by the operator; no semantic object recognition.",
                                "Full original ROI is measured. No height filtering, hole filling or temporal smoothing.",
                                "Median and temporal P90-P10 are computed over per-frame cell medians.",
                                "Spatial P90-P10 is the median within-frame height spread over observed frames.",
                                "Coverage is sufficiently sampled frames / received captured frames, not a camera accuracy measure.",
                                "Object support uses v04 point support before edge exclusion and temporal votes.",
                                "Insufficient samples are null/NaN, never zero. Maps require >=10 frames and >=50% coverage.",
                                "No automatic causal diagnosis, calibration update or depth correction."])
    with (directory / "summary.json").open("x", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, allow_nan=False)
    with (directory / "samples.npz").open("xb") as output:
        np.savez_compressed(output, times_s=np.asarray(capture.times), **capture.arrays())
    image = render(table, result, cfg, "DONE - report saved", selected, mode)
    if not cv2.imwrite(str(directory / "report.png"), image):
        raise OSError(f"Failed to save PNG; partial report directory: {directory}")
    print(f"Saved report: {directory.resolve()}")
    print(f"Reliable cells: {np.count_nonzero(result['reliable'])}/{np.count_nonzero(table['table_mask'])}")
    if not np.any(result["reliable"]):
        print("WARNING: no cells have enough samples; do not interpret this as a flat/clean desk.")
    return directory


@contextmanager
def camera_source():
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError("Install pyrealsense2, or use --demo for synthetic data") from error
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    started = False
    try:
        pipeline.start(config)
        started = True
        pointcloud = rs.pointcloud()
        for _ in range(20):
            pipeline.wait_for_frames(1000)

        def read():
            depth = pipeline.wait_for_frames(1000).get_depth_frame()
            if not depth:
                return np.empty((0, 3))
            points = pointcloud.calculate(depth)
            return np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3).copy()

        yield read
    finally:
        if started:
            pipeline.stop()


def run_live(args, cfg, table, source):
    status, selected, capture, result = "READY - press C", None, None, None
    start = None
    done = False
    gui_open = False

    def mouse(event, x, y, flags, userdata):
        nonlocal selected
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = clicked_cell(x, y, table)

    try:
        with camera_source() as read:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            gui_open = True
            cv2.setMouseCallback(WINDOW, mouse)
            while True:
                if not done:
                    frame = frame_grid(read(), table, cfg)
                    now = time.monotonic()
                    result = preview_result(frame)
                    if capture is not None:
                        elapsed = now - start
                        capture.add(frame, elapsed)
                        result["n_frames"], result["elapsed_seconds"] = len(capture.frames), elapsed
                        status = f"CAPTURING {min(elapsed, args.seconds):.1f}/{args.seconds:g}s (maps: current frame)"
                        if elapsed >= args.seconds:
                            result = capture.summarize(table, elapsed)
                            save_report(args.output_dir, table, cfg, capture, result, "LIVE", source, selected)
                            status, done = "DONE - click a cell; S saves", True
                cv2.imshow(WINDOW, render(table, result, cfg, status, selected))
                key = cv2.waitKey(30 if done else 1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("c") and (capture is None or done):
                    capture, start, done, selected = Capture(table), time.monotonic(), False, None
                elif key == ord("s") and done:
                    save_report(args.output_dir, table, cfg, capture, result, "LIVE", source, selected)
    finally:
        if gui_open:
            cv2.destroyAllWindows()


def synthetic_frame(table, cfg, rng, index):
    """Stable +20mm, temporal +/-12mm, -15mm and sparse-coverage patches."""
    y, x = np.nonzero(table["table_mask"])
    uv = np.column_stack([x + .5, y + .5]) * table["grid_m"]
    u, v = uv.T
    sparse = (u > 1.0) & (v < .22)
    keep = rng.random(len(x)) > np.where(sparse, .9, .08)
    uv, u, v = uv[keep], u[keep], v[keep]
    h = np.zeros(len(uv))
    h[(u > .12) & (u < .32) & (v > .55) & (v < .72)] = .020
    h[(u > .65) & (u < .90) & (v > .5) & (v < .7)] = .012 if index % 2 else -.012
    h[(u > .15) & (u < .35) & (v > .12) & (v < .28)] = -.015
    h += rng.normal(0, .0008, len(h))
    uv, h = np.repeat(uv, 5, axis=0), np.repeat(h, 5)
    h += rng.normal(0, .001, len(h))
    return (table["origin"] + uv[:, :1] * table["u_axis"]
            + uv[:, 1:] * table["v_axis"] + h[:, None] * table["normal"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--grid-mm", type=float, default=10)
    parser.add_argument("--edge-margin-mm", type=float, default=20, help="Draw v04 active boundary; does NOT exclude diagnostic samples")
    parser.add_argument("--min-height-mm", type=float, default=10, help="Reference for object support, NOT a height sampling cutoff")
    parser.add_argument("--min-cell-points", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "height_diagnostics")
    parser.add_argument("--demo", action="store_true", help="Save a synthetic report and exit; no camera or GUI")
    args = parser.parse_args(argv)
    cfg = v04.Settings(grid_m=args.grid_mm / 1000, edge_margin_m=args.edge_margin_mm / 1000,
                       min_height_m=args.min_height_mm / 1000, min_cell_points=args.min_cell_points)
    try:
        cfg.validate()
        if not np.isfinite(args.seconds) or not 1 <= args.seconds <= 30:
            raise ValueError("Capture duration must be between 1 and 30 seconds")
    except ValueError as error:
        parser.error(str(error))
    return args, cfg


def main(argv=None):
    args, cfg = parse_args(argv)
    try:
        if args.demo:
            table, source = v04.demo_table(cfg), {"kind": "synthetic", "seed": 4}
            capture, rng = Capture(table), np.random.default_rng(4)
            frames = round(args.seconds * 30)
            for i in range(frames):
                capture.add(frame_grid(synthetic_frame(table, cfg, rng, i), table, cfg), (i + 1) * args.seconds / frames)
            result = capture.summarize(table, args.seconds)
            save_report(args.output_dir, table, cfg, capture, result, "SYNTHETIC - NOT CAMERA DATA", source)
        else:
            data = args.calibration.read_bytes()
            table = v04.table_from_data(json.loads(data), cfg.grid_m, cfg.edge_margin_m)
            source = {"path": str(args.calibration.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
            print("Close v04 first. Clear the desk (or at least the suspect patch); keep camera/light fixed.")
            print("Press C when ready. Maps use FULL original ROI; no height clipping or calibration writes.")
            print(f"Capture {args.seconds:g}s, then auto-save to {args.output_dir.resolve()}. Click a cell; S saves another report.")
            run_live(args, cfg, table, source)
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}")
        print("Check calibration, output permissions, desktop display and camera ownership/USB. No calibration was changed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
