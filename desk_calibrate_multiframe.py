"""Refine an existing desk plane from a multi-frame EMPTY-desk capture.

C: capture a candidate; V: collect a separate validation; S: save view; Q: quit.
Never overwrites or automatically activates calibration.json. --demo is synthetic.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

import desk_height_diagnostic as diag
import desk_orientation_v04 as v04


ROOT = Path(__file__).resolve().parent
WINDOW = "Multi-frame desk calibration - C fit, V validate, Q quit"
MIN_FRAMES = 30
MIN_COVERAGE = .80
MAX_HEIGHT_M = .030
MAX_TIME_SPAN_MM = 20.
INLIER_M = .006
HUBER_M = .003
MAX_TILT_DEG = 3.


def grid_uv(table):
    y, x = np.indices(table["table_mask"].shape)
    return np.stack([table["u_min"] + (x + .5) * table["grid_m"],
                     table["v_min"] + (y + .5) * table["grid_m"]], axis=-1)


def require_spread(mask, table, fraction=.50):
    """Reject too few samples or samples concentrated in just one desk patch."""
    active = table["active_mask"] > 0
    count, total = np.count_nonzero(mask), np.count_nonzero(active)
    if count < max(100, int(np.ceil(total * fraction))):
        raise ValueError("Too few well-observed desk cells; clear desk and improve coverage/lighting")
    uv = grid_uv(table)
    full, used = uv[active], uv[mask]
    if np.any(np.ptp(used, axis=0) < .70 * np.ptp(full, axis=0)):
        raise ValueError("Fit cells do not span enough of the desk")
    middle = np.median(full, axis=0)
    quadrant = (uv[..., 0] > middle[0]).astype(int) + 2 * (uv[..., 1] > middle[1])
    for q in range(4):
        required = max(10, int(np.ceil(.25 * np.count_nonzero(active & (quadrant == q)))))
        if np.count_nonzero(mask & (quadrant == q)) < required:
            raise ValueError("Fit lacks support in one desk quadrant; do not calibrate from a small patch")


def fit_plane(result, table):
    """Fit h=a*u+b*v+c to per-cell temporal medians, in meters.

    Each cell has equal weight (not each raw depth point). Deterministic RANSAC
    initializes the fit, followed by bounded Huber refinement of the consensus.
    This is a small correction of an existing plane, not a new scene detector.
    """
    if result["n_frames"] < MIN_FRAMES:
        raise ValueError(f"Need at least {MIN_FRAMES} captured frames")
    h = result["median_mm"] / 1000
    mask = ((table["active_mask"] > 0) & result["reliable"]
            & (result["coverage_pct"] >= MIN_COVERAGE * 100)
            & np.isfinite(h) & (np.abs(h) <= MAX_HEIGHT_M)
            & (result["temporal_span_mm"] <= MAX_TIME_SPAN_MM))
    require_spread(mask, table)
    uv = grid_uv(table)[mask]
    center, scale = uv.mean(axis=0), np.ptp(uv, axis=0)
    X = np.column_stack([(uv - center) / scale, np.ones(len(uv))])
    z = h[mask]
    rng = np.random.default_rng(4)
    best, best_score = None, (-1, -np.inf)
    for _ in range(256):
        indices = rng.choice(len(z), 3, replace=False)
        if np.linalg.cond(X[indices]) > 1000:
            continue
        trial = np.linalg.solve(X[indices], z[indices])
        residual = np.abs(z - X @ trial)
        inliers = residual <= INLIER_M
        score = (int(inliers.sum()), -float(np.median(residual[inliers])))
        if score > best_score:
            best, best_score = trial, score
    if best is None or best_score[0] < .65 * len(z):
        raise ValueError("No dominant near-plane consensus; clear desk/check calibration and lighting")
    for _ in range(15):
        residual = np.abs(z - X @ best)
        weights = np.where(residual <= INLIER_M,
                           np.minimum(1, HUBER_M / np.maximum(residual, 1e-12)), 0)
        if np.count_nonzero(weights) < .65 * len(z):
            raise ValueError("Plane consensus became unstable")
        root = np.sqrt(weights)
        updated = np.linalg.lstsq(X * root[:, None], z * root, rcond=None)[0]
        if np.linalg.norm(updated - best) < 1e-9:
            best = updated
            break
        best = updated
    residual = z - X @ best
    inliers = np.abs(residual) <= INLIER_M
    if np.count_nonzero(inliers) < .65 * len(z):
        raise ValueError("Too many non-plane cells remain")
    inlier_mask = np.zeros(mask.shape, bool)
    inlier_mask[mask] = inliers
    require_spread(inlier_mask, table, fraction=.40)
    a, b = best[:2] / scale
    c = best[2] - np.dot([a, b], center)
    beta = np.array([a, b, c])
    tilt = float(np.degrees(np.arctan(np.linalg.norm(beta[:2]))))
    corner_heights = table["polygon_uv"] @ beta[:2] + c
    if tilt > MAX_TILT_DEG or np.max(np.abs(corner_heights)) > MAX_HEIGHT_M:
        raise ValueError("Correction too large for safe refinement; redo full four-corner calibration")
    stats = dict(method="cell temporal median + RANSAC + bounded Huber",
                 selected_cells=int(mask.sum()), inlier_cells=int(inliers.sum()),
                 active_cells=int(np.count_nonzero(table["active_mask"])),
                 correction_a_b_c_m=beta.tolist(), tilt_degrees=tilt,
                 corner_height_corrections_mm=(corner_heights * 1000).tolist(),
                 inlier_abs_residual_p90_mm=float(np.quantile(np.abs(residual[inliers]), .9) * 1000),
                 selected_abs_residual_p90_mm=float(np.quantile(np.abs(residual), .9) * 1000),
                 gates=dict(min_frames=MIN_FRAMES, min_coverage=MIN_COVERAGE,
                            max_abs_height_mm=MAX_HEIGHT_M * 1000, max_time_span_mm=MAX_TIME_SPAN_MM,
                            inlier_mm=INLIER_M * 1000, huber_mm=HUBER_M * 1000))
    return beta, stats, inlier_mask


def candidate_calibration(original, table, beta, source_hash, stats):
    """Update the complete orthonormal frame; preserve depth-camera corner rays."""
    a, b, c = np.asarray(beta, float)
    n, u, v, o = [table[key] for key in ["normal", "u_axis", "v_axis", "origin"]]
    raw_n = n - a * u - b * v
    length = np.linalg.norm(raw_n)
    new_n = raw_n / length
    new_d = float((table["d"] + a * np.dot(u, o) + b * np.dot(v, o) - c) / length)
    if np.dot(new_n, n) <= 0 or new_d <= 0:
        raise ValueError("Refined plane faces away from the camera")
    new_o = o - (np.dot(new_n, o) + new_d) * new_n
    new_u = u - np.dot(u, new_n) * new_n
    new_u /= np.linalg.norm(new_u)
    handedness = np.sign(np.dot(np.cross(u, v), n))
    new_v = handedness * np.cross(new_n, new_u)
    # Existing origin may be slightly off its plane; UV is unchanged by this correction.
    old_o_on_plane = o - (np.dot(n, o) + table["d"]) * n
    poly = table["polygon_uv"]
    corners = old_o_on_plane + poly[:, :1] * u + poly[:, 1:] * v
    denominators = corners @ new_n
    if np.any(np.abs(denominators) < 1e-6):
        raise ValueError("A corner ray is parallel to the candidate plane")
    factors = -new_d / denominators
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("A candidate corner lies behind the camera")
    new_corners = corners * factors[:, None]
    new_poly = np.column_stack([(new_corners - new_o) @ new_u, (new_corners - new_o) @ new_v])
    result = deepcopy(original)
    result.update(normal=new_n.tolist(), d=new_d, origin=new_o.tolist(),
                  u_axis=new_u.tolist(), v_axis=new_v.tolist(), polygon_uv=new_poly.tolist())
    had_baseline = "baseline" in result
    result.pop("baseline", None)  # Occupancy baseline belongs to the old plane/ROI.
    result["multiframe_refinement"] = dict(source_sha256=source_hash, candidate_only=True,
                                           independent_validation_required=True,
                                           corner_policy="preserve depth-camera rays, intersect new plane",
                                           old_baseline_removed=had_baseline, fit=stats)
    # Existing consumers must accept the candidate without changing their algorithms.
    v04.table_from_data(result, table["grid_m"], table["edge_margin_m"])
    return result


def corrected_fit_result(result, table, beta):
    """Training display ONLY: UV cell-center approximation, not validation."""
    normalizer = np.sqrt(1 + np.dot(beta[:2], beta[:2]))
    trend_mm = (grid_uv(table) @ beta[:2] + beta[2]) * 1000
    corrected = dict(result)
    corrected["median_mm"] = (result["median_mm"] - trend_mm) / normalizer
    corrected["temporal_span_mm"] = result["temporal_span_mm"] / normalizer
    corrected["object_support_pct"] = np.full(trend_mm.shape, np.nan)
    return corrected


def measurement_table(table, candidate):
    """Same original UV cells, but exact candidate signed height of every point.

    This is a comparison view, NOT a valid calibration frame. Never serialize it
    as calibration. Same raw points and same UV groups avoid ROI-change bias.
    """
    return dict(table, normal=np.asarray(candidate["normal"]), d=candidate["d"])


def comparison(before, after, table):
    active = table["active_mask"] > 0
    common = active & before["reliable"] & after["reliable"]
    metrics = dict(common_reliable_cells=int(common.sum()), active_cells=int(active.sum()),
                   comparison_scope="same original active ROI and same captured frames")
    for name, result in [("original", before), ("candidate", after)]:
        if not common.any():
            metrics[name] = None
            continue
        heights = result["median_mm"][common]
        supported = result["object_support_pct"][common]
        supported = supported[np.isfinite(supported)]
        metrics[name] = dict(median_abs_height_mm=float(np.median(np.abs(heights))),
                             p90_abs_height_mm=float(np.quantile(np.abs(heights), .9)),
                             median_temporal_span_mm=float(np.median(result["temporal_span_mm"][common])),
                             mean_object_support_pct=float(np.mean(supported)) if len(supported) else None)
    metrics["note"] = "No pass/fail certification or automatic activation; object support is NOT temporally filtered occupancy."
    return metrics


def render(table, before, after, status, selected=None):
    canvas = np.full((740, 1128, 3), 24, np.uint8)

    def text(line, y, x=12, scale=.47, color=(225, 225, 225)):
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    text("Multi-frame desk plane | " + status, 26, scale=.57)
    text("C: NEW FIT (desk EMPTY) | V: INDEPENDENT validation | S: save selected view | Q: quit", 52)
    text("Keep camera / desk / non-direct lighting FIXED. Clear the WHOLE ROI; do not move anything during a capture.", 78)
    text("Original calibration stays untouched. Candidate is NEVER activated automatically. PCA / thresholds unchanged.", 103)
    metrics = comparison(before, after, table)
    if metrics["original"] is not None:
        text(f"Common cells: {metrics['common_reliable_cells']} | Median |height|: "
             f"{metrics['original']['median_abs_height_mm']:.2f} -> {metrics['candidate']['median_abs_height_mm']:.2f}mm"
             f" | Frames: {before['n_frames']}", 127)
    specs = [("Original height median (mm)", before["median_mm"], -30, 30,
              [(220, 90, 40), (240, 240, 240), (30, 45, 225)], ["<= -30", "0", ">= +30 mm"]),
             ("Candidate height median (mm)", after["median_mm"], -30, 30,
              [(220, 90, 40), (240, 240, 240), (30, 45, 225)], ["<= -30", "0", ">= +30 mm"]),
             ("Candidate temporal P90-P10 (mm)", after["temporal_span_mm"], 0, 20,
              [(240, 240, 240), (35, 200, 245), (20, 35, 220)], ["0", "10", ">= 20 mm"])]
    for i, ((title, values, low, high, palette, ticks), (x0, y0, dw, dh)) in enumerate(zip(specs, diag.panel_layout(table))):
        left = 12 + i * 372
        text(title, 146, x=left)
        panel = diag.colorize(values, low, high, palette)
        panel[table["table_mask"] == 0] = 0
        cv2.polylines(panel, [table["poly_grid"]], True, (0, 225, 245), 1)
        cv2.drawContours(panel, table["active_contours"], -1, (235, 220, 20), 1)
        canvas[y0:y0 + dh, x0:x0 + dw] = cv2.resize(cv2.flip(panel, 0), (dw, dh), interpolation=cv2.INTER_NEAREST)
        if selected is not None:
            y, x = selected
            point = (x0 + int((x + .5) * dw / table["cols"]),
                     y0 + int((table["rows"] - y - .5) * dh / table["rows"]))
            cv2.drawMarker(canvas, point, (0, 0, 0), cv2.MARKER_CROSS, 15, 3)
            cv2.drawMarker(canvas, point, (255, 255, 255), cv2.MARKER_CROSS, 15, 1)
        bar = diag.colorize(np.linspace(low, high, 330)[None, :], low, high, palette)
        canvas[540:552, left:left + 330] = np.repeat(bar, 12, axis=0)
        for tick, offset in zip(ticks, [0, 150, 245]):
            text(tick, 572, x=left + offset, scale=.43)
    if selected is not None:
        y, x = selected
        def value(result, key):
            number = result[key][y, x]
            return f"{number:.1f}" if np.isfinite(number) else "N/A"
        text(f"Cell row={y} col={x}: height {value(before, 'median_mm')} -> {value(after, 'median_mm')}mm"
             f" | time spread {value(before, 'temporal_span_mm')} -> {value(after, 'temporal_span_mm')}mm", 603)
        text(f"Coverage {value(before, 'coverage_pct')} -> {value(after, 'coverage_pct')}%"
             f" | Object-supported frames {value(before, 'object_support_pct')} -> {value(after, 'object_support_pct')}%", 629)
    else:
        text("Click the far corner in any panel to compare that same cell. Gray means insufficient observations.", 603)
        text("Small fit residual is NOT independent validation. Press V after the candidate is saved.", 629)
    text("FIT result: approximate cell-center correction on training data. VALIDATION: fresh raw points, exact plane distances.", 655, scale=.45)
    text("All panels share OLD UV cells and OLD ROI outlines for comparison. Candidate JSON contains its own updated frame/ROI.", 681, scale=.45)
    text("This changes the reference plane, NOT the temporal noise. Strong local depth errors may remain.", 707)
    text("No smoothing, hole filling, baseline subtraction, or automatic object recognition is added.", 731, scale=.44)
    return canvas


def new_directory(base, prefix):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    path = base / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    path.mkdir()  # Fail rather than overwrite any previous capture.
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)


def write_samples(path, capture):
    with Path(path).open("xb") as output:
        np.savez_compressed(output, times_s=np.asarray(capture.times), **capture.arrays())


def map_json(result):
    return {k: diag.json_grid(v) if isinstance(v, np.ndarray) else v for k, v in result.items()}


def save_fit(base, original_bytes, candidate, table, capture, before, after, stats, inliers, cfg, mode):
    path = new_directory(base, "plane_")
    with (path / "original_calibration.json").open("xb") as output:
        output.write(original_bytes)
    write_json(path / "candidate_calibration.json", candidate)
    write_json(path / "fit_report.json", dict(mode=mode, stage="FIT - not independent validation",
               fit=stats, grid_mm=cfg.grid_m * 1000, edge_margin_mm=cfg.edge_margin_m * 1000,
               object_height_range_mm=[cfg.min_height_m * 1000, cfg.max_height_m * 1000],
               min_cell_points=cfg.min_cell_points, support_fraction=cfg.support_fraction,
               depth_range_m=[cfg.min_depth_m, cfg.max_depth_m],
               fit_inlier_mask=diag.json_grid(inliers), original=map_json(before), candidate_approx=map_json(after),
               metrics=comparison(before, after, table)))
    write_samples(path / "fit_samples.npz", capture)
    v04.save_image(path / "fit_comparison.png", render(table, before, after, mode + " | FIT ONLY"))
    print(f"Candidate saved (NOT activated): {path / 'candidate_calibration.json'}")
    print("Next: press V to collect a separate validation. Do not use the fit picture as proof of improvement.")
    return path


def save_validation(bundle, table, old_capture, new_capture, before, after, candidate, mode):
    path = new_directory(bundle, "validation_")
    metrics = comparison(before, after, table)
    write_json(path / "validation.json", dict(mode=mode, stage="INDEPENDENT CAPTURE - candidate held fixed",
               candidate_sha256=hashlib.sha256((Path(bundle) / "candidate_calibration.json").read_bytes()).hexdigest(),
               original_geometry={k: table[k].tolist() for k in ["origin", "normal", "u_axis", "v_axis", "polygon_uv"]},
               original_plane_d=table["d"], u_min=table["u_min"], v_min=table["v_min"], grid_m=table["grid_m"],
               original_mask=diag.json_grid(table["table_mask"] > 0), active_mask=diag.json_grid(table["active_mask"] > 0),
               candidate_plane=dict(normal=candidate["normal"], d=candidate["d"]),
               original=map_json(before), candidate=map_json(after), metrics=metrics))
    write_samples(path / "samples_original.npz", old_capture)
    write_samples(path / "samples_candidate.npz", new_capture)
    v04.save_image(path / "comparison.png", render(table, before, after, mode + " | INDEPENDENT VALIDATION"))
    print("Independent comparison:", json.dumps(metrics, ensure_ascii=False))
    print(f"Saved validation: {path}")
    return path


def run_live(args, cfg, original_bytes):
    original = json.loads(original_bytes)
    table = v04.table_from_data(original, cfg.grid_m, cfg.edge_margin_m)
    source_hash = hashlib.sha256(original_bytes).hexdigest()
    candidate = bundle = current_dir = None
    stage, active_capture, new_capture, selected = "READY - C starts a fit", None, None, None
    operation = None
    gui_open = False

    def mouse(event, x, y, flags, userdata):
        nonlocal selected
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = diag.clicked_cell(x, y, table)

    try:
        with diag.camera_source() as read:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            gui_open = True
            cv2.setMouseCallback(WINDOW, mouse)
            while True:
                if active_capture is not None or candidate is None:
                    vertices = read()
                    old_frame = diag.frame_grid(vertices, table, cfg)
                    before = diag.preview_result(old_frame)
                    if candidate is not None:
                        new_frame = diag.frame_grid(vertices, measurement_table(table, candidate), cfg)
                        after = diag.preview_result(new_frame)
                    else:
                        after = dict(before, median_mm=np.full(before["median_mm"].shape, np.nan),
                                     reliable=np.zeros(before["reliable"].shape, bool))
                    if active_capture is not None:
                        elapsed = time.monotonic() - started
                        active_capture.add(old_frame, elapsed)
                        if operation == "validate":
                            new_capture.add(new_frame, elapsed)
                        stage = f"{operation.upper()} {min(elapsed, args.seconds):.1f}/{args.seconds:g}s - CURRENT FRAME"
                        if elapsed >= args.seconds:
                            before = active_capture.summarize(table, elapsed)
                            if operation == "fit":
                                try:
                                    beta, stats, inliers = fit_plane(before, table)
                                    proposal = candidate_calibration(original, table, beta, source_hash, stats)
                                    corrected = corrected_fit_result(before, table, beta)
                                except ValueError as error:
                                    print(f"Fit rejected: {error}. Old calibration unchanged; press C to retry.")
                                    stage = "FIT REJECTED - see terminal; C retries"
                                    candidate = None
                                else:
                                    bundle = save_fit(args.output_dir, original_bytes, proposal, table,
                                                      active_capture, before, corrected, stats, inliers, cfg, "LIVE")
                                    candidate, after, current_dir = proposal, corrected, bundle
                                    stage = "FIT SAVED - NOT VALIDATED; press V"
                            else:
                                after = new_capture.summarize(table, elapsed)
                                current_dir = save_validation(bundle, table, active_capture, new_capture,
                                                              before, after, candidate, "LIVE")
                                stage = "INDEPENDENT VALIDATION - saved; click a cell"
                            active_capture = new_capture = None
                cv2.imshow(WINDOW, render(table, before, after, stage, selected))
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if active_capture is None and key == ord("c"):
                    # A new fit invalidates the in-memory candidate, never saved files.
                    candidate = bundle = current_dir = None
                    active_capture, operation, started, selected = diag.Capture(table), "fit", time.monotonic(), None
                elif active_capture is None and key == ord("v"):
                    if candidate is None:
                        print("Capture a valid candidate with C first.")
                    else:
                        active_capture, new_capture = diag.Capture(table), diag.Capture(table)
                        operation, started, selected = "validate", time.monotonic(), None
                elif active_capture is None and key == ord("s") and current_dir is not None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    v04.save_image(current_dir / f"selected_{stamp}.png", render(table, before, after, stage, selected))
    finally:
        if gui_open:
            cv2.destroyAllWindows()


def synthetic_points(table, rng, beta, frame):
    """Known plane tilt, random noise and local contaminating points; not hardware."""
    uv = grid_uv(table)[table["table_mask"] > 0]
    uv = uv[rng.random(len(uv)) > .05]
    h = uv @ beta[:2] + beta[2] + rng.normal(0, .003, len(uv))
    # A small fixed patch of non-plane samples tests robust fitting, not semantic safety.
    h[(uv[:, 0] > .75) & (uv[:, 0] < .90) & (uv[:, 1] > .3) & (uv[:, 1] < .45)] += .022
    uv, h = np.repeat(uv, 5, axis=0), np.repeat(h, 5)
    h += rng.normal(0, .0008, len(h))
    return (table["origin"] + uv[:, :1] * table["u_axis"] + uv[:, 1:] * table["v_axis"]
            + h[:, None] * table["normal"])


def run_demo(args, cfg):
    table = v04.demo_table(cfg)
    original = {k: table[k].tolist() for k in ["origin", "normal", "u_axis", "v_axis", "polygon_uv"]}
    original.update(d=table["d"], grid_size=cfg.grid_m)
    original_bytes = json.dumps(original).encode()
    known_beta = np.array([-.014, .007, .008])
    rng = np.random.default_rng(11)
    capture = diag.Capture(table)
    count = round(args.seconds * 30)
    for i in range(count):
        capture.add(diag.frame_grid(synthetic_points(table, rng, known_beta, i), table, cfg), (i + 1) * args.seconds / count)
    before = capture.summarize(table, args.seconds)
    beta, stats, inliers = fit_plane(before, table)
    stats["synthetic_ground_truth_a_b_c_m"] = known_beta.tolist()
    candidate = candidate_calibration(original, table, beta, hashlib.sha256(original_bytes).hexdigest(), stats)
    after = corrected_fit_result(before, table, beta)
    bundle = save_fit(args.output_dir, original_bytes, candidate, table, capture, before, after, stats, inliers, cfg, "SYNTHETIC")
    old_capture, new_capture = diag.Capture(table), diag.Capture(table)
    rng = np.random.default_rng(29)  # Independent scene noise, not reused fitting samples.
    for i in range(count):
        points = synthetic_points(table, rng, known_beta, i)
        elapsed = (i + 1) * args.seconds / count
        old_capture.add(diag.frame_grid(points, table, cfg), elapsed)
        new_capture.add(diag.frame_grid(points, measurement_table(table, candidate), cfg), elapsed)
    before, after = old_capture.summarize(table, args.seconds), new_capture.summarize(table, args.seconds)
    save_validation(bundle, table, old_capture, new_capture, before, after, candidate, "SYNTHETIC")
    return bundle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--grid-mm", type=float, default=10)
    parser.add_argument("--edge-margin-mm", type=float, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plane_calibrations")
    parser.add_argument("--demo", action="store_true", help="Headless synthetic fit and independent validation; no camera")
    args = parser.parse_args(argv)
    cfg = v04.Settings(grid_m=args.grid_mm / 1000, edge_margin_m=args.edge_margin_mm / 1000)
    try:
        cfg.validate()
        if not np.isfinite(args.seconds) or not 3 <= args.seconds <= 15:
            raise ValueError("Capture duration must be between 3 and 15 seconds")
    except ValueError as error:
        parser.error(str(error))
    return args, cfg


def main(argv=None):
    args, cfg = parse_args(argv)
    try:
        if args.demo:
            print("SYNTHETIC ONLY - not a live calibration or sensor accuracy test")
            run_demo(args, cfg)
        else:
            original_bytes = args.calibration.read_bytes()
            print("Close other camera programs. Clear the WHOLE desk ROI, keep camera and non-direct lighting fixed.")
            print("C fits a candidate. V records a separate validation. No original file is overwritten or auto-activated.")
            run_live(args, cfg, original_bytes)
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
        print(f"Error: {error}\nOriginal calibration was NOT overwritten. Check scene, camera, calibration and output permissions.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
