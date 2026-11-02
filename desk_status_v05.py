"""Desk v0.5: visible free-space score, quality gates and PCA diagnostics.

This is an occupancy-based availability proxy, NOT semantic tidiness.
Live mode requires an explicit --calibration. Existing calibration is read-only.
--demo --headless saves synthetic PNG + JSON without a camera or GUI.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

import desk_orientation_v04 as core
import desk_height_background as background
from desk_height_diagnostic import camera_source


ROOT = Path(__file__).resolve().parent
WINDOW = "Desk v0.5 - free space / observation quality / PCA"
SCENES = ("empty", "objects", "rotated", "removed", "missing")


@dataclass(frozen=True)
class ScorePolicy:
    min_known: float = .90
    min_current: float = .90
    max_held: float = .10

    def validate(self):
        values = [self.min_known, self.min_current, self.max_held]
        if not np.isfinite(values).all() or not (0 < self.min_known <= 1
                and 0 < self.min_current <= 1 and 0 <= self.max_held <= 1):
            raise ValueError("Invalid score coverage/hold policy")


def measure(stable, evidence, held, table, policy=ScorePolicy(), temporal=True, paused=False):
    """Count ALL stable occupied cells, not just large/PCA-eligible components.

    Unknown and excluded cells are never counted as free. A score uses the known
    denominator; quality percentages use the entire ACTIVE ROI denominator.
    Quality gates are MVP display rules, not probabilities of sensor accuracy.
    """
    policy.validate()
    active = table["active_mask"] > 0
    for grid in (stable, evidence, held):
        if grid.shape != active.shape:
            raise ValueError("State shape does not match active ROI")
    if not np.isin(stable, [core.UNKNOWN, core.FREE, core.OCCUPIED]).all() or not np.isin(
            evidence, [core.UNKNOWN, core.FREE, core.OCCUPIED]).all():
        raise ValueError("Invalid three-state grid")
    if held.dtype != bool or np.any(held & (stable == core.UNKNOWN)):
        raise ValueError("Held must mark known states only")
    total = int(active.sum())
    if total == 0:
        raise ValueError("No active ROI cells")
    free = int(np.count_nonzero(active & (stable == core.FREE)))
    occupied = int(np.count_nonzero(active & (stable == core.OCCUPIED)))
    known = free + occupied
    current = int(np.count_nonzero(active & (evidence != core.UNKNOWN)))
    held_count = int(np.count_nonzero(active & held))
    known_fraction, current_fraction = known / total, current / total
    held_fraction = held_count / total
    reasons = []
    if paused:
        reasons.append("PAUSED")
    if not temporal:
        reasons.append("TEMPORAL OFF")
    if known_fraction < policy.min_known:
        reasons.append("LOW STABLE COVERAGE")
    if current_fraction < policy.min_current:
        reasons.append("LOW CURRENT COVERAGE")
    if held_fraction > policy.max_held:
        reasons.append("TOO MUCH HELD STATE")
    return dict(free_space_score=100 * free / known if known and not reasons else None,
                score_valid=not reasons, status="READY" if not reasons else reasons[0], reasons=reasons,
                active_cells=total, free_cells=free, occupied_cells=occupied,
                unknown_cells=total - known, known_cells=known, current_known_cells=current,
                held_cells=held_count, stable_known_pct=100 * known_fraction,
                current_known_pct=100 * current_fraction, held_pct=100 * held_fraction,
                unknown_active_pct=100 * (total - known) / total,
                occupied_active_pct=100 * occupied / total,
                occupied_known_pct=100 * occupied / known if known else None,
                temporal_enabled=temporal, paused=paused)


def render(table, observation, stable, held, labels, regions, cfg, policy, metrics,
           temporal=True, paused=False, mode="LIVE", calibration_name="", detection_info=None):
    """Reuse v04's unmodified three panels, adding the v05 metric header."""
    legacy = core.render(table, observation, stable, held, labels, regions, cfg, temporal, mode, paused)
    # The panels, labels and legend are unchanged. The old diagnostic header is replaced.
    canvas = np.full((legacy.shape[0] + 120, legacy.shape[1], 3), 24, np.uint8)
    canvas[252:] = legacy[132:]

    def text(line, x, y, scale=.46, color=(225, 225, 225)):
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    title = f"Desk v0.5 | {mode}{' | PAUSED' if paused else ''}"
    text(title, 12, 27, .62)
    cv2.rectangle(canvas, (828, 5), (1116, 36), (35, 95, 35) if temporal else (30, 85, 165), -1)
    text(f"Temporal {'ON' if temporal else 'OFF'} | T -> {'OFF' if temporal else 'ON'}", 838, 27, .53)
    score = metrics["free_space_score"]
    occupied = metrics["occupied_known_pct"]
    valid_axes = sum(r["angle"] is not None for r in regions)
    cards = [("VISIBLE FREE SPACE", "N/A" if score is None else f"{score:.1f} / 100", metrics["status"]),
             ("OCCUPIED / KNOWN", "N/A" if occupied is None else f"{occupied:.1f}%", "Stable cells; not object volume"),
             ("UNKNOWN / ACTIVE ROI", f"{metrics['unknown_active_pct']:.1f}%", "Never treated as free space"),
             ("CANDIDATE REGIONS", str(len(regions)), f"Valid PCA axes: {valid_axes} (not identities)")]
    for i, (label, value, detail) in enumerate(cards):
        x = 12 + i * 279
        cv2.rectangle(canvas, (x, 50), (x + 267, 145), (40, 40, 40), -1)
        text(label, x + 10, 71, .43, (170, 190, 180))
        text(value, x + 10, 111, .95, (100, 220, 160) if i == 0 and score is not None else (235, 235, 235))
        text(detail, x + 10, 134, .36)
    text(f"Stable known: {metrics['stable_known_pct']:.1f}% | Classified NOW: {metrics['current_known_pct']:.1f}%"
         f" | Held: {metrics['held_pct']:.1f}% | Occupied / ACTIVE ROI: {metrics['occupied_active_pct']:.1f}%", 12, 169)
    if detection_info and detection_info['mode'] == 'local_height':
        lo, _, hi = detection_info['gate_min_median_max_mm']
        hp = detection_info['policy']
        text(f"LOCAL delta gate {lo:.1f}-{hi:.1f}mm | Grid {cfg.grid_m * 1000:g}mm"
             f" | Inset {cfg.edge_margin_m * 1000:g}mm | Height median {hp['min_samples']}/{hp['window']}"
             f" | Votes {cfg.votes}/{cfg.window} | Hold {cfg.hold_seconds:g}s", 12, 191, .44)
        canvas[252:276] = 24
        text("CURRENT: local delta / cell median", 12, 261, .42)
        text("STABLE: height median + time votes", 384, 261, .42)
        text("REGIONS: PCA axes / table edge", 756, 261, .42)
    else:
        text(f"Grid {cfg.grid_m * 1000:g}mm | Height {cfg.min_height_m * 1000:g}-{cfg.max_height_m * 1000:g}mm"
             f" | Inset {cfg.edge_margin_m * 1000:g}mm | Votes {cfg.votes}/{cfg.window} | Hold {cfg.hold_seconds:g}s"
             f" | Calibration: {calibration_name[:40]}", 12, 191, .44)
    canvas[252:276, 748:] = 24
    area_gate = f"area >= {cfg.min_pca_area_cm2:g}cm2" if cfg.min_pca_area_cm2 else "area gate OFF"
    text(f"REGIONS: PCA r >= {cfg.pca_ratio:g} | {area_gate}", 756, 261, .38)
    reasons = ", ".join(metrics["reasons"]) if metrics["reasons"] else "READY"
    text(f"Score gate: stable >= {policy.min_known:.0%}, now >= {policy.min_current:.0%}, held <= {policy.max_held:.0%}"
         f" | {reasons}", 12, 214, .44, (100, 190, 245) if reasons != "READY" else (190, 220, 190))
    text("Free space = free / known. Not semantic tidiness. PCA / region count do not affect the score.", 12, 238, .46)
    # Replace v04's keyboard legend because v05 saves paired reports and cycles scenes.
    canvas[-35:] = 24
    text("[T] temporal  [R] reset  [SPACE] pause  [S] save PNG+JSON  [N] next demo scene  [Q] quit", 12, canvas.shape[0] - 17)
    return canvas


def demo_vertices(table, cfg, rng, scene, low_profile=False):
    """Synthetic only: two shapes, depth noise, sparse outliers and 3% dropout."""
    if scene not in SCENES:
        raise ValueError("Unknown demo scene")
    if scene == "missing":
        return np.empty((0, 3))
    mask = np.zeros((table["rows"], table["cols"]), np.uint8)
    if scene in ("objects", "rotated"):
        angle = 60 if scene == "rotated" else 0
        g = cfg.grid_m
        for center, size, theta in [((.38 / g, .4 / g), (.30 / g, .10 / g), angle),
                                    ((.95 / g, .4 / g), (.15 / g, .15 / g), 0)]:
            cv2.fillConvexPoly(mask, np.rint(cv2.boxPoints((center, size, theta))).astype(np.int32), 1)
    y, x = np.nonzero(table["table_mask"])
    keep = rng.random(len(x)) >= .03
    x, y = np.repeat(x[keep], 5), np.repeat(y[keep], 5)
    uv = (np.column_stack([x, y]) + .5 + rng.uniform(-.3, .3, (len(x), 2))) * cfg.grid_m
    heights = np.where(mask[y, x] > 0, .008 if low_profile else .045, 0.) + rng.normal(0, .003, len(x))
    heights[rng.random(len(x)) < .02] = .025
    if low_profile:
        # Same spatial bias as training, fresh noise and object placement only at runtime.
        noise = rng.normal(0, .002, mask.shape)
        heights += background.synthetic_bias(table)[y, x] / 1000 + noise[y, x]
    return (table["origin"] + uv[:, :1] * table["u_axis"] + uv[:, 1:] * table["v_axis"]
            + heights[:, None] * table["normal"])


def load_context(args, cfg):
    if args.demo:
        return core.demo_table(cfg), dict(kind="synthetic", seed=4, calibration_sha256=None)
    raw = args.calibration.read_bytes()
    table = core.table_from_data(json.loads(raw), cfg.grid_m, cfg.edge_margin_m)
    return table, dict(kind="live", calibration_name=args.calibration.name,
                       calibration_sha256=hashlib.sha256(raw).hexdigest())


def make_report(table, metrics, regions, cfg, policy, source, scene, frame_index, detection_info=None):
    return dict(schema_version=2, app="desk_status_v05", captured_at=datetime.now(timezone.utc).isoformat(),
                source=dict(source, demo_scene=scene if source["kind"] == "synthetic" else None),
                detection=detection_info or dict(mode='absolute_height',
                    point_rule='v04: min object points AND object fraction; see settings',
                    current_quality_source='current point-supported classification'),
                frame_index=frame_index, metrics=metrics, settings=asdict(cfg), score_policy=asdict(policy),
                score_definition="100 * stable FREE / (stable FREE + stable OCCUPIED), ACTIVE ROI only",
                regions=[{key: r[key] for key in ["label", "area_cells", "area_cm2", "angle", "ratio", "span_m", "reason"]}
                         for r in regions],
                geometry={**{key: table[key].tolist() for key in ["normal", "origin", "u_axis", "v_axis", "polygon_uv"]},
                          "d": table["d"], "grid_m": table["grid_m"]},
                runtime=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__),
                limitations=["Visible space availability, not a semantic tidiness judgement.",
                             "Unknown and excluded cells are NOT free; score covers known active cells only.",
                             "Coverage gates are display rules, not sensor accuracy or confidence estimates.",
                             "Regions are connected components, not object identities; PCA may be N/A.",
                             "Small occupied components still count toward occupancy, even without a PCA axis.",
                             "No RGB, raw point cloud, or depth recording is included in this report."])


def save_result(base, picture, report):
    """Create a new directory; never overwrite a screenshot/report or calibration."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    directory = base / ("v05_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir()
    with (directory / "report.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
    core.save_image(directory / "view.png", picture)
    print(f"Saved v05 result: {directory.resolve()}")
    return directory


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, help="Required for LIVE; choose a reviewed calibration explicitly")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--background", type=Path, help="Optional EMPTY reference from desk_height_background.py; LIVE only")
    parser.add_argument("--demo-background", action="store_true", help="Synthetic local-height mode with 8mm shapes; requires --demo")
    parser.add_argument("--demo-scene", choices=SCENES, default="objects")
    parser.add_argument("--headless", action="store_true", help="With --demo: warm up, save PNG+JSON, exit without GUI")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--grid-mm", type=float, default=10)
    parser.add_argument("--min-height-mm", type=float, default=10)
    parser.add_argument("--max-height-mm", type=float, default=400)
    parser.add_argument("--edge-margin-mm", type=float, default=20)
    parser.add_argument("--pca-ratio", type=float, default=2.5,
                        help="Direction-only eigenvalue ratio gate (>1; default 2.5). Does not change occupancy or score")
    parser.add_argument("--min-pca-area-cm2", type=float, default=0,
                        help="Minimum occupied region area for direction only (cm2; default 0 disables). Keeps small regions and score")
    parser.add_argument("--no-temporal", action="store_true", help="Diagnostic only; free-space score is disabled")
    args = parser.parse_args(argv)
    if args.demo and args.calibration is not None:
        parser.error("--demo uses synthetic geometry; do not combine it with --calibration")
    if args.background is not None and (args.demo or args.demo_background):
        parser.error("--background is LIVE only; use --demo --demo-background for synthetic data")
    if args.demo_background and not args.demo:
        parser.error("--demo-background requires --demo")
    if (args.background is not None or args.demo_background) and args.min_height_mm != 10:
        parser.error("--min-height-mm is for absolute mode; local mode uses the saved per-cell gates")
    if args.headless and not args.demo:
        parser.error("--headless requires --demo")
    if not args.demo and args.calibration is None:
        parser.error("LIVE requires --calibration PATH; no latest/default calibration is silently selected")
    cfg = core.Settings(grid_m=args.grid_mm / 1000, min_height_m=args.min_height_mm / 1000,
                        max_height_m=args.max_height_mm / 1000, edge_margin_m=args.edge_margin_mm / 1000,
                        pca_ratio=args.pca_ratio, min_pca_area_cm2=args.min_pca_area_cm2)
    try:
        cfg.validate()
    except ValueError as error:
        parser.error(str(error))
    return args, cfg


def detection_context(args, table, cfg, source):
    if args.background is not None:
        return background.load_reference(args.background, table, cfg, source['calibration_sha256'])
    if args.demo_background:
        data, _ = background.synthetic_reference(table, cfg)
        detector = background.LocalHeightDetector(data, table, cfg, None)
        return detector, background.reference_info(detector, table, 'synthetic-empty', None, synthetic=True)
    return None, None


def quality_inputs(observation, stable, held, local=False):
    """Height-history evidence is NOT a fresh measurement for quality scoring."""
    current = observation.raw if local else observation.evidence
    if local:
        held = held | ((stable != core.UNKNOWN) & (stable != current))
    return current, held


def run(args, cfg):
    table, source = load_context(args, cfg)
    detector, detection_info = detection_context(args, table, cfg, source)
    policy = ScorePolicy()
    stabilizer = core.TemporalGrid(table["active_mask"].shape, cfg)
    temporal, paused, gui_open = not args.no_temporal, False, False
    scene, frame_index = args.demo_scene, 0
    rng = np.random.default_rng(4)
    print("SYNTHETIC DEMO - not sensor validation" if args.demo else f"LIVE | read-only calibration: {args.calibration.resolve()}")
    print("Visible free-space proxy, NOT semantic tidiness. Score requires Temporal ON and adequate current/stable coverage.")
    print(f"Settings: {cfg}\nScore policy: {policy}\nT temporal | R reset | SPACE pause | S PNG+JSON | N demo scene | Q quit")
    if detector is not None:
        print(f"LOCAL HEIGHT | reference: {detection_info['background_name']} | gates min/median/max (mm): "
              f"{detection_info['gate_min_median_max_mm']}\nAbsolute min_height/support_fraction are NOT used in this mode. "
              "T controls BOTH the height median and state votes. Independent empty/object validation is required.")

    def reset_history():
        stabilizer.reset()
        if detector is not None:
            detector.reset()

    try:
        with (nullcontext(None) if args.demo else camera_source()) as read:
            while True:
                if not paused:
                    vertices = demo_vertices(table, cfg, rng, scene, args.demo_background) if args.demo else read()
                    now = frame_index / 30 if args.headless else time.monotonic()
                    observation = (detector.observe(vertices, now, smooth=temporal) if detector is not None
                                   else core.make_observation(vertices, table, cfg))
                    if temporal:
                        stable, held = stabilizer.update(observation.evidence, now)
                    else:
                        stable, held = observation.evidence.copy(), np.zeros(table["active_mask"].shape, bool)
                    current, held = quality_inputs(observation, stable, held, detector is not None)
                    labels, regions = core.extract_regions(stable, table, cfg)
                    frame_index += 1
                metrics = measure(stable, current, held, table, policy, temporal, paused)
                mode = f"SYNTHETIC | {scene}" if args.demo else "LIVE"
                picture = render(table, observation, stable, held, labels, regions, cfg, policy, metrics,
                                 temporal, paused, mode, source.get("calibration_name", "synthetic"), detection_info)
                if args.headless:
                    if frame_index >= max(30, cfg.window):
                        return save_result(args.output_dir, picture,
                                           make_report(table, metrics, regions, cfg, policy, source, scene, frame_index, detection_info))
                    continue
                if not gui_open:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
                    gui_open = True
                cv2.imshow(WINDOW, picture)
                key = cv2.waitKey(30 if args.demo or paused else 1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(" "):
                    paused = not paused
                    if not paused:
                        reset_history()
                elif key == ord("t"):
                    temporal, paused = not temporal, False
                    reset_history()
                elif key == ord("r"):
                    reset_history()
                    paused = False
                elif key == ord("n") and args.demo:
                    scene, paused = SCENES[(SCENES.index(scene) + 1) % len(SCENES)], False
                elif key == ord("s"):
                    save_result(args.output_dir, picture,
                                make_report(table, metrics, regions, cfg, policy, source, scene, frame_index, detection_info))
    finally:
        if gui_open:
            cv2.destroyAllWindows()


def main(argv=None):
    args, cfg = parse_args(argv)
    try:
        run(args, cfg)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}\nNo calibration was overwritten. Close other camera programs and check the chosen file/device.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
