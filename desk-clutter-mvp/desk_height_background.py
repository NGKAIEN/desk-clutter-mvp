"""Capture an explicitly EMPTY desk reference for optional v05 local-height mode.

This is NOT plane calibration and never writes calibration.json. C confirms the
whole desk ROI is empty and records a new reference; Q quits. References are
immutable files bound to exact calibration bytes and sampling settings.
"""

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import cv2
import numpy as np

import desk_height_diagnostic as diag
import desk_orientation_v04 as core


ROOT = Path(__file__).resolve().parent
WINDOW = "EMPTY desk height reference - C capture, Q quit"


@dataclass(frozen=True)
class HeightPolicy:
    window: int = 5
    min_samples: int = 3
    max_age_s: float = .25
    floor_mm: float = 6.
    noise_quantile: float = .99
    noise_margin_mm: float = 2.
    max_gate_mm: float = 12.
    max_offset_mm: float = 20.
    min_capture_frames: int = 60
    min_coverage: float = .90
    min_usable_fraction: float = .90

    def validate(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Height policy must be finite")
        if (not isinstance(self.window, int) or not isinstance(self.min_samples, int)
                or not isinstance(self.min_capture_frames, int)
                or not 1 <= self.min_samples <= self.window <= 15
                or self.min_capture_frames < 30 or not 0 < self.max_age_s <= 1
                or not 0 < self.floor_mm <= self.max_gate_mm < 50
                or not .5 < self.noise_quantile < 1 or self.noise_margin_mm <= 0
                or not 0 < self.max_offset_mm <= 30
                or not .5 < self.min_coverage <= 1 or not .5 < self.min_usable_fraction <= 1):
            raise ValueError("Invalid height reference policy")


class HeightMedian:
    """Fresh sample required; old/missing frames are never filled from history."""
    def __init__(self, shape, policy):
        policy.validate()
        self.shape, self.policy = shape, policy
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.policy.window)
        self.last_time = None

    def update(self, values, now):
        if values.shape != self.shape or np.isinf(values).any():
            raise ValueError("Invalid height frame")
        if not np.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError("Height timestamps must be finite and monotonic")
        self.last_time = now
        self.history.append((float(now), np.asarray(values, float).copy()))
        while self.history and now - self.history[0][0] > self.policy.max_age_s:
            self.history.popleft()
        stack = np.stack([v for _, v in self.history])
        supported = np.isfinite(values) & (np.isfinite(stack).sum(axis=0) >= self.policy.min_samples)
        result = np.full(self.shape, np.nan)
        if supported.any():
            result[supported] = np.nanmedian(stack[:, supported], axis=0)
        return result


def fingerprint(table, cfg, calibration_sha256):
    return dict(calibration_sha256=calibration_sha256, grid_m=cfg.grid_m,
                edge_margin_m=cfg.edge_margin_m, min_cell_points=cfg.min_cell_points,
                min_depth_m=cfg.min_depth_m, max_depth_m=cfg.max_depth_m,
                shape=list(table['table_mask'].shape), u_min=table['u_min'], v_min=table['v_min'])


def fit_reference(capture, table, cfg, calibration_sha256, elapsed, policy=HeightPolicy()):
    """Estimate local offset and conservative noise gates from EMPTY samples only.

    A Q99 envelope of the SAME short-median filter plus a 2mm margin sets each
    cell's threshold (at least 6mm). This is a heuristic, not a false-alarm bound.
    Independent empty/object trials remain required after capture.
    """
    policy.validate()
    result = capture.summarize(table, elapsed)
    if result['n_frames'] < policy.min_capture_frames:
        raise ValueError(f"Need >= {policy.min_capture_frames} frames; retry or increase --seconds")
    active = table['active_mask'] > 0
    offsets = result['median_mm'].astype(float)
    valid = (active & result['reliable'] & (result['coverage_pct'] >= policy.min_coverage * 100)
             & np.isfinite(offsets) & (np.abs(offsets) <= policy.max_offset_mm))
    median_filter = HeightMedian(active.shape, policy)
    filtered = []
    for frame, stamp in zip(capture.frames, capture.times):
        values = np.where(valid & (frame.point_counts >= cfg.min_cell_points), frame.median_mm - offsets, np.nan)
        filtered.append(median_filter.update(values, stamp))
    filtered = np.stack(filtered)
    valid &= np.isfinite(filtered).sum(axis=0) >= max(30, int(np.ceil(.75 * result['n_frames'])))
    gates = np.full(active.shape, np.nan)
    if valid.any():
        gates[valid] = np.maximum(policy.floor_mm,
                                 np.nanquantile(np.abs(filtered[:, valid]), policy.noise_quantile, axis=0)
                                 + policy.noise_margin_mm)
    valid &= np.isfinite(gates) & (gates <= policy.max_gate_mm)
    usable = int(valid.sum()) / int(active.sum())
    if usable < policy.min_usable_fraction:
        raise ValueError(f"Only {usable:.1%} usable reference cells; need {policy.min_usable_fraction:.0%}. "
                         "Clear ROI/check lighting, motion and coverage; do not force noisy cells to zero")
    offsets[~valid] = np.nan
    gates[~valid] = np.nan
    return dict(schema_version=1, kind='empty_height_reference', operator_confirmed_empty=True,
                created_at=datetime.now(timezone.utc).isoformat(), policy=asdict(policy),
                sampling=fingerprint(table, cfg, calibration_sha256),
                offsets_mm=diag.json_grid(offsets), gates_mm=diag.json_grid(gates), valid=valid.tolist(),
                capture=dict(n_frames=result['n_frames'], elapsed_seconds=float(elapsed), usable_pct=usable * 100,
                             gate_min_median_max_mm=np.quantile(gates[valid], [0, .5, 1]).tolist()),
                limitations=['Operator must clear the WHOLE ROI; no semantic check for flat objects.',
                             'Not calibration; reference never adapts while objects are present.',
                             'Noise gate uses training data, not an independent accuracy guarantee.',
                             'Changing camera/desk/light/device requires a new reference even if the file hash is unchanged.'])


def validate_reference(data, table, cfg, calibration_sha256):
    if (not isinstance(data, dict) or data.get('kind') != 'empty_height_reference' or data.get('schema_version') != 1
            or data.get('operator_confirmed_empty') is not True):
        raise ValueError("Not an explicit empty reference; ordinary diagnostic/object reports cannot be used")
    if data.get('sampling') != fingerprint(table, cfg, calibration_sha256):
        raise ValueError("Background calibration/grid/sampling mismatch; capture again with the same settings")
    policy = HeightPolicy(**data['policy'])
    policy.validate()
    offsets = np.asarray(data['offsets_mm'], float)
    gates = np.asarray(data['gates_mm'], float)
    valid = np.asarray(data['valid'])
    active = table['active_mask'] > 0
    if any(a.shape != active.shape for a in [offsets, gates, valid]) or valid.dtype != bool:
        raise ValueError("Invalid background grid shape/type")
    if (np.any(valid & ~active) or np.isinf(offsets).any() or np.isinf(gates).any()
            or not np.array_equal(np.isfinite(offsets), valid)
            or not np.array_equal(np.isfinite(gates), valid)
            or np.any(np.abs(offsets[valid]) > policy.max_offset_mm)
            or np.any(gates[valid] < policy.floor_mm) or np.any(gates[valid] > policy.max_gate_mm)
            or valid.sum() / active.sum() < policy.min_usable_fraction):
        raise ValueError("Invalid background offsets/gates/coverage")
    return offsets, gates, valid, policy


class LocalHeightDetector:
    def __init__(self, data, table, cfg, calibration_sha256):
        self.offsets, self.gates, self.valid, self.policy = validate_reference(data, table, cfg, calibration_sha256)
        self.table, self.cfg = table, cfg
        self.filter = HeightMedian(self.valid.shape, self.policy)

    def reset(self):
        self.filter.reset()

    def classify(self, values):
        state = np.zeros(self.valid.shape, np.uint8)
        known = self.valid & np.isfinite(values)
        occupied = known & (values > self.gates) & (values < self.cfg.max_height_m * 1000)
        free = known & (np.abs(values) < self.cfg.plane_tolerance_m * 1000) & ~occupied
        state[free], state[occupied] = core.FREE, core.OCCUPIED
        return state

    def observe_frame(self, frame, now, smooth=True):
        sampled = (frame.point_counts >= self.cfg.min_cell_points) & np.isfinite(frame.median_mm) & self.valid
        delta = np.where(sampled, frame.median_mm - self.offsets, np.nan)
        current = self.classify(delta)
        filtered = self.filter.update(delta, now) if smooth else delta
        # Raw is CURRENT cell-median classification, not v04's any-point mask.
        return core.Observation(current, self.classify(filtered), frame.point_counts)

    def observe(self, vertices, now, smooth=True):
        return self.observe_frame(diag.frame_grid(vertices, self.table, self.cfg), now, smooth)


def load_reference(path, table, cfg, calibration_sha256):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    detector = LocalHeightDetector(data, table, cfg, calibration_sha256)
    if data.get('synthetic'):
        raise ValueError('Synthetic references are for --demo-background only, not LIVE')
    return detector, reference_info(detector, table, Path(path).name, hashlib.sha256(raw).hexdigest())


def reference_info(detector, table, name, sha256, synthetic=False):
    return dict(mode='local_height', background_name=name, synthetic=synthetic,
                background_sha256=sha256, policy=asdict(detector.policy),
                reference_usable_pct=100 * float(detector.valid.sum()) / np.count_nonzero(table['active_mask']),
                gate_min_median_max_mm=np.quantile(detector.gates[detector.valid], [0, .5, 1]).tolist(),
                current_quality_source='current cell median delta, not height-history classification',
                point_rule='>= min_cell_points, per-frame median; NOT the absolute-mode 60% support rule',
                unused_absolute_settings=['min_height_m', 'support_fraction'],
                independent_validation_required=True)


def save_reference(base, data, capture, table, cfg, result):
    directory = Path(base) / ('background_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'background.json').open('x', encoding='utf-8') as output:
        json.dump(data, output, ensure_ascii=False, allow_nan=False, indent=2)
    with (directory / 'samples.npz').open('xb') as output:
        np.savez_compressed(output, times_s=np.asarray(capture.times), **capture.arrays())
    core.save_image(directory / 'capture.png', capture_view(table, result, cfg, 'SAVED - Q quits; C starts NEW empty capture'))
    print(f"Saved EMPTY height reference (not activated): {directory / 'background.json'}")
    print(f"Usable: {data['capture']['usable_pct']:.1f}% | gate min/median/max (mm): {data['capture']['gate_min_median_max_mm']}")
    return directory


def capture_view(table, result, cfg, status):
    image = diag.render(table, result, cfg, status)
    image[:132] = 24
    lines = [('EMPTY desk height reference | ' + status, 26),
             ('Clear ALL objects, including laptop/cables. Fix camera/desk/light. C confirms EMPTY and captures.', 51),
             (f"Frames: {result['n_frames']} | Grid {cfg.grid_m * 1000:g}mm | Candidate plane unchanged; no automatic activation.", 76),
             ('Offsets + noise gates are saved separately. Missing/noisy cells stay unknown. No online background learning.', 101),
             ('C: start a NEW 5s capture (or --seconds) | Q / Esc: quit. Do NOT capture with objects on the desk.', 125)]
    for line, y in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .46, (225, 225, 225), 1, cv2.LINE_AA)
    # Redraw below the replacement header; do not clip the original title glyphs.
    image[132:150] = 24
    titles = ['CURRENT height (mm)' if result.get('preview') else 'Empty height offset (mm)',
              'Temporal spread: P90-P10 (mm)', 'Observed frames / captured frames']
    for x, title in zip([12, 384, 756], titles):
        cv2.putText(image, title, (x, 145), cv2.FONT_HERSHEY_SIMPLEX, .46, (225, 225, 225), 1, cv2.LINE_AA)
    image[589:] = 24
    for line, y in [('Left: height offset | Middle: raw temporal P90-P10 | Right: sample coverage', 611),
                    ('Local detection gate = max(6mm, Q99(abs(short-median empty residual)) + 2mm).', 639),
                    ('Gates above 12mm / offsets above 20mm / poor coverage are rejected, not filled.', 665),
                    ('A saved reference still needs a separate EMPTY then LAPTOP trial. This picture is training data.', 693),
                    ('A fixed flat object can look like desk: the operator must ensure the ROI is actually empty.', 721)]:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .46, (225, 225, 225), 1, cv2.LINE_AA)
    return image


def synthetic_bias(table):
    y, x = np.indices(table['table_mask'].shape)
    return -4 + 2 * np.sin(x * table['grid_m'] * 3) + .5 * np.cos(y * table['grid_m'] * 4)


def synthetic_reference(table, cfg, calibration_sha256=None):
    capture = diag.Capture(table)
    rng = np.random.default_rng(924)
    bias = synthetic_bias(table)
    for i in range(90):
        heights = bias + rng.normal(0, 2.5, bias.shape)
        sampled = (table['table_mask'] > 0) & (rng.random(bias.shape) > .02)
        frame = diag.FrameGrid(np.where(sampled, heights, np.nan), np.where(sampled, 1., np.nan),
                               np.where(sampled, 8, 0), np.zeros(bias.shape, bool))
        capture.add(frame, i / 30)
    reference = fit_reference(capture, table, cfg, calibration_sha256, 3.)
    reference['synthetic'] = True
    return reference, capture


def run_capture(args, cfg):
    raw = args.calibration.read_bytes()
    calibration_sha256 = hashlib.sha256(raw).hexdigest()
    table = core.table_from_data(json.loads(raw), cfg.grid_m, cfg.edge_margin_m)
    capture, result, done = None, None, False
    status, gui_open = 'READY - clear whole ROI, then C', False
    try:
        with diag.camera_source() as read:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            gui_open = True
            while True:
                if not done:
                    frame = diag.frame_grid(read(), table, cfg)
                    result = diag.preview_result(frame)
                    if capture is not None:
                        elapsed = time.monotonic() - started
                        capture.add(frame, elapsed)
                        result.update(n_frames=len(capture.frames), elapsed_seconds=elapsed)
                        status = f'CAPTURING {min(elapsed, args.seconds):.1f}/{args.seconds:g}s'
                        if elapsed >= args.seconds:
                            result = capture.summarize(table, elapsed)
                            try:
                                data = fit_reference(capture, table, cfg, calibration_sha256, elapsed)
                            except ValueError as error:
                                print(f'Reference rejected: {error}. C retries; calibration unchanged.')
                                status = 'REJECTED - see terminal; C retries'
                            else:
                                directory = save_reference(args.output_dir, data, capture, table, cfg, result)
                                command = [sys.executable, str(ROOT / 'desk_status_v05.py'),
                                           '--calibration', str(args.calibration.resolve()),
                                           '--background', str((directory / 'background.json').resolve()),
                                           '--grid-mm', str(args.grid_mm), '--edge-margin-mm', str(args.edge_margin_mm)]
                                print('After Q closes this camera, run:\n' + shlex.join(command))
                                status = 'SAVED - Q quits; C starts NEW capture'
                            capture, done = None, True
                cv2.imshow(WINDOW, capture_view(table, result, cfg, status))
                key = cv2.waitKey(20) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('c') and capture is None:
                    capture, started, done = diag.Capture(table), time.monotonic(), False
    finally:
        if gui_open:
            cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--grid-mm', type=float, default=10)
    parser.add_argument('--edge-margin-mm', type=float, default=20)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'height_backgrounds')
    args = parser.parse_args(argv)
    cfg = core.Settings(grid_m=args.grid_mm / 1000, edge_margin_m=args.edge_margin_mm / 1000)
    try:
        cfg.validate()
        if not np.isfinite(args.seconds) or not 3 <= args.seconds <= 15:
            raise ValueError('Capture duration must be 3-15 seconds')
    except ValueError as error:
        parser.error(str(error))
    try:
        print('Clear the WHOLE ROI. C explicitly confirms EMPTY. No old diagnostic reports are imported.')
        run_capture(args, cfg)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
        print(f'Error: {error}\nCalibration was not modified.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
