"""Optional empty-reference detector; synthetic data only, no device or private files."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import desk_height_background as bg
import desk_height_diagnostic as diag
import desk_orientation_v04 as core
import desk_status_v05 as app


class MedianTests(unittest.TestCase):
    def setUp(self):
        self.filter = bg.HeightMedian((1, 1), bg.HeightPolicy())

    def update(self, value, now):
        return self.filter.update(np.array([[value]], float), now)[0, 0]

    def test_needs_three_fresh_samples(self):
        self.assertTrue(np.isnan(self.update(8, 0)))
        self.assertTrue(np.isnan(self.update(8, .03)))
        self.assertEqual(self.update(8, .06), 8)

    def test_isolated_spike_is_rejected(self):
        for i, value in enumerate([0, 0, 40, 0, 0]):
            last = self.update(value, i / 30)
        self.assertEqual(last, 0)

    def test_missing_current_is_not_filled(self):
        for i in range(5):
            self.update(8, i / 30)
        self.assertTrue(np.isnan(self.update(np.nan, .2)))

    def test_age_expiry_and_reset_require_new_samples(self):
        for i in range(5):
            self.update(8, i / 30)
        self.assertTrue(np.isnan(self.update(8, 1)))
        self.filter.reset()
        self.assertTrue(np.isnan(self.update(8, 0)))

    def test_very_slow_stream_cannot_use_stale_heights(self):
        for i in range(10):
            self.assertTrue(np.isnan(self.update(8, i * .3)))

    def test_invalid_shape_time_and_infinity_rejected(self):
        self.update(0, 1)
        for value, stamp in [(np.inf, 2), (0, np.nan), (0, .5)]:
            with self.subTest(value=value, stamp=stamp), self.assertRaises(ValueError):
                self.update(value, stamp)
        with self.assertRaises(ValueError):
            self.filter.update(np.zeros((2, 2)), 2)

    def test_invalid_policies_rejected(self):
        for policy in [bg.HeightPolicy(window=2), bg.HeightPolicy(min_samples=0),
                       bg.HeightPolicy(floor_mm=13), bg.HeightPolicy(max_age_s=0),
                       bg.HeightPolicy(noise_quantile=1), bg.HeightPolicy(noise_margin_mm=np.nan),
                       bg.HeightPolicy(min_coverage=.4)]:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                policy.validate()


class Base(unittest.TestCase):
    def setUp(self):
        self.cfg = core.Settings(grid_m=.05)
        self.table = core.demo_table(self.cfg)
        self.shape = self.table['active_mask'].shape
        self.active = self.table['active_mask'] > 0
        self.cell = tuple(np.argwhere(self.active)[20])

    def frame(self, height=-4., counts=8):
        values = np.broadcast_to(np.asarray(height, float), self.shape).copy()
        return diag.FrameGrid(values, np.ones(self.shape), np.full(self.shape, counts),
                              np.zeros(self.shape, bool))

    def capture(self, values=None, n=60, step=1 / 30):
        capture = diag.Capture(self.table)
        for i in range(n):
            capture.add(self.frame(-4. if values is None else values(i)), i * step)
        return capture

    def reference(self, capture=None):
        capture = capture or self.capture()
        return bg.fit_reference(capture, self.table, self.cfg, 'calibration-hash', capture.times[-1] + .01)

    def detector(self, data=None):
        return bg.LocalHeightDetector(data or self.reference(), self.table, self.cfg, 'calibration-hash')


class ReferenceTests(Base):
    def test_signed_offsets_and_gate_floor_round_trip(self):
        data = json.loads(json.dumps(self.reference(), allow_nan=False))
        offsets, gates, valid, _ = bg.validate_reference(data, self.table, self.cfg, 'calibration-hash')
        np.testing.assert_array_equal(valid, self.active)
        self.assertTrue((offsets[valid] == -4).all())
        self.assertTrue((gates[valid] == 6).all())
        self.assertTrue(np.isnan(offsets[~valid]).all())

    def test_moderate_noise_raises_gate(self):
        data = self.reference(self.capture(lambda i: -4 + (9 if i % 20 < 10 else -9)))
        self.assertAlmostEqual(data['capture']['gate_min_median_max_mm'][1], 11)

    def test_excessive_noise_or_offset_rejected(self):
        for values in [lambda i: -4 + (15 if i % 20 < 10 else -15), lambda i: 30]:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'usable reference'):
                self.reference(self.capture(values))

    def test_insufficient_frames_and_low_frame_rate_rejected(self):
        for capture in [self.capture(n=59), self.capture(step=.3)]:
            with self.subTest(capture=capture), self.assertRaises(ValueError):
                self.reference(capture)

    def test_poor_coverage_is_not_zero_filled(self):
        with self.assertRaisesRegex(ValueError, 'usable reference'):
            self.reference(self.capture(lambda i: np.nan if i % 5 == 0 else -4))

    def test_isolated_bad_reference_cell_remains_unknown(self):
        values = np.full(self.shape, -4.)
        values[self.cell] = np.nan
        detector = self.detector(self.reference(self.capture(lambda i: values)))
        for i in range(5):
            observation = detector.observe_frame(self.frame(30), i / 30)
        self.assertEqual(observation.raw[self.cell], core.UNKNOWN)
        self.assertEqual(observation.evidence[self.cell], core.UNKNOWN)

    def test_hash_grid_inset_and_sampling_mismatch_rejected(self):
        data = self.reference()
        for cfg, digest in [(self.cfg, 'changed'), (replace(self.cfg, grid_m=.04), 'calibration-hash'),
                            (replace(self.cfg, edge_margin_m=.03), 'calibration-hash'),
                            (replace(self.cfg, min_cell_points=4), 'calibration-hash'),
                            (replace(self.cfg, max_depth_m=3), 'calibration-hash')]:
            with self.subTest(cfg=cfg, digest=digest), self.assertRaisesRegex(ValueError, 'mismatch'):
                bg.validate_reference(data, self.table, cfg, digest)

    def test_ordinary_diagnostics_and_non_objects_rejected(self):
        for data in [[], {}, {'frame_count': 127, 'maps': {}}, dict(self.reference(), operator_confirmed_empty=False)]:
            with self.subTest(data=type(data)), self.assertRaises(ValueError):
                bg.validate_reference(data, self.table, self.cfg, 'calibration-hash')

    def test_corrupt_arrays_rejected(self):
        original = self.reference()
        y, x = self.cell
        for name, value in [('offsets_mm', None), ('offsets_mm', float('inf')),
                            ('gates_mm', 100), ('valid', False)]:
            data = deepcopy(original)
            data[name][y][x] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                bg.validate_reference(data, self.table, self.cfg, 'calibration-hash')

    def test_invalid_shape_and_non_boolean_mask_rejected(self):
        for data in [dict(self.reference(), offsets_mm=[[0]]),
                     dict(self.reference(), valid=self.active.astype(int).tolist())]:
            with self.assertRaises(ValueError):
                bg.validate_reference(data, self.table, self.cfg, 'calibration-hash')

    def test_flat_object_contamination_cannot_be_semantically_detected(self):
        # Explicit limitation: capture must be operator-confirmed EMPTY.
        data = self.reference(self.capture(lambda i: 8.))
        self.assertIn('no semantic check', data['limitations'][0])
        self.assertEqual(data['offsets_mm'][self.cell[0]][self.cell[1]], 8.)

    def test_save_load_hash_and_no_overwrite(self):
        capture = self.capture()
        data = self.reference(capture)
        result = capture.summarize(self.table, 2.)
        with tempfile.TemporaryDirectory() as base, patch.object(bg, 'datetime') as clock:
            clock.now.return_value.strftime.return_value = 'fixed'
            directory = bg.save_reference(base, data, capture, self.table, self.cfg, result)
            detector, info = bg.load_reference(directory / 'background.json', self.table, self.cfg, 'calibration-hash')
            self.assertEqual(info['background_sha256'], hashlib.sha256((directory / 'background.json').read_bytes()).hexdigest())
            self.assertEqual(info['background_name'], 'background.json')
            self.assertEqual(detector.valid.sum(), self.active.sum())
            self.assertEqual(cv2.imread(str(directory / 'capture.png')).shape, (740, 1128, 3))
            with np.load(directory / 'samples.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['median_mm'].shape[0], 60)
            with self.assertRaises(FileExistsError):
                bg.save_reference(base, data, capture, self.table, self.cfg, result)
            self.assertEqual(json.loads((directory / 'background.json').read_text()), data)

    def test_synthetic_reference_cannot_load_as_live(self):
        data = dict(self.reference(), synthetic=True)
        with tempfile.TemporaryDirectory() as base:
            path = Path(base) / 'synthetic.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Synthetic'):
                bg.load_reference(path, self.table, self.cfg, 'calibration-hash')


class DetectionTests(Base):
    def test_eight_mm_relative_change_detected_despite_four_mm_absolute(self):
        detector = self.detector()
        for i in range(5):
            observation = detector.observe_frame(self.frame(4), i / 30)
        self.assertTrue((observation.evidence[self.active] == core.OCCUPIED).all())
        self.assertTrue((observation.raw[self.active] == core.OCCUPIED).all())

    def test_empty_and_large_negative_or_too_tall_are_distinct(self):
        detector = self.detector()
        for height, expected in [(-4, core.FREE), (-20, core.UNKNOWN), (410, core.UNKNOWN), (2, core.FREE)]:
            with self.subTest(height=height):
                detector.reset()
                observation = detector.observe_frame(self.frame(height), 0, smooth=False)
                self.assertTrue((observation.evidence[self.active] == expected).all())

    def test_sparse_and_missing_current_stay_unknown(self):
        detector = self.detector()
        for i in range(5):
            detector.observe_frame(self.frame(4), i / 30)
        for frame in [self.frame(np.nan), self.frame(4, counts=2)]:
            observation = detector.observe_frame(frame, .2)
            self.assertFalse(observation.raw.any())
            self.assertFalse(observation.evidence.any())

    def test_missing_current_immediately_disables_score_despite_history(self):
        detector = self.detector()
        stabilizer = core.TemporalGrid(self.shape, self.cfg)
        for i in range(10):
            observation = detector.observe_frame(self.frame(), i / 30)
            stable, held = stabilizer.update(observation.evidence, i / 30)
        observation = detector.observe_frame(self.frame(np.nan), 10 / 30)
        stable, held = stabilizer.update(observation.evidence, 10 / 30)
        current, held = app.quality_inputs(observation, stable, held, local=True)
        metrics = app.measure(stable, current, held, self.table)
        self.assertEqual(metrics['stable_known_pct'], 100)
        self.assertEqual(metrics['current_known_pct'], 0)
        self.assertIsNone(metrics['free_space_score'])

    def test_unfiltered_disagreement_counts_as_held(self):
        detector = self.detector()
        for i in range(5):
            observation = detector.observe_frame(self.frame(), i / 30)
        observation = detector.observe_frame(self.frame(4), .2)
        stable = observation.evidence.copy()
        current, held = app.quality_inputs(observation, stable, np.zeros(self.shape, bool), local=True)
        self.assertTrue((stable[self.active] == core.FREE).all())
        self.assertTrue((current[self.active] == core.OCCUPIED).all())
        self.assertTrue(held[self.active].all())
        self.assertEqual(app.measure(stable, current, held, self.table)['status'], 'TOO MUCH HELD STATE')

    def test_actual_points_pass_through_median_not_absolute_support(self):
        detector = self.detector()
        y, x = self.cell
        uv = np.array([x + .5, y + .5]) * self.cfg.grid_m
        points = (self.table['origin'] + uv[0] * self.table['u_axis'] + uv[1] * self.table['v_axis']
                  + np.full((5, 1), .004) * self.table['normal'])
        self.assertEqual(core.make_observation(points, self.table, self.cfg).evidence[self.cell], core.FREE)
        self.assertEqual(detector.observe(points, 0, smooth=False).evidence[self.cell], core.OCCUPIED)


class IntegrationTests(unittest.TestCase):
    def test_local_lifecycle_uses_independent_noise_and_preserves_pca(self):
        cfg = core.Settings()
        table = core.demo_table(cfg)
        data, _ = bg.synthetic_reference(table, cfg)
        detector = bg.LocalHeightDetector(data, table, cfg, None)
        state_filter = core.TemporalGrid(table['active_mask'].shape, cfg)
        rng, number = np.random.default_rng(42), 0

        def advance(scene, count=30):
            nonlocal number
            for _ in range(count):
                points = app.demo_vertices(table, cfg, rng, scene, low_profile=True)
                observation = detector.observe(points, number / 30)
                stable, held = state_filter.update(observation.evidence, number / 30)
                current, held = app.quality_inputs(observation, stable, held, local=True)
                number += 1
            _, regions = core.extract_regions(stable, table, cfg)
            return app.measure(stable, current, held, table), regions

        empty, regions = advance('empty')
        self.assertGreater(empty['free_space_score'], 99)
        self.assertEqual(len(regions), 0)
        objects, regions = advance('objects')
        self.assertLess(objects['free_space_score'], 96)
        self.assertEqual(len(regions), 2)
        rotated, regions = advance('rotated')
        self.assertAlmostEqual(rotated['free_space_score'], objects['free_space_score'], delta=1)
        angle = next(r['angle'] for r in regions if r['angle'] is not None)
        self.assertAlmostEqual(angle, 60, delta=5)
        removed, regions = advance('removed')
        self.assertGreater(removed['free_space_score'], 99)
        self.assertEqual(len(regions), 0)
        missing, _ = advance('missing', 1)
        self.assertIsNone(missing['free_space_score'])
        missing, _ = advance('missing', 20)
        self.assertEqual(missing['unknown_active_pct'], 100)

    def test_local_headless_has_no_camera_or_gui_and_exports_mode(self):
        with tempfile.TemporaryDirectory() as base, \
                patch.object(app, 'camera_source', side_effect=AssertionError('No camera')), \
                patch.object(app.cv2, 'imshow', side_effect=AssertionError('No GUI')):
            self.assertEqual(app.main(['--demo', '--demo-background', '--headless', '--output-dir', base]), 0)
            report = json.loads(next(Path(base).glob('v05_*/report.json')).read_text())
            self.assertEqual(report['detection']['mode'], 'local_height')
            self.assertTrue(report['detection']['synthetic'])
            self.assertTrue(report['metrics']['score_valid'])
            self.assertIsNone(report['detection']['background_sha256'])

    def test_cli_rejects_ambiguous_modes_and_ignored_threshold(self):
        for options in [['--demo-background'], ['--demo', '--background', 'x.json'],
                        ['--demo', '--demo-background', '--min-height-mm', '3'],
                        ['--calibration', 'x.json', '--demo-background']]:
            with self.subTest(options=options), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args(options)

    def test_bad_reference_fails_before_camera(self):
        cfg = core.Settings()
        with patch.object(app, 'load_context', return_value=(core.demo_table(cfg), dict(calibration_sha256='x'))), \
                patch.object(app, 'camera_source', side_effect=AssertionError('No camera')):
            self.assertEqual(app.main(['--calibration', 'unused', '--background', '/nonexistent/empty.json']), 1)

    def test_keys_reset_both_filters_and_off_does_not_smooth(self):
        cfg = core.Settings()
        table = core.demo_table(cfg)
        data, _ = bg.synthetic_reference(table, cfg)
        detector = bg.LocalHeightDetector(data, table, cfg, None)
        info = bg.reference_info(detector, table, 'synthetic', None, synthetic=True)
        keys = [-1] * 7 + [ord('t'), ord('s'), ord('t'), ord('r'), ord(' '), ord(' '), ord('q')]
        with tempfile.TemporaryDirectory() as base, ExitStack() as stack:
            stack.enter_context(patch.object(app, 'detection_context', return_value=(detector, info)))
            reset = stack.enter_context(patch.object(detector, 'reset', wraps=detector.reset))
            observe = stack.enter_context(patch.object(detector, 'observe', wraps=detector.observe))
            stack.enter_context(patch.object(app.time, 'monotonic', side_effect=np.arange(1000) / 30))
            for name in ['namedWindow', 'imshow', 'destroyAllWindows']:
                stack.enter_context(patch.object(app.cv2, name))
            stack.enter_context(patch.object(app.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(app.cv2, 'waitKey', side_effect=keys))
            self.assertEqual(app.main(['--demo', '--demo-background', '--output-dir', base]), 0)
            self.assertEqual(reset.call_count, 4)
            self.assertTrue(any(call.kwargs['smooth'] is False for call in observe.call_args_list))
            report = json.loads(next(Path(base).glob('v05_*/report.json')).read_text())
            self.assertEqual(report['metrics']['status'], 'TEMPORAL OFF')


class CaptureWorkflowTests(Base):
    def test_no_capture_without_c_and_c_saves_separately(self):
        raw = json.dumps({**{k: self.table[k].tolist() for k in ['normal', 'origin', 'u_axis', 'v_axis', 'polygon_uv']},
                          'd': self.table['d']}).encode()
        for capture_requested in [False, True]:
            closed = []

            @contextmanager
            def source():
                try:
                    yield lambda: np.empty((0, 3))
                finally:
                    closed.append(True)

            keys = [ord('c')] + [-1] * 105 + [ord('q')] if capture_requested else [ord('q')]
            with self.subTest(capture=capture_requested), tempfile.TemporaryDirectory() as base, ExitStack() as stack:
                path = Path(base) / 'calibration.json'
                path.write_bytes(raw)
                stack.enter_context(patch.object(diag, 'camera_source', source))
                stack.enter_context(patch.object(diag, 'frame_grid', return_value=self.frame()))
                stack.enter_context(patch.object(bg.time, 'monotonic', side_effect=np.arange(300) / 30))
                for name in ['namedWindow', 'imshow']:
                    stack.enter_context(patch.object(bg.cv2, name))
                destroyed = stack.enter_context(patch.object(bg.cv2, 'destroyAllWindows'))
                stack.enter_context(patch.object(bg.cv2, 'getWindowProperty', return_value=1))
                stack.enter_context(patch.object(bg.cv2, 'waitKey', side_effect=keys))
                self.assertEqual(bg.main(['--calibration', str(path), '--grid-mm', '50', '--seconds', '3', '--output-dir', base]), 0)
                self.assertEqual(len(list(Path(base).glob('background_*/background.json'))), int(capture_requested))
                self.assertEqual(path.read_bytes(), raw)
                self.assertEqual(closed, [True])
                destroyed.assert_called_once()

    def test_invalid_duration_is_rejected_without_camera(self):
        for duration in ['0', 'nan', '20']:
            with self.subTest(duration=duration), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                bg.main(['--calibration', 'unused', '--seconds', duration])


if __name__ == '__main__':
    unittest.main()
