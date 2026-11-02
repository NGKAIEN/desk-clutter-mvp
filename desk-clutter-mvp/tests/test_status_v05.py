"""v05 score semantics, scene lifecycle and packaging entrypoint tests; no hardware."""

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import desk_status_v05 as app
import desk_orientation_v04 as core


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.table = dict(active_mask=np.ones((10, 10), np.uint8))
        self.state = np.full((10, 10), core.FREE, np.uint8)
        self.evidence = self.state.copy()
        self.held = np.zeros((10, 10), bool)

    def measure(self, **kwargs):
        return app.measure(self.state, self.evidence, self.held, self.table, **kwargs)

    def test_empty_is_100(self):
        result = self.measure()
        self.assertEqual(result['free_space_score'], 100)
        self.assertEqual(result['status'], 'READY')

    def test_fully_occupied_is_zero(self):
        self.state[:] = self.evidence[:] = core.OCCUPIED
        self.assertEqual(self.measure()['free_space_score'], 0)

    def test_known_denominator_and_unknown_not_free(self):
        self.state.flat[:30] = core.OCCUPIED
        self.state.flat[90:] = core.UNKNOWN
        self.evidence[:] = self.state
        result = self.measure()
        self.assertAlmostEqual(result['free_space_score'], 100 * 60 / 90)
        self.assertAlmostEqual(result['occupied_known_pct'], 100 * 30 / 90)
        self.assertEqual(result['occupied_active_pct'], 30)
        self.assertEqual(result['unknown_active_pct'], 10)

    def test_low_known_coverage_disables_score(self):
        self.state.flat[:11] = core.UNKNOWN
        self.assertIsNone(self.measure()['free_space_score'])
        self.assertIn('LOW STABLE COVERAGE', self.measure()['reasons'])

    def test_current_missing_disables_score_even_with_known_history(self):
        self.evidence[:] = core.UNKNOWN
        self.held[:] = True
        result = self.measure()
        self.assertEqual(result['stable_known_pct'], 100)
        self.assertIsNone(result['free_space_score'])
        self.assertIn('LOW CURRENT COVERAGE', result['reasons'])

    def test_current_partial_failure_disables_score(self):
        self.evidence.flat[:11] = core.UNKNOWN
        self.assertIsNone(self.measure()['free_space_score'])

    def test_held_gate_includes_opposite_classification(self):
        self.evidence.flat[:11] = core.OCCUPIED
        self.held.flat[:11] = True
        self.assertEqual(self.measure()['status'], 'TOO MUCH HELD STATE')

    def test_exact_policy_boundaries_accepted(self):
        self.evidence.flat[:10] = core.UNKNOWN
        self.held.flat[:10] = True
        self.assertEqual(self.measure()['free_space_score'], 100)

    def test_all_unknown_not_perfect_score(self):
        self.state[:] = self.evidence[:] = core.UNKNOWN
        result = self.measure()
        self.assertIsNone(result['free_space_score'])
        self.assertIsNone(result['occupied_known_pct'])
        self.assertEqual(result['unknown_active_pct'], 100)

    def test_excluded_cells_do_not_affect_counts_or_gates(self):
        self.table['active_mask'][:2] = 0
        self.state[:2] = core.OCCUPIED
        self.evidence[:2] = core.UNKNOWN
        self.held[:2] = True
        result = self.measure()
        self.assertEqual(result['active_cells'], 80)
        self.assertEqual(result['occupied_cells'], 0)
        self.assertEqual(result['free_space_score'], 100)

    def test_pause_and_temporal_off_disable_score(self):
        for option in [dict(paused=True), dict(temporal=False)]:
            with self.subTest(option=option):
                result = self.measure(**option)
                self.assertFalse(result['score_valid'])
                self.assertIsNone(result['free_space_score'])

    def test_score_has_no_angle_or_component_count_input(self):
        self.state.flat[:15] = core.OCCUPIED
        self.evidence[:] = self.state
        self.assertEqual(self.measure()['free_space_score'], 85)

    def test_invalid_shapes_states_and_held_rejected(self):
        with self.assertRaises(ValueError):
            app.measure(self.state[:1], self.evidence, self.held, self.table)
        self.state[0, 0] = 99
        with self.assertRaises(ValueError):
            self.measure()
        self.state[0, 0] = core.UNKNOWN
        self.held[0, 0] = True
        with self.assertRaises(ValueError):
            self.measure()

    def test_empty_roi_rejected(self):
        self.table['active_mask'][:] = 0
        with self.assertRaises(ValueError):
            self.measure()

    def test_invalid_policy_rejected(self):
        for policy in [app.ScorePolicy(min_known=0), app.ScorePolicy(min_current=np.nan),
                       app.ScorePolicy(max_held=1.1)]:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.measure(policy=policy)


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.cfg = core.Settings()
        self.table = core.demo_table(self.cfg)
        self.filter = core.TemporalGrid(self.table['active_mask'].shape, self.cfg)
        self.rng = np.random.default_rng(4)
        self.frame = 0

    def advance(self, scene, count=30):
        for _ in range(count):
            points = app.demo_vertices(self.table, self.cfg, self.rng, scene)
            observation = core.make_observation(points, self.table, self.cfg)
            stable, held = self.filter.update(observation.evidence, self.frame / 30)
            self.frame += 1
        labels, regions = core.extract_regions(stable, self.table, self.cfg)
        metrics = app.measure(stable, observation.evidence, held, self.table)
        return observation, stable, held, labels, regions, metrics

    def test_empty_objects_rotation_removal_and_loss_lifecycle(self):
        empty = self.advance('empty')
        self.assertAlmostEqual(empty[-1]['free_space_score'], 100, delta=.1)
        self.assertEqual(len(empty[-2]), 0)
        objects = self.advance('objects')
        self.assertTrue(objects[-1]['score_valid'])
        self.assertLess(objects[-1]['free_space_score'], 96)
        self.assertEqual(len(objects[-2]), 2)
        self.assertEqual(sum(r['angle'] is not None for r in objects[-2]), 1)
        rotated = self.advance('rotated')
        self.assertAlmostEqual(objects[-1]['free_space_score'], rotated[-1]['free_space_score'], delta=.5)
        angle = next(r['angle'] for r in rotated[-2] if r['angle'] is not None)
        self.assertAlmostEqual(angle, 60, delta=3)
        removed = self.advance('removed')
        self.assertAlmostEqual(removed[-1]['free_space_score'], 100, delta=.1)
        self.assertEqual(len(removed[-2]), 0)
        missing = self.advance('missing', 1)
        self.assertGreater(missing[-1]['stable_known_pct'], 90)
        self.assertIsNone(missing[-1]['free_space_score'])
        expired = self.advance('missing', 20)
        self.assertEqual(expired[-1]['unknown_active_pct'], 100)
        restored = self.advance('empty')
        self.assertTrue(restored[-1]['score_valid'])

    def test_warmup_is_not_100(self):
        self.assertIsNone(self.advance('empty', 1)[-1]['free_space_score'])

    def test_small_component_without_pca_still_counts(self):
        stable = np.full(self.table['active_mask'].shape, core.FREE, np.uint8)
        stable[20, 20] = core.OCCUPIED
        labels, regions = core.extract_regions(stable, self.table, self.cfg)
        metrics = app.measure(stable, stable, np.zeros_like(stable, bool), self.table)
        self.assertEqual(regions, [])
        self.assertEqual(metrics['occupied_cells'], 1)
        self.assertLess(metrics['free_space_score'], 100)

    def test_render_and_report_ready_and_unknown(self):
        for scene in ['objects', 'missing']:
            with self.subTest(scene=scene):
                observation, stable, held, labels, regions, metrics = self.advance(scene)
                picture = app.render(self.table, observation, stable, held, labels, regions,
                                     self.cfg, app.ScorePolicy(), metrics)
                self.assertEqual(picture.shape, (794, 1128, 3))
                report = app.make_report(self.table, metrics, regions, self.cfg, app.ScorePolicy(),
                                         dict(kind='synthetic', seed=4), scene, self.frame)
                json.dumps(report, allow_nan=False)
                self.assertEqual(report['source']['demo_scene'], scene)
                self.assertNotIn('path', report['source'])

    def test_pca_gate_does_not_change_score(self):
        _, stable, held, _, _, metrics = self.advance('objects')
        strict = replace(self.cfg, pca_ratio=9999)
        _, regions = core.extract_regions(stable, self.table, strict)
        self.assertTrue(all(r['angle'] is None for r in regions))
        self.assertEqual(metrics['free_space_score'], app.measure(stable, stable, held, self.table)['free_space_score'])


class PcaGateTests(unittest.TestCase):
    def test_cli_default_and_only_ratio_changes(self):
        _, default = app.parse_args(['--demo'])
        _, trial = app.parse_args(['--demo', '--pca-ratio', '1.5'])
        self.assertEqual(default.pca_ratio, 2.5)
        self.assertEqual(trial, replace(default, pca_ratio=1.5))

    def test_invalid_ratio_rejected(self):
        for value in ['1', '0', '-2', 'nan', 'inf', '-inf']:
            with self.subTest(value=value), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args(['--demo', '--pca-ratio=' + value])

    def test_moderate_rectangle_rotation_at_trial_gate(self):
        _, cfg = app.parse_args(['--demo', '--pca-ratio', '1.5'])
        y, x = np.indices((10, 13))
        points = np.column_stack([x.ravel(), y.ravel()]) * .01
        for degrees in [0, 30, 60, 120, 179]:
            rad = np.radians(degrees)
            rotation = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            rotated = points @ rotation.T
            strict = core.pca_direction(rotated, replace(cfg, pca_ratio=2.5))
            trial = core.pca_direction(rotated, cfg)
            self.assertIsNone(strict[0])
            self.assertEqual(strict[4], 'no dominant axis')
            self.assertGreater(trial[2], 1.5)
            self.assertLess(trial[2], 2.5)
            self.assertAlmostEqual((trial[0] - degrees + 90) % 180 - 90, 0)

    def test_square_and_short_axis_still_rejected(self):
        _, cfg = app.parse_args(['--demo', '--pca-ratio', '1.5'])
        y, x = np.indices((10, 10))
        square = np.column_stack([x.ravel(), y.ravel()]) * .01
        self.assertEqual(core.pca_direction(square, cfg)[4], 'no dominant axis')
        tiny = np.array([[0, 0], [.01, 0], [.02, 0], [.03, .01]])
        self.assertEqual(core.pca_direction(tiny, cfg)[4], 'too small')

    def test_region_labels_and_score_unchanged(self):
        _, cfg = app.parse_args(['--demo'])
        trial = replace(cfg, pca_ratio=1.5)
        table = core.demo_table(cfg)
        state = np.full(table['active_mask'].shape, core.FREE, np.uint8)
        state[20:30, 20:33] = core.OCCUPIED
        state[table['active_mask'] == 0] = core.OCCUPIED  # Excluded border must stay excluded.
        before = state.copy()
        metrics = app.measure(state, state, np.zeros_like(state, bool), table)
        labels, strict_regions = core.extract_regions(state, table, cfg)
        relaxed_labels, relaxed_regions = core.extract_regions(state, table, trial)
        np.testing.assert_array_equal(labels, relaxed_labels)
        np.testing.assert_array_equal(state, before)
        self.assertTrue(np.all(relaxed_labels[table['active_mask'] == 0] == 0))
        self.assertEqual(len(strict_regions), 1)
        self.assertIsNone(strict_regions[0]['angle'])
        self.assertIsNotNone(relaxed_regions[0]['angle'])
        self.assertEqual(strict_regions[0]['area_cells'], relaxed_regions[0]['area_cells'])
        self.assertEqual(metrics, app.measure(state, state, np.zeros_like(state, bool), table))

    def test_existing_reference_compatible_and_local_observations_unchanged(self):
        _, cfg = app.parse_args(['--demo'])
        trial = replace(cfg, pca_ratio=1.5)
        table = core.demo_table(cfg)
        reference, _ = app.background.synthetic_reference(table, cfg)
        strict_detector = app.background.LocalHeightDetector(reference, table, cfg, None)
        trial_detector = app.background.LocalHeightDetector(reference, table, trial, None)
        rng = np.random.default_rng(4)
        for i in range(8):
            points = app.demo_vertices(table, cfg, rng, 'objects', low_profile=True)
            strict = strict_detector.observe(points, i / 30)
            relaxed = trial_detector.observe(points, i / 30)
            np.testing.assert_array_equal(strict.raw, relaxed.raw)
            np.testing.assert_array_equal(strict.evidence, relaxed.evidence)

    def test_trial_export_records_ratio_and_ui_label(self):
        with tempfile.TemporaryDirectory() as base, patch.object(app.cv2, 'putText', wraps=cv2.putText) as draw:
            self.assertEqual(app.main(['--demo', '--demo-background', '--headless',
                                       '--pca-ratio', '1.5', '--output-dir', base]), 0)
            report = json.loads(next(Path(base).glob('v05_*/report.json')).read_text())
            self.assertEqual(report['settings']['pca_ratio'], 1.5)
            self.assertEqual(report['detection']['mode'], 'local_height')
            self.assertTrue(any('PCA r >= 1.5' in call.args[1] for call in draw.call_args_list))


class PcaAreaGateTests(unittest.TestCase):
    def setUp(self):
        _, self.cfg = app.parse_args(['--demo', '--pca-ratio', '1.5'])
        self.trial = replace(self.cfg, min_pca_area_cm2=50)
        self.table = dict(active_mask=np.ones((100, 100), np.uint8), grid_m=.01,
                          u_min=0., v_min=0., reference_deg=0.)
        self.state = np.full((100, 100), core.FREE, np.uint8)

    def test_cli_default_trial_and_invalid(self):
        self.assertEqual(self.cfg.min_pca_area_cm2, 0)
        _, cfg = app.parse_args(['--demo', '--pca-ratio', '1.5', '--min-pca-area-cm2', '50'])
        self.assertEqual(cfg, self.trial)
        for value in ['-1', 'nan', 'inf', '-inf']:
            with self.subTest(value=value), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args(['--demo', '--min-pca-area-cm2=' + value])

    def test_small_elongated_region_kept_but_direction_suppressed(self):
        self.state[10:12, 10:18] = core.OCCUPIED
        before = self.state.copy()
        metrics = app.measure(self.state, self.state, np.zeros_like(self.state, bool), self.table)
        labels, original = core.extract_regions(self.state, self.table, self.cfg)
        new_labels, regions = core.extract_regions(self.state, self.table, self.trial)
        self.assertIsNotNone(original[0]['angle'])
        self.assertEqual(len(regions), len(original))
        self.assertEqual(regions[0]['area_cm2'], 16)
        self.assertIsNone(regions[0]['angle'])
        self.assertEqual(regions[0]['reason'], 'area below PCA minimum')
        np.testing.assert_array_equal(labels, new_labels)
        np.testing.assert_array_equal(self.state, before)
        self.assertEqual(metrics, app.measure(self.state, self.state, np.zeros_like(self.state, bool), self.table))
        self.assertEqual(metrics['occupied_cells'], 16)

    def test_area_boundary_is_inclusive(self):
        for count in [49, 50]:
            with self.subTest(count=count):
                self.state[:] = core.FREE
                self.state[10, 10:10 + count] = core.OCCUPIED
                _, regions = core.extract_regions(self.state, self.table, self.trial)
                self.assertEqual(regions[0]['area_cm2'], count)
                self.assertEqual(regions[0]['angle'] is not None, count == 50)

    def test_physical_area_not_cell_count_across_grids(self):
        for grid, shape, accepted in [(.005, (2, 20), False), (.005, (5, 40), True),
                                       (.02, (2, 6), False), (.02, (1, 13), True)]:
            with self.subTest(grid=grid, shape=shape):
                self.state[:] = core.FREE
                h, w = shape
                self.state[10:10 + h, 10:10 + w] = core.OCCUPIED
                table = dict(self.table, grid_m=grid)
                _, regions = core.extract_regions(self.state, table, replace(self.trial, grid_m=grid))
                self.assertAlmostEqual(regions[0]['area_cm2'], h * w * (grid * 100) ** 2)
                self.assertEqual(regions[0]['angle'] is not None, accepted)

    def test_actual_occupied_area_not_bounding_box(self):
        self.state[10, 10:30] = core.OCCUPIED
        self.state[10:30, 10] = core.OCCUPIED
        _, regions = core.extract_regions(self.state, self.table, self.trial)
        self.assertEqual(regions[0]['area_cm2'], 39)
        self.assertEqual(regions[0]['reason'], 'area below PCA minimum')

    def test_large_rectangle_passes_but_square_and_excluded_cells_do_not(self):
        self.state[10:30, 10:40] = core.OCCUPIED
        _, original = core.extract_regions(self.state, self.table, self.cfg)
        _, trial = core.extract_regions(self.state, self.table, self.trial)
        self.assertIsNotNone(trial[0]['angle'])
        self.assertEqual(original[0]['angle'], trial[0]['angle'])
        self.state[:] = core.FREE
        self.state[10:20, 10:20] = core.OCCUPIED
        _, square = core.extract_regions(self.state, self.table, self.trial)
        self.assertEqual(square[0]['reason'], 'no dominant axis')
        self.table['active_mask'][10:20, 10:20] = 0
        _, excluded = core.extract_regions(self.state, self.table, self.trial)
        self.assertEqual(excluded, [])

    def test_local_reference_and_classification_unchanged(self):
        table = core.demo_table(self.cfg)
        reference, _ = app.background.synthetic_reference(table, self.cfg)
        original = app.background.LocalHeightDetector(reference, table, self.cfg, None)
        trial = app.background.LocalHeightDetector(reference, table, self.trial, None)
        rng = np.random.default_rng(4)
        for i in range(8):
            points = app.demo_vertices(table, self.cfg, rng, 'objects', low_profile=True)
            a, b = original.observe(points, i / 30), trial.observe(points, i / 30)
            np.testing.assert_array_equal(a.raw, b.raw)
            np.testing.assert_array_equal(a.evidence, b.evidence)

    def test_trial_export_and_ui(self):
        with tempfile.TemporaryDirectory() as base, patch.object(app.cv2, 'putText', wraps=cv2.putText) as draw:
            self.assertEqual(app.main(['--demo', '--demo-background', '--headless', '--pca-ratio', '1.5',
                                       '--min-pca-area-cm2', '50', '--output-dir', base]), 0)
            report = json.loads(next(Path(base).glob('v05_*/report.json')).read_text())
            self.assertEqual(report['settings']['min_pca_area_cm2'], 50)
            self.assertTrue(any('area >= 50cm2' in call.args[1] for call in draw.call_args_list))


class WorkflowTests(unittest.TestCase):
    def test_headless_has_no_camera_gui_or_private_calibration(self):
        with tempfile.TemporaryDirectory() as base, \
                patch.object(app, 'camera_source', side_effect=AssertionError('No camera')), \
                patch.object(app.cv2, 'imshow', side_effect=AssertionError('No GUI')):
            self.assertEqual(app.main(['--demo', '--headless', '--output-dir', base]), 0)
            folders = list(Path(base).glob('v05_*'))
            self.assertEqual(len(folders), 1)
            report = json.loads((folders[0] / 'report.json').read_text())
            self.assertTrue(report['metrics']['score_valid'])
            self.assertEqual(report['source']['kind'], 'synthetic')
            self.assertIsNone(report['source']['calibration_sha256'])
            self.assertEqual(cv2.imread(str(folders[0] / 'view.png')).shape, (794, 1128, 3))

    def test_headless_unknown_and_temporal_off_export_null(self):
        for options in [['--demo-scene', 'missing'], ['--no-temporal']]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as base:
                self.assertEqual(app.main(['--demo', '--headless', '--output-dir', base] + options), 0)
                report = json.loads(next(Path(base).glob('v05_*/report.json')).read_text())
                self.assertIsNone(report['metrics']['free_space_score'])

    def test_invalid_cli_cannot_silently_select_calibration(self):
        for options in [[], ['--headless'], ['--demo', '--calibration', 'x.json'],
                        ['--demo', '--grid-mm', '0'], ['--demo', '--min-height-mm', 'nan']]:
            with self.subTest(options=options), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args(options)

    def test_missing_calibration_fails_before_camera(self):
        with tempfile.TemporaryDirectory() as base, patch.object(app, 'camera_source', side_effect=AssertionError('No camera')):
            self.assertEqual(app.main(['--calibration', str(Path(base) / 'missing.json')]), 1)

    def test_save_collision_never_overwrites(self):
        with tempfile.TemporaryDirectory() as base, patch.object(app, 'datetime') as clock:
            clock.now.return_value.strftime.return_value = 'fixed'
            path = app.save_result(base, np.zeros((10, 10, 3), np.uint8), {'first': True})
            with self.assertRaises(FileExistsError):
                app.save_result(base, np.zeros((10, 10, 3), np.uint8), {'changed': True})
            self.assertEqual(json.loads((path / 'report.json').read_text()), {'first': True})

    def test_live_keys_reports_readonly_calibration_and_cleanup(self):
        cfg = core.Settings()
        table = core.demo_table(cfg)
        original = {k: table[k].tolist() for k in ['normal', 'origin', 'u_axis', 'v_axis', 'polygon_uv']}
        original['d'] = table['d']
        original['baseline'] = .123
        raw = json.dumps(original).encode()
        rng = np.random.default_rng(4)
        closed = []
        @contextmanager
        def source():
            try:
                yield lambda: app.demo_vertices(table, cfg, rng, 'objects')
            finally:
                closed.append(True)
        keys = [-1] * 8 + [ord('s'), ord(' '), ord('s'), ord(' ')] + [-1] * 8
        keys += [ord('t'), ord('s'), ord('t'), ord('r'), ord('n'), ord('q')]
        with tempfile.TemporaryDirectory() as base, ExitStack() as stack:
            calibration = Path(base) / 'calibration.json'
            calibration.write_bytes(raw)
            stack.enter_context(patch.object(app, 'camera_source', source))
            for method in ['namedWindow', 'imshow']:
                stack.enter_context(patch.object(app.cv2, method))
            destroyed = stack.enter_context(patch.object(app.cv2, 'destroyAllWindows'))
            stack.enter_context(patch.object(app.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(app.cv2, 'waitKey', side_effect=keys))
            stack.enter_context(patch.object(app.time, 'monotonic', side_effect=np.arange(1000) / 30))
            self.assertEqual(app.main(['--calibration', str(calibration), '--output-dir', base]), 0)
            reports = [json.loads(p.read_text()) for p in sorted(Path(base).glob('v05_*/report.json'))]
            self.assertEqual(len(reports), 3)
            self.assertTrue(reports[0]['metrics']['score_valid'])
            self.assertIsNone(reports[1]['metrics']['free_space_score'])
            self.assertEqual(reports[1]['metrics']['status'], 'PAUSED')
            self.assertEqual(reports[2]['metrics']['status'], 'TEMPORAL OFF')
            self.assertEqual(calibration.read_bytes(), raw)
            self.assertEqual(closed, [True])
            destroyed.assert_called_once()

    def test_camera_and_gui_released_on_display_failure(self):
        closed = []
        @contextmanager
        def source():
            try:
                yield lambda: np.empty((0, 3))
            finally:
                closed.append(True)
        cfg = core.Settings()
        table = core.demo_table(cfg)
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, 'load_context', return_value=(table, dict(kind='live'))))
            stack.enter_context(patch.object(app, 'camera_source', source))
            stack.enter_context(patch.object(app.cv2, 'namedWindow'))
            stack.enter_context(patch.object(app.cv2, 'imshow', side_effect=RuntimeError('display failure')))
            destroyed = stack.enter_context(patch.object(app.cv2, 'destroyAllWindows'))
            self.assertEqual(app.main(['--calibration', 'unused.json']), 1)
            self.assertEqual(closed, [True])
            destroyed.assert_called_once()


if __name__ == '__main__':
    unittest.main()
