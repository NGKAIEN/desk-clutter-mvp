"""Offline calibration tests. Real camera and GUI are never opened."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import desk_calibrate_multiframe as app
import desk_height_diagnostic as diag
import desk_orientation_v04 as v04


class Base(unittest.TestCase):
    def setUp(self):
        self.cfg = v04.Settings(grid_m=.020)
        self.table = v04.demo_table(self.cfg)
        self.original = {k: self.table[k].tolist() for k in
                         ["origin", "normal", "u_axis", "v_axis", "polygon_uv"]}
        self.original.update(d=self.table["d"], grid_size=.02, baseline=.17, extra={"keep": True})
        self.raw = (json.dumps(self.original, indent=2) + "\n").encode()
        self.beta = np.array([-.014, .007, .008])

    def result(self, beta=None):
        beta = self.beta if beta is None else np.asarray(beta)
        heights = (app.grid_uv(self.table) @ beta[:2] + beta[2]) * 1000
        shape = heights.shape
        return dict(median_mm=heights, temporal_span_mm=np.full(shape, 8.),
                    spatial_span_mm=np.full(shape, 1.), object_support_pct=np.zeros(shape),
                    coverage_pct=np.full(shape, 100.), reliable=self.table["table_mask"] > 0,
                    valid_frames=np.full(shape, 60), n_frames=60, elapsed_seconds=3., required_frames=30)

    def candidate(self, beta=None, original=None, table=None):
        return app.candidate_calibration(self.original if original is None else original,
                                         self.table if table is None else table,
                                         self.beta if beta is None else np.asarray(beta),
                                         hashlib.sha256(self.raw).hexdigest(), {})

    def capture(self, seed=11, candidate=None):
        rng = np.random.default_rng(seed)
        old, new = diag.Capture(self.table), diag.Capture(self.table)
        for i in range(30):
            points = app.synthetic_points(self.table, rng, self.beta, i)
            old.add(diag.frame_grid(points, self.table, self.cfg), (i + 1) / 10)
            if candidate is not None:
                new.add(diag.frame_grid(points, app.measurement_table(self.table, candidate), self.cfg), (i + 1) / 10)
        return old, new


class FitTests(Base):
    def test_exact_plane_and_units(self):
        beta, stats, inliers = app.fit_plane(self.result(), self.table)
        np.testing.assert_allclose(beta, self.beta, atol=1e-12)
        self.assertEqual(stats["inlier_cells"], np.count_nonzero(self.table["active_mask"]))
        self.assertTrue(np.all(inliers <= (self.table["active_mask"] > 0)))

    def test_noisy_plane_local_outliers_and_determinism(self):
        result = self.result()
        result["median_mm"] += np.random.default_rng(8).normal(0, .5, result["median_mm"].shape)
        uv = app.grid_uv(self.table)
        mask = (uv[..., 0] > .75) & (uv[..., 0] < .90) & (uv[..., 1] > .3) & (uv[..., 1] < .45)
        result["median_mm"][mask] += 22
        first, second = app.fit_plane(result, self.table), app.fit_plane(result, self.table)
        np.testing.assert_allclose(first[0], self.beta, atol=.0002)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertFalse(first[2][mask].any())

    def test_too_few_frames(self):
        result = self.result()
        result["n_frames"] = 29
        with self.assertRaisesRegex(ValueError, "30 captured frames"):
            app.fit_plane(result, self.table)

    def test_poor_quality_rejected(self):
        for key, value in [("coverage_pct", 79), ("median_mm", 31),
                           ("temporal_span_mm", 21), ("temporal_span_mm", np.nan), ("reliable", False)]:
            with self.subTest(key=key, value=value):
                result = self.result()
                result[key][:] = value
                with self.assertRaises(ValueError):
                    app.fit_plane(result, self.table)

    def test_excluded_band_not_used(self):
        result = self.result()
        result["median_mm"][self.table["excluded_mask"] > 0] = 500
        beta, _, inliers = app.fit_plane(result, self.table)
        np.testing.assert_allclose(beta, self.beta, atol=1e-12)
        self.assertFalse(inliers[self.table["excluded_mask"] > 0].any())

    def test_missing_quadrant_rejected(self):
        result = self.result()
        uv = app.grid_uv(self.table)
        result["reliable"][(uv[..., 0] < .6) & (uv[..., 1] < .4)] = False
        with self.assertRaisesRegex(ValueError, "quadrant"):
            app.fit_plane(result, self.table)

    def test_concentrated_strip_rejected(self):
        uv = app.grid_uv(self.table)
        mask = (self.table["active_mask"] > 0) & (uv[..., 1] > .18) & (uv[..., 1] < .62)
        with self.assertRaisesRegex(ValueError, "span"):
            app.require_spread(mask, self.table)

    def test_no_consensus_rejected(self):
        result = self.result()
        result["median_mm"] = np.random.default_rng(21).uniform(-29, 29, result["median_mm"].shape)
        with self.assertRaisesRegex(ValueError, "consensus"):
            app.fit_plane(result, self.table)

    def test_excessive_correction_rejected(self):
        for beta in [[.060, 0, -.036], [.040, 0, -.015]]:
            with self.subTest(beta=beta), self.assertRaisesRegex(ValueError, "Correction too large"):
                app.fit_plane(self.result(beta), self.table)


class GeometryTests(Base):
    def test_orthonormal_frame_and_origin(self):
        table = v04.table_from_data(self.candidate(), self.cfg.grid_m, self.cfg.edge_margin_m)
        basis = np.stack([table[k] for k in ["u_axis", "v_axis", "normal"]])
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis), 1)
        self.assertAlmostEqual(table["origin"] @ table["normal"] + table["d"], 0)

    def test_exact_height_identity(self):
        candidate = self.candidate()
        points = app.synthetic_points(self.table, np.random.default_rng(3), self.beta, 0)
        relative = points - self.table["origin"]
        uv = np.column_stack([relative @ self.table["u_axis"], relative @ self.table["v_axis"]])
        old_height = points @ self.table["normal"] + self.table["d"]
        expected = (old_height - uv @ self.beta[:2] - self.beta[2]) / np.sqrt(1 + self.beta[:2] @ self.beta[:2])
        np.testing.assert_allclose(points @ candidate["normal"] + candidate["d"], expected, atol=1e-12)

    def test_corner_rays_preserved(self):
        candidate = self.candidate()
        old = (self.table["origin"] + self.table["polygon_uv"][:, :1] * self.table["u_axis"]
               + self.table["polygon_uv"][:, 1:] * self.table["v_axis"])
        poly = np.asarray(candidate["polygon_uv"])
        new = (np.asarray(candidate["origin"]) + poly[:, :1] * candidate["u_axis"] + poly[:, 1:] * candidate["v_axis"])
        np.testing.assert_allclose(np.cross(old, new), 0, atol=1e-12)
        np.testing.assert_allclose(new @ candidate["normal"] + candidate["d"], 0, atol=1e-12)
        np.testing.assert_allclose(new[:, :2] / new[:, 2:], old[:, :2] / old[:, 2:], atol=1e-12)

    def test_zero_correction(self):
        candidate = self.candidate([0, 0, 0])
        for key in ["origin", "normal", "u_axis", "v_axis", "polygon_uv", "d"]:
            np.testing.assert_allclose(candidate[key], self.original[key], atol=1e-12)

    def test_original_metadata_and_baseline_protected(self):
        original = deepcopy(self.original)
        candidate = self.candidate()
        self.assertNotIn("baseline", candidate)
        self.assertEqual(self.original, original)
        self.assertEqual(candidate["extra"], original["extra"])
        candidate["extra"]["keep"] = False
        self.assertTrue(self.original["extra"]["keep"])
        self.assertTrue(candidate["multiframe_refinement"]["candidate_only"])
        self.assertTrue(candidate["multiframe_refinement"]["old_baseline_removed"])

    def test_opposite_handedness(self):
        original = deepcopy(self.original)
        original["v_axis"] = (-np.asarray(original["v_axis"])).tolist()
        polygon = np.asarray(original["polygon_uv"])
        polygon[:, 1] *= -1
        original["polygon_uv"] = polygon.tolist()
        table = v04.table_from_data(original, self.cfg.grid_m, self.cfg.edge_margin_m)
        candidate = self.candidate(original=original, table=table)
        self.assertAlmostEqual(np.linalg.det(np.stack([candidate[k] for k in ["u_axis", "v_axis", "normal"]])), -1)

    def test_measurement_table_uses_same_cells(self):
        candidate = self.candidate()
        normal = self.table["normal"].copy()
        adjusted = app.measurement_table(self.table, candidate)
        points = app.synthetic_points(self.table, np.random.default_rng(3), self.beta, 0)
        old = diag.frame_grid(points, self.table, self.cfg)
        new = diag.frame_grid(points, adjusted, self.cfg)
        np.testing.assert_array_equal(old.point_counts, new.point_counts)
        np.testing.assert_array_equal(self.table["normal"], normal)
        self.assertIs(adjusted["table_mask"], self.table["table_mask"])
        relative = points - self.table["origin"]
        x = np.floor(((relative @ self.table["u_axis"]) - self.table["u_min"]) / self.cfg.grid_m).astype(int)
        y = np.floor(((relative @ self.table["v_axis"]) - self.table["v_min"]) / self.cfg.grid_m).astype(int)
        expected = np.median((points[(y == 12) & (x == 14)] @ candidate["normal"] + candidate["d"]) * 1000)
        self.assertAlmostEqual(new.median_mm[12, 14], expected, places=5)

    def test_training_approximation_does_not_invent_support(self):
        result = self.result()
        after = app.corrected_fit_result(result, self.table, self.beta)
        np.testing.assert_allclose(after["median_mm"], 0, atol=1e-10)
        self.assertTrue(np.isnan(after["object_support_pct"]).all())
        self.assertGreater(np.max(np.abs(result["median_mm"])), 0)

    def test_no_common_samples_explicit(self):
        before, after = self.result(), self.result()
        after["reliable"][:] = False
        metrics = app.comparison(before, after, self.table)
        self.assertEqual(metrics["common_reliable_cells"], 0)
        self.assertIsNone(metrics["original"])
        self.assertIsNone(metrics["candidate"])


class WorkflowTests(Base):
    def test_fresh_validation_improves_bias_not_noise(self):
        train, _ = self.capture(seed=11)
        beta, _, _ = app.fit_plane(train.summarize(self.table, 3), self.table)
        old, new = self.capture(seed=29, candidate=self.candidate(beta))
        metrics = app.comparison(old.summarize(self.table, 3), new.summarize(self.table, 3), self.table)
        self.assertLess(metrics["candidate"]["median_abs_height_mm"], 1)
        self.assertGreater(metrics["original"]["median_abs_height_mm"], 3)
        self.assertAlmostEqual(metrics["original"]["median_temporal_span_mm"],
                               metrics["candidate"]["median_temporal_span_mm"], delta=.1)
        self.assertFalse(np.array_equal(train.frames[0].median_mm, old.frames[0].median_mm, equal_nan=True))
        self.assertEqual(old.times, new.times)

    def test_file_roundtrip_and_exact_backup(self):
        capture, _ = self.capture()
        before = capture.summarize(self.table, 3)
        beta, stats, inliers = app.fit_plane(before, self.table)
        candidate = self.candidate(beta)
        after = app.corrected_fit_result(before, self.table, beta)
        with tempfile.TemporaryDirectory() as base:
            bundle = app.save_fit(base, self.raw, candidate, self.table, capture, before, after, stats, inliers, self.cfg, "TEST")
            self.assertEqual((bundle / "original_calibration.json").read_bytes(), self.raw)
            old, new = self.capture(seed=29, candidate=candidate)
            directory = app.save_validation(bundle, self.table, old, new,
                                            old.summarize(self.table, 3), new.summarize(self.table, 3), candidate, "TEST")
            for path in bundle.rglob("*.json"):
                json.loads(path.read_text(), parse_constant=lambda value: self.fail(f"Invalid JSON: {value}"))
            report = json.loads((directory / "validation.json").read_text())
            self.assertEqual(report["candidate_sha256"], hashlib.sha256((bundle / "candidate_calibration.json").read_bytes()).hexdigest())
            with np.load(directory / "samples_candidate.npz", allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays["times_s"], old.times)
                np.testing.assert_allclose(arrays["median_mm"], new.arrays()["median_mm"], equal_nan=True)
            self.assertEqual(cv2.imread(str(directory / "comparison.png")).shape, (740, 1128, 3))

    def test_never_overwrites(self):
        with tempfile.TemporaryDirectory() as base, patch.object(app, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "fixed"
            directory = app.new_directory(base, "plane_")
            with self.assertRaises(FileExistsError):
                app.new_directory(base, "plane_")
            path = directory / "test.json"
            app.write_json(path, {"original": True})
            with self.assertRaises(FileExistsError):
                app.write_json(path, {"changed": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_demo_no_camera_or_gui(self):
        with tempfile.TemporaryDirectory() as base, \
                patch.object(diag, "camera_source", side_effect=AssertionError("No camera")), \
                patch.object(app.cv2, "imshow", side_effect=AssertionError("No GUI")), \
                patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(app.main(["--demo", "--seconds", "3", "--grid-mm", "20", "--output-dir", base]), 0)
            self.assertEqual(len(list(Path(base).glob("plane_*/candidate_calibration.json"))), 1)
            self.assertEqual(len(list(Path(base).glob("plane_*/validation_*/validation.json"))), 1)

    def test_invalid_cli(self):
        for options in [["--seconds", "2"], ["--seconds", "nan"], ["--seconds", "16"],
                        ["--grid-mm", "0"], ["--edge-margin-mm", "-1"]]:
            with self.subTest(options=options), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args(options)

    def test_unknown_preview_render(self):
        result = diag.preview_result(diag.frame_grid([], self.table, self.cfg))
        self.assertEqual(app.render(self.table, result, result, "READY", (12, 14)).shape, (740, 1128, 3))

    def live_context(self, stack, keys, points):
        closed = []
        @contextmanager
        def source():
            try:
                yield lambda: points
            finally:
                closed.append(True)
        stack.enter_context(patch.object(diag, "camera_source", source))
        for method in ["namedWindow", "setMouseCallback", "imshow"]:
            stack.enter_context(patch.object(app.cv2, method))
        destroyed = stack.enter_context(patch.object(app.cv2, "destroyAllWindows"))
        stack.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
        stack.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
        stack.enter_context(patch.object(app.time, "monotonic", side_effect=np.arange(1000) * .1))
        return closed, destroyed

    def test_live_fit_validate_save_recapture(self):
        keys = [ord("v"), ord("c")] + [-1] * 31 + [ord("v")] + [-1] * 31 + [ord("s"), ord("c"), ord("q")]
        points = app.synthetic_points(self.table, np.random.default_rng(1), self.beta, 0)
        with tempfile.TemporaryDirectory() as base, ExitStack() as stack:
            closed, destroyed = self.live_context(stack, keys, points)
            app.run_live(SimpleNamespace(seconds=3, output_dir=Path(base)), self.cfg, self.raw)
            self.assertEqual(len(list(Path(base).glob("plane_*/candidate_calibration.json"))), 1)
            self.assertEqual(len(list(Path(base).glob("plane_*/validation_*/selected_*.png"))), 1)
            self.assertEqual(closed, [True])
            destroyed.assert_called_once()

    def test_live_rejection_cannot_validate_or_save(self):
        keys = [ord("c")] + [-1] * 31 + [ord("v"), ord("s"), ord("q")]
        with tempfile.TemporaryDirectory() as base, ExitStack() as stack:
            closed, destroyed = self.live_context(stack, keys, np.empty((0, 3)))
            app.run_live(SimpleNamespace(seconds=3, output_dir=Path(base)), self.cfg, self.raw)
            self.assertEqual(list(Path(base).iterdir()), [])
            self.assertEqual(closed, [True])
            destroyed.assert_called_once()

    def test_cleanup_on_error(self):
        with tempfile.TemporaryDirectory() as base, ExitStack() as stack:
            closed, destroyed = self.live_context(stack, [], np.empty((0, 3)))
            stack.enter_context(patch.object(diag, "frame_grid", side_effect=RuntimeError("read failed")))
            with self.assertRaisesRegex(RuntimeError, "read failed"):
                app.run_live(SimpleNamespace(seconds=3, output_dir=Path(base)), self.cfg, self.raw)
            self.assertEqual(closed, [True])
            destroyed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
