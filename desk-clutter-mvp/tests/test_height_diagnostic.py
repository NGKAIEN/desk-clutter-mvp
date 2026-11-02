"""Offline signed-height diagnostics; no real camera or desktop needed."""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import desk_height_diagnostic as app
import desk_orientation_v04 as v04


class Base(unittest.TestCase):
    def setUp(self):
        self.cfg = v04.Settings()
        self.table = v04.demo_table(self.cfg)
        self.cell = (20, 25)

    def points(self, heights_mm, u=.255, v=.205):
        h = np.asarray(heights_mm) / 1000
        return (self.table["origin"] + u * self.table["u_axis"] + v * self.table["v_axis"]
                + h[:, None] * self.table["normal"])

    def frame(self, heights_mm=(20, 20, 20, 20, 20)):
        return app.frame_grid(self.points(heights_mm), self.table, self.cfg)

    def capture(self, frames):
        capture = app.Capture(self.table)
        for i, frame in enumerate(frames):
            capture.add(frame, i / 30)
        return capture, capture.summarize(self.table, len(frames) / 30)


class FrameTests(Base):
    def test_signed_height_and_quantiles_match_numpy(self):
        values = [-20, -5, 0, 10, 30]
        frame = self.frame(values)
        self.assertAlmostEqual(frame.median_mm[self.cell], np.median(values), places=4)
        self.assertAlmostEqual(frame.spatial_span_mm[self.cell], np.ptp(np.quantile(values, [.1, .9])), places=4)
        self.assertEqual(frame.point_counts[self.cell], 5)

    def test_negative_and_above_object_ceiling_are_not_discarded(self):
        for height in [-40, 450]:
            with self.subTest(height=height):
                frame = self.frame([height] * 5)
                self.assertAlmostEqual(frame.median_mm[self.cell], height, places=4)
                self.assertFalse(frame.object_support[self.cell])

    def test_missing_and_invalid_points_are_nan_not_zero(self):
        points = [[0, 0, 0], [0, 0, .1], [0, 0, 3], [np.nan, 0, 1], [0, np.inf, 1]]
        for vertices in [[], points]:
            frame = app.frame_grid(vertices, self.table, self.cfg)
            self.assertTrue(np.isnan(frame.median_mm).all())
            self.assertFalse(frame.point_counts.any())
            self.assertFalse(frame.object_support.any())

    def test_sparse_cells_are_unobserved(self):
        frame = self.frame([25, 25])
        self.assertEqual(frame.point_counts[self.cell], 2)
        self.assertTrue(np.isnan(frame.median_mm[self.cell]))
        self.assertFalse(frame.object_support[self.cell])

    def test_original_excluded_band_is_measured(self):
        frame = app.frame_grid(self.points([20] * 5, u=.005), self.table, self.cfg)
        self.assertTrue(self.table["excluded_mask"][20, 0])
        self.assertAlmostEqual(frame.median_mm[20, 0], 20)
        self.assertTrue(frame.object_support[20, 0])

    def test_object_support_matches_v04_before_time_voting(self):
        for heights in [[0] * 5, [0, 0, 30, 30, 30], [0, 0, 0, 0, 30], [5] * 5]:
            for threshold in [.003, .010]:
                with self.subTest(heights=heights, threshold=threshold):
                    cfg = replace(self.cfg, min_height_m=threshold)
                    vertices = self.points(heights)
                    frame = app.frame_grid(vertices, self.table, cfg)
                    old = v04.make_observation(vertices, self.table, cfg)
                    self.assertEqual(frame.object_support[self.cell], old.evidence[self.cell] == v04.OCCUPIED)

    def test_cells_do_not_mix_when_points_are_shuffled(self):
        first, second = [-30, 0, 10, 20], [15, 15, 15, 15, 15, 15, 15]
        points = np.concatenate([self.points(first), self.points(second, u=.455)])
        np.random.default_rng(4).shuffle(points)
        frame = app.frame_grid(points, self.table, self.cfg)
        self.assertAlmostEqual(frame.median_mm[20, 25], 5)
        self.assertAlmostEqual(frame.median_mm[20, 45], 15)
        self.assertEqual(frame.point_counts[20, 45], 7)

    def test_negative_grid_coordinate_cannot_leak_into_first_cell(self):
        frame = app.frame_grid(self.points([20] * 5, u=-.001), self.table, self.cfg)
        self.assertFalse(frame.point_counts.any())


class CaptureTests(Base):
    def test_stable_bias_is_high_median_low_temporal_spread(self):
        _, result = self.capture([self.frame()] * 20)
        self.assertAlmostEqual(result["median_mm"][self.cell], 20)
        self.assertAlmostEqual(result["temporal_span_mm"][self.cell], 0)
        self.assertEqual(result["coverage_pct"][self.cell], 100)
        self.assertEqual(result["object_support_pct"][self.cell], 100)

    def test_alternating_heights_have_large_time_spread(self):
        _, result = self.capture([self.frame([-12] * 5), self.frame([12] * 5)] * 10)
        self.assertAlmostEqual(result["median_mm"][self.cell], 0)
        self.assertAlmostEqual(result["temporal_span_mm"][self.cell], 24)
        self.assertEqual(result["object_support_pct"][self.cell], 50)

    def test_spatial_noise_is_retained_even_when_temporal_median_is_stable(self):
        _, result = self.capture([self.frame([-10, 0, 20])] * 20)
        self.assertEqual(result["temporal_span_mm"][self.cell], 0)
        self.assertAlmostEqual(result["spatial_span_mm"][self.cell], 24)

    def test_low_coverage_does_not_report_false_low_noise(self):
        empty = app.frame_grid([], self.table, self.cfg)
        _, result = self.capture([self.frame()] * 5 + [empty] * 15)
        self.assertEqual(result["coverage_pct"][self.cell], 25)
        self.assertFalse(result["reliable"][self.cell])
        self.assertTrue(np.isnan(result["median_mm"][self.cell]))
        self.assertTrue(np.isnan(result["temporal_span_mm"][self.cell]))

    def test_half_coverage_with_ten_samples_is_accepted_without_zero_filling(self):
        empty = app.frame_grid([], self.table, self.cfg)
        _, result = self.capture([self.frame()] * 10 + [empty] * 10)
        self.assertTrue(result["reliable"][self.cell])
        self.assertAlmostEqual(result["median_mm"][self.cell], 20)
        self.assertEqual(result["object_support_pct"][self.cell], 100)
        self.assertEqual(result["coverage_pct"][self.cell], 50)

    def test_less_than_ten_frames_is_insufficient(self):
        _, result = self.capture([self.frame()] * 9)
        self.assertFalse(result["reliable"].any())
        self.assertTrue(np.isnan(result["median_mm"]).all())

    def test_all_missing_capture_has_no_reliable_height(self):
        _, result = self.capture([app.frame_grid([], self.table, self.cfg)] * 20)
        self.assertFalse(result["coverage_pct"].any())
        self.assertFalse(result["reliable"].any())
        self.assertTrue(np.isnan(result["temporal_span_mm"]).all())

    def test_timestamps_duration_and_empty_capture_validation(self):
        capture = app.Capture(self.table)
        with self.assertRaises(ValueError):
            capture.summarize(self.table, 0)
        capture.add(self.frame(), 1)
        for stamp in [-1, .5, float("nan"), float("inf")]:
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                capture.add(self.frame(), stamp)
        with self.assertRaises(ValueError):
            capture.summarize(self.table, .9)

    def test_capture_has_a_memory_guard(self):
        capture = app.Capture(self.table)
        with patch.object(app, "MAX_CELL_SAMPLES", self.table["table_mask"].size):
            capture.add(self.frame(), 0)
            with self.assertRaises(ValueError):
                capture.add(self.frame(), 1)


class DisplayAndReportTests(Base):
    def test_clicks_map_same_cell_in_all_three_panels(self):
        y, x = self.cell
        for x0, y0, dw, dh in app.panel_layout(self.table):
            px = x0 + int((x + .5) * dw / self.table["cols"])
            py = y0 + int((self.table["rows"] - y - .5) * dh / self.table["rows"])
            self.assertEqual(app.clicked_cell(px, py, self.table), self.cell)
        self.assertIsNone(app.clicked_cell(0, 0, self.table))

    def test_gray_nan_differs_from_zero_and_color_clipping_is_display_only(self):
        values = np.array([[np.nan, 0, 30, 100]])
        picture = app.colorize(values, -30, 30, [(255, 0, 0), (255, 255, 255), (0, 0, 255)])
        np.testing.assert_array_equal(picture[0, 0], [65, 65, 65])
        np.testing.assert_array_equal(picture[0, 1], [255, 255, 255])
        np.testing.assert_array_equal(picture[0, 2], picture[0, 3])
        self.assertEqual(values[0, 3], 100)

    def test_preview_has_no_temporal_spread(self):
        result = app.preview_result(self.frame())
        self.assertTrue(result["preview"])
        self.assertEqual(result["n_frames"], 0)
        self.assertTrue(np.isnan(result["temporal_span_mm"]).all())
        picture = app.render(self.table, result, self.cfg, "READY", self.cell)
        self.assertEqual(picture.shape, (740, 1128, 3))

    def test_report_round_trip_nan_json_npz_and_selected_cell(self):
        capture, result = self.capture([self.frame()] * 10)
        with tempfile.TemporaryDirectory() as base:
            directory = app.save_report(base, self.table, self.cfg, capture, result, "TEST", {"kind": "synthetic"}, self.cell)
            report = json.loads((directory / "summary.json").read_text(),
                                parse_constant=lambda value: self.fail(f"Non-JSON value: {value}"))
            self.assertEqual(report["selected_cell"]["median_mm"], 20)
            self.assertIsNone(report["maps"]["median_mm"][0][0])
            self.assertEqual(report["frame_count"], 10)
            with np.load(directory / "samples.npz", allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays["median_mm"][:, 20, 25], 20)
                np.testing.assert_allclose(arrays["times_s"], capture.times)
            self.assertEqual(cv2.imread(str(directory / "report.png")).shape, (740, 1128, 3))

    def test_existing_report_is_never_overwritten(self):
        capture, result = self.capture([self.frame()] * 10)
        with tempfile.TemporaryDirectory() as base, patch.object(app, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "fixed"
            directory = app.save_report(base, self.table, self.cfg, capture, result, "TEST", {})
            before = hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):
                app.save_report(base, self.table, self.cfg, capture, result, "TEST", {})
            self.assertEqual(hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(), before)

    def test_demo_is_headless_and_does_not_open_camera(self):
        with tempfile.TemporaryDirectory() as base, \
                patch.object(app, "camera_source", side_effect=AssertionError("No camera")), \
                patch.object(app.cv2, "imshow", side_effect=AssertionError("No GUI")):
            self.assertEqual(app.main(["--demo", "--seconds", "1", "--output-dir", base]), 0)
            self.assertEqual(len(list(Path(base).glob("height_*/report.png"))), 1)

    def test_cli_rejects_invalid_measurement_parameters(self):
        for options in [["--seconds", "0"], ["--seconds", "nan"], ["--seconds", "31"],
                        ["--grid-mm", "0"], ["--min-cell-points", "0"]]:
            with self.subTest(options=options), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args(options)

    def test_live_keys_capture_save_recapture_and_cleanup_without_hardware(self):
        closed = []

        @contextmanager
        def fake_source():
            try:
                yield lambda: self.points([20] * 5)
            finally:
                closed.append(True)

        keys = [ord("c")] + [-1] * 14 + [ord("s"), ord("c"), -1, -1, ord("q")]
        with tempfile.TemporaryDirectory() as base:
            args, cfg = app.parse_args(["--seconds", "1", "--output-dir", base])
            with patch.object(app, "camera_source", fake_source), \
                    patch.object(app.time, "monotonic", side_effect=np.arange(100) * .2), \
                    patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "setMouseCallback"), \
                    patch.object(app.cv2, "imshow"), patch.object(app.cv2, "waitKey", side_effect=keys), \
                    patch.object(app.cv2, "getWindowProperty", return_value=1), \
                    patch.object(app.cv2, "destroyAllWindows") as close:
                app.run_live(args, cfg, self.table, {})
                close.assert_called_once()
            self.assertEqual(len(list(Path(base).glob("height_*/report.png"))), 2)
            self.assertEqual(closed, [True])

    def test_camera_is_released_if_warmup_fails(self):
        pipeline = MagicMock()
        pipeline.wait_for_frames.side_effect = RuntimeError("Disconnected")
        fake_rs = SimpleNamespace(pipeline=lambda: pipeline, config=MagicMock(), pointcloud=MagicMock(),
                                  stream=SimpleNamespace(depth=1), format=SimpleNamespace(z16=1))
        with patch.dict("sys.modules", pyrealsense2=fake_rs), self.assertRaises(RuntimeError):
            with app.camera_source():
                self.fail("Warmup must fail first")
        pipeline.stop.assert_called_once()

    def test_live_entry_reads_exact_calibration_without_modifying_it(self):
        data = {key: self.table[key].tolist() for key in ["origin", "normal", "u_axis", "v_axis", "polygon_uv"]}
        data.update(d=self.table["d"], grid_size=.008, baseline=12.34)
        with tempfile.TemporaryDirectory() as base:
            path = Path(base) / "calibration.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            before = path.read_bytes()
            with patch.object(app, "run_live") as run:
                self.assertEqual(app.main(["--calibration", str(path), "--output-dir", base]), 0)
                table, source = run.call_args.args[2:]
                self.assertEqual(table["grid_m"], .01)
                self.assertEqual(source["sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
