"""Headless, deterministic v04 tests. No RealSense import or camera required."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import desk_orientation_v04 as app


def axis_error(a, b):
    gap = abs(a - b) % 180
    return min(gap, 180 - gap)


def rectangle_points(angle, length=.30, width=.10):
    x, y = np.meshgrid(np.linspace(-length / 2, length / 2, 31),
                       np.linspace(-width / 2, width / 2, 11))
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return np.column_stack([x.ravel(), y.ravel()]) @ rotation.T


class GridTests(unittest.TestCase):
    def setUp(self):
        self.cfg = app.Settings()
        self.table = app.demo_table(self.cfg)
        self.u, self.v = .255, .205

    def points(self, heights, u=None, v=None):
        u = self.u if u is None else u
        v = self.v if v is None else v
        h = np.asarray(heights)
        return (self.table["origin"] + u * self.table["u_axis"]
                + v * self.table["v_axis"] + h[:, None] * self.table["normal"])

    def cell(self, observation, field="evidence"):
        return getattr(observation, field)[20, 25]

    def test_empty_frame_is_unknown(self):
        observation = app.make_observation(np.empty((0, 3)), self.table, self.cfg)
        self.assertFalse(observation.raw.any())
        self.assertFalse(observation.evidence.any())

    def test_valid_plane_is_free(self):
        observation = app.make_observation(self.points([0] * 5), self.table, self.cfg)
        self.assertEqual(self.cell(observation), app.FREE)

    def test_valid_object_is_occupied(self):
        observation = app.make_observation(self.points([.03] * 5), self.table, self.cfg)
        self.assertEqual(self.cell(observation), app.OCCUPIED)

    def test_one_outlier_marks_raw_but_not_supported_grid(self):
        observation = app.make_observation(self.points([0, 0, 0, 0, .03]), self.table, self.cfg)
        self.assertEqual(self.cell(observation, "raw"), app.OCCUPIED)
        self.assertEqual(self.cell(observation), app.FREE)

    def test_sparse_object_points_are_unknown(self):
        observation = app.make_observation(self.points([.03, .03]), self.table, self.cfg)
        self.assertEqual(self.cell(observation), app.UNKNOWN)

    def test_conflicting_support_is_unknown(self):
        observation = app.make_observation(self.points([0] * 3 + [.03] * 3), self.table, self.cfg)
        self.assertEqual(self.cell(observation), app.UNKNOWN)

    def test_3mm_overlap_does_not_cast_free_votes(self):
        cfg = replace(self.cfg, min_height_m=.003)
        observation = app.make_observation(self.points([.005] * 5), self.table, cfg)
        self.assertEqual(self.cell(observation), app.OCCUPIED)

    def test_below_plane_and_over_height_are_unknown(self):
        for h in [-.03, .45]:
            with self.subTest(height=h):
                observation = app.make_observation(self.points([h] * 5), self.table, self.cfg)
                self.assertEqual(self.cell(observation), app.UNKNOWN)

    def test_invalid_depth_is_not_empty_desk(self):
        vertices = np.array([[0, 0, 0], [np.nan, 0, 1], [0, 0, np.inf], [0, 0, 5]])
        observation = app.make_observation(vertices, self.table, self.cfg)
        self.assertFalse(observation.evidence.any())

    def test_negative_coordinate_does_not_leak_into_boundary_cell(self):
        observation = app.make_observation(self.points([.03] * 5, u=-.001), self.table, self.cfg)
        self.assertFalse(observation.raw.any())

    def test_polygon_excludes_points_inside_bounding_box_only(self):
        data = dict(origin=[-.6, .4, 1], normal=[0, 0, -1], d=1,
                    u_axis=[1, 0, 0], v_axis=[0, -1, 0],
                    polygon_uv=[[.5, 0], [1, .5], [.5, 1], [0, .5]])
        table = app.table_from_data(data, .01)
        vertices = table["origin"] + np.array([[.02, -.02, -.03]] * 5)
        observation = app.make_observation(vertices, table, self.cfg)
        self.assertFalse(observation.evidence.any())


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = app.Settings()
        self.filter = app.TemporalGrid((1, 1), self.cfg)

    def update(self, state, now):
        return self.filter.update(np.array([[state]], np.uint8), now)

    def confirm_object(self):
        for i in range(3):
            self.update(app.OCCUPIED, i / 30)

    def test_requires_three_votes_at_startup(self):
        for i in range(2):
            state, _ = self.update(app.OCCUPIED, i / 30)
            self.assertEqual(state[0, 0], app.UNKNOWN)
        state, _ = self.update(app.OCCUPIED, 2 / 30)
        self.assertEqual(state[0, 0], app.OCCUPIED)

    def test_single_frame_spike_is_rejected(self):
        for i, value in enumerate([app.FREE] * 3 + [app.OCCUPIED] + [app.FREE] * 3):
            state, _ = self.update(value, i / 30)
        self.assertEqual(state[0, 0], app.FREE)

    def test_missing_observation_is_held_not_free(self):
        self.confirm_object()
        state, held = self.update(app.UNKNOWN, .10)
        self.assertEqual(state[0, 0], app.OCCUPIED)
        self.assertTrue(held[0, 0])

    def test_missing_observations_expire_by_time(self):
        self.confirm_object()
        state, held = self.update(app.UNKNOWN, .70)
        self.assertEqual(state[0, 0], app.UNKNOWN)
        self.assertFalse(held[0, 0])

    def test_observed_removal_clears_after_three_free_votes(self):
        self.confirm_object()
        for i in range(3):
            state, _ = self.update(app.FREE, .1 + i / 30)
        self.assertEqual(state[0, 0], app.FREE)

    def test_sparse_opposite_evidence_does_not_keep_old_object_forever(self):
        self.confirm_object()
        for i in range(1, 22):
            state, _ = self.update(app.FREE if i % 3 == 0 else app.UNKNOWN, .07 + i / 30)
        self.assertEqual(state[0, 0], app.UNKNOWN)

    def test_returned_arrays_do_not_alias_filter_state(self):
        self.confirm_object()
        state, _ = self.update(app.OCCUPIED, .1)
        state.fill(app.FREE)
        self.assertEqual(self.filter.state[0, 0], app.OCCUPIED)

    def test_reset_discards_old_votes(self):
        self.confirm_object()
        self.filter.reset()
        state, _ = self.update(app.FREE, 1)
        self.assertEqual(state[0, 0], app.UNKNOWN)

    def test_old_votes_do_not_confirm_new_scene_after_stall(self):
        self.confirm_object()
        state, _ = self.update(app.OCCUPIED, 2)
        self.assertEqual(state[0, 0], app.UNKNOWN)

    def test_non_monotonic_time_rejected(self):
        self.update(app.FREE, 1)
        with self.assertRaises(ValueError):
            self.update(app.FREE, .9)


class PCATests(unittest.TestCase):
    def setUp(self):
        self.cfg = app.Settings()

    def test_rectangle_angles_and_180_degree_equivalence(self):
        for target in [0, 1, 30, 90, 179, 180]:
            with self.subTest(angle=target):
                angle, _, ratio, _, reason = app.pca_direction(rectangle_points(target), self.cfg)
                self.assertEqual(reason, "ok")
                self.assertGreater(ratio, self.cfg.pca_ratio)
                self.assertLess(axis_error(angle, target), 1e-6)

    def test_square_has_no_valid_direction(self):
        x, y = np.meshgrid(np.linspace(0, .2, 21), np.linspace(0, .2, 21))
        result = app.pca_direction(np.column_stack([x.ravel(), y.ravel()]), self.cfg)
        self.assertIsNone(result[0])
        self.assertEqual(result[-1], "no dominant axis")

    def test_tiny_rectangle_has_no_valid_direction(self):
        result = app.pca_direction(rectangle_points(30, .02, .005), self.cfg)
        self.assertIsNone(result[0])
        self.assertEqual(result[-1], "too small")

    def test_translation_does_not_change_orientation(self):
        points = rectangle_points(30)
        angle = app.pca_direction(points + [100, -20], self.cfg)[0]
        self.assertLess(axis_error(angle, 30), 1e-6)

    def test_small_holes_do_not_destroy_direction_on_synthetic_rectangle(self):
        points = rectangle_points(30)
        keep = np.random.default_rng(7).random(len(points)) > .2
        angle = app.pca_direction(points[keep], self.cfg)[0]
        self.assertLess(axis_error(angle, 30), 3)

    def test_table_reference_not_screen_x_axis(self):
        table = app.demo_table(self.cfg)
        table["reference_deg"] = 30
        state = np.zeros((table["rows"], table["cols"]), np.uint8)
        state[10:20, 20:50] = app.OCCUPIED
        _, regions = app.extract_regions(state, table, self.cfg)
        self.assertEqual(len(regions), 1)
        self.assertLess(axis_error(regions[0]["angle"], 150), 1e-6)

    def test_region_area_is_physical_not_fixed_cell_count(self):
        cfg = replace(self.cfg, grid_m=.005)
        table = app.demo_table(cfg)
        state = np.zeros((table["rows"], table["cols"]), np.uint8)
        state[10:14, 10:14] = app.OCCUPIED  # 16 cells * .25 cm2 < 8 cm2
        _, regions = app.extract_regions(state, table, cfg)
        self.assertEqual(regions, [])


class EdgeAndDisplayTests(unittest.TestCase):
    def setUp(self):
        self.cfg = app.Settings()
        self.table = app.demo_table(self.cfg)

    def border_observation(self, table=None):
        table = self.table if table is None else table
        # An elongated, well-supported false object along the lower ROI edge.
        # Without the inset it is large enough for both a region and a PCA axis.
        y, x = np.mgrid[0:2, 10:60]
        uv = np.repeat(np.column_stack([x.ravel() + .5, y.ravel() + .5]), 5, axis=0) * .01
        vertices = (table["origin"] + uv[:, :1] * table["u_axis"]
                    + uv[:, 1:] * table["v_axis"] + .03 * table["normal"])
        return app.make_observation(vertices, table, self.cfg)

    def test_default_and_disabled_margin_cli(self):
        _, cfg = app.parse_args([])
        self.assertEqual(cfg.edge_margin_m, .02)
        _, cfg = app.parse_args(["--edge-margin-mm", "0"])
        self.assertEqual(cfg.edge_margin_m, 0)
        _, cfg = app.parse_args(["--edge-margin-mm", "40"])
        self.assertEqual(cfg.edge_margin_m, .04)

    def test_negative_and_nonfinite_margins_rejected(self):
        for margin in [-.01, float("nan"), float("inf")]:
            with self.subTest(margin=margin):
                cfg = replace(self.cfg, edge_margin_m=margin)
                with self.assertRaises(ValueError):
                    cfg.validate()
                with self.assertRaises(ValueError):
                    app.demo_table(cfg)

    def test_rectangular_inset_preserves_only_whole_interior_cells(self):
        expected = np.zeros_like(self.table["table_mask"])
        expected[2:78, 2:118] = 255
        np.testing.assert_array_equal(self.table["active_mask"], expected)
        self.assertFalse(np.any((expected > 0) & self.table["excluded_mask"]))
        np.testing.assert_array_equal((expected > 0) | self.table["excluded_mask"],
                                      self.table["table_mask"] > 0)

    def test_zero_margin_restores_original_roi_without_aliasing(self):
        table = app.demo_table(replace(self.cfg, edge_margin_m=0))
        np.testing.assert_array_equal(table["active_mask"], table["table_mask"])
        self.assertFalse(table["excluded_mask"].any())
        table["active_mask"].fill(0)
        self.assertTrue(table["table_mask"].any())

    def test_rotated_inset_uses_metric_edges_in_both_windings_and_grid_sizes(self):
        polygon = np.array([[0, 0], [1.2, 0], [1.1, .8], [.1, .8]])
        theta = np.radians(31)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        polygon = polygon @ rotation.T + [-.1, .2]
        for grid in [.005, .01, .02]:
            previous = None
            for poly in [polygon, polygon[::-1]]:
                with self.subTest(grid=grid, clockwise=previous is not None):
                    data = dict(self.table, polygon_uv=poly)
                    table = app.table_from_data(data, grid, .02)
                    y, x = np.nonzero(table["active_mask"])
                    self.assertGreater(len(x), 500)
                    # Independently check all FOUR corners, not just centers.
                    corners = ((np.column_stack([x, y])[:, None, :]
                                + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) * grid
                               + [table["u_min"], table["v_min"]])
                    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
                        edge = end - start
                        relative = corners - start
                        distance = (edge[0] * relative[..., 1] - edge[1] * relative[..., 0]) / np.linalg.norm(edge)
                        self.assertGreaterEqual(float(distance.min()), .02 - 1e-10)
                    if previous is not None:
                        np.testing.assert_array_equal(table["active_mask"], previous)
                    previous = table["active_mask"]

    def test_wider_margin_only_removes_more_cells(self):
        wider = app.demo_table(replace(self.cfg, edge_margin_m=.04))
        self.assertLess(np.count_nonzero(wider["active_mask"]),
                        np.count_nonzero(self.table["active_mask"]))
        self.assertFalse(np.any((wider["active_mask"] > 0) & (self.table["active_mask"] == 0)))

    def test_excessive_margin_fails_without_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "no active ROI"):
            app.demo_table(replace(self.cfg, edge_margin_m=.5))

    def test_border_noise_never_reaches_temporal_or_pca(self):
        observation = self.border_observation()
        self.assertEqual(np.count_nonzero(observation.raw == app.OCCUPIED), 100)
        self.assertFalse(observation.evidence.any())
        temporal = app.TemporalGrid(observation.raw.shape, self.cfg)
        for frame in range(10):
            stable, held = temporal.update(observation.evidence, frame / 30)
        self.assertFalse(stable.any())
        self.assertFalse(held.any())
        labels, regions = app.extract_regions(stable, self.table, self.cfg)
        self.assertFalse(labels.any())
        self.assertEqual(regions, [])
        self.assertEqual(app.active_percent(observation.raw == app.OCCUPIED, self.table), 0)

    def test_disabled_margin_allows_same_border_noise_for_comparison(self):
        table = app.demo_table(replace(self.cfg, edge_margin_m=0))
        observation = self.border_observation(table)
        _, regions = app.extract_regions(observation.evidence, table, self.cfg)
        self.assertEqual(len(regions), 1)
        self.assertLess(axis_error(regions[0]["angle"], 0), 1e-6)

    def test_even_injected_border_states_are_masked_before_components(self):
        state = np.zeros_like(self.table["table_mask"])
        state[self.table["excluded_mask"]] = app.OCCUPIED
        labels, regions = app.extract_regions(state, self.table, self.cfg)
        self.assertFalse(labels.any())
        self.assertEqual(regions, [])

    def test_interior_object_keeps_direction_with_border_noise_present(self):
        state = np.zeros_like(self.table["table_mask"])
        state[self.table["excluded_mask"]] = app.OCCUPIED
        state[20:30, 30:60] = app.OCCUPIED
        labels, regions = app.extract_regions(state, self.table, self.cfg)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["area_cells"], 300)
        self.assertLess(axis_error(regions[0]["angle"], 0), 1e-6)
        self.assertFalse(labels[self.table["excluded_mask"]].any())

    def test_statistics_exclude_band_from_numerator_and_denominator(self):
        active = self.table["active_mask"] > 0
        self.assertEqual(app.active_percent(active, self.table), 100)
        self.assertEqual(app.active_percent(self.table["table_mask"] > 0, self.table), 100)
        self.assertEqual(app.active_percent(self.table["excluded_mask"], self.table), 0)
        one_cell = np.zeros_like(active)
        one_cell[20, 20] = True
        self.assertAlmostEqual(app.active_percent(one_cell, self.table), 100 / active.sum())

    def test_exclusion_display_does_not_depend_on_classification(self):
        unknown = np.zeros_like(self.table["table_mask"])
        occupied = np.full_like(unknown, app.OCCUPIED)
        first, second = app.state_image(unknown, self.table), app.state_image(occupied, self.table)
        excluded = self.table["excluded_mask"]
        np.testing.assert_array_equal(first[excluded], second[excluded])
        self.assertTrue(np.any(first[excluded] != 0))
        self.assertFalse(np.any(np.all(second[excluded] == [45, 60, 230], axis=1)))

    def test_held_changes_counter_but_not_panel_brightness(self):
        state = np.zeros_like(self.table["table_mask"])
        state[self.table["active_mask"] > 0] = app.FREE
        state[20:30, 30:60] = app.OCCUPIED
        observation = app.Observation(state, state, np.zeros_like(state))
        labels, regions = app.extract_regions(state, self.table, self.cfg)
        first = app.render(self.table, observation, state, np.zeros_like(state, bool),
                           labels, regions, self.cfg)
        second = app.render(self.table, observation, state, state != app.UNKNOWN,
                            labels, regions, self.cfg)
        np.testing.assert_array_equal(first[156:556], second[156:556])
        self.assertFalse(np.array_equal(first[75:100], second[75:100]))

    def test_temporal_badge_is_visible_and_changes_with_state(self):
        state = np.zeros_like(self.table["table_mask"])
        observation = app.Observation(state, state, state)
        images = [app.render(self.table, observation, state, state.astype(bool),
                             state, [], self.cfg, temporal=enabled) for enabled in [True, False]]
        np.testing.assert_array_equal(images[0][6, 829], [35, 95, 35])
        np.testing.assert_array_equal(images[1][6, 829], [30, 85, 165])
        self.assertFalse(np.array_equal(images[0][10:30, 837:1116], images[1][10:30, 837:1116]))


class IntegrationTests(unittest.TestCase):
    def test_invalid_settings_rejected(self):
        for change in [dict(grid_m=0), dict(min_height_m=.5), dict(support_fraction=.5),
                       dict(window=5, votes=2), dict(hold_seconds=0), dict(pca_ratio=1),
                       dict(min_height_m=float("nan"))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(app.Settings(), **change).validate()

    def test_invalid_calibration_rejected(self):
        data = dict(origin=[0, 0, 1], normal=[0, 0, -1], d=1,
                    u_axis=[1, 0, 0], v_axis=[1, 0, 0],
                    polygon_uv=[[0, 0], [1, 0], [1, 1], [0, 1]])
        with self.assertRaises(ValueError):
            app.table_from_data(data, .01)

    def test_loading_and_running_preserve_calibration_bytes(self):
        data = dict(origin=[0, 0, 1], normal=[0, 0, -1], d=1,
                    u_axis=[1, 0, 0], v_axis=[0, -1, 0],
                    polygon_uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
                    grid_size=.008, baseline=12.34)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            table = app.load_table(path, .01)
            app.make_observation(np.empty((0, 3)), table, app.Settings())
            self.assertEqual(table["grid_m"], .01)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_demo_0_30_90_and_square_through_complete_pipeline(self):
        cfg = app.Settings()
        table = app.demo_table(cfg)
        for target in [0, 30, 90]:
            with self.subTest(angle=target):
                temporal = app.TemporalGrid((table["rows"], table["cols"]), cfg)
                rng = np.random.default_rng(4)
                for frame in range(18):
                    vertices = app.demo_vertices(table, cfg, rng, target)
                    observation = app.make_observation(vertices, table, cfg)
                    stable, held = temporal.update(observation.evidence, frame / 30)
                labels, regions = app.extract_regions(stable, table, cfg)
                valid = [r for r in regions if r["angle"] is not None]
                self.assertEqual(len(regions), 2)
                self.assertEqual(len(valid), 1)
                self.assertLess(axis_error(valid[0]["angle"], target), 3)
                image = app.render(table, observation, stable, held, labels, regions, cfg)
                self.assertEqual(image.shape, (674, 1128, 3))
                self.assertEqual(image.dtype, np.uint8)

    def test_headless_snapshot_does_not_require_camera_or_gui(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.png"
            with patch.object(app.cv2, "imshow", side_effect=AssertionError("GUI forbidden")):
                self.assertEqual(app.main(["--demo", "--snapshot", str(path)]), 0)
            self.assertIsNotNone(cv2.imread(str(path)))

    def test_screenshot_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.png"
            app.save_image(path, np.zeros((10, 10, 3), np.uint8))
            with self.assertRaises(FileExistsError):
                app.save_image(path, np.zeros((10, 10, 3), np.uint8))

    def test_demo_keyboard_controls_and_cleanup(self):
        args, cfg = app.parse_args(["--demo"])
        keys = [ord("t"), ord(" "), ord(" "), ord("r"), ord("n"), ord("q")]
        with patch.object(app.cv2, "imshow"), patch.object(app.cv2, "waitKey", side_effect=keys), \
                patch.object(app.cv2, "getWindowProperty", return_value=1), \
                patch.object(app.cv2, "destroyAllWindows") as close:
            app.run(args, cfg)
            close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
