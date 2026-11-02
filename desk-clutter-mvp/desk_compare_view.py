"""RGB / bird's-eye comparison rendering; importing this module opens no camera."""

from dataclasses import dataclass

import cv2
import numpy as np


YELLOW = (0, 230, 255)
GREEN = (50, 210, 70)
CYAN = (255, 220, 60)
BACKGROUND = (25, 28, 33)
PANEL_WIDTH, PANEL_HEIGHT = 600, 450
MARGIN, HEADER, FOOTER = 12, 96, 64


@dataclass(frozen=True)
class ImagePlacement:
    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int

    def from_source(self, point):
        """Map a source-image point into the displayed panel for linked cursors."""
        x, y = point
        if (not np.isfinite([x, y]).all()
                or not (0 <= x < self.source_width and 0 <= y < self.source_height)):
            return None
        return (int(round(self.x + (x + .5) * self.width / self.source_width - .5)),
                int(round(self.y + (y + .5) * self.height / self.source_height - .5)))

    def to_source(self, x, y):
        """Reject clicks in the header, other panel, or letterboxing."""
        if not (self.x <= x < self.x + self.width
                and self.y <= y < self.y + self.height):
            return None
        return (
            min(self.source_width - 1, int((x - self.x) * self.source_width / self.width)),
            min(self.source_height - 1, int((y - self.y) * self.source_height / self.height)),
        )


def project_roi(rs, polygon_uv, origin, u_axis, v_axis, normal, d, camera):
    """Project table corners from the original depth frame into the RGB sensor."""
    uv = np.asarray(polygon_uv, dtype=np.float64).reshape(-1, 2)
    if len(uv) == 0:
        return np.empty((0, 2), dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    points = (np.asarray(origin) + uv[:, :1] * np.asarray(u_axis)
              + uv[:, 1:] * np.asarray(v_axis))
    # The mean of RANSAC inliers need not lie exactly on the fitted plane.
    points -= ((points @ normal + d) / (normal @ normal))[:, None] * normal
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    for i, point in enumerate(points):
        color_point = rs.rs2_transform_point_to_point(camera["extrinsics"], point.tolist())
        if np.isfinite(color_point).all() and color_point[2] > 0.01:
            pixel = rs.rs2_project_point_to_pixel(camera["intrinsics"], color_point)
            if np.isfinite(pixel).all() and np.max(np.abs(pixel)) < 1_000_000:
                pixels[i] = pixel
    return pixels


def rectified_plane_photo(rs, color, pixel_to_uv, origin, u_axis, v_axis,
                          normal, d, camera, width, height, step=16):
    """Sample the RGB photo on the same fitted-plane coordinates as the BEV.

    This is a flat-plane rectification, not a reconstruction of object tops.
    SDK projections on a small grid preserve color-camera distortion; bilinear
    interpolation supplies the dense remap without a Python loop per pixel.
    """
    nx, ny = max(2, int(np.ceil(width / step)) + 1), max(2, int(np.ceil(height / step)) + 1)
    x, y = np.meshgrid(np.linspace(0, width - 1, nx), np.linspace(0, height - 1, ny))
    u, v = pixel_to_uv(x, y)
    pixels = project_roi(rs, np.column_stack((u.ravel(), v.ravel())),
                         origin, u_axis, v_axis, normal, d, camera).reshape(ny, nx, 2)
    valid = np.isfinite(pixels).all(axis=2)
    pixels[~valid] = -10000
    map_x, map_y = np.meshgrid(np.linspace(0, nx - 1, width, dtype=np.float32),
                               np.linspace(0, ny - 1, height, dtype=np.float32))
    source_x = cv2.remap(pixels[:, :, 0].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    source_y = cv2.remap(pixels[:, :, 1].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    covered = cv2.remap(valid.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR) > .999
    covered &= ((source_x >= 0) & (source_x <= color.shape[1] - 1)
                & (source_y >= 0) & (source_y <= color.shape[0] - 1))
    image = cv2.remap(color, source_x, source_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND)
    image[~covered] = BACKGROUND
    return image, covered


def draw_view_footprint(image, corners):
    """Outline which part of the camera photo supplies the right-hand view."""
    points = np.asarray(corners).reshape(-1, 2)
    height, width = image.shape[:2]
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        if not np.isfinite([a, b]).all():
            continue
        visible, start, end = cv2.clipLine((0, 0, width, height),
                                          tuple(np.rint(a).astype(int)),
                                          tuple(np.rint(b).astype(int)))
        if visible:
            cv2.line(image, start, end, CYAN, 2, cv2.LINE_AA)


def draw_linked_cursor(canvas, placements, rgb_pixel, bev_pixel):
    """Connect the same plane location across both panels without selecting it."""
    a = placements["rgb"].from_source(rgb_pixel)
    b = placements["bev"].from_source(bev_pixel)
    if a is None or b is None:
        return False
    cv2.line(canvas, a, b, (15, 20, 25), 3, cv2.LINE_AA)
    cv2.line(canvas, a, b, CYAN, 1, cv2.LINE_AA)
    for point in (a, b):
        cv2.circle(canvas, point, 11, (10, 10, 10), 4, cv2.LINE_AA)
        cv2.circle(canvas, point, 11, CYAN, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, point, CYAN, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        label = (min(point[0] + 16, canvas.shape[1] - 112), max(20, point[1] - 14))
        cv2.putText(canvas, "SAME SPOT", label, cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(canvas, "SAME SPOT", label, cv2.FONT_HERSHEY_SIMPLEX,
                    .45, CYAN, 1, cv2.LINE_AA)
    return True


def color_pixel_to_table(rs, pixel, origin, u_axis, v_axis, normal, d, camera,
                         depth_limits=(0.20, 2.00)):
    """Intersect an RGB pixel's ray with the fitted plane in depth coordinates.

    Invert the SDK's projection numerically so forward-distorted color models
    work too (the SDK's deprojection does not support all such models).
    No depth sample at the clicked corner is required.
    """
    intrinsics = camera["intrinsics"]
    target = np.asarray(pixel, dtype=np.float64)
    if (not np.isfinite(target).all() or intrinsics.fx <= 0 or intrinsics.fy <= 0
            or not (0 <= target[0] < intrinsics.width and 0 <= target[1] < intrinsics.height)):
        return None
    ray_xy = (target - [intrinsics.ppx, intrinsics.ppy]) / [intrinsics.fx, intrinsics.fy]

    def project(xy):
        return np.asarray(rs.rs2_project_point_to_pixel(
            intrinsics, [float(xy[0]), float(xy[1]), 1.]), dtype=np.float64)

    try:
        for _ in range(12):
            predicted = project(ray_xy)
            error = predicted - target
            if not np.isfinite(error).all():
                return None
            if np.linalg.norm(error) < 1e-4:
                break
            epsilon = 1e-4
            jacobian = np.column_stack((
                (project(ray_xy + [epsilon, 0]) - predicted) / epsilon,
                (project(ray_xy + [0, epsilon]) - predicted) / epsilon,
            ))
            ray_xy -= np.linalg.solve(jacobian, error)
        if not np.isfinite(ray_xy).all() or np.linalg.norm(project(ray_xy) - target) > .05:
            return None
    except (RuntimeError, np.linalg.LinAlgError):
        return None

    extrinsics = camera["extrinsics"]
    rotation = np.asarray(extrinsics.rotation).reshape(3, 3, order="F")
    translation = np.asarray(extrinsics.translation)
    ray_origin = -rotation.T @ translation
    ray_direction = rotation.T @ np.array([*ray_xy, 1.])
    normal = np.asarray(normal)
    denominator = normal @ ray_direction
    if abs(denominator) < 1e-8:
        return None
    distance = -(normal @ ray_origin + d) / denominator
    point = ray_origin + distance * ray_direction
    if (distance <= 0 or not np.isfinite(point).all()
            or not depth_limits[0] < point[2] < depth_limits[1]):
        return None
    relative = point - np.asarray(origin)
    return np.array([relative @ u_axis, relative @ v_axis], dtype=np.float64)


def ordered_table_corners(corners, min_area=0.0001):
    """Validate four outer corners before calibration is allowed to overwrite a file."""
    polygon = np.asarray(corners, dtype=np.float64)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("Select four table corners first.")
    center = polygon.mean(axis=0)
    polygon = polygon[np.argsort(np.arctan2(polygon[:, 1] - center[1],
                                            polygon[:, 0] - center[0]))]
    contour = polygon.astype(np.float32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= min_area:
        raise ValueError("Choose four distinct outer corners. Use U to undo or R to reset.")
    return polygon


def draw_corners(image, corners, closed=True):
    """Draw clipped ROI edges and matching corner IDs without changing coordinates."""
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    height, width = image.shape[:2]
    if len(points) == 0:
        return
    valid = np.isfinite(points).all(axis=1)
    edge_count = len(points) if closed and len(points) > 2 else len(points) - 1
    for i in range(edge_count):
        j = (i + 1) % len(points)
        if valid[i] and valid[j]:
            visible, a, b = cv2.clipLine(
                (0, 0, width, height),
                tuple(np.rint(points[i]).astype(int)),
                tuple(np.rint(points[j]).astype(int)),
            )
            if visible:
                cv2.line(image, a, b, YELLOW, 2, cv2.LINE_AA)
    for i, point in enumerate(points):
        if not valid[i]:
            continue
        x, y = np.rint(point).astype(int)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (int(x), int(y)), 5, YELLOW, -1, cv2.LINE_AA)
            label = (int(np.clip(x + 9, 3, width - 24)),
                     int(np.clip(y - 8, 20, height - 5)))
            cv2.putText(image, str(i + 1), label, cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, str(i + 1), label, cv2.FONT_HERSHEY_SIMPLEX,
                        .65, YELLOW, 2, cv2.LINE_AA)


def camera_overlay(color, texture_uv, plane_mask, corners, enabled=True):
    """Tint only measured plane points; texture UVs come from RealSense map_to."""
    result = color.copy()
    if enabled:
        uv = np.asarray(texture_uv)
        selected = (np.asarray(plane_mask, dtype=bool)
                    & np.isfinite(uv).all(axis=1)
                    & (uv[:, 0] >= 0) & (uv[:, 0] < 1)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < 1))
        height, width = color.shape[:2]
        pixels = (uv[selected] * [width, height]).astype(np.int32)
        mask = np.zeros((height, width), dtype=bool)
        mask[pixels[:, 1], pixels[:, 0]] = True
        result[mask] = (result[mask] * .65 + np.asarray(GREEN) * .35).astype(np.uint8)
    draw_corners(result, corners, closed=len(corners) == 4)
    return result


def _place(canvas, source, x, nearest=False):
    height, width = source.shape[:2]
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    target_w = max(1, min(PANEL_WIDTH, round(width * scale)))
    target_h = max(1, min(PANEL_HEIGHT, round(height * scale)))
    left = x + (PANEL_WIDTH - target_w) // 2
    top = HEADER + (PANEL_HEIGHT - target_h) // 2
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    canvas[top:top + target_h, left:left + target_w] = cv2.resize(
        source, (target_w, target_h), interpolation=interpolation)
    return ImagePlacement(left, top, target_w, target_h, width, height)


def comparison_view(color, bev, title, status, controls, calibration=False,
                    right_title=None, legend=None):
    """Compose both views and return their exact click-coordinate transforms."""
    width = PANEL_WIDTH * 2 + MARGIN * 3
    canvas = np.full((HEADER + PANEL_HEIGHT + FOOTER, width, 3), BACKGROUND, np.uint8)
    cv2.putText(canvas, title, (MARGIN, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .7, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, status, (MARGIN, 53), cv2.FONT_HERSHEY_SIMPLEX,
                .53, (210, 215, 220), 1, cv2.LINE_AA)
    left_title = "CAMERA / CLICK 4 TABLE CORNERS HERE" if calibration else "CAMERA / RGB"
    cv2.putText(canvas, left_title, (MARGIN, 83), cv2.FONT_HERSHEY_SIMPLEX,
                .52, (245, 245, 245), 1, cv2.LINE_AA)
    right_x = PANEL_WIDTH + MARGIN * 2
    if right_title is None:
        right_title = "TOP VIEW / MATCHING CORNERS" if calibration else "TOP VIEW / OCCUPANCY"
    cv2.putText(canvas, right_title, (right_x, 83), cv2.FONT_HERSHEY_SIMPLEX,
                .52, (245, 245, 245), 1, cv2.LINE_AA)
    color_placement = _place(canvas, color, MARGIN)
    bev_placement = _place(canvas, bev, right_x, nearest=True)
    bottom = HEADER + PANEL_HEIGHT
    if legend is None:
        legend = "Green = measured plane   Red = objects   Yellow = ROI / matching corner IDs"
    cv2.putText(canvas, legend,
                (MARGIN, bottom + 23), cv2.FONT_HERSHEY_SIMPLEX,
                .51, (205, 215, 205), 1, cv2.LINE_AA)
    cv2.putText(canvas, controls, (MARGIN, bottom + 48), cv2.FONT_HERSHEY_SIMPLEX,
                .53, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas, {"rgb": color_placement, "bev": bev_placement}
