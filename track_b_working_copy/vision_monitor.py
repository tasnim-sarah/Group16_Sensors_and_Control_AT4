"""Display-only RGB-D and estimated-state monitoring for the Track B demo."""

import ctypes
import ctypes.wintypes
import os
import time

import cv2
import numpy as np


class VisionMonitor:
    """Render already-captured wrist-camera frames without affecting control."""

    WINDOW_NAME = "Track B RGB-D Vision Monitor"
    OBJECT_COLORS = {
        "Red Cube": (40, 40, 255),
        "Yellow Cube": (0, 230, 255),
        "Blue Cube": (255, 150, 40),
        "Pink Tray": (220, 80, 220),
        "Green Tray": (60, 220, 60),
        "Purple Tray": (200, 80, 180),
    }

    def __init__(self, image_width, image_height, max_depth_m=1.5):
        if image_width <= 0 or image_height <= 0 or max_depth_m <= 0:
            raise ValueError("image dimensions and maximum depth must be positive")
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.max_depth_m = float(max_depth_m)
        self.enabled = True
        self.window_width = 1280
        self.window_height = 660
        self._timings = []
        self._last_fps_time = time.perf_counter()
        self._fps_frames = 0
        self.fps = 0.0
        self.display_time_ms = 0.0

    def open(self):
        """Create and place the monitor window next to the MuJoCo window."""
        screen_width = 1920
        screen_height = 1080
        if os.name == "nt":
            user32 = ctypes.windll.user32
            screen_width = int(user32.GetSystemMetrics(0))
            screen_height = int(user32.GetSystemMetrics(1))

        self.window_width = min(1280, max(800, screen_width // 2))
        self.window_width -= self.window_width % 2
        pane_width = self.window_width // 2
        pane_height = round(pane_width * self.image_height / self.image_width)
        self.window_height = min(screen_height, pane_height + 220)

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.window_width, self.window_height)
        preview = np.zeros(
            (self.window_height, self.window_width, 3), dtype=np.uint8
        )
        self._put_text(
            preview,
            "Waiting for the first wrist RGB-D frame...",
            (24, min(self.window_height - 18, pane_height + 38)),
            scale=0.8,
            thickness=2,
        )
        cv2.imshow(self.WINDOW_NAME, preview)
        cv2.waitKey(1)
        cv2.moveWindow(self.WINDOW_NAME, screen_width - self.window_width, 0)
        positioned = self._position_simulation_window(
            screen_width - self.window_width,
            self.window_width,
            self.window_height,
        )
        if positioned:
            print("RGB-D monitor opened beside the MuJoCo viewer (press Q to close monitor only).")
        else:
            print(
                "RGB-D monitor opened. The MuJoCo window could not be positioned "
                "automatically; arrange the two windows side by side."
            )

    def _position_simulation_window(self, x, width, height):
        if os.name != "nt":
            return False

        user32 = ctypes.windll.user32
        process_id = os.getpid()
        candidates = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        @callback_type
        def visit(hwnd, _):
            window_process_id = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_process_id))
            if (
                window_process_id.value != process_id
                or not user32.IsWindowVisible(hwnd)
            ):
                return True

            title_length = user32.GetWindowTextLengthW(hwnd)
            if title_length <= 0:
                return True

            title = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(hwnd, title, len(title))
            if title.value == self.WINDOW_NAME:
                return True

            rect = ctypes.wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
                candidates.append((area, hwnd))
            return True

        user32.EnumWindows(visit, 0)
        if not candidates:
            return False

        hwnd = max(candidates, key=lambda item: item[0])[1]
        no_z_order = 0x0004
        no_activate = 0x0010
        return bool(
            user32.SetWindowPos(
                hwnd, None, 0, 0, x, height, no_z_order | no_activate
            )
        )

    @staticmethod
    def _put_text(image, text, origin, color=(235, 235, 235), scale=0.48, thickness=1):
        cv2.putText(
            image,
            str(text),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _annotated_rgb(self, rgb, detections):
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for name, detection in detections.items():
            contour = detection.get("contour")
            if contour is None:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            color = self.OBJECT_COLORS.get(name, (255, 255, 255))
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            self._put_text(
                image,
                f"{name} {detection['depth_m']:.2f}m",
                (x, max(18, y - 7)),
                color,
                scale=0.55,
                thickness=2,
            )
        self._put_text(image, "WRIST RGB  |  detected objects", (12, 24), scale=0.62, thickness=2)
        return image

    def _annotated_depth(self, depth, detections):
        values = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(values) & (values > 0.0)
        scaled = np.zeros(values.shape, dtype=np.uint8)
        scaled[valid] = np.clip(
            values[valid] / self.max_depth_m * 255.0, 0, 255
        ).astype(np.uint8)
        image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        image[~valid] = (0, 0, 0)
        for name, detection in detections.items():
            contour = detection.get("contour")
            if contour is None:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            color = self.OBJECT_COLORS.get(name, (255, 255, 255))
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            self._put_text(
                image,
                f"{detection['depth_m']:.3f} m",
                (x, min(self.image_height - 8, y + height + 18)),
                color,
                scale=0.52,
                thickness=2,
            )
        self._put_text(image, f"DEPTH  |  0-{self.max_depth_m:.1f}m", (12, 24), scale=0.62, thickness=2)
        return image

    def compose(
        self,
        rgb,
        depth,
        detections,
        target_name,
        target_status,
        raw_position,
        filtered_position,
        pbvs_target,
        pbvs_error,
        stage,
        q_estimated,
        q_ground_truth,
        observer_std,
        lqr_active,
    ):
        """Build the combined display frame; does not run control or capture images."""
        pane_width = self.window_width // 2
        pane_height = round(pane_width * self.image_height / self.image_width)
        rgb_view = cv2.resize(self._annotated_rgb(rgb, detections), (pane_width, pane_height))
        depth_view = cv2.resize(self._annotated_depth(depth, detections), (pane_width, pane_height))
        image_row = np.hstack((rgb_view, depth_view))

        info_height = self.window_height - pane_height
        info = np.zeros((info_height, self.window_width, 3), dtype=np.uint8)
        cv2.rectangle(info, (8, 8), (self.window_width - 8, info_height - 8), (65, 65, 65), 1)

        raw_text = "n/a" if raw_position is None else np.array2string(
            np.asarray(raw_position), precision=3, suppress_small=True
        )
        filtered_text = "n/a" if filtered_position is None else np.array2string(
            np.asarray(filtered_position), precision=3, suppress_small=True
        )
        target_text = "n/a" if pbvs_target is None else np.array2string(
            np.asarray(pbvs_target), precision=3, suppress_small=True
        )
        error_text = "n/a" if pbvs_error is None else f"{pbvs_error:.3f}m"
        status_color = {
            "VISIBLE": (70, 230, 70),
            "PREDICTED": (0, 200, 255),
            "OCCLUDED": (0, 200, 255),
            "LOST": (40, 40, 255),
            "IDLE": (180, 180, 180),
            "FALLBACK": (0, 160, 255),
        }.get(target_status, (230, 230, 230))
        self._put_text(
            info,
            f"TARGET: {target_name or 'none'}  STATUS: {target_status}  STAGE: {stage}",
            (18, 30),
            status_color,
            scale=0.58,
            thickness=2,
        )
        self._put_text(info, f"RGB-D raw: {raw_text}", (18, 58))
        self._put_text(info, f"Kalman filtered: {filtered_text}", (18, 82))
        self._put_text(
            info,
            f"PBVS target: {target_text}  |  TCP position error: {error_text}",
            (18, 106),
        )
        control_color = (70, 230, 70) if lqr_active else (40, 40, 255)
        self._put_text(
            info,
            f"LQR STATE FEEDBACK: {'ACTIVE' if lqr_active else 'INACTIVE'}",
            (18, 132),
            control_color,
            scale=0.55,
            thickness=2,
        )
        q_estimated = np.asarray(q_estimated, dtype=float).reshape(6)
        q_ground_truth = np.asarray(q_ground_truth, dtype=float).reshape(6)
        observer_std = np.asarray(observer_std, dtype=float).reshape(6)
        row_text = "  ".join(
            f"J{i + 1}:{q_estimated[i]:+.2f}/{q_ground_truth[i]:+.2f}"
            for i in range(6)
        )
        self._put_text(info, f"q_est / q_true (rad): {row_text}", (18, 158), scale=0.43)
        uncertainty_text = "  ".join(
            f"J{i + 1}:{observer_std[i]:.4f}"
            for i in range(6)
        )
        self._put_text(
            info,
            f"Observer sigma (rad): {uncertainty_text}",
            (18, min(info_height - 18, 183)),
            scale=0.43,
        )
        self._put_text(
            info,
            f"Monitor render/event pump: {self.display_time_ms:.1f}ms  {self.fps:.1f} FPS",
            (18, min(info_height - 4, 208)),
            scale=0.43,
        )
        return np.vstack((image_row, info))

    def update(self, frame, **status):
        """Display a previously captured frame and service the monitor window."""
        if not self.enabled:
            return
        if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            self.enabled = False
            return
        started = time.perf_counter()
        display = self.compose(
            frame["rgb"], frame["depth"], frame["detections"], **status
        )
        cv2.imshow(self.WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.enabled = False
            cv2.destroyWindow(self.WINDOW_NAME)
        elapsed = time.perf_counter() - started
        self.display_time_ms = elapsed * 1000.0
        self._timings.append(self.display_time_ms)
        if len(self._timings) > 300:
            self._timings.pop(0)
        self._fps_frames += 1
        now = time.perf_counter()
        if now - self._last_fps_time >= 1.0:
            self.fps = self._fps_frames / (now - self._last_fps_time)
            self._fps_frames = 0
            self._last_fps_time = now

    def mean_display_time_ms(self):
        """Return mean composition, display, and event-pump time."""
        return float(np.mean(self._timings)) if self._timings else 0.0

    def close(self):
        if self.enabled:
            cv2.destroyWindow(self.WINDOW_NAME)
            self.enabled = False
