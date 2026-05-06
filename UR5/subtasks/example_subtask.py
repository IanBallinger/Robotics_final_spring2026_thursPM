import csv
import time
import math
import json
import threading
import os
from pathlib import Path #filesystem, not robot point paths

from arm import UR5Arm

GRIPPER_OPEN_MM_MAX = 85.0

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    from robotiq_gripper_control import RobotiqGripper
except Exception:
    RobotiqGripper = None


CM_PIXEL = 54.0 / 275.0
CENTER_X = 337.5
CENTER_Y = 337.5

CAMERA_COLOR_THRESHOLDS = {
    "purple": ((156, 83, 0), (180, 176, 143)),
    "yellow": ((13, 255, 120), (98, 255, 208)),
    "green": ((53, 90, 128), (87, 180, 221)),
    "blue": ((95, 105, 158), (103, 255, 255)),
    "red": ((1, 180, 131), (3, 255, 236)),
}

CAMERA_DEFAULT_MIN_AREA = 500
CAMERA_DEFAULT_KERNEL_SIZE = 5
CAMERA_DEFAULT_OPEN_ITER = 3
CAMERA_DEFAULT_CLOSE_ITER = 3
VISION_CAMERA_SCAN_MAX_INDEX = 12
VISION_EXCLUDED_CAMERA_INDICES = set()
VISION_FIRST_FRAME_TIMEOUT_S = 5.0

DEFAULT_XY_CALIBRATION = {
    "x_scale": 1.0,
    "x_offset_m": 0.0,
    "y_scale": 1.0,
    "y_offset_m": 0.0,
    "xy_cross": 0.0,
    "yx_cross": 0.0,
}

DEFAULT_XZ_CALIBRATION = {
    "x_scale": 1.0,
    "x_offset_m": 0.0,
    "y_offset_m": 0.36,
    "z_scale": 1.0,
    "z_offset_m": 0.06,
    "xz_cross": 0.0,
    "zx_cross": 0.0,
}

CAMERA_TARGET_SPECS = {
    2: {
        "axis_pair": ("x", "y"),
        "targets": [
            ("blue plate", "blue"),
            ("purple cup", "purple"),
            ("red bowl", "red"),
            ("yellow cup", "yellow"),
            ("green bottle", "green"),
        ],
    },
    3: {
        "axis_pair": ("x", "z"),
        "targets": [
            ("blue microwave door handle", "blue"),
            ("red stop button", "red"),
            ("blue plate edge", "blue"),
            ("red bowl edge", "red"),
        ],
    },
}

VISION_TARGET_ALIASES = {
    "blue plate": ["blue plate", "microwavable_plate", "plate"],
    "purple cup": ["purple cup", "cup_for_drink", "cup"],
    "yellow cup": ["yellow cup"],
    "red bowl": ["red bowl", "microwavable_bowl", "bowl"],
    "green bottle": ["green bottle", "bottle_to_fill_cup", "bottle"],
    "blue microwave door handle": ["blue microwave door handle", "microwave_door_handle", "door_handle"],
    "red stop button": ["red stop button", "microwave_stop_button", "stop_button"],
    "blue plate edge": ["blue plate edge", "plate_edge"],
    "red bowl edge": ["red bowl edge", "bowl_edge"],
}

TASK_DEFAULT_TARGETS = {
    "open_microwave_door": "blue microwave door handle",
    "close_microwave_door": "blue microwave door handle",
    "press_microwave_stop": "red stop button",
    "acquire_plate": "blue plate",
    "acquire_bowl": "red bowl",
    "acquire_cup": "purple cup",
    "acquire_bottle": "green bottle",
    "place_plate_in_microwave": "blue plate edge",
    "take_plate_out_to_tray": "blue plate edge",
    "place_bowl_in_microwave": "red bowl edge",
    "take_bowl_out_to_tray": "red bowl edge",
}


def _norm_label(label):
    return str(label or "").strip().lower().replace("_", " ")


ALIAS_TO_PRIMARY = {}
for _primary, _aliases in VISION_TARGET_ALIASES.items():
    ALIAS_TO_PRIMARY[_norm_label(_primary)] = _primary
    for _alias in _aliases:
        ALIAS_TO_PRIMARY[_norm_label(_alias)] = _primary


def _resolve_primary_target_label(label):
    return ALIAS_TO_PRIMARY.get(_norm_label(label), "")


def _camera_point_from_pixel(xc, yc, axis_pair):
    x_m = ((float(xc) - CENTER_X) * CM_PIXEL) / 100.0
    second_m = ((CENTER_Y - float(yc)) * CM_PIXEL) / 100.0
    return {
        "x": x_m,
        axis_pair[1]: second_m,
    }


def _default_plane_calibration(axis_pair):
    second = str(axis_pair[1]).lower() if len(axis_pair) >= 2 else "y"
    if second == "z":
        return dict(DEFAULT_XZ_CALIBRATION)
    return dict(DEFAULT_XY_CALIBRATION)


def _coerce_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _sanitize_plane_calibration(raw, axis_pair):
    second = str(axis_pair[1]).lower() if len(axis_pair) >= 2 else "y"
    defaults = _default_plane_calibration(axis_pair)
    payload = raw if isinstance(raw, dict) else {}

    sanitized = {
        "x_scale": _coerce_float(payload.get("x_scale", defaults["x_scale"]), defaults["x_scale"]),
        "x_offset_m": _coerce_float(payload.get("x_offset_m", defaults["x_offset_m"]), defaults["x_offset_m"]),
    }
    if second == "z":
        sanitized["y_offset_m"] = _coerce_float(payload.get("y_offset_m", defaults["y_offset_m"]), defaults["y_offset_m"])
        sanitized["z_scale"] = _coerce_float(payload.get("z_scale", defaults["z_scale"]), defaults["z_scale"])
        sanitized["z_offset_m"] = _coerce_float(payload.get("z_offset_m", defaults["z_offset_m"]), defaults["z_offset_m"])
        sanitized["xz_cross"] = _coerce_float(payload.get("xz_cross", defaults["xz_cross"]), defaults["xz_cross"])
        sanitized["zx_cross"] = _coerce_float(payload.get("zx_cross", defaults["zx_cross"]), defaults["zx_cross"])
        raw_target_offsets = payload.get("target_z_offset_m", {})
        target_offsets = {}
        if isinstance(raw_target_offsets, dict):
            for target_name, raw_val in raw_target_offsets.items():
                target_offsets[str(target_name)] = _coerce_float(raw_val, 0.0)
        sanitized["target_z_offset_m"] = target_offsets
    else:
        sanitized["y_scale"] = _coerce_float(payload.get("y_scale", defaults["y_scale"]), defaults["y_scale"])
        sanitized["y_offset_m"] = _coerce_float(payload.get("y_offset_m", defaults["y_offset_m"]), defaults["y_offset_m"])
        sanitized["xy_cross"] = _coerce_float(payload.get("xy_cross", defaults["xy_cross"]), defaults["xy_cross"])
        sanitized["yx_cross"] = _coerce_float(payload.get("yx_cross", defaults["yx_cross"]), defaults["yx_cross"])
    return sanitized


class _DualCameraVisionFeeds:
    def __init__(
        self,
        camera_indices=None,
        max_camera_index=VISION_CAMERA_SCAN_MAX_INDEX,
        excluded_camera_indices=None,
    ):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = {}
        self._spec_keys = sorted(CAMERA_TARGET_SPECS.keys())
        self._spec_choices = [None] + self._spec_keys
        self._camera_enabled = {}
        self._camera_spec_selection = {}
        self._item_settings = {}
        self._window_names = {}
        self._view_modes = {}
        self._trackbar_ready = {}
        self._trackbar_seeded = {}
        self._last_spec_for_camera = {}
        self._settings_file_path = None
        self._settings_dirty = False
        self._last_settings_save_ts = 0.0
        self._settings_save_interval_s = 0.5
        self._camera_plane_calibration = {}

        self._excluded_camera_indices = set(VISION_EXCLUDED_CAMERA_INDICES)
        if excluded_camera_indices is not None:
            self._excluded_camera_indices = {int(v) for v in excluded_camera_indices}

        self._available_camera_indices = list(camera_indices) if camera_indices else self._discover_available_camera_indices(max_camera_index)
        self._available_camera_indices = [
            idx for idx in self._available_camera_indices if idx not in self._excluded_camera_indices
        ]
        if not self._available_camera_indices:
            self._available_camera_indices = sorted(CAMERA_TARGET_SPECS.keys())
            print("[vision] Warning: no cameras discovered; using configured indices:", self._available_camera_indices)

        for camera_index in self._available_camera_indices:
            if camera_index in self._spec_keys:
                self._camera_spec_selection[camera_index] = self._spec_choices.index(camera_index)
                self._camera_enabled[camera_index] = 1
            else:
                self._camera_spec_selection[camera_index] = 0
                self._camera_enabled[camera_index] = 0
            self._last_spec_for_camera[camera_index] = self._active_spec_key(camera_index)
            self._view_modes[camera_index] = "dashboard"
            self._window_names[camera_index] = f"vision_cam_{camera_index}"
            self._trackbar_ready[camera_index] = False
            self._trackbar_seeded[camera_index] = False
            axis_pair = tuple(CAMERA_TARGET_SPECS.get(self._active_spec_key(camera_index), {}).get("axis_pair", ("x", "y")))
            self._camera_plane_calibration[camera_index] = _default_plane_calibration(axis_pair)

        for spec in CAMERA_TARGET_SPECS.values():
            for color_name, target_label in self._iter_target_pairs_for_spec(spec):
                if target_label in self._item_settings:
                    continue
                low_raw, high_raw = CAMERA_COLOR_THRESHOLDS.get(color_name, ((0, 0, 0), (255, 255, 255)))
                self._item_settings[target_label] = {
                    "h_lo": int(low_raw[0]),
                    "s_lo": int(low_raw[1]),
                    "v_lo": int(low_raw[2]),
                    "h_hi": int(high_raw[0]),
                    "s_hi": int(high_raw[1]),
                    "v_hi": int(high_raw[2]),
                    "min_area": int(CAMERA_DEFAULT_MIN_AREA),
                    "kernel": int(CAMERA_DEFAULT_KERNEL_SIZE),
                    "open_iter": int(CAMERA_DEFAULT_OPEN_ITER),
                    "close_iter": int(CAMERA_DEFAULT_CLOSE_ITER),
                }

        self._buffers = {
            idx: {
                "timestamp": 0.0,
                "axis_pair": tuple(CAMERA_TARGET_SPECS.get(self._active_spec_key(idx), {}).get("axis_pair", ("x", "y"))),
                "spec_key": self._active_spec_key(idx),
                "enabled": int(self._camera_enabled.get(idx, 0)),
                "targets": {},
            }
            for idx in self._available_camera_indices
        }

        print("[vision] Active camera indices:", self._available_camera_indices)

    def _discover_available_camera_indices(self, max_camera_index):
        discovered = []
        max_idx = max(0, int(max_camera_index))
        for camera_index in range(max_idx + 1):
            if camera_index in self._excluded_camera_indices:
                continue
            cap = self._open_capture(camera_index)
            is_open = bool(cap is not None and cap.isOpened())
            if is_open:
                discovered.append(camera_index)
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
        return discovered

    def configure_persistence(self, settings_file_path=None):
        if settings_file_path is None:
            return

        normalized = str(settings_file_path).strip()
        if not normalized:
            self._settings_file_path = None
            return

        path = Path(normalized)
        if self._settings_file_path == path:
            return

        self._settings_file_path = path
        self._load_settings_from_disk()

    def _settings_payload(self):
        camera_orientation = {}
        camera_plane_calibration = {}
        for cam_idx in self._available_camera_indices:
            spec_key = self._active_spec_key(cam_idx)
            spec = CAMERA_TARGET_SPECS.get(spec_key, {}) if spec_key is not None else {}
            axis_pair = tuple(spec.get("axis_pair", ("x", "y")))
            plane = "xz" if (len(axis_pair) >= 2 and str(axis_pair[1]).lower() == "z") else "xy"
            calibration = _sanitize_plane_calibration(self._camera_plane_calibration.get(cam_idx, {}), axis_pair)
            self._camera_plane_calibration[cam_idx] = calibration
            camera_orientation[str(cam_idx)] = {
                "axis_pair": [str(axis_pair[0]), str(axis_pair[1])],
                "plane": plane,
            }
            if plane == "xz":
                camera_plane_calibration[str(cam_idx)] = {
                    "x_scale": float(calibration.get("x_scale", 1.0)),
                    "x_offset_m": float(calibration.get("x_offset_m", 0.0)),
                    "y_offset_m": float(calibration.get("y_offset_m", 0.36)),
                    "z_scale": float(calibration.get("z_scale", 1.0)),
                    "z_offset_m": float(calibration.get("z_offset_m", 0.06)),
                    "xz_cross": float(calibration.get("xz_cross", 0.0)),
                    "zx_cross": float(calibration.get("zx_cross", 0.0)),
                    "target_z_offset_m": {
                        str(k): float(v)
                        for k, v in dict(calibration.get("target_z_offset_m", {})).items()
                    },
                }

        return {
            "version": 1,
            "camera_orientation": camera_orientation,
            "camera_plane_calibration": camera_plane_calibration,
            "camera_spec_selection": {
                str(cam_idx): int(sel)
                for cam_idx, sel in self._camera_spec_selection.items()
            },
            "item_settings": {
                str(target): {
                    "h_lo": int(item.get("h_lo", 0)),
                    "s_lo": int(item.get("s_lo", 0)),
                    "v_lo": int(item.get("v_lo", 0)),
                    "h_hi": int(item.get("h_hi", 255)),
                    "s_hi": int(item.get("s_hi", 255)),
                    "v_hi": int(item.get("v_hi", 255)),
                    "min_area": int(item.get("min_area", CAMERA_DEFAULT_MIN_AREA)),
                    "kernel": int(item.get("kernel", CAMERA_DEFAULT_KERNEL_SIZE)),
                    "open_iter": int(item.get("open_iter", CAMERA_DEFAULT_OPEN_ITER)),
                    "close_iter": int(item.get("close_iter", CAMERA_DEFAULT_CLOSE_ITER)),
                }
                for target, item in self._item_settings.items()
            },
        }

    def _load_settings_from_disk(self):
        if self._settings_file_path is None or not self._settings_file_path.exists():
            return

        try:
            with self._settings_file_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print(f"[vision] Warning: could not read settings file '{self._settings_file_path}': {exc}")
            return

        if not isinstance(payload, dict):
            return

        # Optional metadata-only field for downstream tooling; accepted for compatibility.
        raw_camera_orientation = payload.get("camera_orientation", {})
        if raw_camera_orientation is not None and not isinstance(raw_camera_orientation, dict):
            raw_camera_orientation = {}

        raw_spec_sel = payload.get("camera_spec_selection", {})
        if isinstance(raw_spec_sel, dict):
            for cam_key, sel in raw_spec_sel.items():
                try:
                    cam_idx = int(cam_key)
                    sel_idx = int(sel)
                except Exception:
                    continue
                if cam_idx not in self._camera_spec_selection:
                    continue
                self._camera_spec_selection[cam_idx] = max(0, min(sel_idx, max(0, len(self._spec_choices) - 1)))

        raw_plane_calib = payload.get("camera_plane_calibration", {})
        if isinstance(raw_plane_calib, dict):
            for cam_key, raw_calib in raw_plane_calib.items():
                try:
                    cam_idx = int(cam_key)
                except Exception:
                    continue
                if cam_idx not in self._camera_spec_selection:
                    continue
                spec_key = self._active_spec_key(cam_idx)
                axis_pair = tuple(CAMERA_TARGET_SPECS.get(spec_key, {}).get("axis_pair", ("x", "y")))
                self._camera_plane_calibration[cam_idx] = _sanitize_plane_calibration(raw_calib, axis_pair)

        raw_items = payload.get("item_settings", {})
        if isinstance(raw_items, dict):
            for target_name, raw_item in raw_items.items():
                if target_name not in self._item_settings or not isinstance(raw_item, dict):
                    continue
                item = self._item_settings[target_name]
                try:
                    h_lo = int(raw_item.get("h_lo", item["h_lo"]))
                    s_lo = int(raw_item.get("s_lo", item["s_lo"]))
                    v_lo = int(raw_item.get("v_lo", item["v_lo"]))
                    h_hi = int(raw_item.get("h_hi", item["h_hi"]))
                    s_hi = int(raw_item.get("s_hi", item["s_hi"]))
                    v_hi = int(raw_item.get("v_hi", item["v_hi"]))
                    item["h_lo"] = max(0, min(255, min(h_lo, h_hi)))
                    item["s_lo"] = max(0, min(255, min(s_lo, s_hi)))
                    item["v_lo"] = max(0, min(255, min(v_lo, v_hi)))
                    item["h_hi"] = max(0, min(255, max(h_lo, h_hi)))
                    item["s_hi"] = max(0, min(255, max(s_lo, s_hi)))
                    item["v_hi"] = max(0, min(255, max(v_lo, v_hi)))
                    item["min_area"] = max(1, int(raw_item.get("min_area", item["min_area"])))
                    item["kernel"] = max(1, int(raw_item.get("kernel", item["kernel"])) | 1)
                    item["open_iter"] = max(0, int(raw_item.get("open_iter", item["open_iter"])))
                    item["close_iter"] = max(0, int(raw_item.get("close_iter", item["close_iter"])))
                except Exception:
                    continue

        print(f"[vision] Loaded tuned specs from {self._settings_file_path}")

    def _persist_settings_if_dirty(self, force=False):
        if self._settings_file_path is None:
            return
        if not self._settings_dirty:
            return
        now = time.time()
        if not force and (now - self._last_settings_save_ts) < self._settings_save_interval_s:
            return

        try:
            self._settings_file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._settings_file_path.with_suffix(self._settings_file_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(self._settings_payload(), fh, indent=2, sort_keys=True)
            tmp_path.replace(self._settings_file_path)
            self._settings_dirty = False
            self._last_settings_save_ts = now
        except Exception as exc:
            print(f"[vision] Warning: failed to persist tuned specs to '{self._settings_file_path}': {exc}")

    @staticmethod
    def _iter_target_pairs_for_spec(spec):
        # spec["targets"] is [(label, color_name), ...]
        for label, color_name in spec.get("targets", []):
            yield color_name, label

    def _window_name(self, camera_index):
        return self._window_names[camera_index]

    def _active_spec_key(self, camera_index):
        sel = int(self._camera_spec_selection.get(camera_index, 0))
        if sel < 0:
            sel = 0
        if sel >= len(self._spec_choices):
            sel = len(self._spec_choices) - 1
        self._camera_spec_selection[camera_index] = sel
        return self._spec_choices[sel]

    def _ensure_trackbars(self, camera_index):
        if self._trackbar_ready.get(camera_index):
            return

        # If persisted settings are available, ensure in-memory values are hydrated
        # before creating UI controls so default slider values don't clobber disk state.
        if self._settings_file_path is not None and self._settings_file_path.exists() and not self._settings_dirty:
            self._load_settings_from_disk()

        window_name = self._window_name(camera_index)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("enabled", window_name, int(self._camera_enabled.get(camera_index, 0)), 1, lambda _v: None)
        cv2.createTrackbar("spec_idx", window_name, int(self._camera_spec_selection.get(camera_index, 0)), max(0, len(self._spec_choices) - 1), lambda _v: None)
        cv2.createTrackbar("target_idx", window_name, 0, 10, lambda _v: None)
        cv2.createTrackbar("h_lo", window_name, 0, 255, lambda _v: None)
        cv2.createTrackbar("s_lo", window_name, 0, 255, lambda _v: None)
        cv2.createTrackbar("v_lo", window_name, 0, 255, lambda _v: None)
        cv2.createTrackbar("h_hi", window_name, 255, 255, lambda _v: None)
        cv2.createTrackbar("s_hi", window_name, 255, 255, lambda _v: None)
        cv2.createTrackbar("v_hi", window_name, 255, 255, lambda _v: None)
        cv2.createTrackbar("min_area", window_name, CAMERA_DEFAULT_MIN_AREA, 10000, lambda _v: None)
        cv2.createTrackbar("kernel", window_name, CAMERA_DEFAULT_KERNEL_SIZE, 31, lambda _v: None)
        cv2.createTrackbar("open_iter", window_name, CAMERA_DEFAULT_OPEN_ITER, 10, lambda _v: None)
        cv2.createTrackbar("close_iter", window_name, CAMERA_DEFAULT_CLOSE_ITER, 10, lambda _v: None)
        cv2.createTrackbar("y_offset_cm", window_name, 36, 100, lambda _v: None)
        cv2.createTrackbar("z_offset_cm", window_name, 6, 100, lambda _v: None)

        cv2.setTrackbarPos("enabled", window_name, int(self._camera_enabled.get(camera_index, 0)))
        cv2.setTrackbarPos("spec_idx", window_name, int(self._camera_spec_selection.get(camera_index, 0)))
        axis_pair = tuple(CAMERA_TARGET_SPECS.get(self._active_spec_key(camera_index), {}).get("axis_pair", ("x", "y")))
        calibration = _sanitize_plane_calibration(self._camera_plane_calibration.get(camera_index, {}), axis_pair)
        self._camera_plane_calibration[camera_index] = calibration
        cv2.setTrackbarPos(
            "y_offset_cm",
            window_name,
            max(0, min(100, int(round(float(calibration.get("y_offset_m", 0.36)) * 100.0)))),
        )
        cv2.setTrackbarPos(
            "z_offset_cm",
            window_name,
            max(0, min(100, int(round(float(calibration.get("z_offset_m", 0.06)) * 100.0)))),
        )
        self._trackbar_ready[camera_index] = True
        self._trackbar_seeded[camera_index] = False

    def _safe_get_trackbar(self, camera_index, name, default=0):
        window_name = self._window_name(camera_index)
        try:
            return int(cv2.getTrackbarPos(name, window_name))
        except Exception:
            # If user closes a window, OpenCV drops the trackbar handle.
            self._trackbar_ready[camera_index] = False
            self._ensure_trackbars(camera_index)
            try:
                return int(cv2.getTrackbarPos(name, window_name))
            except Exception:
                return int(default)

    def _sync_trackbars_from_item(self, camera_index, item):
        window_name = self._window_name(camera_index)
        cv2.setTrackbarPos("h_lo", window_name, int(item["h_lo"]))
        cv2.setTrackbarPos("s_lo", window_name, int(item["s_lo"]))
        cv2.setTrackbarPos("v_lo", window_name, int(item["v_lo"]))
        cv2.setTrackbarPos("h_hi", window_name, int(item["h_hi"]))
        cv2.setTrackbarPos("s_hi", window_name, int(item["s_hi"]))
        cv2.setTrackbarPos("v_hi", window_name, int(item["v_hi"]))
        cv2.setTrackbarPos("min_area", window_name, int(item["min_area"]))
        cv2.setTrackbarPos("kernel", window_name, int(item["kernel"]))
        cv2.setTrackbarPos("open_iter", window_name, int(item["open_iter"]))
        cv2.setTrackbarPos("close_iter", window_name, int(item["close_iter"]))

    def _read_item_settings_from_trackbars(self, camera_index, item):
        window_name = self._window_name(camera_index)
        previous = dict(item)

        h_lo = int(self._safe_get_trackbar(camera_index, "h_lo", item.get("h_lo", 0)))
        s_lo = int(self._safe_get_trackbar(camera_index, "s_lo", item.get("s_lo", 0)))
        v_lo = int(self._safe_get_trackbar(camera_index, "v_lo", item.get("v_lo", 0)))
        h_hi = int(self._safe_get_trackbar(camera_index, "h_hi", item.get("h_hi", 255)))
        s_hi = int(self._safe_get_trackbar(camera_index, "s_hi", item.get("s_hi", 255)))
        v_hi = int(self._safe_get_trackbar(camera_index, "v_hi", item.get("v_hi", 255)))

        item["h_lo"] = min(h_lo, h_hi)
        item["s_lo"] = min(s_lo, s_hi)
        item["v_lo"] = min(v_lo, v_hi)
        item["h_hi"] = max(h_lo, h_hi)
        item["s_hi"] = max(s_lo, s_hi)
        item["v_hi"] = max(v_lo, v_hi)
        item["min_area"] = max(1, int(self._safe_get_trackbar(camera_index, "min_area", item.get("min_area", CAMERA_DEFAULT_MIN_AREA))))
        item["kernel"] = max(1, int(self._safe_get_trackbar(camera_index, "kernel", item.get("kernel", CAMERA_DEFAULT_KERNEL_SIZE))) | 1)
        item["open_iter"] = max(0, int(self._safe_get_trackbar(camera_index, "open_iter", item.get("open_iter", CAMERA_DEFAULT_OPEN_ITER))))
        item["close_iter"] = max(0, int(self._safe_get_trackbar(camera_index, "close_iter", item.get("close_iter", CAMERA_DEFAULT_CLOSE_ITER))))
        return previous != item

    def _render_window(self, camera_index, frame, overlay_frame, selected_mask, selected_opened, selected_closed, selected_target_name, spec_key):
        mode = self._view_modes.get(camera_index, "overlay")

        def _to_bgr(img):
            if img is None:
                return np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        def _panel(img, label):
            panel = cv2.resize(_to_bgr(img), (360, 240))
            cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return panel

        if mode == "dashboard":
            top = np.hstack([
                _panel(frame, "raw"),
                _panel(overlay_frame, "overlay"),
                _panel(selected_mask, "mask"),
            ])
            bottom = np.hstack([
                _panel(selected_opened, "opened"),
                _panel(selected_closed, "closed"),
                _panel(None, "controls: press v"),
            ])
            shown = np.vstack([top, bottom])
        elif mode == "raw":
            shown = frame
        elif mode == "mask" and selected_mask is not None:
            shown = selected_mask
        elif mode == "opened" and selected_opened is not None:
            shown = selected_opened
        elif mode == "closed" and selected_closed is not None:
            shown = selected_closed
        else:
            shown = overlay_frame

        if len(shown.shape) == 2:
            shown = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)

        info = f"cam={camera_index} spec={spec_key} target={selected_target_name or '-'} view={mode} (press v)"
        cv2.putText(
            shown,
            info,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow(self._window_name(camera_index), shown)

    def _handle_view_key(self, camera_index):
        modes = ["dashboard", "overlay", "raw", "mask", "opened", "closed"]
        current = self._view_modes.get(camera_index, "overlay")
        try:
            idx = modes.index(current)
        except ValueError:
            idx = 0
        self._view_modes[camera_index] = modes[(idx + 1) % len(modes)]
        print(f"[vision] camera {camera_index} view mode -> {self._view_modes[camera_index]}")

    def start(self):
        if cv2 is None or np is None:
            print("[vision] OpenCV/numpy unavailable; camera-aware offsets disabled")
            return
        self._stop.clear()
        for camera_index in self._available_camera_indices:
            existing = self._threads.get(camera_index)
            if existing is not None and existing.is_alive():
                continue
            self._threads.pop(camera_index, None)
            t = threading.Thread(target=self._run_camera_loop, args=(camera_index,), daemon=True)
            self._threads[camera_index] = t
            t.start()

    def _run_camera_loop(self, camera_index):
        self._ensure_trackbars(camera_index)
        cap = self._open_capture(camera_index)
        if not cap.isOpened():
            print(f"[vision] Warning: camera index {camera_index} unavailable; terminating camera thread")
            with self._lock:
                self._camera_enabled[camera_index] = 0
                if camera_index in self._buffers:
                    self._buffers[camera_index]["enabled"] = 0
                    self._buffers[camera_index]["targets"] = {}
                self._persist_settings_if_dirty(force=True)
            try:
                cap.release()
            except Exception:
                pass
            return

        first_frame_deadline = time.time() + VISION_FIRST_FRAME_TIMEOUT_S
        got_any_frame = False
        gui_disable_deadline = None

        last_selected_target_name = ""

        while not self._stop.is_set():
            if not cap.isOpened():
                cap.release()
                cap = self._open_capture(camera_index)
                if not cap.isOpened():
                    print(f"[vision] Warning: camera index {camera_index} lost and failed to reinitialize; terminating camera thread")
                    with self._lock:
                        self._camera_enabled[camera_index] = 0
                        if camera_index in self._buffers:
                            self._buffers[camera_index]["enabled"] = 0
                            self._buffers[camera_index]["targets"] = {}
                        self._persist_settings_if_dirty(force=True)
                    break
                time.sleep(0.25)
                continue

            ok, frame = cap.read()
            if not ok:
                if not got_any_frame and time.time() > first_frame_deadline:
                    print(
                        f"[vision] Warning: camera index {camera_index} produced no frames "
                        f"within {VISION_FIRST_FRAME_TIMEOUT_S:.1f}s; disabling and terminating thread"
                    )
                    with self._lock:
                        self._camera_enabled[camera_index] = 0
                        if camera_index in self._buffers:
                            self._buffers[camera_index]["enabled"] = 0
                            self._buffers[camera_index]["targets"] = {}
                        self._persist_settings_if_dirty(force=True)
                    break
                time.sleep(0.02)
                continue

            got_any_frame = True

            enabled_now = 1 if self._safe_get_trackbar(camera_index, "enabled", self._camera_enabled.get(camera_index, 0)) > 0 else 0
            enabled_prev = int(self._camera_enabled.get(camera_index, 0))
            if enabled_prev != enabled_now:
                self._camera_enabled[camera_index] = enabled_now
                if enabled_now == 0:
                    gui_disable_deadline = time.time() + VISION_FIRST_FRAME_TIMEOUT_S
                    print(
                        f"[vision] camera {camera_index} disabled in UI; "
                        f"terminating thread in {VISION_FIRST_FRAME_TIMEOUT_S:.1f}s"
                    )
                else:
                    gui_disable_deadline = None

            if enabled_now == 0 and gui_disable_deadline is not None and time.time() >= gui_disable_deadline:
                print(
                    f"[vision] camera {camera_index} remained disabled for "
                    f"{VISION_FIRST_FRAME_TIMEOUT_S:.1f}s; terminating camera thread"
                )
                with self._lock:
                    self._camera_enabled[camera_index] = 0
                    if camera_index in self._buffers:
                        self._buffers[camera_index]["enabled"] = 0
                        self._buffers[camera_index]["targets"] = {}
                break

            spec_idx_max = max(0, len(self._spec_choices) - 1)
            selected_spec_idx = self._safe_get_trackbar(camera_index, "spec_idx", 0)
            if selected_spec_idx > spec_idx_max:
                selected_spec_idx = spec_idx_max
                cv2.setTrackbarPos("spec_idx", self._window_name(camera_index), selected_spec_idx)
            if self._camera_spec_selection.get(camera_index) != selected_spec_idx:
                self._camera_spec_selection[camera_index] = selected_spec_idx
                self._settings_dirty = True
            else:
                self._camera_spec_selection[camera_index] = selected_spec_idx

            spec_key = self._active_spec_key(camera_index)
            spec = CAMERA_TARGET_SPECS.get(spec_key)
            axis_pair = tuple(spec.get("axis_pair", ("x", "y"))) if spec is not None else ("x", "y")
            self._camera_plane_calibration[camera_index] = _sanitize_plane_calibration(
                self._camera_plane_calibration.get(camera_index, {}),
                axis_pair,
            )
            calibration_before = dict(self._camera_plane_calibration[camera_index])
            y_offset_cm = self._safe_get_trackbar(camera_index, "y_offset_cm", int(round(float(calibration_before.get("y_offset_m", 0.36)) * 100.0)))
            calibration_after = dict(calibration_before)
            calibration_after["y_offset_m"] = max(0.0, min(1.0, float(y_offset_cm) / 100.0))
            axis_pair_second = str(axis_pair[1]).lower() if len(axis_pair) >= 2 else "y"
            if axis_pair_second == "z":
                z_offset_cm = self._safe_get_trackbar(camera_index, "z_offset_cm", int(round(float(calibration_before.get("z_offset_m", 0.06)) * 100.0)))
                calibration_after["z_offset_m"] = max(0.0, min(1.0, float(z_offset_cm) / 100.0))
            if calibration_after != calibration_before:
                self._camera_plane_calibration[camera_index] = calibration_after
                self._settings_dirty = True
            if self._last_spec_for_camera.get(camera_index) != spec_key:
                self._last_spec_for_camera[camera_index] = spec_key
                print(f"[vision] camera {camera_index} now using target spec {spec_key if spec_key is not None else 'none'}")

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            overlay_frame = frame.copy()

            target_pairs = list(self._iter_target_pairs_for_spec(spec)) if (enabled_now and spec is not None) else []
            target_idx_max = max(0, len(target_pairs) - 1)
            selected_target_idx = self._safe_get_trackbar(camera_index, "target_idx", 0)
            if selected_target_idx > target_idx_max:
                selected_target_idx = target_idx_max
                cv2.setTrackbarPos("target_idx", self._window_name(camera_index), selected_target_idx)

            selected_mask = None
            selected_opened = None
            selected_closed = None
            selected_target_name = ""

            if target_pairs:
                _, selected_target_name = target_pairs[selected_target_idx]
                selected_item = self._item_settings[selected_target_name]
                if selected_target_name != last_selected_target_name:
                    self._sync_trackbars_from_item(camera_index, selected_item)
                    last_selected_target_name = selected_target_name
                    self._trackbar_seeded[camera_index] = False
                elif not self._trackbar_seeded.get(camera_index, False):
                    self._trackbar_seeded[camera_index] = True
                elif self._read_item_settings_from_trackbars(camera_index, selected_item):
                    self._settings_dirty = True

            targets = {}
            now_ts = time.time()
            for pair_idx, (_color_name, target_name) in enumerate(target_pairs):
                item = self._item_settings[target_name]
                low = np.array([item["h_lo"], item["s_lo"], item["v_lo"]], dtype=np.uint8)
                high = np.array([item["h_hi"], item["s_hi"], item["v_hi"]], dtype=np.uint8)
                mask = cv2.inRange(hsv, low, high)

                kernel_size = int(item["kernel"])
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(item["open_iter"]))
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=int(item["close_iter"]))

                contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                detections = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < float(item["min_area"]):
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append(
                        {
                            "xc": x + (w // 2),
                            "yc": y + (h // 2),
                            "area": float(area),
                            "bbox": (x, y, w, h),
                        }
                    )

                detections.sort(key=lambda d: d["area"], reverse=True)
                if detections:
                    det = detections[0]
                    point = _camera_point_from_pixel(det["xc"], det["yc"], spec["axis_pair"])
                    point = self._apply_plane_calibration(camera_index, axis_pair, point, target_name)
                    point.update(
                        {
                            "camera_index": camera_index,
                            "timestamp": now_ts,
                            "axis_pair": spec["axis_pair"],
                            "spec_key": spec_key,
                            "target_name": target_name,
                        }
                    )
                    targets[target_name] = point

                    x, y, w, h = det["bbox"]
                    cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        overlay_frame,
                        target_name,
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                if pair_idx == selected_target_idx:
                    selected_mask = mask
                    selected_opened = opened
                    selected_closed = closed

            self._render_window(
                camera_index,
                frame,
                overlay_frame,
                selected_mask,
                selected_opened,
                selected_closed,
                selected_target_name,
                spec_key,
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("v"):
                self._handle_view_key(camera_index)

            with self._lock:
                self._buffers[camera_index]["timestamp"] = now_ts
                self._buffers[camera_index]["axis_pair"] = tuple(spec.get("axis_pair", ("x", "y"))) if spec is not None else ("x", "y")
                self._buffers[camera_index]["spec_key"] = spec_key
                self._buffers[camera_index]["enabled"] = int(enabled_now)
                self._buffers[camera_index]["targets"] = targets
                self._persist_settings_if_dirty(force=False)

        cap.release()
        with self._lock:
            self._persist_settings_if_dirty(force=True)
        try:
            cv2.destroyWindow(self._window_name(camera_index))
        except Exception:
            pass
        with self._lock:
            self._threads.pop(camera_index, None)

    def _apply_plane_calibration(self, camera_index, axis_pair, point, target_name=""):
        calibration = _sanitize_plane_calibration(
            self._camera_plane_calibration.get(camera_index, {}),
            axis_pair,
        )
        second = str(axis_pair[1]).lower() if len(axis_pair) >= 2 else "y"

        x_raw = float(point.get("x", 0.0))
        second_raw = float(point.get(second, 0.0))

        if second == "z":
            x_out = (
                calibration["x_scale"] * x_raw
                + calibration.get("zx_cross", 0.0) * second_raw
                + calibration["x_offset_m"]
            )
            z_out = (
                calibration["z_scale"] * second_raw
                + calibration.get("xz_cross", 0.0) * x_raw
                + calibration["z_offset_m"]
            )
            target_z_offset_m = calibration.get("target_z_offset_m", {})
            if isinstance(target_z_offset_m, dict):
                z_out += float(target_z_offset_m.get(str(target_name), 0.0))
            point["x"] = float(x_out)
            point["y"] = float(calibration.get("y_offset_m", 0.36))
            point["z"] = float(z_out)
        else:
            x_out = (
                calibration["x_scale"] * x_raw
                + calibration.get("yx_cross", 0.0) * second_raw
                + calibration["x_offset_m"]
            )
            y_out = (
                calibration["y_scale"] * second_raw
                + calibration.get("xy_cross", 0.0) * x_raw
                + calibration["y_offset_m"]
            )
            point["x"] = float(x_out)
            point["y"] = float(y_out)

        point["calibration"] = dict(calibration)
        return point

    def _open_capture(self, camera_index):
        # On Windows laptops, built-in webcams often behave better with DirectShow.
        if os.name == "nt":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(camera_index)

    def stop(self):
        self._stop.set()
        with self._lock:
            self._persist_settings_if_dirty(force=True)

    def snapshot(self):
        with self._lock:
            return {
                idx: {
                    "timestamp": buf["timestamp"],
                    "axis_pair": tuple(buf["axis_pair"]),
                    "targets": {k: dict(v) for k, v in buf["targets"].items()},
                }
                for idx, buf in self._buffers.items()
            }

    def get_target(self, target_label):
        primary = _resolve_primary_target_label(target_label)
        if not primary:
            return None
        with self._lock:
            newest = None
            for _, buf in self._buffers.items():
                point = buf["targets"].get(primary)
                if point is None:
                    continue
                if newest is None or float(point.get("timestamp", 0.0)) > float(newest.get("timestamp", 0.0)):
                    newest = dict(point)
            return newest


_VISION_FEEDS = None
_VISION_FEEDS_LOCK = threading.Lock()


def _resolve_vision_specs_path(params):
    params = params or {}
    explicit = str(params.get("vision_specs_file", "") or "").strip()
    if explicit:
        return Path(explicit)

    task_graph = str(
        params.get("task_graph_file")
        or params.get("task_graph_path")
        or ""
    ).strip()
    if not task_graph:
        task_graph = "UR5/master_task_graph.json"

    graph_path = Path(task_graph)
    return graph_path.with_name("vision_tuned_specs.json")


def _get_or_start_vision_feeds(params=None):
    global _VISION_FEEDS
    with _VISION_FEEDS_LOCK:
        params = params or {}
        settings_path = _resolve_vision_specs_path(params)
        max_scan_index = int(params.get("vision_camera_scan_max_index", VISION_CAMERA_SCAN_MAX_INDEX))
        excluded_raw = params.get("vision_excluded_camera_indices", None)
        if excluded_raw is None:
            excluded = set(VISION_EXCLUDED_CAMERA_INDICES)
        elif isinstance(excluded_raw, (list, tuple, set)):
            excluded = {int(v) for v in excluded_raw}
        else:
            excluded = {
                int(v.strip())
                for v in str(excluded_raw).split(",")
                if str(v).strip() != ""
            }

        should_recreate = _VISION_FEEDS is None
        if not should_recreate:
            try:
                should_recreate = bool(_VISION_FEEDS._stop.is_set())
            except Exception:
                should_recreate = True

        if not should_recreate:
            try:
                current_excluded = set(getattr(_VISION_FEEDS, "_excluded_camera_indices", set()))
                if current_excluded != excluded:
                    should_recreate = True
                else:
                    discovered_now = _VISION_FEEDS._discover_available_camera_indices(max_scan_index)
                    discovered_now = [idx for idx in discovered_now if idx not in excluded]
                    current_indices = list(getattr(_VISION_FEEDS, "_available_camera_indices", []))
                    if discovered_now != current_indices:
                        should_recreate = True
            except Exception:
                should_recreate = True

        if should_recreate:
            if _VISION_FEEDS is not None:
                try:
                    _VISION_FEEDS.stop()
                except Exception:
                    pass
            _VISION_FEEDS = _DualCameraVisionFeeds(
                max_camera_index=max_scan_index,
                excluded_camera_indices=excluded,
            )
        _VISION_FEEDS.configure_persistence(settings_path)
        # Always ensure all available camera threads are started on each call.
        _VISION_FEEDS.start()
        return _VISION_FEEDS


def _load_joint_trace(csv_path: Path):
    records = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ts = float(row["timestamp"])
                joints = [float(row[f"actual_q_{i}"]) for i in range(6)]
            except (KeyError, ValueError, TypeError):
                continue
            records.append((ts, joints))
    return records


def _estimate_trace_hz(records):
    if len(records) < 2:
        return None

    deltas = []
    prev_ts = records[0][0]
    for ts, _ in records[1:]:
        dt = float(ts) - float(prev_ts)
        if dt > 0:
            deltas.append(dt)
        prev_ts = ts

    if not deltas:
        return None

    # Median is robust to occasional timing jitter in recorded traces.
    deltas.sort()
    mid = len(deltas) // 2
    median_dt = deltas[mid] if len(deltas) % 2 == 1 else 0.5 * (deltas[mid - 1] + deltas[mid])
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def _decimate_records(records, target_hz):
    source_hz = _estimate_trace_hz(records)
    if source_hz is None or target_hz <= 0:
        return records, 1, source_hz

    if target_hz >= source_hz:
        return records, 1, source_hz

    step = max(1, int(math.floor(source_hz / target_hz)))
    return records[::step], step, source_hz


def _load_named_waypoints(csv_path: Path, task_id: str = "", arm_prefix: str = "right"):
    def _task_id_aliases(raw_task_id):
        tid = str(raw_task_id or "").strip().lower()
        if not tid:
            return {""}
        aliases = {
            tid,
        }
        legacy_groups = [
            {"open_microwave_door", "door_open"},
            {"close_microwave_door", "close_door"},
            {"press_microwave_stop", "press_stop"},
            {"place_bowl_in_microwave", "put_bowl"},
            {"take_bowl_out_to_tray", "bowl_to_tray"},
            {"place_plate_in_microwave", "put_plate"},
            {"take_plate_out_to_tray", "plate_to_tray"},
            {"place_cup_on_tray", "cup_on_tray"},
        ]
        for group in legacy_groups:
            if tid in group:
                aliases.update(group)
        return aliases

    def _try_float(row, key):
        raw = row.get(key)
        if raw is None:
            return None
        txt = str(raw).strip()
        if txt == "" or txt.lower() == "nothing":
            return None
        try:
            return float(txt)
        except (ValueError, TypeError):
            return None

    def _extract_q_position(row):
        candidate_sets = [
            [f"{arm_prefix}_q_{i}" for i in range(6)],
            [f"q_position_{i}" for i in range(6)],
            [f"actual_q_{i}" for i in range(6)],
            [f"q_{i}" for i in range(6)],
        ]
        for keys in candidate_sets:
            values = [_try_float(row, key) for key in keys]
            if all(v is not None for v in values):
                return values
        return None

    def _try_bool(row, key):
        raw = row.get(key)
        if raw is None:
            return None
        txt = str(raw).strip().lower()
        if txt == "" or txt == "nothing":
            return None
        if txt in {"true", "1", "yes", "y", "open"}:
            return True
        if txt in {"false", "0", "no", "n", "closed", "close"}:
            return False
        return None

    def _try_pct(row, key):
        val = _try_float(row, key)
        if val is None:
            return None
        return max(0.0, min(100.0, float(val)))

    allowed_task_ids = _task_id_aliases(task_id)
    waypoints = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_task_id = str(row.get("task_id", "")).strip()
            if task_id and row_task_id and str(row_task_id).strip().lower() not in allowed_task_ids:
                continue

            try:
                idx = int(float(row.get("waypoint_index", "0")))
            except (ValueError, TypeError):
                continue

            tcp_position = [
                _try_float(row, f"{arm_prefix}_x"),
                _try_float(row, f"{arm_prefix}_y"),
                _try_float(row, f"{arm_prefix}_z"),
                _try_float(row, f"{arm_prefix}_rx"),
                _try_float(row, f"{arm_prefix}_ry"),
                _try_float(row, f"{arm_prefix}_rz"),
            ]
            if not all(v is not None for v in tcp_position):
                tcp_position = None

            q_position = _extract_q_position(row)

            if tcp_position is None and q_position is None:
                continue

            tracked_items = {}
            tracked_items_raw = str(row.get("tracked_items_json", "")).strip()
            if tracked_items_raw:
                try:
                    parsed = json.loads(tracked_items_raw)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if not isinstance(item, dict):
                                continue
                            label = _resolve_primary_target_label(item.get("label", ""))
                            pos = item.get("position", None)
                            if not label or not isinstance(pos, list) or len(pos) < 2:
                                continue
                            tracked_items[label] = {
                                "x": float(pos[0]),
                                "y": float(pos[1]),
                                "z": float(pos[2]) if len(pos) >= 3 else None,
                            }
                except Exception:
                    pass

            gripper_open_bool = _try_bool(row, f"{arm_prefix}_gripper_open")
            gripper_open_pct = _try_pct(row, f"{arm_prefix}_gripper_open_pct")
            if gripper_open_pct is None:
                if gripper_open_bool is True:
                    gripper_open_pct = 100.0
                elif gripper_open_bool is False:
                    gripper_open_pct = 0.0

            waypoints.append(
                {
                    "index": idx,
                    "name": str(row.get("waypoint_name", "")).strip(),
                    "tcp_position": tcp_position,
                    "q_position": q_position,
                    "gripper_open": gripper_open_bool,
                    "gripper_open_pct": gripper_open_pct,
                    "tracked_items": tracked_items,
                }
            )

    waypoints.sort(key=lambda item: item["index"])
    return waypoints


def register_subtasks(registry):
    """Register simple team-editable subtasks."""

    def _gripper_disabled(supervisor, params):
        if bool(params.get("no_gripper", False)):
            return True
        if bool(getattr(supervisor, "_no_gripper", False)):
            return True
        return False

    def _call_with_timeout(task_name, label, timeout_s, fn, require_truthy_result=False):
        state = {"ok": False, "exc": None, "result": None}

        def _worker():
            try:
                state["result"] = fn()
                state["ok"] = True
            except Exception as exc:
                state["exc"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=max(0.01, float(timeout_s)))
        if t.is_alive():
            print(f"[{task_name}] Warning: {label} timed out after {float(timeout_s):.2f}s")
            return False
        if state["exc"] is not None:
            print(f"[{task_name}] Warning: {label} failed: {state['exc']}")
            return False
        if require_truthy_result and not bool(state["result"]):
            print(f"[{task_name}] Warning: {label} returned failure")
            return False
        return True

    def _init_gripper_best_effort(task_name, rtde_control, force, speed, timeout_s):
        if RobotiqGripper is None:
            print(f"[{task_name}] robotiq_gripper_control not available; gripper waypoint safeguards disabled")
            return None

        holder = {"gripper": None}

        def _setup():
            g = RobotiqGripper(rtde_control)
            g.set_force(int(force))
            g.set_speed(int(speed))
            holder["gripper"] = g

        ok = _call_with_timeout(
            task_name,
            label="gripper initialization",
            timeout_s=timeout_s,
            fn=_setup,
        )
        if not ok:
            return None

        print(f"[{task_name}] Gripper controller ready (force={int(force)}, speed={int(speed)})")
        return holder["gripper"]

    def _connect_arm_with_timeout(task_name, arm_ip, timeout_s):
        holder = {"arm": None}

        def _connect():
            holder["arm"] = UR5Arm(arm_ip, verbose=False)

        print(f"[{task_name}] Connecting to arm {arm_ip} (timeout={float(timeout_s):.2f}s)")
        ok = _call_with_timeout(
            task_name,
            label=f"arm connect {arm_ip}",
            timeout_s=timeout_s,
            fn=_connect,
        )
        if not ok or holder["arm"] is None:
            raise RuntimeError(f"Failed to connect to arm {arm_ip} within timeout")
        return holder["arm"]

    def _execute_waypoint_motion(
        task_name,
        arm,
        wp,
        speed,
        acceleration,
        motion_timeout_s,
        prefer_tcp=True,
        use_linear=False,
    ):
        tcp_position = wp.get("tcp_position")
        q_position = wp.get("q_position")

        if prefer_tcp and tcp_position is not None:
            motion_label = (
                f"move_linear_to_pose wp#{wp['index']}"
                if use_linear
                else f"move_to_pose wp#{wp['index']}"
            )
            if use_linear:
                motion_fn = lambda: arm.move_linear_to_pose(
                    tcp_position,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                )
            else:
                motion_fn = lambda: arm.move_to_pose(
                    tcp_position,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                )
            ok_motion = _call_with_timeout(
                task_name,
                label=motion_label,
                timeout_s=motion_timeout_s,
                fn=motion_fn,
                require_truthy_result=True,
            )
        elif q_position is not None:
            motion_label = f"move_to_joint_position wp#{wp['index']}"
            ok_motion = _call_with_timeout(
                task_name,
                label=motion_label,
                timeout_s=motion_timeout_s,
                fn=lambda: arm.move_to_joint_position(
                    q_position,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                ),
                require_truthy_result=True,
            )
        elif tcp_position is not None:
            # Fallback to TCP-space when q is missing.
            motion_label = f"move_to_pose wp#{wp['index']}"
            ok_motion = _call_with_timeout(
                task_name,
                label=motion_label,
                timeout_s=motion_timeout_s,
                fn=lambda: arm.move_to_pose(
                    tcp_position,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                ),
                require_truthy_result=True,
            )
        else:
            raise RuntimeError(
                f"Waypoint missing both tcp_position and q_position at index={wp['index']}"
            )

        if not ok_motion:
            try:
                arm.stop_arm(deceleration=10.0, asynchronous=True, use_linear=False)
            except Exception:
                pass
            return False
        return True

    def _example(supervisor, params):
        print("[example_subtask] params:", params)
        frames = supervisor.compute_task_frames()
        print("[example_subtask] task frames:", frames)
        return True

    registry["example"] = {
        "description": "Minimal collaborative subtask template",
        "runner": _example,
    }

    def _run_named_waypoint_task(
        supervisor,
        params,
        *,
        task_name,
        default_csv,
        default_task_id,
        arm_side,
    ):
        waypoints_csv = Path(params.get("named_waypoints_csv", default_csv))
        task_id = str(params.get("task_id", default_task_id))
        speed = params.get("speed", None)
        acceleration = params.get("acceleration", None)
        arm_connect_timeout_s = float(params.get("arm_connect_timeout_s", 5.0))
        motion_timeout_s = float(params.get("motion_timeout_s", 20.0))

        if not waypoints_csv.exists():
            raise FileNotFoundError(f"Named waypoints CSV not found: {waypoints_csv}")

        waypoints = _load_named_waypoints(waypoints_csv, task_id=task_id, arm_prefix=arm_side)
        if not waypoints:
            raise RuntimeError(
                f"No valid {arm_side}-arm waypoints found in {waypoints_csv} for task_id={task_id}"
            )

        arm_ip = supervisor.right_ip if arm_side == "right" else supervisor.left_ip
        arm = None
        gripper = None
        gripper_force = int(params.get("gripper_force", 100))
        gripper_speed = int(params.get("gripper_speed", 100))
        gripper_settle_s = float(params.get("gripper_settle_s", 0.15))
        gripper_call_timeout_s = float(params.get("gripper_call_timeout_s", 1.0))

        try:
            arm = _connect_arm_with_timeout(task_name, arm_ip, timeout_s=arm_connect_timeout_s)
            if _gripper_disabled(supervisor, params):
                print(f"[{task_name}] Gripper disabled by runtime flag")
            else:
                gripper = _init_gripper_best_effort(
                    task_name,
                    arm.rtde_control,
                    force=gripper_force,
                    speed=gripper_speed,
                    timeout_s=gripper_call_timeout_s,
                )
            print(f"[{task_name}] Replaying {len(waypoints)} waypoints from {waypoints_csv}")
            for wp in waypoints:
                target_gripper_open_pct = wp.get("gripper_open_pct", None)
                ok = _execute_waypoint_motion(
                    task_name,
                    arm,
                    wp,
                    speed=speed,
                    acceleration=acceleration,
                    motion_timeout_s=motion_timeout_s,
                    prefer_tcp=True,
                    use_linear=False,
                )
                if not ok:
                    raise RuntimeError(
                        f"Failed at waypoint index={wp['index']} name={wp['name'] or '<unnamed>'}"
                    )

                # Team policy: assert desired gripper state after each motion step.
                if gripper is not None and target_gripper_open_pct is not None:
                    # Gripper open percentage maps to 0..85mm.
                    pos_mm = float(target_gripper_open_pct) * GRIPPER_OPEN_MM_MAX / 100.0
                    ok_gripper = _call_with_timeout(
                        task_name,
                        label=f"gripper move at waypoint {wp['index']}",
                        timeout_s=gripper_call_timeout_s,
                        fn=lambda: gripper.move(int(round(pos_mm))),
                    )
                    if not ok_gripper:
                        gripper = None
                    if gripper_settle_s > 0:
                        time.sleep(gripper_settle_s)

            arm.stop_arm(use_linear=False)
            print(f"[{task_name}] Waypoint replay complete")
            return True
        except Exception as exc:
            print(f"[{task_name}] Waypoint replay aborted: {exc}")
            return False
        finally:
            try:
                if arm is not None:
                    arm.disconnect()
            except Exception:
                pass

    def _run_named_waypoint_task_camera_aware(
        supervisor,
        params,
        *,
        task_name,
        default_csv,
        default_task_id,
        arm_side,
    ):
        waypoints_csv = Path(params.get("named_waypoints_csv", default_csv))
        task_id = str(params.get("task_id", default_task_id))
        speed = params.get("speed", None)
        acceleration = params.get("acceleration", None)
        arm_connect_timeout_s = float(params.get("arm_connect_timeout_s", 5.0))
        motion_timeout_s = float(params.get("motion_timeout_s", 20.0))
        gripper_force = int(params.get("gripper_force", 100))
        gripper_speed = int(params.get("gripper_speed", 100))
        gripper_settle_s = float(params.get("gripper_settle_s", 0.15))
        gripper_call_timeout_s = float(params.get("gripper_call_timeout_s", 1.0))
        offset_gain = float(params.get("vision_offset_gain", 1.0))
        use_camera_offsets = bool(params.get("use_camera_offsets", True))
        no_camera = bool(params.get("no_camera", False)) or bool(getattr(supervisor, "_no_camera", False))
        if no_camera:
            use_camera_offsets = False

        if not waypoints_csv.exists():
            raise FileNotFoundError(f"Named waypoints CSV not found: {waypoints_csv}")

        waypoints = _load_named_waypoints(waypoints_csv, task_id=task_id, arm_prefix=arm_side)
        if not waypoints:
            raise RuntimeError(
                f"No valid {arm_side}-arm waypoints found in {waypoints_csv} for task_id={task_id}"
            )

        raw_target_label = (
            params.get("vision_target_label")
            or params.get("dependent_item_label")
            or params.get("target_label")
            or params.get("object_label")
            or params.get("source_label")
            or TASK_DEFAULT_TARGETS.get(task_name, "")
        )
        target_label = _resolve_primary_target_label(raw_target_label)

        vision_feeds = _get_or_start_vision_feeds(params=params) if use_camera_offsets else None
        if no_camera:
            print(f"[{task_name}] Camera disabled by runtime flag")
        if use_camera_offsets and not target_label:
            print(f"[{task_name}] Vision offsets enabled but no recognizable target label was provided")

        arm_ip = supervisor.right_ip if arm_side == "right" else supervisor.left_ip
        arm = None
        gripper = None

        try:
            arm = _connect_arm_with_timeout(task_name, arm_ip, timeout_s=arm_connect_timeout_s)
            if _gripper_disabled(supervisor, params):
                print(f"[{task_name}] Gripper disabled by runtime flag")
            else:
                gripper = _init_gripper_best_effort(
                    task_name,
                    arm.rtde_control,
                    force=gripper_force,
                    speed=gripper_speed,
                    timeout_s=gripper_call_timeout_s,
                )
            print(f"[{task_name}] Replaying {len(waypoints)} waypoints from {waypoints_csv}")
            if target_label:
                print(f"[{task_name}] Camera offset target: {target_label}")

            is_press_stop_task = str(task_name).strip() == "press_microwave_stop" or str(task_id).strip() == "press_stop"

            for wp in waypoints:
                target_gripper_open_pct = wp.get("gripper_open_pct", None)
                move_wp = dict(wp)
                move_wp["tcp_position"] = list(wp["tcp_position"]) if wp.get("tcp_position") else None

                if use_camera_offsets and vision_feeds is not None and target_label and move_wp.get("tcp_position"):
                    tracked = wp.get("tracked_items", {})
                    ref = tracked.get(target_label)
                    live = vision_feeds.get_target(target_label)

                    if ref is not None and live is not None:
                        secondary_axis = str(live.get("axis_pair", ("x", "y"))[1])
                        if secondary_axis in ref and secondary_axis in live:
                            dx = (float(live["x"]) - float(ref["x"])) * offset_gain
                            d_secondary = (float(live[secondary_axis]) - float(ref[secondary_axis])) * offset_gain
                            move_wp["tcp_position"][0] = float(move_wp["tcp_position"][0]) + dx
                            # We apply secondary camera axis to tool Y correction.
                            move_wp["tcp_position"][1] = float(move_wp["tcp_position"][1]) + d_secondary
                            print(
                                f"[{task_name}] wp#{wp['index']} offset applied: "
                                f"dx={dx:+.4f}, d{secondary_axis}={d_secondary:+.4f}"
                            )

                ok = _execute_waypoint_motion(
                    task_name,
                    arm,
                    move_wp,
                    speed=speed,
                    acceleration=acceleration,
                    motion_timeout_s=motion_timeout_s,
                    prefer_tcp=True if is_press_stop_task else bool(use_camera_offsets and move_wp.get("tcp_position") is not None),
                    use_linear=True if is_press_stop_task else False,
                )

                if not ok:
                    raise RuntimeError(
                        f"Failed at waypoint index={wp['index']} name={wp['name'] or '<unnamed>'}"
                    )

                # Team policy: assert desired gripper state after each motion step.
                if gripper is not None and target_gripper_open_pct is not None:
                    # Gripper open percentage maps to 0..85mm.
                    pos_mm = float(target_gripper_open_pct) * GRIPPER_OPEN_MM_MAX / 100.0
                    ok_gripper = _call_with_timeout(
                        task_name,
                        label=f"gripper move at waypoint {wp['index']}",
                        timeout_s=gripper_call_timeout_s,
                        fn=lambda: gripper.move(int(round(pos_mm))),
                    )
                    if not ok_gripper:
                        gripper = None
                    if gripper_settle_s > 0:
                        time.sleep(gripper_settle_s)

            arm.stop_arm(use_linear=False)
            print(f"[{task_name}] Camera-aware waypoint replay complete")
            return True
        except Exception as exc:
            print(f"[{task_name}] Camera-aware waypoint replay aborted: {exc}")
            return False
        finally:
            try:
                if arm is not None:
                    arm.disconnect()
            except Exception:
                pass

    def _register_stub_task(task_name, arm_side="right", default_csv=None):
        """Register a lightweight runner for graph tasks not fully implemented yet.

        Behavior:
        - If a matching waypoint CSV exists (or is supplied via params), replay it.
        - Otherwise, log a no-op stub execution and return success.
        """

        def _runner(supervisor, params):
            params = params or {}
            resolved_default_csv = default_csv or f"UR5/waypoints_{task_name}.csv"
            csv_path = Path(params.get("named_waypoints_csv", resolved_default_csv))

            # Allow explicit task_id override while using task_name as default.
            task_id = str(params.get("task_id", task_name))

            if csv_path.exists():
                print(f"[{task_name}] Using waypoint replay from {csv_path}")
                return _run_named_waypoint_task_camera_aware(
                    supervisor,
                    dict(params, named_waypoints_csv=str(csv_path), task_id=task_id),
                    task_name=task_name,
                    default_csv=str(csv_path),
                    default_task_id=task_id,
                    arm_side=str(params.get("arm_side", arm_side)),
                )

            print(
                f"[{task_name}] STUB runner: no waypoint CSV found at {csv_path}. "
                "Returning success (no-op)."
            )
            return True

        registry[task_name] = {
            "description": f"Stub runner for {task_name}; replays waypoints if CSV exists, else no-op success",
            "runner": _runner,
        }

    # Task names observed in simulator output (UR5/subtasks/Untitled-1.txt).
    # Includes existing acquire/open/close flows so all task runners are list-registered.
    for _task_name, _arm_side, _default_csv in [
        ("acquire_bowl", "right", "UR5/waypoints_acquire_bowl.csv"),
        ("open_microwave_door", "left", "UR5/waypoints_door_open_for_unload.csv"),
        ("close_microwave_door", "left", "UR5/waypoints_close_microwave_door.csv"),
        ("place_bowl_in_microwave", "right", "UR5/waypoints_place_bowl_in_microwave.csv"),
        ("right_arm_safe_retract", "right", "UR5/waypoints_right_arm_safe_retract.csv"),
        ("press_microwave_stop", "right", "UR5/waypoints_press_microwave_stop.csv"),
        ("take_bowl_out_to_tray", "right", "UR5/waypoints_take_bowl_out_to_tray.csv"),
        ("acquire_plate", "right", "UR5/waypoints_acquire_plate.csv"),
        ("place_plate_in_microwave", "right", "UR5/waypoints_place_plate_in_microwave.csv"),
        ("take_plate_out_to_tray", "right", "UR5/waypoints_take_plate_out_to_tray.csv"),
        ("move_tray", "right", "UR5/waypoints_move_tray.csv"),
        ("acquire_cup", "right", "UR5/waypoints_acquire_cup.csv"),
        ("acquire_bottle", "right", "UR5/waypoints_acquire_bottle.csv"),
        ("pour_drink_into_cup", "right", "UR5/waypoints_pour_drink_into_cup.csv"),
        ("return_bottle", "left", "UR5/waypoints_return_bottle.csv"),
        ("place_cup_on_tray", "right", "UR5/waypoints_place_cup_on_tray.csv"),
        ("stir_cup", "right", "UR5/waypoints_stir_cup.csv"),
    ]:
        if _task_name not in registry:
            _register_stub_task(_task_name, arm_side=_arm_side, default_csv=_default_csv)

    def _total_replay(supervisor, params):
        """Replay full dual-arm joint traces by stepping point-by-point (default 10 Hz)."""
        left_csv = Path(params.get("left_csv", "robot_data_left.csv"))
        right_csv = Path(params.get("right_csv", "robot_data_right.csv"))
        sample_hz = float(params.get("sample_hz", 10.0))
        time_scale = float(params.get("time_scale", 1.0))
        speed = params.get("speed", None)
        acceleration = params.get("acceleration", None)

        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        if time_scale <= 0:
            raise ValueError("time_scale must be > 0")

        left_records = _load_joint_trace(left_csv)
        right_records = _load_joint_trace(right_csv)
        if not left_records or not right_records:
            raise RuntimeError("Both left/right CSV files must contain timestamp + actual_q_* data")

        left_records, left_step, left_src_hz = _decimate_records(left_records, sample_hz)
        right_records, right_step, right_src_hz = _decimate_records(right_records, sample_hz)

        if left_step > 1 or right_step > 1:
            left_src_str = f"{left_src_hz:.2f}" if left_src_hz else "unknown"
            right_src_str = f"{right_src_hz:.2f}" if right_src_hz else "unknown"
            print(
                "[total_replay] Decimated traces for sample_hz "
                f"{sample_hz:.2f}: left step={left_step} (src_hz={left_src_str}), "
                f"right step={right_step} (src_hz={right_src_str})"
            )

        total_points = min(len(left_records), len(right_records))
        if total_points <= 0:
            raise RuntimeError("No replay points found in overlapping left/right trace lengths")

        dt_trace = 1.0 / sample_hz
        dt_wall = dt_trace / time_scale

        left_arm = UR5Arm(supervisor.left_ip, verbose=False)
        right_arm = UR5Arm(supervisor.right_ip, verbose=False)
        try:
            print(
                "[total_replay] Starting replay "
                f"{left_csv} + {right_csv} at {sample_hz:.2f} Hz "
                f"(time_scale={time_scale:.2f}, points={total_points})"
            )

            for idx in range(total_points):
                loop_start = time.time()
                left_joints = left_records[idx][1]
                right_joints = right_records[idx][1]

                ok_l = left_arm.move_to_joint_position(
                    left_joints,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                )
                ok_r = right_arm.move_to_joint_position(
                    right_joints,
                    speed=speed,
                    acceleration=acceleration,
                    asynchronous=False,
                )
                if not ok_l or not ok_r:
                    raise RuntimeError(
                        f"Move command failed at index={idx} (left_ok={ok_l}, right_ok={ok_r})"
                    )

                sleep_for = dt_wall - (time.time() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # Ensure final state settles.
            left_arm.stop_arm(use_linear=False)
            right_arm.stop_arm(use_linear=False)
            print("[total_replay] Replay complete")
            return True
        finally:
            left_arm.disconnect()
            right_arm.disconnect()

    registry["total_replay"] = {
        "description": "Replay full left/right CSV joint traces point-by-point at requested sample rate (default 10 Hz)",
        "runner": _total_replay,
    }
