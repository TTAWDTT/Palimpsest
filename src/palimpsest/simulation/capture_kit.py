"""Deterministic screen stimuli and provenance checks for controlled recapture.

These are *digital inputs* to an experiment, not a model of emitted light. A
compositor screenshot checks the submitted frame only; it cannot measure panel
luminance, subpixel shape, PWM, or camera exposure.
"""

from __future__ import annotations

from palimpsest.io.hashing import file_sha256 as sha256

import argparse
import csv
import json
import math
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "screen-capture-kit-v1"
LOG_FIELDS = [
    "capture_id",
    "pattern_id",
    "condition_id",
    "repeat_index",
    "captured_at_iso",
    "distance_mm",
    "yaw_deg",
    "pitch_deg",
    "focus_setting",
    "exposure_s",
    "iso",
    "aperture_f",
    "white_balance",
    "display_brightness_setting",
    "jpeg_path",
    "jpeg_sha256",
    "raw_path",
    "raw_sha256",
    "notes",
]


def save_json(path: Path, value: dict) -> None:
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot encode {type(value).__name__}")

    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=scalar)
        + "\n"
    ).encode("utf-8")
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f"existing file differs: {path}")
    if not path.exists():
        path.write_bytes(data)


def _flat(width: int, height: int, value: int) -> tuple[np.ndarray, list]:
    return np.full((height, width, 3), value, dtype=np.uint8), [
        {
            "name": "central_flat",
            "xyxy": [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
        }
    ]


def _geometry(width: int, height: int) -> tuple[np.ndarray, list]:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    x0, x1 = round(width * 0.1), round(width * 0.9)
    y0, y1 = round(height * 0.1), round(height * 0.9)
    xs = np.linspace(x0, x1, 13, dtype=int)
    ys = np.linspace(y0, y1, 9, dtype=int)
    for row in range(8):
        for column in range(12):
            image[ys[row] : ys[row + 1], xs[column] : xs[column + 1]] = (
                224 if (row + column) % 2 else 32
            )
    marker_size = max(12, min(width, height) // 30)
    positions = [
        (round(width * 0.025), round(height * 0.025), (255, 0, 0)),
        (
            width - round(width * 0.025) - marker_size,
            round(height * 0.025),
            (0, 255, 0),
        ),
        (
            width - round(width * 0.025) - marker_size,
            height - round(height * 0.025) - marker_size,
            (0, 0, 255),
        ),
        (
            round(width * 0.025),
            height - round(height * 0.025) - marker_size,
            (255, 255, 0),
        ),
    ]
    for x, y, color in positions:
        image[y : y + marker_size, x : x + marker_size] = color
    return image, [{"name": "checkerboard", "xyxy": [x0, y0, x1, y1], "tiles": [12, 8]}]


def _color_chart(width: int, height: int) -> tuple[np.ndarray, list]:
    colors = [
        (16, 16, 16),
        (64, 64, 64),
        (128, 128, 128),
        (224, 224, 224),
        (224, 32, 32),
        (32, 224, 32),
        (32, 32, 224),
        (224, 224, 32),
        (224, 32, 224),
        (32, 224, 224),
        (224, 128, 32),
        (32, 128, 224),
        (128, 32, 224),
        (128, 224, 32),
        (224, 128, 128),
        (128, 224, 224),
    ]
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    xs = np.linspace(round(width * 0.06), round(width * 0.94), 5, dtype=int)
    ys = np.linspace(round(height * 0.06), round(height * 0.94), 5, dtype=int)
    rois = []
    for i, color in enumerate(colors):
        row, col = divmod(i, 4)
        x0, x1 = xs[col], xs[col + 1]
        y0, y1 = ys[row], ys[row + 1]
        margin = max(3, min(x1 - x0, y1 - y0) // 12)
        image[y0 + margin : y1 - margin, x0 + margin : x1 - margin] = color
        rois.append(
            {
                "name": f"patch_{i:02d}",
                "xyxy": [
                    x0 + 2 * margin,
                    y0 + 2 * margin,
                    x1 - 2 * margin,
                    y1 - 2 * margin,
                ],
                "rgb_8bit": list(color),
            }
        )
    return image, rois


def _slanted_edges(width: int, height: int) -> tuple[np.ndarray, list]:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    rois = []
    xs = np.linspace(round(width * 0.08), round(width * 0.92), 3, dtype=int)
    ys = np.linspace(round(height * 0.08), round(height * 0.92), 3, dtype=int)
    tangent = math.tan(math.radians(5))
    for row in range(2):
        for col in range(2):
            x0, x1, y0, y1 = xs[col], xs[col + 1], ys[row], ys[row + 1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if row == 0:
                bright = xx - cx > (1 if col else -1) + tangent * (yy - cy)
                orientation = "near_vertical"
            else:
                bright = yy - cy > (1 if col else -1) + tangent * (xx - cx)
                orientation = "near_horizontal"
            image[y0:y1, x0:x1] = np.where(bright[..., None], 224, 32).astype(np.uint8)
            pad = min(x1 - x0, y1 - y0) // 10
            rois.append(
                {
                    "name": f"edge_{row}_{col}",
                    "xyxy": [x0 + pad, y0 + pad, x1 - pad, y1 - pad],
                    "orientation": orientation,
                    "slant_deg": 5,
                }
            )
    return image, rois


PERIODS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80)


def _frequencies(
    width: int, height: int, axis: str, amplitude: int
) -> tuple[np.ndarray, list]:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    xs = np.linspace(0, width, 5, dtype=int)
    ys = np.linspace(0, height, 4, dtype=int)
    rois = []
    for i, period in enumerate(PERIODS):
        row, col = divmod(i, 4)
        x0, x1, y0, y1 = xs[col], xs[col + 1], ys[row], ys[row + 1]
        coordinates = np.arange(x0, x1) if axis == "x" else np.arange(y0, y1)
        # Cosine keeps the 2-pixel Nyquist probe alternating. A zero-phase sine
        # would sample to a constant at every integer pixel for period 2.
        wave = (
            np.rint(128 + amplitude * np.cos(2 * np.pi * coordinates / period))
            .clip(0, 255)
            .astype(np.uint8)
        )
        patch = np.broadcast_to(
            wave[None, :] if axis == "x" else wave[:, None], (y1 - y0, x1 - x0)
        )
        image[y0:y1, x0:x1] = patch[..., None]
        margin = max(3, min(x1 - x0, y1 - y0) // 16)
        rois.append(
            {
                "name": f"frequency_{i:02d}",
                "xyxy": [x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                "axis": axis,
                "period_display_px": period,
                "amplitude_8bit": amplitude,
            }
        )
    return image, rois


def _rgb_pixel_stripes(width: int, height: int) -> tuple[np.ndarray, list]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x = np.arange(width)
    for channel in range(3):
        image[:, x % 3 == channel, channel] = 255
    return image, [
        {
            "name": "rgb_pixel_stripes",
            "xyxy": [0, 0, width, height],
            "period_display_px": 3,
        }
    ]


def _gray_ramp(width: int, height: int) -> tuple[np.ndarray, list]:
    values = np.rint(np.linspace(0, 255, width)).astype(np.uint8)
    image = np.broadcast_to(values[None, :, None], (height, width, 3)).copy()
    return image, [{"name": "gray_ramp", "xyxy": [0, 0, width, height]}]


def pattern_specs(width: int, height: int):
    yield (
        "geometry",
        "calibration",
        "projective geometry and corner orientation",
        _geometry(width, height),
    )
    for value in (0, 16, 64, 128, 192, 255):
        yield (
            f"flat_{value:03d}",
            "calibration",
            "flat field and temporal/noise repeat",
            _flat(width, height, value),
        )
    for channel, name in enumerate(("red", "green", "blue")):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, channel] = 255
        yield (
            f"primary_{name}",
            "calibration",
            "display/camera channel response",
            (frame, []),
        )
    yield (
        "color_chart",
        "calibration",
        "composite color response",
        _color_chart(width, height),
    )
    yield (
        "slanted_edges",
        "calibration",
        "composite edge response",
        _slanted_edges(width, height),
    )
    yield (
        "frequency_x_high",
        "calibration",
        "horizontal sampling response",
        _frequencies(width, height, "x", 90),
    )
    yield (
        "frequency_y_high",
        "holdout",
        "directional sampling check",
        _frequencies(width, height, "y", 90),
    )
    yield (
        "frequency_x_low",
        "holdout",
        "weak artifact response",
        _frequencies(width, height, "x", 16),
    )
    yield (
        "rgb_pixel_stripes",
        "holdout",
        "color phase/alias stress test",
        _rgb_pixel_stripes(width, height),
    )
    yield "gray_ramp", "holdout", "unseen tone mapping check", _gray_ramp(width, height)


def generate(output_dir: Path, width: int, height: int) -> dict:
    if width < 640 or height < 480:
        raise ValueError("frame must be at least 640x480 pixels")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    patterns = []
    for pattern_id, role, purpose, (array, rois) in pattern_specs(width, height):
        assert array.shape == (height, width, 3) and array.dtype == np.uint8
        path = frame_dir / f"{pattern_id}.png"
        staging = path.with_suffix(".png.tmp")
        Image.fromarray(array, mode="RGB").save(staging, format="PNG")
        if path.exists():
            if sha256(path) != sha256(staging):
                staging.unlink()
                raise FileExistsError(f"existing pattern differs: {path}")
            staging.unlink()
        else:
            staging.replace(path)
        patterns.append(
            {
                "pattern_id": pattern_id,
                "role": role,
                "purpose": purpose,
                "file": f"frames/{pattern_id}.png",
                "sha256": sha256(path),
                "rois": rois,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "width": width,
        "height": height,
        "color_encoding": "8-bit RGB PNG values; emitted panel radiance is unmeasured",
        "patterns": patterns,
    }
    save_json(output_dir / "pattern_manifest.json", manifest)
    session_template = {
        "session_id": "unknown",
        "display_model": "unknown",
        "display_native_width_px": "unknown",
        "display_native_height_px": "unknown",
        "display_current_mode_width_px": width,
        "display_current_mode_height_px": height,
        "display_physical_width_mm": "unknown",
        "display_physical_height_mm": "unknown",
        "os_display_scale_percent": "unknown",
        "display_brightness_setting": "unknown",
        "refresh_hz": "unknown",
        "display_color_mode": "unknown",
        "camera_model": "unknown",
        "lens_model": "unknown",
        "camera_native_jpeg": "unknown",
        "camera_raw_available": "unknown",
        "auto_exposure_locked": "unknown",
        "white_balance_locked": "unknown",
        "ambient_light_notes": "unknown",
        "screen_rect_in_compositor_xyxy": "unknown",
        "pattern_manifest_sha256": "fill with the SHA-256 of pattern_manifest.json after copying this template",
    }
    save_json(output_dir / "session_template.json", session_template)
    log_template = output_dir / "capture_log_template.csv"
    if not log_template.exists():
        with log_template.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(LOG_FIELDS)
    elif log_template.read_text(encoding="utf-8").strip() != ",".join(LOG_FIELDS):
        raise FileExistsError(f"existing log template differs: {log_template}")
    return manifest


def validate_kit(output_dir: Path) -> tuple[dict, list[str]]:
    manifest = json.loads(
        (output_dir / "pattern_manifest.json").read_text(encoding="utf-8")
    )
    errors = []
    if manifest.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    ids = [row.get("pattern_id") for row in manifest.get("patterns", [])]
    if len(ids) != len(set(ids)):
        errors.append("duplicate pattern IDs")
    for row in manifest.get("patterns", []):
        path = (output_dir / row["file"]).resolve()
        if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
            errors.append(f"missing or escaped frame: {row['pattern_id']}")
            continue
        if sha256(path) != row["sha256"]:
            errors.append(f"SHA-256 mismatch: {row['pattern_id']}")
        with Image.open(path) as frame:
            if (
                frame.size != (manifest["width"], manifest["height"])
                or frame.mode != "RGB"
            ):
                errors.append(f"size or RGB mode mismatch: {row['pattern_id']}")
    return manifest, errors


def init_session(kit_dir: Path, session_dir: Path, session_id: str) -> None:
    manifest, errors = validate_kit(kit_dir)
    if errors:
        raise RuntimeError("; ".join(errors))
    if not session_id or any(char in session_id for char in "/\\\r\n"):
        raise ValueError("session ID must be a nonempty single-line name")
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(f"session directory is not empty: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)
    template = json.loads(
        (kit_dir / "session_template.json").read_text(encoding="utf-8")
    )
    template["session_id"] = session_id
    template["pattern_manifest_sha256"] = sha256(kit_dir / "pattern_manifest.json")
    assert (
        template["display_current_mode_width_px"],
        template["display_current_mode_height_px"],
    ) == (manifest["width"], manifest["height"])
    save_json(session_dir / "session.json", template)
    shutil.copyfile(
        kit_dir / "capture_log_template.csv", session_dir / "capture_log.csv"
    )
    (session_dir / "camera_originals").mkdir()


def record_capture(kit_dir: Path, session_dir: Path, record: dict) -> None:
    """Append one immutable camera capture; hash image bytes automatically."""
    manifest, errors = validate_kit(kit_dir)
    if errors:
        raise RuntimeError("; ".join(errors))
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    if session.get("pattern_manifest_sha256") != sha256(
        kit_dir / "pattern_manifest.json"
    ):
        raise RuntimeError("session references a different pattern manifest")
    if record["pattern_id"] not in {row["pattern_id"] for row in manifest["patterns"]}:
        raise ValueError(f"unknown pattern: {record['pattern_id']}")
    if not record["capture_id"] or any(c in record["capture_id"] for c in "/\\\r\n"):
        raise ValueError("capture ID must be a nonempty single-line name")
    log = session_dir / "capture_log.csv"
    with log.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != LOG_FIELDS:
            raise ValueError("capture log header differs from template")
        rows = list(reader)
    if any(row["capture_id"] == record["capture_id"] for row in rows):
        raise ValueError(f"duplicate capture ID: {record['capture_id']}")
    row = {field: str(record.get(field, "unknown")) for field in LOG_FIELDS}
    for field, hash_field in (("jpeg_path", "jpeg_sha256"), ("raw_path", "raw_sha256")):
        if row[field] in ("", "unknown"):
            row[field] = ""
            row[hash_field] = ""
            continue
        path = (session_dir / row[field]).resolve()
        if not path.is_relative_to(session_dir.resolve()) or not path.is_file():
            raise ValueError(
                f"{field} must be an existing file inside session directory"
            )
        row[hash_field] = sha256(path)
        if field == "jpeg_path":
            with Image.open(path) as camera_jpeg:
                if camera_jpeg.format != "JPEG":
                    raise ValueError("jpeg_path is not a JPEG file")
                camera_jpeg.verify()
    if not row["jpeg_path"] and not row["raw_path"]:
        raise ValueError("at least one camera image is required")
    staging = log.with_suffix(".csv.tmp")
    with staging.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows + [row])
    staging.replace(log)


def validate_session(
    kit_dir: Path, session_dir: Path, manifest: dict
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    expected = sha256(kit_dir / "pattern_manifest.json")
    if session.get("pattern_manifest_sha256") != expected:
        errors.append("session references a different pattern manifest")
    if (
        session.get("display_current_mode_width_px"),
        session.get("display_current_mode_height_px"),
    ) != (manifest["width"], manifest["height"]):
        errors.append("session display mode differs from pattern size")
    for field in (
        "session_id",
        "display_model",
        "camera_model",
        "os_display_scale_percent",
        "display_brightness_setting",
        "refresh_hz",
        "display_color_mode",
        "auto_exposure_locked",
        "white_balance_locked",
        "display_native_width_px",
        "display_native_height_px",
        "screen_rect_in_compositor_xyxy",
    ):
        if session.get(field) in (None, "", "unknown"):
            warnings.append(f"session field unknown: {field}")
    if session.get("display_native_width_px") not in (
        None,
        "",
        "unknown",
    ) and session.get("display_native_height_px") not in (None, "", "unknown"):
        try:
            native_size = (
                int(session["display_native_width_px"]),
                int(session["display_native_height_px"]),
            )
            if native_size != (manifest["width"], manifest["height"]):
                warnings.append(
                    "display current mode differs from panel native resolution; panel scaling must be modelled"
                )
        except ValueError:
            errors.append("invalid native display resolution")
    allowed = {row["pattern_id"] for row in manifest["patterns"]}
    with (session_dir / "capture_log.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != LOG_FIELDS:
            return ["capture log header differs from template"], warnings
        seen = set()
        count = 0
        for row in reader:
            count += 1
            capture_id = row["capture_id"]
            if not capture_id or capture_id in seen:
                errors.append(f"empty/duplicate capture_id: {capture_id!r}")
            seen.add(capture_id)
            if row["pattern_id"] not in allowed:
                errors.append(f"unknown pattern ID: {row['pattern_id']}")
            for path_field, hash_field in (
                ("jpeg_path", "jpeg_sha256"),
                ("raw_path", "raw_sha256"),
            ):
                value = row[path_field]
                if not value:
                    if path_field == "jpeg_path" and not row["raw_path"]:
                        errors.append(f"capture {capture_id}: no image file")
                    continue
                path = (session_dir / value).resolve()
                if not path.is_relative_to(session_dir.resolve()) or not path.is_file():
                    errors.append(f"capture {capture_id}: missing/escaped {path_field}")
                elif sha256(path) != row[hash_field]:
                    errors.append(f"capture {capture_id}: {hash_field} mismatch")
                elif path_field == "jpeg_path":
                    try:
                        with Image.open(path) as camera_jpeg:
                            if camera_jpeg.format != "JPEG":
                                errors.append(
                                    f"capture {capture_id}: JPEG field is {camera_jpeg.format}"
                                )
                            camera_jpeg.verify()
                    except Exception as exc:
                        errors.append(
                            f"capture {capture_id}: JPEG decode failed: {exc}"
                        )
            for field in (
                "condition_id",
                "repeat_index",
                "captured_at_iso",
                "distance_mm",
                "yaw_deg",
                "pitch_deg",
                "focus_setting",
                "exposure_s",
                "iso",
                "white_balance",
                "display_brightness_setting",
            ):
                if row[field] in ("", "unknown"):
                    warnings.append(f"capture {capture_id}: {field} unknown")
            for field in (
                "repeat_index",
                "distance_mm",
                "exposure_s",
                "iso",
                "aperture_f",
            ):
                if row[field] not in ("", "unknown"):
                    try:
                        value = float(row[field])
                        if (
                            not math.isfinite(value)
                            or value <= 0
                            or (field == "repeat_index" and not value.is_integer())
                        ):
                            raise ValueError("must be positive and finite")
                    except ValueError:
                        errors.append(
                            f"capture {capture_id}: invalid {field}={row[field]!r}"
                        )
            for field in ("yaw_deg", "pitch_deg"):
                if row[field] not in ("", "unknown"):
                    try:
                        value = float(row[field])
                        if not math.isfinite(value) or abs(value) >= 90:
                            raise ValueError("outside (-90, 90)")
                    except ValueError:
                        errors.append(
                            f"capture {capture_id}: invalid {field}={row[field]!r}"
                        )
        if count == 0:
            warnings.append("no captured images logged yet")
    return errors, warnings


def present(kit_dir: Path, first_pattern: str) -> None:
    """Display full-size PNGs on the primary screen; S saves a compositor check."""
    if platform.system() == "Windows":
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except OSError:
            pass
    import tkinter as tk

    from PIL import ImageGrab, ImageTk

    manifest, errors = validate_kit(kit_dir)
    if errors:
        raise RuntimeError("; ".join(errors))
    root = tk.Tk()
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    if (width, height) != (manifest["width"], manifest["height"]):
        root.destroy()
        raise RuntimeError(
            f"primary screen is {width}x{height}; kit is {manifest['width']}x{manifest['height']}"
        )
    ids = [row["pattern_id"] for row in manifest["patterns"]]
    index = ids.index(first_pattern) if first_pattern else 0
    root.geometry(f"{width}x{height}+0+0")
    root.attributes("-fullscreen", True)
    root.configure(cursor="none", bg="black")
    root.lift()
    root.focus_force()
    label = tk.Label(root, borderwidth=0, highlightthickness=0)
    label.pack(fill="both", expand=True)
    current = {"photo": None}

    def show() -> None:
        path = kit_dir / "frames" / f"{ids[index]}.png"
        current["photo"] = ImageTk.PhotoImage(Image.open(path))
        label.configure(image=current["photo"])
        print(f"displaying {ids[index]} | sha256 {sha256(path)}", flush=True)

    def move(delta: int) -> None:
        nonlocal index
        index = (index + delta) % len(ids)
        show()

    def screenshot() -> None:
        observed = ImageGrab.grab(all_screens=False).convert("RGB")
        expected = Image.open(kit_dir / "frames" / f"{ids[index]}.png").convert("RGB")
        check_dir = kit_dir / "compositor_checks"
        check_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        png = check_dir / f"{stamp}_{ids[index]}.png"
        observed.save(png)
        a, b = (
            np.asarray(observed, dtype=np.int16),
            np.asarray(expected, dtype=np.int16),
        )
        report = {
            "pattern_id": ids[index],
            "compositor_png": str(png),
            "compositor_sha256": sha256(png),
            "captured_at_utc": stamp,
            "size_matches": a.shape == b.shape,
            "pixel_exact_fraction": float(np.mean(np.all(a == b, axis=2)))
            if a.shape == b.shape
            else None,
            "mean_absolute_channel_error": float(np.mean(np.abs(a - b)))
            if a.shape == b.shape
            else None,
            "scope": "compositor output only; panel emission not measured",
        }
        (check_dir / f"{stamp}_{ids[index]}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report), flush=True)

    root.bind("<Escape>", lambda _: root.destroy())
    root.bind("<Right>", lambda _: move(1))
    root.bind("<Left>", lambda _: move(-1))
    root.bind("<s>", lambda _: root.after(250, screenshot))
    show()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("generate")
    make.add_argument("--output-dir", type=Path, required=True)
    make.add_argument("--width", type=int, required=True)
    make.add_argument("--height", type=int, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--kit-dir", type=Path, required=True)
    check.add_argument("--session-dir", type=Path)
    start = commands.add_parser("init-session")
    start.add_argument("--kit-dir", type=Path, required=True)
    start.add_argument("--session-dir", type=Path, required=True)
    start.add_argument("--session-id", required=True)
    record = commands.add_parser("record")
    record.add_argument("--kit-dir", type=Path, required=True)
    record.add_argument("--session-dir", type=Path, required=True)
    record.add_argument("--capture-id", required=True)
    record.add_argument("--pattern-id", required=True)
    record.add_argument("--condition-id", required=True)
    record.add_argument("--repeat-index", type=int, required=True)
    record.add_argument("--jpeg-path", default="")
    record.add_argument("--raw-path", default="")
    for name in (
        "captured_at_iso",
        "distance_mm",
        "yaw_deg",
        "pitch_deg",
        "focus_setting",
        "exposure_s",
        "iso",
        "aperture_f",
        "white_balance",
        "display_brightness_setting",
        "notes",
    ):
        record.add_argument("--" + name.replace("_", "-"), default="unknown")
    view = commands.add_parser("present")
    view.add_argument("--kit-dir", type=Path, required=True)
    view.add_argument("--pattern-id", default="geometry")
    args = parser.parse_args()
    if args.command == "generate":
        manifest = generate(args.output_dir, args.width, args.height)
        print(
            json.dumps(
                {
                    "kit": str(args.output_dir),
                    "frames": len(manifest["patterns"]),
                    "manifest_sha256": sha256(
                        args.output_dir / "pattern_manifest.json"
                    ),
                },
                indent=2,
            )
        )
    elif args.command == "init-session":
        init_session(args.kit_dir, args.session_dir, args.session_id)
        print(
            json.dumps(
                {"session": str(args.session_dir), "session_id": args.session_id},
                indent=2,
            )
        )
    elif args.command == "record":
        values = {
            field: getattr(args, field)
            for field in LOG_FIELDS
            if field not in ("jpeg_sha256", "raw_sha256")
        }
        record_capture(args.kit_dir, args.session_dir, values)
        print(
            json.dumps(
                {
                    "session": str(args.session_dir),
                    "capture_id": args.capture_id,
                    "pattern_id": args.pattern_id,
                },
                indent=2,
            )
        )
    elif args.command == "validate":
        manifest, errors = validate_kit(args.kit_dir)
        warnings = []
        if args.session_dir:
            more_errors, warnings = validate_session(
                args.kit_dir, args.session_dir, manifest
            )
            errors.extend(more_errors)
        print(
            json.dumps(
                {
                    "kit": str(args.kit_dir),
                    "patterns": len(manifest["patterns"]),
                    "errors": errors,
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if errors:
            raise SystemExit(1)
    else:
        present(args.kit_dir, args.pattern_id)


if __name__ == "__main__":
    main()
