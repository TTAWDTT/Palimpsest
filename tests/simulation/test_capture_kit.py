"""Checks for the two main capture-kit risks: lost pixel probes and provenance."""

from PIL import Image

from palimpsest.simulation.capture_kit import (
    generate,
    init_session,
    record_capture,
    validate_kit,
    validate_session,
)


def test_nyquist_probe_is_not_constant_and_kit_detects_tampering(tmp_path):
    kit = tmp_path / "kit"
    generate(kit, 640, 480)
    manifest, errors = validate_kit(kit)
    assert not errors
    assert len(manifest["patterns"]) == 17
    with Image.open(kit / "frames/frequency_x_high.png") as frame:
        assert [frame.getpixel((x, 10))[0] for x in range(4)] == [218, 38, 218, 38]
    png = kit / "frames/geometry.png"
    png.write_bytes(png.read_bytes() + b"unexpected trailing bytes")
    _, errors = validate_kit(kit)
    assert any("SHA-256 mismatch: geometry" in error for error in errors)


def test_capture_log_is_bound_to_a_pattern_manifest_and_image_bytes(tmp_path):
    kit = tmp_path / "kit"
    generate(kit, 640, 480)
    manifest, _ = validate_kit(kit)
    session = tmp_path / "session"
    init_session(kit, session, "pilot-001")
    image = session / "camera_originals/capture.jpg"
    Image.new("RGB", (16, 12), (32, 96, 160)).save(image, format="JPEG")
    record_capture(
        kit,
        session,
        {
            "capture_id": "pilot-001-01",
            "pattern_id": "geometry",
            "condition_id": "baseline",
            "repeat_index": 1,
            "captured_at_iso": "2026-09-24T10:00:00+08:00",
            "jpeg_path": "camera_originals/capture.jpg",
            "raw_path": "",
        },
    )
    errors, warnings = validate_session(kit, session, manifest)
    assert not errors
    assert warnings  # Template leaves physical controls deliberately unknown.
    image.write_bytes(b"changed capture")
    errors, _ = validate_session(kit, session, manifest)
    assert any("jpeg_sha256 mismatch" in error for error in errors)
