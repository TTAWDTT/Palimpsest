import csv
import io
import json
import pickle
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from palimpsest.data.cifar10 import load_originals
from palimpsest.data.rr import load_pairs, load_small_image
from palimpsest.data.video import extract_frame, ffprobe_frames


def test_video_decode_preserves_planar_shape_and_little_endian_counts(monkeypatch):
    counts = np.array([[0, 1023, 255], [512, 256, 1]], dtype="<u2")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=counts.tobytes())

    monkeypatch.setattr("subprocess.run", run)
    actual = extract_frame("fake.mkv", 80, "gray16le", 1, "<u2", width=3, height=2)
    np.testing.assert_array_equal(actual, counts)
    assert actual.flags.owndata
    assert "select=eq(n\\,80)" in calls[0]
    with pytest.raises(RuntimeError, match="expected 8"):
        extract_frame("fake.mkv", 80, "gray16le", 1, "<u2", width=4, height=2)


def test_timestamp_reading_does_not_round_or_reorder(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"frames":[{"best_effort_timestamp_time":"0.033367"},{"best_effort_timestamp_time":"0.100100"}]}'
        ),
    )
    assert ffprobe_frames("fake.mkv") == [0.033367, 0.100100]


def test_cifar_batch_lookup_preserves_rgb_channels_and_row_selection(tmp_path):
    batch = np.zeros((2, 3072), dtype=np.uint8)
    batch[0] = 17
    batch[1, :1024], batch[1, 1024:2048], batch[1, 2048:] = 11, 22, 33
    encoded = pickle.dumps({b"data": batch})
    path = tmp_path / "trusted_fixture.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("cifar-10-batches-py/data_batch_1")
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    rows = [{"prefix": "second", "cifar_batch": "data_batch_1", "row": "1"}]
    actual = load_originals(rows, path)["second"]
    assert actual.shape == (32, 32, 3)
    np.testing.assert_array_equal(actual[0, 0], [11, 22, 33])
    assert np.all(actual == actual[0, 0])


def test_rr_pairs_exclude_whole_source_and_reject_duplicate_condition(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"trainval_overlap_audit": {"excluded_source_ids": ["ai/leaked"]}})
    )
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"label": "ai", "source_id": "leaked", "condition": "original"},
        {"label": "real", "source_id": "clean", "condition": "original"},
        {"label": "real", "source_id": "clean", "condition": "redigital"},
    ]

    def write():
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    write()
    assert set(
        load_pairs(manifest, audit, expected_sources=1, expected_redigital=1)
    ) == {"real/clean"}
    rows.append(rows[-1])
    write()
    with pytest.raises(ValueError, match="duplicate"):
        load_pairs(manifest, audit, expected_sources=1, expected_redigital=1)


def test_coarse_image_loading_records_jpeg_encoding_and_float_range(tmp_path):
    Image.new("RGB", (16, 12), (128, 64, 32)).save(tmp_path / "source.jpg", quality=95)
    rgb, fingerprint = load_small_image({"filename": "source.jpg"}, 8, root=tmp_path)
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.float32
    assert 0 <= rgb.min() <= rgb.max() <= 1
    assert len(fingerprint) == 16
    assert rgb[0, 0, 0] > rgb[0, 0, 1] > rgb[0, 0, 2]
