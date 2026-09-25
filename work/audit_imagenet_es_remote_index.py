"""Audit ImageNet-ES's official remote ZIP central directory via HTTP Range."""

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import requests

from work.probe_dragotti_range import RangeFile, read_member


URL = "https://huggingface.co/datasets/Edw2n/ImageNet-ES/resolve/main/ImageNet-ES.zip"
OUT = Path("work/imagenet_es_remote_index.json")


class RequestsRangeFile(RangeFile):
    """Use requests for HF's CDN redirect; verify every returned byte range."""

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n == 0:
            return b""
        start = self.pos
        if self.cache_start <= start and start + n <= self.cache_start + len(self.cache):
            offset = start - self.cache_start
            self.pos += n
            return self.cache[offset:offset + n]
        size = min(max(n, 1_048_576), self.size - start)
        end = start + size - 1
        response = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                timeout=120)
        response.raise_for_status()
        expected = f"bytes {start}-{end}/{self.size}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
            raise OSError(f"unexpected HTTP range {response.status_code} {response.headers.get('Content-Range')}")
        data = response.content
        if len(data) != size:
            raise OSError(f"short range read {len(data)} != {size}")
        self.cache_start, self.cache = start, data
        self.pos += n
        self.request_count += 1
        self.bytes_requested += size
        return data[:n]


def main() -> None:
    head = requests.head(URL, allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    remote = RequestsRangeFile(head.url, size)
    with zipfile.ZipFile(remote) as archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        roots = Counter("/".join(name.split("/")[:4]) for name in names)
        extensions = Counter(Path(name).suffix.lower() for name in names)
        segments = Counter(part for name in names for part in name.split("/")
                           if any(term in part.lower() for term in
                                  ("aperture", "shutter", "iso", "param", "light")))
        samples = {
            "param_control": [name for name in names if "/param_control/" in name][:15],
            "reference": [name for name in names if "sampled_tin_no_resize" in name][:8],
            "auto_exposure": [name for name in names if "/auto_exposure/" in name][:8],
        }
        metadata = []
        for index, info in enumerate(i for i in infos if i.filename.lower().endswith(".json")):
            data = read_member(remote, info)
            path = Path(f"work/imagenet_es_metadata_{index}.json")
            path.write_bytes(data)
            parsed = json.loads(data)
            metadata.append({"member": info.filename, "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest(),
                             "local_copy": str(path),
                             "top_level_keys": list(parsed)[:25] if isinstance(parsed, dict) else None})
        record = {"url": URL, "archive_bytes": size,
                  "etag": head.headers.get("ETag"),
                  "members": len(infos), "uncompressed_bytes": sum(i.file_size for i in infos),
                  "compressed_bytes": sum(i.compress_size for i in infos),
                  "prefix4_counts": roots.most_common(45),
                  "extension_counts": extensions.most_common(),
                  "parameter_named_segments": segments.most_common(70),
                  "sample_members": samples,
                  "metadata": metadata,
                  "range_requests": remote.request_count,
                  "range_bytes": remote.bytes_requested}
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
