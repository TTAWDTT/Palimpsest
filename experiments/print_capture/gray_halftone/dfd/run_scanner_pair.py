"""Extract two scanner-labeled copies of one DFD print for a visual audit.

The filename correspondence alone is not proof that the same sheet was scanned.
Raw samples stay on E: for analysis; outputs get measurements only.
"""

from palimpsest.paths import DATA_ROOT

from pathlib import Path
import tarfile

from PIL import Image


ARCHIVE = DATA_ROOT / "raw/dfd_halftone/HalftoneImages-BW.tar.gz"
DEST = DATA_ROOT / "derived/dfd_probe"
NAMES = {
    f"HalftoneImages-BW/D1_HP4350/full_600_600_600_110216_1_ a_Scanner{index}.png"
    for index in (1, 2)
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    found = []
    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            if member.name not in NAMES:
                continue
            target = DEST / Path(member.name).name
            with tar.extractfile(member) as source, target.open("wb") as sink:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    sink.write(chunk)
            if target.stat().st_size != member.size:
                raise ValueError(f"truncated extraction: {member.name}")
            found.append(target)
    if {f"HalftoneImages-BW/D1_HP4350/{p.name}" for p in found} != NAMES:
        raise ValueError(f"found only {len(found)} of 2 target images")
    previews = []
    for target in sorted(found):
        with Image.open(target) as source:
            preview = source.convert("RGB")
            preview.thumbnail((600, 850))
            previews.append(preview.copy())
            print(target.name, source.size, source.info.get("dpi"))
    sheet = Image.new(
        "RGB",
        (sum(p.width for p in previews), max(p.height for p in previews)),
        "white",
    )
    x = 0
    for p in previews:
        sheet.paste(p, (x, 0))
        x += p.width
    sheet.save(DEST / "D1_same_print_two_scanners_preview.jpg", quality=90)


if __name__ == "__main__":
    main()
