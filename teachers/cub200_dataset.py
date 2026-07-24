"""Shared official CUB-200-2011 parsing, validation, and download helpers."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive


ARCHIVE_NAME = "CUB_200_2011.tgz"
ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
DOWNLOAD_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/"
    "CUB_200_2011.tgz?download=1"
)
DATASET_DIRECTORY = "CUB_200_2011"
EXPECTED_IMAGES = 11_788
EXPECTED_TRAIN_IMAGES = 5_994
EXPECTED_TEST_IMAGES = 5_794
EXPECTED_CLASSES = 200


@dataclass(frozen=True)
class CubRecord:
    image_id: int
    relative_path: str
    target: int
    is_train: bool


def _read_indexed_values(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CUB metadata file not found: {path}")
    values: dict[int, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = raw_line.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed metadata at {path}:{line_number}")
        image_id = int(fields[0])
        if image_id in values:
            raise ValueError(f"Duplicate image id {image_id} in {path}")
        values[image_id] = fields[1]
    return values


def resolve_dataset_root(root: Path) -> Path:
    """Resolve either the archive parent or the extracted CUB directory."""

    root = root.expanduser()
    candidates = (
        root,
        root / DATASET_DIRECTORY,
        root / "cub200" / DATASET_DIRECTORY,
        root / "CUB200" / DATASET_DIRECTORY,
    )
    for candidate in candidates:
        if (candidate / "images.txt").is_file():
            return candidate.resolve()
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "CUB-200-2011 is not extracted. Expected images.txt under one of: "
        f"{expected}"
    )


def read_records(dataset_root: Path) -> list[CubRecord]:
    paths = _read_indexed_values(dataset_root / "images.txt")
    labels = _read_indexed_values(dataset_root / "image_class_labels.txt")
    splits = _read_indexed_values(dataset_root / "train_test_split.txt")
    if not (paths.keys() == labels.keys() == splits.keys()):
        raise ValueError("CUB metadata files contain different image-id sets")

    records: list[CubRecord] = []
    for image_id in sorted(paths):
        class_id = int(labels[image_id])
        split_value = int(splits[image_id])
        if class_id < 1 or class_id > EXPECTED_CLASSES:
            raise ValueError(
                f"Class id out of range for image {image_id}: {class_id}"
            )
        if split_value not in (0, 1):
            raise ValueError(
                f"Split flag must be 0 or 1 for image {image_id}: {split_value}"
            )
        relative_path = paths[image_id]
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(f"Unsafe CUB image path: {relative_path!r}")
        records.append(
            CubRecord(
                image_id=image_id,
                relative_path=relative_path,
                target=class_id - 1,
                is_train=bool(split_value),
            )
        )
    return records


def validate_official_layout(dataset_root: Path) -> None:
    records = read_records(dataset_root)
    train_count = sum(record.is_train for record in records)
    test_count = len(records) - train_count
    classes = {record.target for record in records}
    problems: list[str] = []
    if len(records) != EXPECTED_IMAGES:
        problems.append(f"images={len(records)} expected={EXPECTED_IMAGES}")
    if train_count != EXPECTED_TRAIN_IMAGES:
        problems.append(
            f"train_images={train_count} expected={EXPECTED_TRAIN_IMAGES}"
        )
    if test_count != EXPECTED_TEST_IMAGES:
        problems.append(f"test_images={test_count} expected={EXPECTED_TEST_IMAGES}")
    if classes != set(range(EXPECTED_CLASSES)):
        problems.append(
            f"class_count={len(classes)} expected={EXPECTED_CLASSES}"
        )
    missing = [
        record.relative_path
        for record in records
        if not (dataset_root / "images" / record.relative_path).is_file()
    ]
    if missing:
        problems.append(
            f"missing_images={len(missing)} first={missing[0]!r}"
        )
    if problems:
        raise RuntimeError("Invalid CUB-200-2011 layout: " + "; ".join(problems))


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cub200(root: Path, *, download: bool = True) -> Path:
    """Return a validated official dataset root, downloading when requested."""

    try:
        dataset_root = resolve_dataset_root(root)
    except FileNotFoundError:
        if not download:
            raise
        root = root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        archive = root / ARCHIVE_NAME
        if not archive.is_file() or _md5(archive) != ARCHIVE_MD5:
            archive.unlink(missing_ok=True)
            partial = archive.with_suffix(archive.suffix + ".part")
            partial.unlink(missing_ok=True)
            request = urllib.request.Request(
                DOWNLOAD_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (IBAM-KD-H200-V2 CUB-200)",
                    "Accept-Encoding": "identity",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    with partial.open("wb") as destination:
                        while chunk := response.read(1024 * 1024):
                            destination.write(chunk)
                actual_md5 = _md5(partial)
                if actual_md5 != ARCHIVE_MD5:
                    raise RuntimeError(
                        "CUB archive MD5 mismatch: "
                        f"expected={ARCHIVE_MD5} actual={actual_md5}"
                    )
                partial.replace(archive)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        extract_archive(str(archive), str(root))
        dataset_root = resolve_dataset_root(root)
    validate_official_layout(dataset_root)
    return dataset_root


class CUB200Dataset(Dataset[tuple[Any, int]]):
    """CUB-200-2011 using the authors' official train/test split."""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        transform: Callable[[Image.Image], Any] | None = None,
        validate_layout: bool = False,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("CUB split must be 'train' or 'test'")
        self.root = resolve_dataset_root(Path(root))
        if validate_layout:
            validate_official_layout(self.root)
        self.transform = transform
        want_train = split == "train"
        self.records = [
            record for record in read_records(self.root) if record.is_train == want_train
        ]
        if not self.records:
            raise RuntimeError(f"CUB split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        path = self.root / "images" / record.relative_path
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.target
