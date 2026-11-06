"""Audit an image album and build a separate, face-focused training dataset.

No face restoration, upscaling, recognition or generative edits. The detector
locates faces; selection combines sharpness, resolution, duplicate checks and
album balance. Review the selected faces before treating the set as identity-consistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image, ImageOps

Record = dict[str, Any]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_detector: cv2.FaceDetectorYN | None = None


def initialize(model: str) -> None:
    global _detector
    cv2.setNumThreads(1)
    _detector = cv2.FaceDetectorYN.create(model, "", (320, 320), 0.85, 0.3, 5000)


def perceptual_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(small)[:8, :8].flatten()
    median = np.median(block[1:])
    bits = block > median
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def audit_one(task: tuple[str, str]) -> Record:
    source, relative = task
    path = Path(source) / relative
    stat = path.stat()
    record: Record = {
        "relative": relative,
        "album": str(Path(relative).parent),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    try:
        raw = path.read_bytes()
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        with Image.open(path) as header:
            width, height = header.size
            orientation = int(header.getexif().get(274, 1))
        if orientation in {5, 6, 7, 8}:
            width, height = height, width
        record.update(width=width, height=height)
        image = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_4
        )
        if image is None:
            raise ValueError("Decoder returned no image")
        scale = min(1.0, 768 / max(image.shape[:2]))
        if scale < 1:
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        ih, iw = image.shape[:2]
        record["phash"] = perceptual_hash(image)
        assert _detector is not None
        _detector.setInputSize((iw, ih))
        _, faces = _detector.detect(image)
        record["face_count"] = 0 if faces is None else len(faces)
        if faces is None or len(faces) != 1:
            return record
        face = faces[0]
        x, y, fw, fh = [float(v) for v in face[:4]]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(iw, math.ceil(x + fw)), min(ih, math.ceil(y + fh))
        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return record
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Normalized face-detail score, a ranking heuristic rather than a quality verdict.
        gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        record.update(
            face_box=[
                x * width / iw,
                y * height / ih,
                fw * width / iw,
                fh * height / ih,
            ],
            face_pixels=min(fw * width / iw, fh * height / ih),
            face_fraction=fh / ih,
            confidence=float(face[14]),
            sharpness=sharpness,
            face_phash=perceptual_hash(roi),
            nose_position=float((face[8] - x) / max(fw, 1)),
            eye_roll=math.degrees(
                math.atan2(float(face[7] - face[5]), float(face[6] - face[4]))
            ),
        )
    except (OSError, ValueError, cv2.error) as error:
        record["error"] = str(error)
    return record


def scan(source: Path, model: Path, state: Path, workers: int) -> None:
    state.mkdir(parents=True, exist_ok=True)
    output = state / "audit.jsonl"
    existing: dict[str, Record] = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = cast(Record, json.loads(line))
            existing[row["relative"]] = row
    paths = sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and not any(
            part.startswith((".", "_")) for part in p.relative_to(source).parts[:-1]
        )
    )
    tasks: list[tuple[str, str]] = []
    for path in paths:
        relative = str(path.relative_to(source))
        old = existing.get(relative)
        stat = path.stat()
        if old and old["bytes"] == stat.st_size and old["mtime_ns"] == stat.st_mtime_ns:
            continue
        tasks.append((str(source), relative))
    (state / "scan.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "model": str(model.resolve()),
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "image_count": len(paths),
            },
            indent=2,
        )
    )
    print(
        f"Audit: {len(paths)} images, {len(tasks)} uncached, {workers} CPU workers",
        flush=True,
    )
    with (
        output.open("a") as log,
        ProcessPoolExecutor(
            max_workers=workers, initializer=initialize, initargs=(str(model),)
        ) as executor,
    ):
        for index, row in enumerate(executor.map(audit_one, tasks, chunksize=8), 1):
            log.write(json.dumps(row) + "\n")
            log.flush()
            existing[row["relative"]] = row
            if index % 100 == 0 or index == len(tasks):
                print(f"Audited {index}/{len(tasks)}", flush=True)
    records = [existing[str(path.relative_to(source))] for path in paths]
    counts = Counter(str(row.get("face_count", "error")) for row in records)
    summary = {
        "images": len(paths),
        "face_counts": dict(counts),
        "decode_errors": sum("error" in row for row in records),
        "exact_unique_images": len({row.get("sha256") for row in records}),
        "single_face_candidates": sum(
            row.get("face_count") == 1 and row.get("face_pixels", 0) >= 250
            for row in records
        ),
    }
    (state / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def portrait_box(
    face: list[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x, y, fw, fh = face
    crop_w = min(float(width), max(fw * 2.45, fh * 1.55))
    crop_h = min(float(height), crop_w * 1.25)
    # Preserve hair above the face and shoulders below, with no alignment/warping.
    left = max(0.0, min(width - crop_w, x + fw / 2 - crop_w / 2))
    top = max(0.0, min(height - crop_h, y - fh * 0.55))
    return round(left), round(top), round(left + crop_w), round(top + crop_h)


def select_records(
    rows: list[Record], count: int, cap: int, excluded: set[str] | None = None
) -> list[Record]:
    excluded = excluded or set()
    candidates = [
        row
        for row in rows
        if row.get("face_count") == 1
        and row.get("face_pixels", 0) >= 250
        and row.get("confidence", 0) >= 0.9
        and row.get("sharpness", 0) >= 30
        and row.get("sha256") not in excluded
    ]
    for row in candidates:
        row["rank_score"] = (
            math.log1p(float(row["sharpness"]))
            + math.log1p(min(float(row["face_pixels"]), 1400)) * 0.4
        )
    candidates.sort(key=lambda row: row["rank_score"], reverse=True)
    selected: list[Record] = []
    album_counts: Counter[str] = Counter()
    hashes: set[str] = set(excluded)
    # Round-robin album coverage first, then permit a second photo per album.
    for round_cap in range(1, cap + 1):
        for row in candidates:
            if len(selected) >= count:
                return selected
            if row["sha256"] in hashes or album_counts[row["album"]] >= round_cap:
                continue
            if any(
                (int(row["phash"], 16) ^ int(old["phash"], 16)).bit_count() <= 6
                for old in selected
            ):
                continue
            selected.append(row)
            hashes.add(row["sha256"])
            album_counts[row["album"]] += 1
    return selected


def reviewed_records(
    rows: list[Record], selection: Record
) -> tuple[list[Record], list[Record]]:
    """Resolve an explicit local review, rejecting unknown or duplicate sources."""
    by_path = {row["relative"]: row for row in rows}
    used: set[str] = set()
    groups: list[list[Record]] = []
    for role in ("portrait", "context"):
        paths = selection.get(role)
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError(f"Selection {role} must be a list of relative image paths")
        chosen: list[Record] = []
        for path in paths:
            if path not in by_path:
                raise ValueError(f"Selected source is missing from the audit: {path}")
            row = by_path[path]
            if row.get("face_count") != 1 or not row.get("face_box"):
                raise ValueError(f"Selected source has no single usable face: {path}")
            if row["sha256"] in used:
                raise ValueError(f"Duplicate selected source: {path}")
            used.add(row["sha256"])
            chosen.append(row)
        groups.append(chosen)
    if not groups[0]:
        raise ValueError("Select at least one portrait")
    return groups[0], groups[1]


def context_crop_box(
    value: object, width: int, height: int, face: list[float]
) -> tuple[int, int, int, int]:
    """Validate a reviewed native-pixel crop without clipping the detected face."""
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        raise ValueError("Context crop must contain four integer pixel coordinates")
    left, top, right, bottom = cast(list[int], value)
    x, y, fw, fh = face
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Context crop extends outside the source")
    if not (left <= x and top <= y and right >= x + fw and bottom >= y + fh):
        raise ValueError("Context crop would clip the detected face")
    return left, top, right, bottom


def build(
    state: Path,
    dataset_root: Path,
    name: str,
    portraits: int,
    context: int,
    cap: int,
    review: Path | None = None,
    selection: Path | None = None,
) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]+", name):
        raise ValueError("Dataset name must be lowercase letters/digits/_/-")
    scan_config = json.loads((state / "scan.json").read_text())
    source = Path(scan_config["source"])
    by_path = {
        row["relative"]: row
        for row in (
            json.loads(line)
            for line in (state / "audit.jsonl").read_text().splitlines()
        )
    }
    rows = [
        row
        for row in by_path.values()
        if not any(
            part.startswith((".", "_")) for part in Path(row["relative"]).parts[:-1]
        )
    ]
    exclusions = json.loads(review.read_text()) if review else {}
    rows = [
        row
        for row in rows
        if row["album"] not in exclusions.get("albums", [])
        and row["relative"] not in exclusions.get("files", [])
    ]
    selection_data = json.loads(selection.read_text()) if selection else {}
    if selection:
        faces, contextual = reviewed_records(rows, selection_data)
    else:
        faces = select_records(rows, portraits, cap)
        used = {row["sha256"] for row in faces}
        contextual = select_records(
            [row for row in rows if row.get("face_fraction", 0) >= 0.12],
            context,
            1,
            used,
        )
    targets = {
        "portrait": dataset_root / (name + "_portraits"),
        "context": dataset_root / (name + "_context"),
    }
    for target in targets.values():
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite dataset: {target}")
    for row in faces + contextual:
        if (
            hashlib.sha256((source / row["relative"]).read_bytes()).hexdigest()
            != row["sha256"]
        ):
            raise ValueError(f"Source changed since audit: {row['relative']}")
    for role, selected in (("portrait", faces), ("context", contextual)):
        crop_overrides = selection_data.get(role + "_crops", {})
        for row in selected:
            if row["relative"] in crop_overrides:
                context_crop_box(
                    crop_overrides[row["relative"]],
                    row["width"],
                    row["height"],
                    row["face_box"],
                )
    for target in targets.values():
        target.mkdir(parents=True)
    manifest: list[Record] = []
    for role, selected in [("portrait", faces), ("context", contextual)]:
        for row in selected:
            original = source / row["relative"]
            stem = re.sub(r"[^a-z0-9-]+", "-", original.stem.lower()).strip("-")
            basename = stem + "-" + row["sha256"][:8]
            metadata: Record = {"role": role, "source": str(original), **row}
            if role == "portrait":
                with Image.open(original) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    override = selection_data.get("portrait_crops", {}).get(
                        row["relative"]
                    )
                    box = (
                        context_crop_box(
                            override, image.width, image.height, row["face_box"]
                        )
                        if override is not None
                        else portrait_box(row["face_box"], image.width, image.height)
                    )
                    cropped = image.crop(box)
                    target = targets[role] / (basename + "-portrait.png")
                    cropped.save(target)
                    metadata.update(
                        crop_box=list(box),
                        output_size=list(cropped.size),
                        face_fraction_after_crop=float(row["face_box"][3])
                        / cropped.height,
                    )
            elif row["relative"] in selection_data.get("context_crops", {}):
                box = context_crop_box(
                    selection_data["context_crops"][row["relative"]],
                    row["width"],
                    row["height"],
                    row["face_box"],
                )
                with Image.open(original) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB").crop(box)
                    target = targets[role] / (basename + "-context.png")
                    image.save(target)
                    metadata.update(crop_box=list(box), output_size=list(image.size))
            else:
                target = targets[role] / (basename + original.suffix.lower())
                shutil.copy2(original, target)
            metadata["output"] = str(target)
            manifest.append(metadata)
    result = {
        "source": str(source),
        "datasets": {key: str(value) for key, value in targets.items()},
        "counts": {"portrait": len(faces), "context": len(contextual)},
        "albums": len({row["album"] for row in manifest}),
        "items": manifest,
    }
    (state / "selected.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({key: value for key, value in result.items() if key != "items"}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("scan")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--model", type=Path, required=True)
    audit.add_argument("--state", type=Path, required=True)
    audit.add_argument("--workers", type=int, default=4)
    create = subparsers.add_parser("build")
    create.add_argument("--state", type=Path, required=True)
    create.add_argument("--dataset-root", type=Path, required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--portraits", type=int, default=64)
    create.add_argument("--context", type=int, default=24)
    create.add_argument("--max-per-album", type=int, default=2)
    create.add_argument(
        "--review", type=Path, help="Local JSON with excluded albums and files"
    )
    create.add_argument(
        "--selection",
        type=Path,
        help="Local JSON with explicit portrait/context relative paths; overrides counts",
    )
    arguments = parser.parse_args()
    if arguments.command == "scan":
        if (
            not arguments.source.is_dir()
            or not arguments.model.is_file()
            or not 1 <= arguments.workers <= 16
        ):
            parser.error("Existing source/model and 1..16 workers are required")
        scan(arguments.source, arguments.model, arguments.state, arguments.workers)
    else:
        if (
            arguments.portraits < 1
            or arguments.context < 0
            or arguments.max_per_album < 1
        ):
            parser.error("Invalid dataset counts")
        build(
            arguments.state,
            arguments.dataset_root,
            arguments.name,
            arguments.portraits,
            arguments.context,
            arguments.max_per_album,
            arguments.review,
            arguments.selection,
        )


if __name__ == "__main__":
    main()
