# Face-focused dataset preparation

This CPU pipeline audits a source album, selects sharp single-face photographs,
checks exact and perceptual duplicates, and balances selection across subfolders.
It exports native-resolution head-and-shoulder crops and separate original-frame
context photographs. It does not restore, upscale, align or synthesize faces.
Face detection does not verify identity; review the selection before training.

Install the shared development environment with `uv sync`. Download the
[OpenCV YuNet detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
to your runtime data directory. Keep source paths, detector files, audit state,
review exclusions and generated datasets outside the repository.

```sh
uv run python dataset_prep/curate.py scan \
  --source "$SOURCE_ALBUM" --model "$YUNET_MODEL" --state "$AUDIT_DIR"
uv run python dataset_prep/curate.py build \
  --state "$AUDIT_DIR" --dataset-root "$DATASET_ROOT" --name character_v2 \
  --portraits 64 --context 24 --max-per-album 2 --review "$REVIEW_JSON"
```

The optional review JSON contains `albums` (relative subfolder names) and `files`
(relative image paths) to exclude after visual inspection. Output directories
must be new. `selected.json` records every source, crop boundary and quality
metric. Source images remain unchanged. The scan resumes unchanged files; use a
new state directory when changing the source root or detector.

Caption the exported images with the [captioning pipeline](../captioning/README.md).
Crops need their own captions because framing and visible clothing/background
change. Use both exported folders as separate Toolkit datasets, with the same
identity trigger. Keep pose, expression and lighting variety, while removing
obstructions or styling that conflicts with the intended character appearance.

Run selection checks with `uv run python -m unittest dataset_prep.test_curate`.

For a manually reviewed selection, use `build --selection "$SELECTION_JSON"`.
This overrides automatic image counts and keeps the reviewed role/order:

```json
{
  "portrait": ["album/image-a.jpg"],
  "context": ["album/image-b.jpg"],
  "context_crops": {"album/image-b.jpg": [0, 120, 2000, 3000]}
}
```

Crop coordinates are native pixels `[left, top, right, bottom]` after EXIF
orientation. Context crops are optional, must retain the detected face, and are
exported as lossless PNGs. Review that they also preserve useful hair/body framing.
Unknown or duplicate sources are rejected. Source hashes must still match the
scan. Store selections and review sheets in the runtime state directory.

The optional `portrait_crops` map uses the same native-pixel format and overrides
the automatic portrait box for reviewed border corrections.
