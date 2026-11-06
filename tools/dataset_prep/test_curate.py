"""Selection and crop geometry safeguards for dataset preparation."""

import unittest

from tools.dataset_prep.curate import (
    context_crop_box,
    portrait_box,
    reviewed_records,
    select_records,
)


def record(album: str, digest: str, phash: str, pixels: int = 500) -> dict[str, object]:
    return {
        "album": album,
        "sha256": digest,
        "phash": phash,
        "face_count": 1,
        "face_pixels": pixels,
        "confidence": 0.99,
        "sharpness": 100,
        "relative": digest + ".jpg",
        "face_box": [0.0, 0.0, 100.0, 100.0],
    }


class SelectionTests(unittest.TestCase):
    def test_duplicate_and_excluded_sources_are_not_selected(self) -> None:
        rows = [
            record("a", "one", "0000000000000000"),
            record("b", "two", "0000000000000001"),
            record("c", "three", "ffffffffffffffff"),
            record("d", "four", "aaaaaaaaaaaaaaaa"),
        ]
        result = select_records(rows, 10, 2, {"three"})
        self.assertEqual({row["sha256"] for row in result}, {"one", "four"})

    def test_album_coverage_precedes_second_image(self) -> None:
        rows = [
            record("a", "one", "0000000000000000"),
            record("a", "two", "ffffffffffffffff"),
            record("b", "three", "aaaaaaaaaaaaaaaa"),
        ]
        result = select_records(rows, 2, 2)
        self.assertEqual([row["album"] for row in result], ["a", "b"])

    def test_small_faces_are_excluded(self) -> None:
        self.assertEqual(select_records([record("a", "one", "0", 249)], 1, 1), [])

    def test_crop_stays_inside_image_at_all_edges(self) -> None:
        for face in (
            [0.0, 0.0, 100.0, 120.0],
            [900.0, 800.0, 100.0, 120.0],
            [400.0, 300.0, 200.0, 240.0],
        ):
            left, top, right, bottom = portrait_box(face, 1000, 1000)
            self.assertTrue(0 <= left < right <= 1000)
            self.assertTrue(0 <= top < bottom <= 1000)
            x, y, width, height = face
            self.assertTrue(left <= x and right >= x + width)
            self.assertTrue(top <= y and bottom >= y + height)

    def test_manual_selection_preserves_roles_and_order(self) -> None:
        rows = [record("a", "one", "0"), record("b", "two", "f")]
        portraits, context = reviewed_records(
            rows, {"portrait": ["two.jpg"], "context": ["one.jpg"]}
        )
        self.assertEqual(portraits[0]["sha256"], "two")
        self.assertEqual(context[0]["sha256"], "one")

    def test_manual_selection_rejects_duplicate_source_across_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reviewed_records(
                [record("a", "one", "0")],
                {"portrait": ["one.jpg"], "context": ["one.jpg"]},
            )

    def test_manual_selection_rejects_unknown_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing from the audit"):
            reviewed_records([], {"portrait": ["unknown.jpg"], "context": []})

    def test_context_crop_preserves_face_and_rejects_out_of_bounds(self) -> None:
        face = [100.0, 120.0, 80.0, 90.0]
        self.assertEqual(
            context_crop_box([20, 30, 300, 400], 500, 600, face), (20, 30, 300, 400)
        )
        with self.assertRaisesRegex(ValueError, "clip"):
            context_crop_box([110, 30, 300, 400], 500, 600, face)
        with self.assertRaisesRegex(ValueError, "outside"):
            context_crop_box([0, 0, 501, 600], 500, 600, face)


if __name__ == "__main__":
    unittest.main()
