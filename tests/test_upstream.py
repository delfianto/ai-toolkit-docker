import subprocess
import unittest
from unittest.mock import patch

from tools.upstream import check


class UpstreamTests(unittest.TestCase):
    def setUp(self):
        self.upstream = "a" * 40
        self.wrapper = "b" * 40
        self.digest = "sha256:" + "c" * 64
        self.parse_base_digest = check.base_digest
        mock = patch.object(check, "base_digest", return_value=self.digest)
        self.base = mock.start()
        self.addCleanup(mock.stop)

    def resolve(self, event="schedule", **overrides):
        values = {
            "image": "ghcr.io/owner/ai-toolkit",
            "base_image": "ostris/aitoolkit:latest",
            "upstream_commit": self.upstream,
            "wrapper_commit": self.wrapper,
            "event": event,
        }
        values.update(overrides)
        return check.resolve(**values)

    def test_unchanged_schedule_skips(self):
        with patch.object(check, "published", return_value=True) as published:
            result = self.resolve()
        self.assertEqual(result["needs_build"], "false")
        published.assert_called_once_with(
            "ghcr.io/owner/ai-toolkit:" + result["build_tag"]
        )
        self.assertEqual(result["base_image"], "ostris/aitoolkit:latest@" + self.digest)

    def test_changed_source_base_or_wrapper_gets_another_build_tag(self):
        with patch.object(check, "published", return_value=False):
            original = self.resolve()
            source = self.resolve(upstream_commit="d" * 40)
            wrapper = self.resolve(wrapper_commit="e" * 40)
            self.base.return_value = "sha256:" + "f" * 64
            base = self.resolve()
        self.assertEqual(
            len({r["build_tag"] for r in [original, source, wrapper, base]}), 4
        )
        self.assertTrue(
            all(r["needs_build"] == "true" for r in [original, source, wrapper, base])
        )

    def test_manual_push_and_pr_always_build_without_package_lookup(self):
        with patch.object(check, "published") as published:
            for event in ("workflow_dispatch", "push", "pull_request"):
                self.assertEqual(self.resolve(event)["needs_build"], "true")
            published.assert_not_called()

    def test_lookup_only_treats_a_missing_manifest_as_absent(self):
        image = "ghcr.io/owner/ai-toolkit:build-123"
        for message in (f"ERROR: {image}: not found", "manifest unknown"):
            with (
                self.subTest(message=message),
                patch.object(
                    check.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 1, "", message),
                ),
            ):
                self.assertFalse(check.published(image))
        for message in (
            "denied: permission_denied: read_package",
            "unauthorized: not found",
            "timeout contacting registry",
            "429 Too Many Requests",
            "unexpected registry failure",
        ):
            with (
                self.subTest(message=message),
                patch.object(
                    check.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 1, "", message),
                ),
                self.assertRaises(RuntimeError),
            ):
                check.published(image)

    def test_head_resolution_rejects_empty_or_ambiguous_responses(self):
        with patch.object(check, "command", return_value=self.upstream + "\tHEAD"):
            self.assertEqual(check.upstream_head(), self.upstream)
        for result in (
            "",
            "bad\tHEAD",
            self.upstream + "\tHEAD\n" + self.wrapper + "\tHEAD",
        ):
            with (
                patch.object(check, "command", return_value=result),
                self.assertRaises(ValueError),
            ):
                check.upstream_head()

    def test_invalid_commit_is_rejected_before_docker_invocation(self):
        with self.assertRaises(ValueError):
            self.resolve(upstream_commit="main; execute-other-command")
        self.base.assert_not_called()

    def test_manifest_digest_parser_rejects_missing_digest(self):
        with patch.object(
            check,
            "command",
            return_value="Name: upstream\nDigest: " + self.digest + "\n",
        ):
            self.assertEqual(self.parse_base_digest("upstream"), self.digest)
        with (
            patch.object(check, "command", return_value="Name: upstream\n"),
            self.assertRaises(ValueError),
        ):
            self.parse_base_digest("upstream")


if __name__ == "__main__":
    unittest.main()
