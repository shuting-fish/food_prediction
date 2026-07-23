"""Smoke tests for the evidence-bound Streamlit presentation."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


APP_PATH = Path(__file__).parents[1] / "presentation" / "streamlit_app.py"
APPTEST_SCRIPT = """
import sys
from streamlit.testing.v1 import AppTest

app_path, page, marker = sys.argv[1:4]
app = AppTest.from_file(app_path, default_timeout=15)
app.run()
if page != "Overview":
    app.sidebar.radio[0].set_value(page).run()

assert len(app.exception) == 0, [str(item.value) for item in app.exception]
assert app.sidebar.radio[0].value == page
assert any(marker in item.value for item in app.markdown)
print(f"PAGE_OK={page}")
"""


class StreamlitPresentationTest(unittest.TestCase):
    """Verify that every presentation page renders without exceptions."""

    def assert_page_renders(self, page: str, marker: str) -> None:
        """Run each page check in isolation from Streamlit's native test runtime."""

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                APPTEST_SCRIPT,
                str(APP_PATH),
                page,
                marker,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                f"Page {page!r} failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )

    def test_all_navigation_targets_render(self) -> None:
        expected_markers = {
            "Overview": "From daily demand signals",
            "Data Landscape": "One governed core",
            "Project Workflow": "A repository path",
            "QA & Evidence": "Uncertainty is visible",
            "Limitations & Next Steps": "A useful draft is honest",
        }

        for page, marker in expected_markers.items():
            with self.subTest(page=page):
                self.assert_page_renders(page, marker)


if __name__ == "__main__":
    unittest.main()
