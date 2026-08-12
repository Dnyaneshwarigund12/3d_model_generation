"""Tests for the Colab notebook, which nothing else can catch until Colab runs it.

Two classes of breakage live here: a cell that no longer matches the code it calls,
and the session-setup cell losing files. The second has a specific history - step 4
tells you to re-run that cell after the restart, and step 3 clones the model
repositories into the project folder, so a cell that replaces the folder deletes
gigabytes it is then forbidden to re-download.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.pipeline import run  # noqa: E402
from app.ui import build_ui  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "colab_photo_to_3d.ipynb"
# Cell 4 is the session-setup cell; the tests that use it assert on what it does.
SETUP_CELL = 4


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def as_python(source: str) -> str:
    """Blank the shell escapes and magics, keeping line numbers intact."""
    return "\n".join(
        "" if line.lstrip().startswith(("!", "%")) else line
        for line in source.splitlines()
    )


def test_every_code_cell_parses(code_cells):
    for index, source in enumerate(code_cells):
        try:
            ast.parse(as_python(source))
        except SyntaxError as exc:  # pragma: no cover - only on a broken notebook
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_calls_into_the_project_match_its_signatures(code_cells):
    signatures = {
        "run": inspect.signature(run),
        "build_ui": inspect.signature(build_ui),
    }
    for index, source in enumerate(code_cells):
        for node in ast.walk(ast.parse(as_python(source))):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                if getattr(node.func.value, "id", "") in {"subprocess", "os", "shutil"}:
                    continue  # subprocess.run is not our run
                name = node.func.attr
            else:
                continue
            if name not in signatures:
                continue
            parameters = signatures[name].parameters
            unknown = [
                kw.arg
                for kw in node.keywords
                if kw.arg and kw.arg not in parameters
            ]
            assert not unknown, f"cell {index}: {name}() has no {unknown}"


def test_settings_attributes_the_notebook_sets_exist(code_cells):
    fields = set(Settings().__dict__)
    for index, source in enumerate(code_cells):
        for node in ast.walk(ast.parse(as_python(source))):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "settings"
                ):
                    assert target.attr in fields, f"cell {index}: no Settings.{target.attr}"


def test_the_tools_the_notebook_runs_exist(notebook):
    root = NOTEBOOK.parent.parent
    body = json.dumps(notebook)
    for relative in (
        "tools/colab_setup.py",
        "tools/doctor.py",
        "tools/make_marker.py",
        "tools/validate.py",
        "tools/colab_launch.py",
    ):
        assert relative in body or relative.split("/")[-1].replace(".py", "") in body, (
            f"notebook no longer mentions {relative}"
        )
        assert (root / relative).exists()


def test_step_7_uses_the_colab_safe_launcher(notebook):
    """The old build_ui().launch() path embeds in Colab's broken Gradio iframe."""
    source = "".join(notebook["cells"][13]["source"])
    assert "from tools.colab_launch import launch" in source
    assert "launch(generator=" in source
    assert "build_ui(" not in source
    assert "demo.launch" not in source
    assert "queue().launch" not in source


def run_setup_cell(notebook, **config) -> None:
    """Execute the session-setup cell with its configuration replaced."""
    source = "".join(notebook["cells"][SETUP_CELL]["source"])
    body = source[source.index("import os") :]
    prelude = "\n".join(f"{key} = {value!r}" for key, value in config.items())
    namespace: dict = {}
    cwd = Path.cwd()
    try:
        exec(compile(prelude + "\n" + body, "<setup cell>", "exec"), namespace)
    finally:
        os.chdir(cwd)  # the cell ends in os.chdir


def test_setup_cell_from_drive_keeps_what_the_install_cloned(notebook, tmp_path):
    drive_project = tmp_path / "drive"
    (drive_project / "app").mkdir(parents=True)
    (drive_project / "app" / "pipeline.py").write_text("# newer\n", encoding="utf-8")

    project = tmp_path / "content" / "project"
    cloned = project / "third_party" / "TripoSR" / "tsr"
    cloned.mkdir(parents=True)
    (cloned / "system.py").write_text("# cloned by step 3\n", encoding="utf-8")
    (project / "constraints-colab.txt").write_text("numpy==2.4.6\n", encoding="utf-8")

    run_setup_cell(
        notebook,
        USE_DRIVE=False,
        CODE_SOURCE="drive",
        GITHUB_REPO="",
        GITHUB_BRANCH="main",
        DRIVE_PROJECT=str(drive_project),
        PROJECT_DIR=str(project),
    )

    assert (cloned / "system.py").exists(), "re-running the cell deleted the clone"
    assert (project / "constraints-colab.txt").exists()
    assert (project / "app" / "pipeline.py").exists(), "fresh code was not copied in"


def test_setup_cell_from_github_is_safe_to_rerun(notebook, tmp_path):
    root = NOTEBOOK.parent.parent
    if not (root / ".git").is_dir():
        pytest.skip("not a git checkout")

    project = tmp_path / "project"
    config = dict(
        USE_DRIVE=False,
        CODE_SOURCE="github",
        # The checkout itself stands in for the remote: same code path, no network.
        GITHUB_REPO=str(root),
        GITHUB_BRANCH="main",
        DRIVE_PROJECT="",
        PROJECT_DIR=str(project),
    )

    run_setup_cell(notebook, **config)
    assert (project / "app" / "pipeline.py").exists()

    marker = project / "third_party" / "Hunyuan3D-2.1" / "hy3dshape"
    marker.mkdir(parents=True)
    run_setup_cell(notebook, **config)
    assert marker.exists(), "the second run deleted what step 3 cloned"
