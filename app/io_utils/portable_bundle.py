"""Helpers for exporting a portable QUINTdeepflow bundle to another PC."""

from __future__ import annotations

import os
import site
import shutil
import sys
from pathlib import Path

from deepslice.pipeline import (
    DEFAULT_DEEPSLICE_PYTHON,
    LEGACY_DEEPSLICE_ENV_VARS,
    PRIMARY_DEEPSLICE_ENV_VAR,
)


PORTABLE_BUNDLE_NAME = "QDF_portable"
PORTABLE_APP_DIRNAME = "app"
PORTABLE_RUNTIME_DIRNAME = "rt"
PORTABLE_APP_RUNTIME_DIRNAME = "py"
PORTABLE_DEEPSLICE_RUNTIME_DIRNAME = "ds"
PORTABLE_ATLAS_DIRNAME = "atlas"
PORTABLE_ATLAS_SUBDIR = "ccf"
PROJECT_TOP_LEVEL_DIRS = (
    "atlas",
    "config",
    "data_models",
    "deepslice",
    "exporters",
    "gui",
    "io_utils",
    "multichannel",
    "overlays",
    "quantification",
    "registration",
    "sample_configs",
)
PROJECT_TOP_LEVEL_FILES = (
    "QUINTdeepflow1.py",
    "QUINTdeepflow2.py",
    "quintnext_cli.py",
    "requirements.txt",
    "README.md",
)
TOOLS_WHITELIST = (
    "deepslice_predict_runner.py",
    "portable_launch.py",
)
PORTABLE_PROJECT_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "*.pyc",
    "*.pyo",
)
PORTABLE_RUNTIME_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pdb",
    "*_d.dll",
    "*_d.exe",
    "*.lib",
    "conda-meta",
)
def export_portable_bundle(source_root: Path, destination_parent: Path) -> Path:
    """Copy both GUIs plus local runtimes into a portable bundle directory."""

    source_root = source_root.resolve()
    destination_parent = destination_parent.resolve()
    bundle_root = destination_parent / PORTABLE_BUNDLE_NAME
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    project_src = source_root / "project_scaffold"
    project_dst = bundle_root / PORTABLE_APP_DIRNAME
    _copy_project_subset(project_src, project_dst)

    atlas_cache_dst = _copy_atlas_cache(source_root, bundle_root)
    app_python_dst = _copy_app_runtime(bundle_root)
    deepslice_python_dst = _copy_deepslice_runtime(source_root, bundle_root)

    _write_launchers(bundle_root, app_python_dst=app_python_dst, deepslice_python_dst=deepslice_python_dst)
    _write_readme(
        bundle_root,
        atlas_cache_dst=atlas_cache_dst,
        app_python_dst=app_python_dst,
        deepslice_python_dst=deepslice_python_dst,
    )
    return bundle_root


def _copy_atlas_cache(source_root: Path, bundle_root: Path) -> Path:
    """Copy Allen atlas cache plus fallback metadata into the portable bundle."""

    atlas_cache_src = source_root / "atlas_cache" / "allen_ccf"
    atlas_cache_dst = bundle_root / PORTABLE_ATLAS_DIRNAME / PORTABLE_ATLAS_SUBDIR
    if atlas_cache_src.exists():
        shutil.copytree(
            atlas_cache_src,
            atlas_cache_dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*__quintnext_cache.raw"),
        )
    atlas_cache_dst.mkdir(parents=True, exist_ok=True)

    qcalign_src = source_root / "QUINTsoftware" / "QCAlign-v0.8" / "ABA_Mouse_CCFv3_2017_25um.cutlas"
    for metadata_name in ("tree.json", "labels.txt"):
        src = qcalign_src / metadata_name
        dst = atlas_cache_dst / metadata_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return atlas_cache_dst


def _copy_project_subset(project_src: Path, project_dst: Path) -> None:
    """Copy only runtime-relevant code and configs into the portable bundle."""

    project_dst.mkdir(parents=True, exist_ok=True)

    for dirname in PROJECT_TOP_LEVEL_DIRS:
        src = project_src / dirname
        if not src.exists():
            continue
        shutil.copytree(
            src,
            project_dst / dirname,
            dirs_exist_ok=True,
            ignore=PORTABLE_PROJECT_IGNORE,
        )

    for filename in PROJECT_TOP_LEVEL_FILES:
        src = project_src / filename
        if src.exists():
            shutil.copy2(src, project_dst / filename)

    tools_src = project_src / "tools"
    tools_dst = project_dst / "tools"
    tools_dst.mkdir(parents=True, exist_ok=True)
    for filename in TOOLS_WHITELIST:
        src = tools_src / filename
        if src.exists():
            shutil.copy2(src, tools_dst / filename)


def _copy_app_runtime(bundle_root: Path) -> Path:
    """Copy the current Python 3.11 runtime plus user site packages."""

    current_python = Path(sys.executable).resolve()
    runtime_src = current_python.parent
    runtime_dst = bundle_root / PORTABLE_RUNTIME_DIRNAME / PORTABLE_APP_RUNTIME_DIRNAME
    shutil.copytree(
        runtime_src,
        runtime_dst,
        dirs_exist_ok=True,
        ignore=PORTABLE_RUNTIME_IGNORE,
    )
    _prune_runtime_root_noise(runtime_dst)

    user_site = Path(site.getusersitepackages()).resolve()
    if user_site.exists():
        target_site = runtime_dst / "Lib" / "site-packages"
        target_site.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            user_site,
            target_site,
            dirs_exist_ok=True,
            ignore=PORTABLE_RUNTIME_IGNORE,
        )

    bundled_python = runtime_dst / current_python.name
    if not bundled_python.exists():
        raise FileNotFoundError(f"Portable app runtime python was not copied correctly: {bundled_python}")
    return bundled_python


def _copy_deepslice_runtime(source_root: Path, bundle_root: Path) -> Path | None:
    """Copy the DeepSlice runtime so QUINTdeepflow1 can run on another PC."""

    source_python = _find_deepslice_python(source_root)
    if source_python is None:
        return None
    env_root = source_python.parent
    runtime_dst = bundle_root / PORTABLE_RUNTIME_DIRNAME / PORTABLE_DEEPSLICE_RUNTIME_DIRNAME
    shutil.copytree(
        env_root,
        runtime_dst,
        dirs_exist_ok=True,
        ignore=PORTABLE_RUNTIME_IGNORE,
    )
    _prune_runtime_root_noise(runtime_dst)
    bundled_python = runtime_dst / source_python.name
    return bundled_python if bundled_python.exists() else None


def _prune_runtime_root_noise(runtime_dst: Path) -> None:
    """Remove large runtime-root directories that are not needed at run time."""

    for dirname in ("Doc", "Include", "include", "libs", "Tools", "conda-meta"):
        target = runtime_dst / dirname
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    site_packages = runtime_dst / "Lib" / "site-packages"
    if not site_packages.exists():
        return

    # Only prune top-level helper directories. Recursively removing every
    # "docs"/"tests" folder breaks packages such as TensorFlow 1.x that import
    # package-internal documentation helpers at runtime.
    removable_dirnames = {"tests", "test", "kernel_tests", "examples", "example", "docs", "doc"}
    for path in site_packages.iterdir():
        if not path.is_dir():
            continue
        if path.name not in removable_dirnames:
            continue
        shutil.rmtree(path, ignore_errors=True)

    for header_dir in (
        site_packages / "tensorflow_core" / "include",
        site_packages / "tensorflow" / "include",
    ):
        if header_dir.exists():
            shutil.rmtree(header_dir, ignore_errors=True)


def _find_deepslice_python(source_root: Path) -> Path | None:
    """Locate a concrete DeepSlice Python that should be bundled."""

    candidates: list[Path] = []
    for env_name in (PRIMARY_DEEPSLICE_ENV_VAR, *LEGACY_DEEPSLICE_ENV_VARS):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            candidates.append(Path(env_value))
    candidates.extend(
        [
            source_root / PORTABLE_RUNTIME_DIRNAME / PORTABLE_DEEPSLICE_RUNTIME_DIRNAME / "python.exe",
            source_root / "portable_runtime" / "deepslice_env" / "python.exe",
            DEFAULT_DEEPSLICE_PYTHON,
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _write_launchers(bundle_root: Path, *, app_python_dst: Path, deepslice_python_dst: Path | None) -> None:
    """Write one-click launchers for the QUINTdeepflow GUIs."""

    app_python_rel = rf"%~dp0{PORTABLE_RUNTIME_DIRNAME}\{PORTABLE_APP_RUNTIME_DIRNAME}\python.exe"
    app_root_rel = rf"%~dp0{PORTABLE_RUNTIME_DIRNAME}\{PORTABLE_APP_RUNTIME_DIRNAME}"
    app_path_setup = (
        f"set \"PORTABLE_APP_ROOT={app_root_rel}\"\r\n"
        "set \"PATH=%PORTABLE_APP_ROOT%;%PORTABLE_APP_ROOT%\\Scripts;%PORTABLE_APP_ROOT%\\DLLs;%PATH%\"\r\n"
    )
    deepslice_python_rel = rf"%~dp0{PORTABLE_RUNTIME_DIRNAME}\{PORTABLE_DEEPSLICE_RUNTIME_DIRNAME}\python.exe"
    launch_wrapper_rel = rf"%~dp0{PORTABLE_APP_DIRNAME}\tools\portable_launch.py"
    (bundle_root / "logs").mkdir(parents=True, exist_ok=True)

    launch_deepflow1 = bundle_root / "launch_QUINTdeepflow1.bat"
    launch_deepflow1.write_text(
        "@echo off\r\n"
        "setlocal\r\n"
        f"set \"{PRIMARY_DEEPSLICE_ENV_VAR}={deepslice_python_rel}\"\r\n"
        f"set \"QUINTDEEPFLOW2_DEEPSLICE_PYTHON={deepslice_python_rel}\"\r\n"
        "set \"QUINT_PORTABLE_BUNDLE_ROOT=%~dp0\"\r\n"
        f"if exist \"{app_python_rel}\" (\r\n"
        + app_path_setup
        + f"  if exist \"{launch_wrapper_rel}\" (\r\n"
        + f"    \"{app_python_rel}\" \"{launch_wrapper_rel}\" --app QUINTdeepflow_deepslice --script \"%~dp0{PORTABLE_APP_DIRNAME}\\QUINTdeepflow1.py\"\r\n"
        + "  ) else (\r\n"
        + f"    \"{app_python_rel}\" \"%~dp0{PORTABLE_APP_DIRNAME}\\QUINTdeepflow1.py\"\r\n"
        + "  )\r\n"
        ") else (\r\n"
        + f"  echo Bundled Python runtime was not found: %~dp0{PORTABLE_RUNTIME_DIRNAME}\\{PORTABLE_APP_RUNTIME_DIRNAME}\\python.exe\r\n"
        "  pause\r\n"
        ")\r\n"
        "endlocal\r\n",
        encoding="utf-8",
    )

    launch_deepflow2 = bundle_root / "launch_QUINTdeepflow2.bat"
    launch_deepflow2.write_text(
        "@echo off\r\n"
        "setlocal\r\n"
        f"set \"{PRIMARY_DEEPSLICE_ENV_VAR}={deepslice_python_rel}\"\r\n"
        f"set \"QUINTDEEPFLOW2_DEEPSLICE_PYTHON={deepslice_python_rel}\"\r\n"
        "set \"QUINT_PORTABLE_BUNDLE_ROOT=%~dp0\"\r\n"
        f"if exist \"{app_python_rel}\" (\r\n"
        + app_path_setup
        + f"  if exist \"{launch_wrapper_rel}\" (\r\n"
        + f"    \"{app_python_rel}\" \"{launch_wrapper_rel}\" --app QUINTdeepflow_quantification --script \"%~dp0{PORTABLE_APP_DIRNAME}\\QUINTdeepflow2.py\"\r\n"
        + "  ) else (\r\n"
        + f"    \"{app_python_rel}\" \"%~dp0{PORTABLE_APP_DIRNAME}\\QUINTdeepflow2.py\"\r\n"
        + "  )\r\n"
        ") else (\r\n"
        + f"  echo Bundled Python runtime was not found: %~dp0{PORTABLE_RUNTIME_DIRNAME}\\{PORTABLE_APP_RUNTIME_DIRNAME}\\python.exe\r\n"
        "  pause\r\n"
        ")\r\n"
        "endlocal\r\n",
        encoding="utf-8",
    )

    launch_deepflow3 = bundle_root / "launch_QUINTdeepflow3.bat"
    launch_deepflow3.write_text(
        "@echo off\r\n"
        "setlocal\r\n"
        f"set \"{PRIMARY_DEEPSLICE_ENV_VAR}={deepslice_python_rel}\"\r\n"
        f"set \"QUINTDEEPFLOW2_DEEPSLICE_PYTHON={deepslice_python_rel}\"\r\n"
        "set \"QUINT_PORTABLE_BUNDLE_ROOT=%~dp0\"\r\n"
        f"if exist \"{app_python_rel}\" (\r\n"
        + app_path_setup
        + f"  if exist \"{launch_wrapper_rel}\" (\r\n"
        + f"    \"{app_python_rel}\" \"{launch_wrapper_rel}\" --app QUINTdeepflow_quality_check --script \"%~dp0{PORTABLE_APP_DIRNAME}\\QUINTdeepflow3.py\"\r\n"
        + "  ) else (\r\n"
        + f"    \"{app_python_rel}\" \"%~dp0{PORTABLE_APP_DIRNAME}\\QUINTdeepflow3.py\"\r\n"
        + "  )\r\n"
        ") else (\r\n"
        + f"  echo Bundled Python runtime was not found: %~dp0{PORTABLE_RUNTIME_DIRNAME}\\{PORTABLE_APP_RUNTIME_DIRNAME}\\python.exe\r\n"
        "  pause\r\n"
        ")\r\n"
        "endlocal\r\n",
        encoding="utf-8",
    )

    if app_python_dst.exists():
        launch_python = bundle_root / "launch_portable_python_console.bat"
        launch_python.write_text(
            "@echo off\r\n"
            "setlocal\r\n"
            f"if exist \"{app_python_rel}\" (\r\n"
            + app_path_setup +
            f"  start \"\" cmd /k \"{app_python_rel}\"\r\n"
            ")\r\n"
            "endlocal\r\n",
            encoding="utf-8",
        )


def _write_readme(
    bundle_root: Path,
    *,
    atlas_cache_dst: Path,
    app_python_dst: Path,
    deepslice_python_dst: Path | None,
) -> None:
    """Document how to use the portable bundle on another PC."""

    deepflow1_runtime_line = (
        f"DeepSlice runtime bundled: {PORTABLE_RUNTIME_DIRNAME}\\{PORTABLE_DEEPSLICE_RUNTIME_DIRNAME}\\python.exe\r\n"
        if deepslice_python_dst is not None
        else f"DeepSlice runtime was not bundled. Install Python 3.11 and set {PRIMARY_DEEPSLICE_ENV_VAR} manually.\r\n"
    )
    readme = bundle_root / "PORTABLE_README.txt"
    readme.write_text(
        "QUINTdeepflow portable bundle\r\n"
        "\r\n"
        "Included apps\r\n"
        "1. launch_QUINTdeepflow1.bat : QUINTdeepflow DeepSlice / JPEG preparation GUI\r\n"
        "2. launch_QUINTdeepflow2.bat : QUINTdeepflow Quantification / visualisation GUI\r\n"
        "3. launch_QUINTdeepflow3.bat : QUINTdeepflow Quality Check / omit GUI\r\n"
        "\r\n"
        "How to use on another PC\r\n"
        "1. Copy this whole folder to the target PC.\r\n"
        "2. Double-click launch_QUINTdeepflow1.bat, launch_QUINTdeepflow2.bat, or launch_QUINTdeepflow3.bat.\r\n"
        "\r\n"
        + f"Portable app runtime bundled: {PORTABLE_RUNTIME_DIRNAME}\\{PORTABLE_APP_RUNTIME_DIRNAME}\\python.exe\r\n"
        + deepflow1_runtime_line
        + f"Atlas cache bundled under: {PORTABLE_ATLAS_DIRNAME}\\{PORTABLE_ATLAS_SUBDIR}\r\n"
        + "Portable launch logs are written to: logs\\last_launch_*.log\r\n"
        "\r\n"
        "Notes\r\n"
        "- QUINTdeepflow1, QUINTdeepflow2, and QUINTdeepflow3 are the desktop entry points for QUINTdeepflow.\r\n"
        f"- QUINTdeepflow1 calls the bundled DeepSlice environment through {PRIMARY_DEEPSLICE_ENV_VAR}.\r\n"
        + f"- {PORTABLE_ATLAS_DIRNAME}\\{PORTABLE_ATLAS_SUBDIR} contains the Allen atlas files used by QUINTdeepflow quantification.\r\n"
        "- If channel-map Excel export is unavailable on the target PC, QUINTdeepflow quantification will fall back to CSV.\r\n",
        encoding="utf-8",
    )
