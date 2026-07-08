"""Portable launcher wrapper that keeps errors visible on another PC."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch QUINTdeepflow apps from a portable bundle.")
    parser.add_argument("--app", required=True, help="App label for log naming.")
    parser.add_argument("--script", required=True, help="Target Python script to execute.")
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    app_name = str(args.app).strip() or "QUINTportable"
    bundle_root = Path(os.environ.get("QUINT_PORTABLE_BUNDLE_ROOT", script_path.parents[2])).resolve()
    logs_dir = bundle_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_app_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in app_name)
    log_path = logs_dir / f"last_launch_{safe_app_name}.log"

    lines: list[str] = [
        f"app={app_name}",
        f"script={script_path}",
        f"cwd={Path.cwd()}",
        f"python={sys.executable}",
        f"python_version={sys.version}",
        f"bundle_root={bundle_root}",
        f"deepslice_python={os.environ.get('QUINTDEEPFLOW_DEEPSLICE_PYTHON', os.environ.get('QUINTDEEPFLOW2_DEEPSLICE_PYTHON', ''))}",
        "",
    ]

    try:
        os.chdir(script_path.parent)
        if str(script_path.parent) not in sys.path:
            sys.path.insert(0, str(script_path.parent))
        runpy.run_path(str(script_path), run_name="__main__")
        log_path.write_text("\n".join(lines + ["status=ok"]), encoding="utf-8")
        return 0
    except Exception:
        tb = traceback.format_exc()
        log_path.write_text("\n".join(lines + ["status=error", "", tb]), encoding="utf-8")
        print(tb)
        print()
        print(f"Portable launch failed. See log: {log_path}")
        try:
            input("Press Enter to close this window...")
        except EOFError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
