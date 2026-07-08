"""Desktop GUI entry point for QDFevo_3_Quantitate."""

from __future__ import annotations

import os

os.environ["QUINTDEEPFLOW_EXPORT_PATCH_IDS"] = "1"
os.environ["QUINTDEEPFLOW_IMPORT_QDF1_OMIT"] = "1"
os.environ["QUINTDEEPFLOW_GUI_TITLE"] = "QDFevo_3_Quantitate"
os.environ["QUINTDEEPFLOW_INPUT_JSON_LABEL"] = "3. Input JSON (QDFevo_2_AtlasFitter)"
os.environ["QUINTDEEPFLOW_INPUT_JSON_CHECK_LABEL"] = "Input JSON Check"

from gui.tk_app import launch_gui


if __name__ == "__main__":
    launch_gui()
