"""Run constrained QDFevo_1_Align fitting experiments from an existing JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from deepslice.evo_constrained_experiments import (  # noqa: E402
    DEFAULT_AP_PRIOR_WEIGHT,
    DEFAULT_AP_SEARCH_HALF_VOX,
    DEFAULT_AP_SEARCH_STEP_VOX,
    DEFAULT_AUTO_MARKER_MIN_CONTOUR_GAIN,
    DEFAULT_INTENSITY_SCORE_GAIN,
    DEFAULT_PREVIEW_MAX_SIDE,
    DEFAULT_SCALE_RANGE,
    DEFAULT_TILT_LIMIT_DEG,
    run_auto_marker_contour_warp,
    run_intensity_landmark_selective_update,
    run_coronal_constrained_projection,
    run_lowdim_deepslice_search,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test constrained atlas-fitting strategies from a DeepSlice/QDFevo_1 JSON "
            "without rerunning DeepSlice."
        )
    )
    parser.add_argument("--json", required=True, type=Path, help="Input DeepSlice/QDFevo_1 JSON.")
    parser.add_argument("--jpg-dir", type=Path, help="Directory containing the slice JPEGs.")
    parser.add_argument("--output-dir", type=Path, help="Directory for experiment outputs.")
    parser.add_argument("--ap-hints", type=Path, help="CSV with AP hints.")
    parser.add_argument("--teacher-json", type=Path, help="Optional manually fitted JSON for metrics.")
    parser.add_argument(
        "--methods",
        choices=("all", "coronal9", "lowdim", "intensity_landmark", "auto_marker_contour"),
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument("--tilt-limit-deg", type=float, default=DEFAULT_TILT_LIMIT_DEG)
    parser.add_argument("--scale-min", type=float, default=DEFAULT_SCALE_RANGE[0])
    parser.add_argument("--scale-max", type=float, default=DEFAULT_SCALE_RANGE[1])
    parser.add_argument("--intensity-scale-min", type=float, default=0.98)
    parser.add_argument("--intensity-scale-max", type=float, default=1.02)
    parser.add_argument("--preview-max-side", type=int, default=DEFAULT_PREVIEW_MAX_SIDE)
    parser.add_argument(
        "--ap-search-half-vox",
        type=int,
        default=DEFAULT_AP_SEARCH_HALF_VOX,
        help="Lowdim AP search radius around the DeepSlice initial AP, in 25 um voxels.",
    )
    parser.add_argument("--ap-search-step-vox", type=int, default=DEFAULT_AP_SEARCH_STEP_VOX)
    parser.add_argument(
        "--ap-prior-weight",
        type=float,
        default=DEFAULT_AP_PRIOR_WEIGHT,
        help="Penalty weight for moving away from the DeepSlice initial AP in lowdim search.",
    )
    parser.add_argument(
        "--min-intensity-score-gain",
        type=float,
        default=DEFAULT_INTENSITY_SCORE_GAIN,
        help="No-teacher fallback threshold for accepting intensity/landmark updates.",
    )
    parser.add_argument("--auto-marker-max", type=int, default=6)
    parser.add_argument("--auto-marker-min-move-px", type=float, default=5.0)
    parser.add_argument("--auto-marker-max-move-fraction", type=float, default=0.14)
    parser.add_argument("--auto-marker-min-contour-gain", type=float, default=DEFAULT_AUTO_MARKER_MIN_CONTOUR_GAIN)
    parser.add_argument("--atlas-root", type=Path, help="Optional atlas root override.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_json = args.json.resolve()
    output_dir = (args.output_dir or (input_json.parent / "qdfevo_constrained_experiments")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_range = (float(args.scale_min), float(args.scale_max))
    if scale_range[0] <= 0 or scale_range[1] <= 0 or scale_range[0] > scale_range[1]:
        raise ValueError("--scale-min and --scale-max must be positive and ordered.")

    outputs = []
    if args.methods in {"all", "coronal9"}:
        outputs.append(
            run_coronal_constrained_projection(
                json_path=input_json,
                jpg_dir=args.jpg_dir,
                output_stem=output_dir / "coronal_constrained9" / "jpgDS_coronal9_results",
                ap_hint_path=args.ap_hints,
                teacher_json_path=args.teacher_json,
                tilt_limit_deg=float(args.tilt_limit_deg),
                scale_range=scale_range,
                preview_max_side=int(args.preview_max_side),
                atlas_root=args.atlas_root,
            )
        )
    if args.methods in {"all", "lowdim"}:
        outputs.append(
            run_lowdim_deepslice_search(
                json_path=input_json,
                jpg_dir=args.jpg_dir,
                output_stem=output_dir / "lowdim_search" / "jpgDS_lowdim_search_results",
                ap_hint_path=args.ap_hints,
                teacher_json_path=args.teacher_json,
                tilt_limit_deg=float(args.tilt_limit_deg),
                scale_range=scale_range,
                preview_max_side=int(args.preview_max_side),
                ap_search_half_vox=int(args.ap_search_half_vox),
                ap_search_step_vox=int(args.ap_search_step_vox),
                ap_prior_weight=float(args.ap_prior_weight),
                atlas_root=args.atlas_root,
            )
        )
    if args.methods in {"all", "intensity_landmark"}:
        outputs.append(
            run_intensity_landmark_selective_update(
                json_path=input_json,
                jpg_dir=args.jpg_dir,
                output_stem=output_dir / "intensity_landmark" / "jpgDS_intensity_landmark_results",
                ap_hint_path=args.ap_hints,
                teacher_json_path=args.teacher_json,
                scale_range=(max(0.01, float(args.intensity_scale_min)), float(args.intensity_scale_max)),
                preview_max_side=int(args.preview_max_side),
                ap_search_half_vox=int(args.ap_search_half_vox),
                ap_search_step_vox=int(args.ap_search_step_vox),
                ap_prior_weight=float(args.ap_prior_weight),
                min_objective_gain=float(args.min_intensity_score_gain),
                atlas_root=args.atlas_root,
            )
        )
    if args.methods in {"all", "auto_marker_contour"}:
        outputs.append(
            run_auto_marker_contour_warp(
                json_path=input_json,
                jpg_dir=args.jpg_dir,
                output_stem=output_dir / "auto_marker_contour" / "jpgDS_auto_marker_contour_results",
                teacher_json_path=args.teacher_json,
                preview_max_side=max(int(args.preview_max_side), 420),
                max_markers=int(args.auto_marker_max),
                min_marker_move_px=float(args.auto_marker_min_move_px),
                max_marker_move_fraction=float(args.auto_marker_max_move_fraction),
                min_contour_gain=float(args.auto_marker_min_contour_gain),
                atlas_root=args.atlas_root,
            )
        )

    for output in outputs:
        print(f"[{output.method}]")
        print(f"  JSON: {output.json_path}")
        print(f"  CSV: {output.csv_path}")
        print(f"  XML: {output.xml_path}")
        print(f"  report: {output.report_path}")
        print(f"  QC: {output.qc_dir}")
        if output.teacher_report_path:
            print(f"  teacher metrics: {output.teacher_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
