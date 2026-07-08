"""External DeepSlice runner used by the QUINTdeepflow1 GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from DeepSlice import DSModel

DEEPSLICE_MOUSE_VOXEL_SIZE_UM = 25.0
DEEPSLICE_MOUSE_QUICKNII_RESOLUTION = (456, 528, 320)
DEEPSLICE_MOUSE_BREGMA_X_UM = 5400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSlice on a prepared JPEG manifest.")
    parser.add_argument("--manifest", required=True, help="JSON file listing JPEG paths to predict.")
    parser.add_argument("--output-stem", required=True, help="Output stem without extension.")
    parser.add_argument("--species", default="mouse", choices=["mouse", "rat"], help="Atlas species.")
    parser.add_argument("--ensemble", action="store_true", help="Use DeepSlice ensemble inference for higher accuracy.")
    parser.add_argument(
        "--same-slicing-angle",
        action="store_true",
        help="Apply DeepSlice angle propagation so all slices share one smoothed slicing angle.",
    )
    parser.add_argument(
        "--ap-hint-json",
        default="",
        help="Optional JSON AP constraints created by QUINTdeepflow before DeepSlice post-processing.",
    )
    parser.add_argument(
        "--ap-hint-mode",
        default="hard-clamp",
        choices=["hard-clamp", "target-center"],
        help="How AP hints constrain the predicted slice center before saving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_stem = Path(args.output_stem)
    image_list = json.loads(manifest_path.read_text(encoding="utf-8"))

    model = DSModel(args.species)
    model.predict(image_list=image_list, ensemble=bool(args.ensemble), section_numbers=False)
    if args.ap_hint_json:
        _apply_ap_hint_constraints(
            model,
            ap_hint_json=Path(args.ap_hint_json),
            mode=args.ap_hint_mode,
            species=args.species,
        )
    if args.same_slicing_angle:
        try:
            model.propagate_angles()
            if args.ap_hint_json:
                _apply_ap_hint_constraints(
                    model,
                    ap_hint_json=Path(args.ap_hint_json),
                    mode=args.ap_hint_mode,
                    species=args.species,
                )
        except Exception as exc:  # pragma: no cover - depends on DeepSlice internals
            print(f"WARNING: propagate_angles failed and will be skipped: {exc}")
    model.save_predictions(output_stem.as_posix())
    return 0


def _apply_ap_hint_constraints(model: DSModel, *, ap_hint_json: Path, mode: str, species: str) -> None:
    """Constrain DeepSlice-predicted AP center inside user-provided AP ranges."""

    if species.lower() != "mouse":
        print("WARNING: AP hint constraints are currently implemented for mouse only.")
        return
    if not ap_hint_json.exists():
        print(f"WARNING: AP hint JSON not found: {ap_hint_json}")
        return
    payload = json.loads(ap_hint_json.read_text(encoding="utf-8"))
    hints = {str(item.get("filename", "")).lower(): item for item in payload.get("hints", [])}
    if not hints:
        return
    predictions = model.predictions.copy()
    applied = 0
    for row_index, row in predictions.iterrows():
        filename = Path(str(row.get("Filenames", ""))).name.lower()
        hint = hints.get(filename)
        if hint is None:
            continue
        target_y = _ap_mm_to_quicknii_y(float(hint["ap_mm"]))
        tolerance_y = max(float(hint.get("tolerance_mm", 0.0)) * 1000.0 / DEEPSLICE_MOUSE_VOXEL_SIZE_UM, 0.0)
        center_y = float(row["oy"]) + (0.5 * float(row["uy"])) + (0.5 * float(row["vy"]))
        if mode == "target-center":
            constrained_center_y = target_y
        else:
            lower = target_y - tolerance_y
            upper = target_y + tolerance_y
            constrained_center_y = min(max(center_y, lower), upper)
        delta_y = constrained_center_y - center_y
        if abs(delta_y) <= 1e-9:
            continue
        predictions.loc[row_index, "oy"] = float(row["oy"]) + delta_y
        applied += 1
    model.predictions = predictions
    print(f"AP hint constraints applied to {applied} section(s) before DeepSlice save.")


def _ap_mm_to_quicknii_y(ap_mm: float) -> float:
    allen_x_um = DEEPSLICE_MOUSE_BREGMA_X_UM - (float(ap_mm) * 1000.0)
    return float(DEEPSLICE_MOUSE_QUICKNII_RESOLUTION[1] - 1) - (allen_x_um / DEEPSLICE_MOUSE_VOXEL_SIZE_UM)


if __name__ == "__main__":
    raise SystemExit(main())
