# One-slice demo data

This folder contains a small demo for both public launchers.

## QDFevo_1_Align demo

Use this input folder:

```text
demo\one_slice\qdf1_align\raw
```

Set `Channel for fitting` to `CH3`.

Optional AP hint file:

```text
demo\one_slice\qdf1_align\ap_hints_xy01.csv
```

The demo TIFF is downsampled from the original section so it stays small enough for GitHub. It is intended for workflow testing, not for publication-grade fitting.

## QDFevo_2_AtlasFitter demo

Open this JSON:

```text
demo\one_slice\qdf2_atlasfitter\jpg\QDFevo_demo_one_slice.json
```

The adjacent JPEG is:

```text
demo\one_slice\qdf2_atlasfitter\jpg\505A_XY01_CH3.jpg
```

The JSON contains only one slice and can be edited/saved with `QDFevo_2_AtlasFitter`.
