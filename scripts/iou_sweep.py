"""
iou_sweep.py — testing the localisation-precision hypothesis
=============================================================

MOTIVATION
----------
The baseline comparison showed that the gap between YOLOv10n and YOLO26n is
NOT uniform across IoU thresholds:

    AP50  (loose localisation)  difference falls within the noise band
    AP75  (strict localisation) difference exceeds it

That pattern implies the two detectors differ in *how precisely they place
boxes*, not in *whether they find objects*. A candidate mechanism is available:
YOLOv10n retains Distribution Focal Loss for box regression (the training log
shows `Freezing layer 'model.23.dfl.conv.weight'`), whereas YOLO26 removes DFL
to support end-to-end NMS-free deployment. DFL exists specifically to sharpen
bounding-box regression.

HYPOTHESIS (H-LOC)
------------------
If the gap is driven by regression precision rather than detection ability,
then the YOLOv10n advantage should grow monotonically as the IoU threshold
tightens, and should be near zero at loose thresholds.

TEST
----
Evaluate AP independently at each IoU threshold from 0.50 to 0.95 for all
three seeds of both architectures, and examine whether the difference curve
rises with IoU. Averaging across seeds and reporting the seed spread keeps the
test consistent with the variance-band protocol used elsewhere in this study.

No retraining is required: the exported predictions from the baseline runs are
reused.

USAGE
-----
    python scripts/iou_sweep.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import DATASET_ROOT, build_gt  # noqa: E402

RESULTS = Path("results/coco")
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
CONFIGS = {
    "YOLOv10n": [f"yolov10n_seed{s}" for s in (0, 1, 2)],
    "YOLO26n": [f"yolo26n_seed{s}" for s in (0, 1, 2)],
}


def ap_at_iou(coco_gt, pred_path: Path, thresholds: np.ndarray, max_dets: int = 1000) -> np.ndarray:
    """AP evaluated independently at each IoU threshold."""
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(str(pred_path))
        e = COCOeval(coco_gt, coco_dt, iouType="bbox")
        e.params.iouThrs = thresholds
        e.params.maxDets = [1, 10, max_dets]
        e.evaluate()
        e.accumulate()

    # precision shape = [IoU, recall, class, area, maxDets]
    prec = e.eval["precision"]
    out = []
    for i in range(len(thresholds)):
        p = prec[i, :, :, 0, 2]  # area=all, our maxDets
        out.append(float(np.mean(p[p > -1])) if (p > -1).any() else np.nan)
    return np.array(out)


def main() -> None:
    from pycocotools.coco import COCO

    gt_path, _ = build_gt("val", DATASET_ROOT, RESULTS)
    print()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))

    curves: dict[str, np.ndarray] = {}
    for arch, runs in CONFIGS.items():
        rows = []
        for run in runs:
            p = RESULTS / f"{run}_pred.json"
            if not p.exists():
                sys.exit(f"[ERROR] missing {p}\nRe-run the experiment or check the results directory.")
            print(f"  evaluating {run} ...")
            rows.append(ap_at_iou(coco_gt, p, IOU_THRESHOLDS))
        curves[arch] = np.vstack(rows)  # [seed, threshold]

    a, b = curves["YOLOv10n"], curves["YOLO26n"]
    diff = a.mean(0) - b.mean(0)  # positive = YOLOv10n ahead
    # Pooled within-configuration sd at each threshold, same convention as the
    # variance band used for the headline comparison.
    sd = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)

    print("\n" + "=" * 78)
    print("AP BY IoU THRESHOLD  (mean of 3 seeds +/- sd)")
    print("=" * 78)
    print(f"{'IoU':>5s} {'YOLOv10n':>18s} {'YOLO26n':>18s} {'v10 - 26':>10s} {'2sd':>8s} {'':>5s}")
    print("-" * 78)
    for i, t in enumerate(IOU_THRESHOLDS):
        mark = "*" if abs(diff[i]) > 2 * sd[i] else ""
        print(
            f"{t:5.2f} {a[:, i].mean():9.4f} +/- {a[:, i].std(ddof=1):5.4f}"
            f" {b[:, i].mean():9.4f} +/- {b[:, i].std(ddof=1):5.4f}"
            f" {diff[i]:+10.4f} {2 * sd[i]:8.4f} {mark:>5s}"
        )
    print("-" * 78)
    print("* = difference exceeds the 2sd variance band at that threshold\n")

    # --- does the gap widen with IoU? -------------------------------------
    # The relative gap is the quantity of interest: absolute AP shrinks toward
    # zero at high IoU, so an absolute difference would shrink even if the two
    # models were diverging proportionally.
    #
    # A ratio is only interpretable where its numerator is itself resolvable.
    # At thresholds where the absolute difference falls inside the 2sd variance
    # band, the relative gap is a ratio of two noise-dominated quantities. Those
    # thresholds are therefore excluded from the trend statistic and reported
    # separately. This is the same 2sd criterion used throughout the study, not
    # a rule introduced for this analysis.
    rel = 100 * diff / b.mean(0)
    resolvable = np.abs(diff) > 2 * sd

    rho_all = float(np.corrcoef(IOU_THRESHOLDS, rel)[0, 1])
    if resolvable.sum() >= 3:
        t_r, rel_r = IOU_THRESHOLDS[resolvable], rel[resolvable]
        rho = float(np.corrcoef(t_r, rel_r)[0, 1])
        slope = float(np.polyfit(t_r, rel_r, 1)[0])
        monotonic = bool(np.all(np.diff(rel_r) > 0))
    else:
        rho, slope, monotonic = rho_all, float("nan"), False

    print("=" * 78)
    print("H-LOC: does the YOLOv10n advantage grow as localisation tightens?")
    print("=" * 78)
    print(f"{'IoU':>5s} {'abs diff':>10s} {'2sd':>9s} {'resolvable':>11s} {'rel gap %':>10s}")
    print("-" * 78)
    for k, t in enumerate(IOU_THRESHOLDS):
        flag = "yes" if resolvable[k] else "NO"
        bar = "#" * max(0, int(round(rel[k]))) if resolvable[k] else ""
        print(f"{t:5.2f} {diff[k]:+10.5f} {2 * sd[k]:9.5f} {flag:>11s} {rel[k]:10.2f}  {bar}")
    print("-" * 78)

    excluded = IOU_THRESHOLDS[~resolvable]
    if len(excluded):
        print("excluded from the trend (difference within the noise band): "
              + ", ".join(f"{t:.2f}" for t in excluded))
    print()
    print(f"  correlation over all thresholds       : {rho_all:+.4f}")
    print(f"  correlation over resolvable thresholds: {rho:+.4f}")
    print(f"  slope                                 : {slope:+.2f} %-points per 1.0 IoU")
    print(f"  strictly monotonic increase           : {monotonic}")
    if resolvable.any():
        lo, hi = IOU_THRESHOLDS[resolvable][0], IOU_THRESHOLDS[resolvable][-1]
        r0, r1 = rel[resolvable][0], rel[resolvable][-1]
        print(f"  gap at IoU={lo:.2f} -> IoU={hi:.2f}          : "
              f"{r0:+.2f}% -> {r1:+.2f}%  ({r1 / r0:.2f}x)")
    print()
    if rho > 0.7:
        print("  -> SUPPORTED: the advantage grows monotonically with IoU,")
        print("     consistent with a difference in box-regression precision")
        print("     rather than in detection ability.")
    elif rho < -0.7:
        print("  -> CONTRADICTED: the advantage shrinks as IoU tightens.")
        print("     The localisation-precision explanation does not hold.")
    else:
        print("  -> INCONCLUSIVE: no clear monotonic trend. Report as such;")
        print("     do not present the DFL account as established.")
    print("=" * 78)

    out = {
        "iou_thresholds": IOU_THRESHOLDS.tolist(),
        "yolov10n_per_seed": curves["YOLOv10n"].tolist(),
        "yolo26n_per_seed": curves["YOLO26n"].tolist(),
        "absolute_difference": diff.tolist(),
        "relative_difference_pct": rel.tolist(),
        "pooled_sd": sd.tolist(),
        "resolvable": resolvable.tolist(),
        "correlation_all_thresholds": rho_all,
        "correlation_resolvable_only": rho,
        "slope_pct_per_iou": slope,
        "strictly_monotonic": monotonic,
    }
    Path("results/iou_sweep.json").write_text(json.dumps(out, indent=2))
    print("\nsaved: results/iou_sweep.json")

    # --- figure ------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

        for arch, c, m in (("YOLOv10n", "#1f77b4", "o"), ("YOLO26n", "#d62728", "s")):
            y = curves[arch]
            ax1.errorbar(
                IOU_THRESHOLDS, y.mean(0), yerr=y.std(0, ddof=1),
                marker=m, color=c, capsize=3, label=f"{arch} (mean of 3 seeds)",
                linewidth=1.6, markersize=5,
            )
        ax1.set_xlabel("IoU threshold")
        ax1.set_ylabel("AP")
        ax1.set_title("(a) AP across IoU thresholds")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # Resolvable points carry the trend; excluded points are shown hollow so
        # the reader can see the full data without the trend line being driven
        # by a ratio of two noise-level quantities.
        ax2.plot(IOU_THRESHOLDS[resolvable], rel[resolvable], marker="D",
                 color="#2ca02c", linewidth=1.8, markersize=6, label="difference resolvable")
        ax2.plot(IOU_THRESHOLDS[~resolvable], rel[~resolvable], linestyle="none",
                 marker="o", markerfacecolor="white", markeredgecolor="#888888",
                 markersize=7, label="within noise band (excluded)")
        if resolvable.sum() >= 2:
            t_r = IOU_THRESHOLDS[resolvable]
            fit = np.polyfit(t_r, rel[resolvable], 1)
            ax2.plot(t_r, np.polyval(fit, t_r), linestyle=":", color="#2ca02c",
                     linewidth=1.2, alpha=0.8)
        ax2.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax2.set_xlabel("IoU threshold")
        ax2.set_ylabel("Relative advantage of YOLOv10n (%)")
        ax2.set_title(f"(b) Gap vs localisation strictness (r = {rho:+.3f})")
        ax2.legend(fontsize=8, loc="lower left")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        Path("results/figures").mkdir(parents=True, exist_ok=True)
        fig.savefig("results/figures/iou_sweep.png", dpi=200)
        print("saved: results/figures/iou_sweep.png")
    except ImportError:
        print("(matplotlib not installed — figure skipped; pip install matplotlib)")


if __name__ == "__main__":
    main()
