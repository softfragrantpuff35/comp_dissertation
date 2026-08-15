"""
class_iou_analysis.py — testing the dual-mechanism hypothesis
==============================================================

MOTIVATION
----------
The per-class decomposition showed that the two detectors do not differ
uniformly. YOLO26n is significantly *better* on pedestrian — the smallest and
second most numerous class — and significantly *worse* on car, bus and bicycle.
That rules out the simple reading that YOLO26n is weaker on small objects, and
suggests two of its changes act in opposite directions:

  STAL (small-target-aware label assignment) changes *which* locations are
  treated as positives, and should therefore affect whether small objects are
  detected at all.

  DFL removal changes *how precisely* box coordinates are regressed, and should
  therefore affect how tightly boxes fit, most visibly where a box must cover a
  large extent accurately.

These two mechanisms make different predictions about how each class's gap
behaves as the IoU threshold tightens, and the predictions are separable:

  H-REG (DFL removal, a regression effect)
      Classes where YOLO26n is behind should fall further behind as IoU
      tightens, because the penalty for imprecise box placement grows.
      Expect: car, bus, bicycle gaps widen with IoU.

  H-ASSIGN (STAL, an assignment effect)
      A class helped by better positive assignment is helped at the point of
      detection, not at the point of box refinement. Its advantage should be
      largest at loose IoU and flat or shrinking as IoU tightens.
      Expect: the pedestrian advantage does not grow with IoU.

If both hold, the two mechanisms are separable in the data. If the pedestrian
advantage also grows with IoU, the assignment account fails and the whole
dual-mechanism reading must be dropped.

This is a falsifiable test, and it requires no retraining: the exported
predictions from the completed runs are sufficient.

USAGE
-----
    python scripts/class_iou_analysis.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import DATASET_ROOT, VISDRONE_NAMES, build_gt  # noqa: E402

RESULTS = Path("results/coco")
IOUS = np.arange(0.50, 0.96, 0.05)
SEEDS = (0, 1, 2)
ARCH = {"YOLOv10n": "yolov10n_seed{}", "YOLO26n": "yolo26n_seed{}"}

# Classes where the aggregate per-class comparison found a resolvable
# difference, with the direction found. Listed here so the predictions being
# tested are fixed before the numbers are produced.
PREDICTIONS = {
    "pedestrian": "26n ahead — expect gap NOT to grow with IoU (assignment)",
    "bicycle": "26n behind — expect gap to widen with IoU (regression)",
    "car": "26n behind — expect gap to widen with IoU (regression)",
    "bus": "26n behind — expect gap to widen with IoU (regression)",
}


def per_class_ap_by_iou(coco_gt, pred_path: Path, max_det: int = 1000) -> np.ndarray:
    """Return an array of shape [n_iou, n_class] of AP values."""
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(str(pred_path))
        e = COCOeval(coco_gt, coco_dt, iouType="bbox")
        e.params.iouThrs = IOUS
        e.params.maxDets = [1, 10, max_det]
        e.evaluate()
        e.accumulate()

    prec = e.eval["precision"]  # [IoU, recall, class, area, maxDets]
    out = np.full((len(IOUS), len(VISDRONE_NAMES)), np.nan)
    for t in range(len(IOUS)):
        for k in range(len(VISDRONE_NAMES)):
            p = prec[t, :, k, 0, 2]
            if (p > -1).any():
                out[t, k] = float(np.mean(p[p > -1]))
    return out


def main() -> None:
    from pycocotools.coco import COCO

    gt_path, _ = build_gt("val", DATASET_ROOT, RESULTS)
    print()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))

    curves = {}
    for arch, pattern in ARCH.items():
        runs = []
        for s in SEEDS:
            p = RESULTS / f"{pattern.format(s)}_pred.json"
            if not p.exists():
                sys.exit(f"[ERROR] missing {p}")
            print(f"  evaluating {pattern.format(s)} ...")
            runs.append(per_class_ap_by_iou(coco_gt, p))
        curves[arch] = np.stack(runs)  # [seed, iou, class]

    a, b = curves["YOLOv10n"], curves["YOLO26n"]
    gap = b.mean(0) - a.mean(0)                                   # [iou, class]; + means 26n ahead
    sd = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)       # [iou, class]

    # ---- table --------------------------------------------------------
    print("\n" + "=" * 96)
    print("GAP BY CLASS AND IoU THRESHOLD   (positive = YOLO26n ahead)")
    print("=" * 96)
    hdr = "class".ljust(18) + "".join(f"{t:>7.2f}" for t in IOUS)
    print(hdr)
    print("-" * len(hdr))
    for k, name in enumerate(VISDRONE_NAMES):
        row = name.ljust(18)
        for t in range(len(IOUS)):
            g, s2 = gap[t, k], 2 * sd[t, k]
            mark = "*" if abs(g) > s2 else " "
            row += f"{g:+6.3f}{mark}"
        print(row)
    print("-" * len(hdr))
    print("* = difference exceeds the 2sd band at that threshold\n")

    # ---- trend test ---------------------------------------------------
    # Only thresholds where the difference is resolvable contribute, matching the
    # criterion used elsewhere in the study. A class with too few resolvable
    # thresholds is reported as untestable rather than given a spurious slope.
    print("=" * 96)
    print("TREND OF THE GAP WITH IoU  (does the difference grow as localisation tightens?)")
    print("=" * 96)
    print(f"{'class':18s}{'n_res':>6s}{'gap@lo':>9s}{'gap@hi':>9s}{'slope':>9s}{'r':>8s}   reading")
    print("-" * 96)

    summary = {}
    for k, name in enumerate(VISDRONE_NAMES):
        res = np.abs(gap[:, k]) > 2 * sd[:, k]
        n = int(res.sum())
        if n < 3:
            print(f"{name:18s}{n:6d}{'—':>9s}{'—':>9s}{'—':>9s}{'—':>8s}   not resolvable")
            summary[name] = {"n_resolvable": n, "testable": False}
            continue
        t_r, g_r = IOUS[res], gap[res, k]
        slope = float(np.polyfit(t_r, g_r, 1)[0])
        r = float(np.corrcoef(t_r, g_r)[0, 1])
        ahead = bool(g_r.mean() > 0)   # cast: `numpy.bool_ is False` is never True
        # "Widening" means the magnitude of the gap grows with IoU, in whichever
        # direction the class sits: a positive slope for a class that is ahead,
        # a negative slope for a class that is behind.
        widening = (slope > 0) if ahead else (slope < 0)
        if ahead:
            reading = "advantage grows with IoU" if slope > 0 else "advantage flat/shrinks with IoU"
        else:
            reading = "deficit grows with IoU" if slope < 0 else "deficit flat/shrinks with IoU"
        print(f"{name:18s}{n:6d}{g_r[0]:+9.4f}{g_r[-1]:+9.4f}{slope:+9.3f}{r:+8.3f}   {reading}")
        summary[name] = {
            "n_resolvable": n, "testable": True, "ahead": bool(ahead),
            "slope": slope, "r": r, "widening": bool(widening),
            "gap_low": float(g_r[0]), "gap_high": float(g_r[-1]),
        }

    # ---- verdict against the pre-stated predictions --------------------
    print("\n" + "=" * 96)
    print("AGAINST THE PRE-STATED PREDICTIONS")
    print("=" * 96)
    for name, expectation in PREDICTIONS.items():
        s = summary.get(name, {})
        print(f"\n  {name}")
        print(f"    expected : {expectation}")
        if not s.get("testable"):
            print(f"    observed : only {s.get('n_resolvable', 0)} resolvable thresholds — untestable")
            continue
        direction = "ahead" if s["ahead"] else "behind"
        print(f"    observed : 26n {direction}; slope {s['slope']:+.3f}, r {s['r']:+.3f}, "
              f"{'widens' if s['widening'] else 'does not widen'} with IoU")

    reg = [n for n in ("bicycle", "car", "bus")
           if summary.get(n, {}).get("testable") and summary[n]["widening"]]
    ped = summary.get("pedestrian", {})
    ped_ok = ped.get("testable") and not ped.get("widening")

    print("\n" + "-" * 96)
    print(f"  H-REG    : {len(reg)}/3 deficit classes widen with IoU  -> {reg}")
    print(f"  H-ASSIGN : pedestrian advantage does not grow with IoU  -> {ped_ok}")
    print()
    if len(reg) >= 2 and ped_ok:
        print("  -> BOTH SUPPORTED. The two effects are separable: a regression-like")
        print("     deficit on the classes where YOLO26n is behind, and an")
        print("     assignment-like advantage on pedestrian that does not depend on")
        print("     localisation strictness.")
    elif len(reg) >= 2:
        print("  -> H-REG supported, H-ASSIGN not. The pedestrian advantage also grows")
        print("     with IoU, so it cannot be attributed to assignment alone.")
    elif ped_ok:
        print("  -> H-ASSIGN supported, H-REG not. The deficit does not behave like a")
        print("     regression-precision effect; the DFL account is not supported here.")
    else:
        print("  -> NEITHER SUPPORTED. Report the class differences descriptively and")
        print("     do not advance the dual-mechanism reading.")
    print("=" * 96)

    Path("results").mkdir(exist_ok=True)
    Path("results/class_iou_analysis.json").write_text(json.dumps({
        "iou_thresholds": IOUS.tolist(),
        "classes": VISDRONE_NAMES,
        "gap": gap.tolist(),
        "pooled_sd": sd.tolist(),
        "per_class": summary,
    }, indent=2))
    print("\nsaved: results/class_iou_analysis.json")

    # ---- figure -------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        focus = [c for c in ("pedestrian", "bicycle", "car", "bus") if c in VISDRONE_NAMES]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        colours = {"pedestrian": "#1f77b4", "bicycle": "#ff7f0e",
                   "car": "#2ca02c", "bus": "#d62728"}
        for name in focus:
            k = VISDRONE_NAMES.index(name)
            res = np.abs(gap[:, k]) > 2 * sd[:, k]
            ax.plot(IOUS, gap[:, k], color=colours.get(name), linewidth=1.6,
                    marker="o", markersize=4, label=name, alpha=0.85)
            ax.plot(IOUS[~res], gap[~res, k], linestyle="none", marker="o",
                    markerfacecolor="white", markeredgecolor=colours.get(name), markersize=6)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("IoU threshold")
        ax.set_ylabel("AP difference (YOLO26n − YOLOv10n)")
        ax.set_title("Per-class difference across IoU thresholds\n"
                     "(hollow markers: difference within the variance band)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        Path("results/figures").mkdir(parents=True, exist_ok=True)
        fig.savefig("results/figures/class_iou_analysis.png", dpi=200)
        print("saved: results/figures/class_iou_analysis.png")
    except ImportError:
        print("(matplotlib not installed — figure skipped)")


if __name__ == "__main__":
    main()
