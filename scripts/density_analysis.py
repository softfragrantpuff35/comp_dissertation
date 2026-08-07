"""
density_analysis.py — Phase 0: scene-density distribution and stratified evaluation
===================================================================================

Two jobs:

1. Measure the per-image object-count distribution and derive LOW / MEDIUM /
   HIGH density bins with roughly balanced image counts. Plan §8.4 requires the
   boundaries to come from the observed distribution rather than being fixed in
   advance, because arbitrary thresholds can leave a bin too small for its AP to
   mean anything.

2. Given a predictions file, report AP per density bin. This answers which model
   degrades more slowly as scenes get denser — the analysis no cited prior work
   performs on this dataset.

Requires the ground truth built by coco_eval.py.

USAGE
-----
    # distribution only
    python scripts/density_analysis.py --split val

    # distribution + per-bin AP for a run
    python scripts/density_analysis.py --split val --pred runs/detect/NAME/predictions.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import DATASET_ROOT, build_gt, evaluate, remap_predictions  # noqa: E402


def load_counts(gt_path: Path) -> tuple[dict[int, int], dict[int, str]]:
    """Return {image_id: n_objects} and {image_id: file_name} from a COCO GT file."""
    gt = json.loads(gt_path.read_text())
    counts = {im["id"]: 0 for im in gt["images"]}
    names = {im["id"]: im["file_name"] for im in gt["images"]}
    for a in gt["annotations"]:
        counts[a["image_id"]] += 1
    return counts, names


def describe(counts: dict[int, int]) -> np.ndarray:
    """Print the distribution and return the sorted count array."""
    arr = np.array(sorted(counts.values()))
    print("=" * 70)
    print("OBJECT-COUNT DISTRIBUTION")
    print("=" * 70)
    print(f"  images            : {len(arr)}")
    print(f"  total objects     : {int(arr.sum()):,}")
    print(f"  mean per image    : {arr.mean():.1f}")
    print(f"  std               : {arr.std():.1f}")
    print(f"  min / max         : {arr.min()} / {arr.max()}")
    print()
    for q in (10, 25, 33, 50, 67, 75, 90, 95, 99):
        print(f"  p{q:<3d}              : {np.percentile(arr, q):.0f}")
    print()

    # Crude text histogram — enough to see the shape without a plotting dependency.
    hist, edges = np.histogram(arr, bins=12)
    peak = hist.max()
    print("  histogram:")
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(40 * h / peak) if peak else ""
        print(f"    {lo:5.0f}-{hi:5.0f} | {h:4d} {bar}")
    print()
    return arr


def make_bins(counts: dict[int, int], arr: np.ndarray) -> dict:
    """Split images into three bins at the 33rd and 67th percentiles."""
    lo_edge = float(np.percentile(arr, 33))
    hi_edge = float(np.percentile(arr, 67))

    bins = {
        "low": {"range": [0, lo_edge], "image_ids": []},
        "medium": {"range": [lo_edge, hi_edge], "image_ids": []},
        "high": {"range": [hi_edge, float(arr.max())], "image_ids": []},
    }
    for img_id, n in counts.items():
        key = "low" if n <= lo_edge else ("medium" if n <= hi_edge else "high")
        bins[key]["image_ids"].append(img_id)

    print("=" * 70)
    print("DENSITY BINS (33rd / 67th percentile — balanced by image count)")
    print("=" * 70)
    for k, b in bins.items():
        ids = b["image_ids"]
        objs = sum(counts[i] for i in ids)
        share = 100 * len(ids) / len(counts)
        lo, hi = b["range"]
        rng = f"<={hi:.0f}" if k == "low" else (f">{lo:.0f}" if k == "high" else f"{lo:.0f}-{hi:.0f}")
        print(f"  {k:7s} {rng:>10s} objects/img : {len(ids):4d} images ({share:4.1f}%), {objs:7,} objects")

    smallest = min(len(b["image_ids"]) for b in bins.values())
    if smallest < 50:
        print(f"\n  [WARN] smallest bin has only {smallest} images — per-bin AP will be noisy.")
        print("         Consider reducing to two bins rather than reporting three unstable ones.")
    print()
    return bins


def evaluate_bins(gt_path: Path, pred_path: Path, bins: dict, max_dets: int) -> dict:
    """Run COCOeval separately over each density bin's image subset."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))

    print("=" * 70)
    print("AP BY SCENE DENSITY")
    print("=" * 70)
    print(f"  {'bin':8s} {'images':>7s} {'AP':>8s} {'AP50':>8s} {'AP_small':>10s}")
    print("  " + "-" * 45)

    out = {}
    for name, b in bins.items():
        e = COCOeval(coco_gt, coco_dt, iouType="bbox")
        e.params.imgIds = sorted(b["image_ids"])  # restrict evaluation to this bin
        e.params.maxDets = [1, 10, max_dets]
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate()
            e.accumulate()
            e.summarize()

        stats = list(e.stats)
        if stats[0] < 0:  # see coco_eval.py — maxDets=100 is hardcoded for stats[0]
            prec = e.eval["precision"][:, :, :, 0, 2]
            stats[0] = float(np.mean(prec[prec > -1])) if (prec > -1).any() else float("nan")

        out[name] = {"n_images": len(b["image_ids"]), "AP": stats[0], "AP50": stats[1], "AP_small": stats[3]}
        print(f"  {name:8s} {len(b['image_ids']):7d} {stats[0]:8.4f} {stats[1]:8.4f} {stats[3]:10.4f}")

    print()
    lo, hi = out["low"]["AP"], out["high"]["AP"]
    if lo > 0:
        print(f"  Degradation low -> high: {100 * (lo - hi) / lo:+.1f}% relative AP")
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--out-dir", default="results/coco")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--max-dets", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    gt_path, id_map = build_gt(args.split, Path(args.root), out_dir)
    print()

    counts, names = load_counts(gt_path)
    arr = describe(counts)
    bins = make_bins(counts, arr)

    # Per-image counts, for any further analysis or plotting during write-up.
    csv_path = out_dir / f"density_{args.split}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name", "n_objects"])
        for i in sorted(counts):
            w.writerow([i, names[i], counts[i]])

    # Bin definitions are frozen to a file so every configuration is evaluated
    # against identical bins.
    bins_path = out_dir / f"density_bins_{args.split}.json"
    bins_path.write_text(json.dumps(bins, indent=2))
    print(f"  per-image counts : {csv_path}")
    print(f"  bin definitions  : {bins_path}\n")

    if args.pred:
        pred_path = remap_predictions(
            Path(args.pred), id_map, out_dir / f"visdrone_{args.split}_pred_remapped.json"
        )
        print()
        res = evaluate_bins(gt_path, pred_path, bins, args.max_dets)
        res_path = out_dir / f"density_metrics_{Path(args.pred).parent.name}_{args.split}.json"
        res_path.write_text(json.dumps(res, indent=2))
        print(f"  saved: {res_path}")


if __name__ == "__main__":
    main()
