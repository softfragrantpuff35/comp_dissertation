"""
audit.py — convergence check and annotation verification
=========================================================

Two checks that need no GPU and no retraining, both raised in review.

1. CONVERGENCE (examiner Q12)
   Completing 60 epochs without early stopping is weak evidence *against*
   convergence: `patience=15` not firing means validation fitness was still
   improving within the last 15 epochs. If one architecture is still climbing at
   epoch 60 while the other has plateaued, the comparison partly measures
   convergence speed rather than final capability.

2. ANNOTATION CONVERSION (examiner Q36)
   Evaluating ground truth against itself returns AP = 1.0 by construction and
   validates nothing about the conversion. A systematic error present on both
   sides — a category mis-map, dropped rows, a coordinate bug — would be
   invisible. This samples converted labels and checks them against the original
   VisDrone annotation files.

USAGE
-----
    python scripts/audit.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

RUNS = Path("runs/detect")
DATASET = Path(r"C:\comp_dissertation\datasets\VisDrone")
ORIG = Path(r"C:\comp_dissertation\_orig")


# ---------------------------------------------------------------------------
# 1. Convergence
# ---------------------------------------------------------------------------
def read_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (epoch, mAP50-95) from a run's results.csv."""
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    key = next((k for k in rows[0] if "mAP50-95" in k), None)
    if key is None:
        return None
    ep = np.array([float(r["epoch"]) for r in rows])
    ap = np.array([float(r[key]) for r in rows])
    return ep, ap


def convergence_report() -> dict:
    print("=" * 92)
    print("CONVERGENCE CHECK")
    print("=" * 92)
    print(f"{'run':32s}{'epochs':>7s}{'best_ep':>8s}{'best':>9s}{'final':>9s}"
          f"{'slope20':>10s}{'slope10':>10s}")
    print("-" * 92)

    out = {}
    for d in sorted(RUNS.glob("*/")):
        f = d / "results.csv"
        if not f.exists() or d.name.endswith("_val") or "smoke" in d.name:
            continue
        curve = read_curve(f)
        if curve is None:
            continue
        ep, ap = curve
        best_i = int(ap.argmax())
        # Slope over the final n epochs, in mAP points per epoch.
        s20 = float(np.polyfit(ep[-20:], ap[-20:], 1)[0]) if len(ep) >= 20 else float("nan")
        s10 = float(np.polyfit(ep[-10:], ap[-10:], 1)[0]) if len(ep) >= 10 else float("nan")
        print(f"{d.name:32s}{len(ep):7d}{int(ep[best_i]):8d}{ap[best_i]:9.4f}"
              f"{ap[-1]:9.4f}{s20:10.5f}{s10:10.5f}")
        out[d.name] = {
            "epochs": len(ep), "best_epoch": int(ep[best_i]),
            "best_map": float(ap[best_i]), "final_map": float(ap[-1]),
            "slope_last20": s20, "slope_last10": s10,
        }

    if not out:
        print("  no results.csv files found — check the RUNS path")
        return out

    print("-" * 92)
    fam = {"YOLOv10n": [], "YOLO26n": []}
    for name, v in out.items():
        key = "YOLOv10n" if name.startswith("yolov10") else "YOLO26n"
        fam[key].append(v)

    print("\nBy architecture family:")
    for k, vs in fam.items():
        if not vs:
            continue
        be = np.array([v["best_epoch"] for v in vs])
        s20 = np.array([v["slope_last20"] for v in vs])
        print(f"  {k:9s} n={len(vs):2d}  best_epoch mean {be.mean():5.1f} "
              f"(max {be.max()})  slope_last20 mean {s20.mean():+.5f}")

    print()
    a = np.array([v["slope_last20"] for v in fam.get("YOLOv10n", [])])
    b = np.array([v["slope_last20"] for v in fam.get("YOLO26n", [])])
    if len(a) and len(b):
        print(f"  final-20-epoch slope: YOLOv10n {a.mean():+.5f}, YOLO26n {b.mean():+.5f}, "
              f"difference {b.mean() - a.mean():+.5f} mAP/epoch")
        # A slope of 0.0002/epoch over 20 epochs is 0.004 mAP — comparable to the
        # variance band, so anything at or above that matters for interpretation.
        if abs(b.mean() - a.mean()) > 2e-4:
            print("  -> The two families differ in how much they were still improving at")
            print("     epoch 60. The comparison partly reflects convergence speed and")
            print("     this must be stated as a limitation on RQ1.")
        else:
            print("  -> Comparable end-of-training slopes. Neither family is markedly")
            print("     further from convergence than the other at epoch 60.")
        late = [n for n, v in out.items() if v["best_epoch"] >= 58]
        print(f"\n  runs whose best epoch is >= 58: {len(late)}/{len(out)}")
        if len(late) > len(out) * 0.5:
            print("  -> Most runs peaked at the very end; 60 epochs is a budget, not a")
            print("     convergence criterion. State this in Limitations.")
    return out


# ---------------------------------------------------------------------------
# 2. Annotation conversion
# ---------------------------------------------------------------------------
def annotation_spot_check(split: str = "val", n_images: int = 20, seed: int = 0) -> dict:
    """Compare converted YOLO labels against the original VisDrone annotations.

    VisDrone annotation columns:
        x, y, w, h, score, category, truncation, occlusion
    The ultralytics converter skips rows with score == 0 (ignored regions) and
    maps category c to class index c - 1.
    """
    print("\n" + "=" * 92)
    print(f"ANNOTATION CONVERSION SPOT-CHECK  ({split})")
    print("=" * 92)

    orig_dir = ORIG / f"VisDrone2019-DET-{split}" / "annotations"
    lbl_dir = DATASET / "labels" / split
    img_dir = DATASET / "images" / split
    for d in (orig_dir, lbl_dir, img_dir):
        if not d.exists():
            print(f"  [skip] not found: {d}")
            return {}

    from PIL import Image

    files = sorted(orig_dir.glob("*.txt"))
    sample = random.Random(seed).sample(files, min(n_images, len(files)))

    n_ok = 0
    cat_seen, cat_dropped = set(), set()
    worst_coord_err = 0.0
    problems = []

    for f in sample:
        lbl = lbl_dir / f.name
        if not lbl.exists():
            problems.append(f"{f.name}: no converted label")
            continue
        with Image.open(img_dir / f"{f.stem}.jpg") as im:
            W, H = im.size

        expected = []
        for line in f.read_text().splitlines():
            p = [x for x in line.replace(",", " ").split() if x]
            if len(p) < 6:
                continue
            x, y, w, h, score, cat = (int(float(v)) for v in p[:6])
            cat_seen.add(cat)
            if score == 0:            # ignored region
                cat_dropped.add(cat)
                continue
            # Convert to the same representation as the YOLO label so that both
            # sides can be sorted on identical keys. Sorting them on different
            # keys pairs the wrong boxes and manufactures large errors.
            expected.append((cat - 1, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))

        got = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            got.append((int(p[0]), *(float(v) for v in p[1:5])))

        if len(expected) != len(got):
            problems.append(f"{f.name}: {len(expected)} expected vs {len(got)} converted")
            continue

        # Compare box by box, converting the original to normalised centre-xywh.
        ok = True
        key = lambda t: (t[0], round(t[1], 5), round(t[2], 5))
        for (c_e, xe, ye, we, he), (c_g, xc, yc, bw, bh) in zip(
                sorted(expected, key=key), sorted(got, key=key)):
            if c_e != c_g:
                problems.append(f"{f.name}: class {c_e} vs {c_g}")
                ok = False
                break
            err = max(abs(xe - xc), abs(ye - yc), abs(we - bw), abs(he - bh))
            worst_coord_err = max(worst_coord_err, err)
            if err > 1e-3:
                problems.append(f"{f.name}: coordinate error {err:.5f}")
                ok = False
                break
        n_ok += ok

    print(f"  images sampled          : {len(sample)}")
    print(f"  matched exactly         : {n_ok}")
    print(f"  original categories seen : {sorted(cat_seen)}")
    print(f"  categories among score=0 : {sorted(cat_dropped)}")
    print(f"  worst normalised coord error : {worst_coord_err:.2e}")
    if problems:
        print(f"\n  PROBLEMS ({len(problems)}):")
        for p in problems[:12]:
            print(f"    {p}")
    else:
        print("\n  No discrepancies. Box counts, class indices and coordinates all agree")
        print("  with the original annotation files on the sampled images.")

    # Category 11 is 'others'. If it appears with score != 0 it maps to class 10,
    # which is outside the ten-class range.
    if 11 in cat_seen and 11 not in cat_dropped:
        print("\n  [WARNING] category 11 ('others') present with non-zero score in the")
        print("  sampled files. Verify it does not reach the label set as class 10.")

    return {"sampled": len(sample), "matched": n_ok, "problems": problems,
            "categories_seen": sorted(cat_seen), "worst_coord_error": worst_coord_err}


if __name__ == "__main__":
    conv = convergence_report()
    ann = annotation_spot_check()
    Path("results").mkdir(exist_ok=True)
    Path("results/audit.json").write_text(json.dumps({"convergence": conv, "annotations": ann}, indent=2))
    print("\nsaved: results/audit.json")
