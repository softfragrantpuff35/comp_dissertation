"""
coco_eval.py — Phase 0 Step 2: COCO-style stratified evaluation for VisDrone
============================================================================

WHY THIS EXISTS
---------------
Ultralytics' built-in COCO evaluation only runs for COCO/LVIS datasets:

    # ultralytics/models/yolo/detect/val.py
    if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):

VisDrone is neither, so AP_small / AP_medium / AP_large — the primary metric
for RQ2 and RQ3 — are NOT produced out of the box. This script supplies them.

WHAT ULTRALYTICS *DOES* GIVE US (verified against v8.4.115 source)
------------------------------------------------------------------
With save_json=True it still writes predictions.json for any dataset:

    image_id    = filename stem; STRING when the stem is non-numeric
                  (VisDrone stems look like '0000137_02220_d_0000163')
    category_id = class_index + 1        (1-based for non-COCO datasets)
    bbox        = [x_topleft, y_topleft, w, h] in ORIGINAL image pixels
    score       = confidence

So we must (a) build a matching COCO ground-truth JSON from the YOLO .txt
labels, and (b) reconcile image ids. We remap string stems to integer ids via
a deterministic sorted mapping, because pycocotools is only reliably tested
with integer ids.

USAGE
-----
    # 1. Sanity check FIRST — must print AP = 1.000
    python scripts/coco_eval.py --split val --selftest

    # 2. Real evaluation of a run's predictions
    python scripts/coco_eval.py --split val --pred runs/detect/NAME/predictions.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# VisDrone class names, in the order used by ultralytics' VisDrone.yaml.
# category_id in both GT and predictions is (index + 1).
VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

DATASET_ROOT = Path(r"C:\comp_dissertation\datasets\VisDrone")


# ----------------------------------------------------------------------------
# Ground-truth construction
# ----------------------------------------------------------------------------
def build_gt(split: str, root: Path, out_dir: Path) -> tuple[Path, dict[str, int]]:
    """Convert YOLO-format labels into a COCO ground-truth JSON.

    Returns the path to the JSON and the stem -> integer image_id mapping.
    """
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    for d in (img_dir, lbl_dir):
        if not d.exists():
            sys.exit(f"[ERROR] missing directory: {d}")

    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not img_paths:
        sys.exit(f"[ERROR] no images found in {img_dir}")

    # Deterministic mapping: sorted stem order -> 1..N. Reproducible across runs.
    id_map = {p.stem: i + 1 for i, p in enumerate(img_paths)}

    images, annotations = [], []
    ann_id = 1
    n_missing_label = 0
    n_skipped_boxes = 0

    for p in img_paths:
        # PIL reads only the header here, so this stays fast on thousands of files.
        with Image.open(p) as im:
            w_img, h_img = im.size

        images.append({"id": id_map[p.stem], "file_name": p.name, "width": w_img, "height": h_img})

        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            n_missing_label += 1
            continue

        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, bw, bh = (float(v) for v in parts[1:5])

            # normalised centre-xywh -> absolute top-left xywh
            x = (xc - bw / 2) * w_img
            y = (yc - bh / 2) * h_img
            bw_abs = bw * w_img
            bh_abs = bh * h_img

            if bw_abs <= 0 or bh_abs <= 0:
                n_skipped_boxes += 1
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": id_map[p.stem],
                    "category_id": cls + 1,  # 1-based, matches ultralytics class_map
                    "bbox": [round(x, 3), round(y, 3), round(bw_abs, 3), round(bh_abs, 3)],
                    # 'area' drives COCOeval's small/medium/large split — required.
                    "area": round(bw_abs * bh_abs, 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    gt = {
        "info": {"description": f"VisDrone {split} converted from YOLO labels"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(VISDRONE_NAMES)],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / f"visdrone_{split}_gt.json"
    gt_path.write_text(json.dumps(gt))
    (out_dir / f"visdrone_{split}_idmap.json").write_text(json.dumps(id_map, indent=2))

    print(f"[GT] split           : {split}")
    print(f"[GT] images          : {len(images)}")
    print(f"[GT] annotations     : {len(annotations)}")
    print(f"[GT] objects/image   : {len(annotations) / max(len(images), 1):.1f}")
    if n_missing_label:
        print(f"[GT] missing labels  : {n_missing_label}")
    if n_skipped_boxes:
        print(f"[GT] degenerate boxes: {n_skipped_boxes} (skipped)")
    print(f"[GT] written         : {gt_path}")
    return gt_path, id_map


# ----------------------------------------------------------------------------
# Prediction remapping
# ----------------------------------------------------------------------------
def remap_predictions(pred_path: Path, id_map: dict[str, int], out_path: Path) -> Path:
    """Rewrite predictions.json so its image_id values are the integer ids used in the GT."""
    preds = json.loads(pred_path.read_text())
    if not preds:
        sys.exit(f"[ERROR] {pred_path} contains no detections")

    out, unmatched = [], set()
    for d in preds:
        raw = d["image_id"]
        stem = Path(str(d.get("file_name", ""))).stem or str(raw)
        if stem in id_map:
            new_id = id_map[stem]
        elif str(raw) in id_map:
            new_id = id_map[str(raw)]
        else:
            unmatched.add(stem or str(raw))
            continue
        out.append({**d, "image_id": new_id})

    if unmatched:
        print(f"[WARN] {len(unmatched)} image ids in predictions had no GT match, e.g. {list(unmatched)[:3]}")
    if not out:
        sys.exit("[ERROR] no predictions could be matched to the ground truth — id mapping is wrong")

    out_path.write_text(json.dumps(out))
    print(f"[PRED] detections    : {len(out)}")
    print(f"[PRED] written       : {out_path}")
    return out_path


def gt_as_predictions(gt_path: Path, out_path: Path) -> Path:
    """Self-test fixture: feed the ground truth back in as perfect detections."""
    gt = json.loads(gt_path.read_text())
    preds = [
        {"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
        for a in gt["annotations"]
    ]
    out_path.write_text(json.dumps(preds))
    return out_path


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(gt_path: Path, pred_path: Path, max_dets: int = 1000, quiet: bool = False) -> dict[str, float]:
    """Run COCOeval and return the 12 standard metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):  # COCO() is noisy on load
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))

    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Default maxDets is [1, 10, 100]. VisDrone scenes routinely exceed 100
    # objects, so leaving this at the default silently caps recall.
    e.params.maxDets = [1, 10, max_dets]

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate()
            e.accumulate()
            e.summarize()
    else:
        e.evaluate()
        e.accumulate()
        e.summarize()

    stats = list(e.stats)

    # pycocotools hardcodes maxDets=100 when computing stats[0] (overall AP).
    # Because we raise params.maxDets to [1, 10, max_dets], 100 is no longer in
    # the list, the lookup fails, and stats[0] silently becomes -1. Recompute it
    # from the precision array at our own maxDets index instead.
    #   precision shape = [IoU, recall, class, area, maxDets]
    if stats[0] < 0:
        prec = e.eval["precision"][:, :, :, 0, 2]  # area=all, maxDets index 2
        stats[0] = float(np.mean(prec[prec > -1])) if (prec > -1).any() else float("nan")

    keys = [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR_1", "AR_10", f"AR_{max_dets}", "AR_small", "AR_medium", "AR_large",
    ]
    return {k: float(v) for k, v in zip(keys, stats)}


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--out-dir", default="results/coco")
    ap.add_argument("--pred", default=None, help="path to a run's predictions.json")
    ap.add_argument("--max-dets", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true", help="feed GT back as predictions; AP must be 1.000")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    gt_path, id_map = build_gt(args.split, Path(args.root), out_dir)

    if args.selftest:
        print("\n" + "=" * 70)
        print("SELF-TEST — ground truth evaluated against itself")
        print("=" * 70)
        pred_path = gt_as_predictions(gt_path, out_dir / f"visdrone_{args.split}_selftest.json")
        stats = evaluate(gt_path, pred_path, args.max_dets)

        ok = stats["AP"] > 0.999 and stats["AP50"] > 0.999
        print("\n" + "=" * 70)
        if ok:
            print(f"PASS — AP = {stats['AP']:.3f}. ID mapping and box format are correct.")
        else:
            print(f"FAIL — AP = {stats['AP']:.3f}, expected 1.000.")
            print("Do NOT trust any evaluation until this passes.")
        print("=" * 70)
        sys.exit(0 if ok else 1)

    if not args.pred:
        print("\nGround truth built. Pass --pred <predictions.json> to evaluate a run,")
        print("or --selftest to verify the pipeline first.")
        return

    pred_path = remap_predictions(
        Path(args.pred), id_map, out_dir / f"visdrone_{args.split}_pred_remapped.json"
    )
    print()
    stats = evaluate(gt_path, pred_path, args.max_dets)

    print("\n" + "=" * 70)
    print("KEY METRICS")
    print("=" * 70)
    for k in ("AP", "AP50", "AP_small", "AP_medium", "AP_large"):
        print(f"  {k:12s}: {stats[k]:.4f}")

    res_path = Path(args.out_dir) / f"metrics_{Path(args.pred).parent.name}_{args.split}.json"
    res_path.write_text(json.dumps(stats, indent=2))
    print(f"\nSaved: {res_path}")


if __name__ == "__main__":
    main()
