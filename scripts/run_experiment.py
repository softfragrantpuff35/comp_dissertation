"""
run_experiment.py — unified training + evaluation entry point
=============================================================

Every run in this study goes through this script. Hardcoding the shared
settings here rather than passing them on the command line each time is
deliberate: plan §6 requires that every configuration is trained and evaluated
under identical conditions, and the surest way to guarantee that is to give the
caller only two knobs — which model, and which seed.

Pipeline (plan §9):
    train -> validate -> select checkpoint -> export predictions
          -> COCO evaluation -> scale-stratified -> density-stratified
          -> per-class -> latency -> result logging

USAGE
-----
    # smoke test: tiny run to prove the pipeline end to end
    python scripts/run_experiment.py --model yolo26n --seed 0 --smoke

    # a real run
    python scripts/run_experiment.py --model yolo26n --seed 0
    python scripts/run_experiment.py --model yolov10n --seed 1

    # evaluate an existing run without retraining
    python scripts/run_experiment.py --model yolo26n --seed 0 --eval-only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import DATASET_ROOT, VISDRONE_NAMES, build_gt, evaluate, remap_predictions  # noqa: E402
from density_analysis import evaluate_bins, load_counts, make_bins  # noqa: E402
from weight_remap import load_pretrained_shifted, verify_transfer  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen experimental configuration — plan §6.
# Changing anything here invalidates comparability with completed runs.
# ---------------------------------------------------------------------------
CONFIG = {
    "data": "VisDrone.yaml",
    "imgsz": 640,
    "batch": 12,   # reduced from 16: 8GB VRAM peaked at 8.49G on dense batches
    "epochs": 60,
    "patience": 15,
    "max_det": 1000,  # plan §12 item 1 — measured max is 317 objects/image
    "amp": True,
    "split": "val",
    "workers": 8,
    "exist_ok": False,  # never silently overwrite a completed run
}

SMOKE_OVERRIDES = {"epochs": 3, "batch": 8, "fraction": 0.05}

RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs/detect")


def set_seeds(seed: int) -> None:
    """Seed every RNG that can affect a run. Ultralytics also seeds internally."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    """Record the exact code state that produced a result."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def measure_latency(model, imgsz: int, warmup: int = 50, iters: int = 200) -> dict[str, float]:
    """Single-image latency, plan §8.5.

    Batch size 1, warm-up before timing, explicit synchronisation. Reported
    separately from the throughput figure ultralytics prints during validation,
    which uses batched inference and is not comparable.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.rand(1, 3, imgsz, imgsz, device=device)
    net = model.model.float().eval().to(device)

    with torch.no_grad():
        for _ in range(warmup):
            net(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            net(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    return {
        "latency_ms_mean": float(t.mean()),
        "latency_ms_std": float(t.std()),
        "latency_ms_p50": float(np.percentile(t, 50)),
        "latency_ms_p95": float(np.percentile(t, 95)),
        "fps": float(1000 / t.mean()),
        "precision": "fp32",
        "batch_size": 1,
        "warmup_iters": warmup,
        "timed_iters": iters,
    }


def per_class_ap(gt_path: Path, pred_path: Path, max_det: int) -> dict[str, float]:
    """AP for each of the ten VisDrone classes, plan §8.3."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        e = COCOeval(coco_gt, coco_dt, iouType="bbox")
        e.params.maxDets = [1, 10, max_det]
        e.evaluate()
        e.accumulate()

    # precision shape = [IoU, recall, class, area, maxDets]
    prec = e.eval["precision"]
    out = {}
    for i, name in enumerate(VISDRONE_NAMES):
        p = prec[:, :, i, 0, 2]
        out[name] = float(np.mean(p[p > -1])) if (p > -1).any() else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="yolo26n | yolov10n | a path to a .yaml or .pt")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--weights", default=None, help="override the initial weights (plan §7.3)")
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. cbam")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    ap.add_argument("--eval-only", action="store_true", help="skip training, evaluate an existing run")
    ap.add_argument("--insert-at", type=int, default=11,
                    help="index at which a module was inserted, for weight remapping")
    args = ap.parse_args()

    from ultralytics import YOLO

    # Custom attention modules must be resolvable before any YAML naming them
    # is parsed. Harmless for the unmodified baselines.
    from attention import register as register_attention

    register_attention()

    cfg = dict(CONFIG)
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES)

    base = Path(args.model).stem
    run_name = f"{base}{'_' + args.tag if args.tag else ''}_seed{args.seed}{'_smoke' if args.smoke else ''}"
    run_dir = RUNS_DIR / run_name

    print("=" * 70)
    print(f"RUN: {run_name}")
    print("=" * 70)
    for k, v in cfg.items():
        print(f"  {k:10s}: {v}")
    print(f"  {'seed':10s}: {args.seed}")
    print(f"  {'commit':10s}: {git_commit()}")
    print("=" * 70 + "\n")

    if run_dir.exists() and not args.eval_only:
        sys.exit(
            f"[ERROR] {run_dir} already exists.\n"
            "Runs are never overwritten (plan §6). Delete it deliberately, or use --eval-only."
        )

    set_seeds(args.seed)
    pretrained_summary = None

    # ---- train ------------------------------------------------------------
    if args.eval_only:
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            sys.exit(f"[ERROR] no checkpoint at {best}")
        model = YOLO(str(best))
        train_seconds = None
    else:
        # When --model names a YAML the model MUST be built from that YAML;
        # --weights then supplies initial weights on top of it via partial
        # loading (plan §7.2). Loading the .pt directly would silently discard
        # the architecture change and produce a baseline run wearing the
        # modified configuration's name.
        if args.model.endswith((".yaml", ".yml")):
            model = YOLO(args.model)
            if args.weights:
                # Ultralytics matches pretrained weights by parameter name, and
                # those names encode layer position. Inserting a module renames
                # every later layer, so a plain load() silently drops the whole
                # neck and detection head (measured: 240/711 transferred, head
                # 0/240). Remap the checkpoint's indices first.
                # P2 is not a single-module insertion: it restructures the head
                # (four detection scales instead of three, different channel
                # widths downstream). Its backbone indices align with the
                # checkpoint's, so no shift applies; the layers that do differ
                # genuinely have no pretrained counterpart.
                n_shift = 0 if "p2" in Path(args.model).stem else 1
                summary = load_pretrained_shifted(
                    model, args.weights, insert_at=args.insert_at, n_inserted=n_shift
                )
                # P2 restructures the head rather than inserting a module, so
                # tensors legitimately differ in shape as well as being absent.
                # The remapping check assumes the insertion case and does not
                # apply; the transfer count is asserted below instead.
                if n_shift:
                    verify_transfer(summary, expected_new_tensors=summary["no_counterpart"])
                if summary["transferred"] < 350:
                    sys.exit(
                        f"[ERROR] only {summary['transferred']} tensors transferred; "
                        "expected ~708. Do not train on this model."
                    )
                pretrained_summary = summary

                # engine/model.py:822 rebuilds the network from YAML at the
                # start of train(), passing weights only when self.ckpt is set.
                # A YAML-built model has no ckpt, so everything loaded above
                # would be discarded and training would start from scratch --
                # silently, and indistinguishably from the module being
                # ineffective. Serialising to a checkpoint makes the modified
                # model enter training through exactly the same path as the
                # unmodified baselines.
                init_ckpt = Path("runs/_init") / f"{run_name}.pt"
                init_ckpt.parent.mkdir(parents=True, exist_ok=True)
                net = model.model
                net.args = {}
                torch.save(
                    {"model": net.half(), "date": "", "version": "",
                     "train_args": {}, "epoch": -1, "best_fitness": None},
                    init_ckpt,
                )
                model = YOLO(str(init_ckpt))
                if not model.ckpt:
                    sys.exit("[ERROR] checkpoint round-trip failed; weights would be lost in train()")
                print(f"[weights] serialised initialised model to {init_ckpt}")
        else:
            model = YOLO(args.weights or f"{args.model}.pt")
        t0 = time.time()
        model.train(
            data=cfg["data"],
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            patience=cfg["patience"],
            seed=args.seed,
            amp=cfg["amp"],
            workers=cfg["workers"],
            max_det=cfg["max_det"],
            name=run_name,
            exist_ok=cfg["exist_ok"],
            deterministic=True,
            **({"fraction": cfg["fraction"]} if "fraction" in cfg else {}),
        )
        train_seconds = time.time() - t0
        # train() and val() resolve `project` differently from one another and
        # from the global runs_dir setting; read the path actually used rather
        # than reconstructing it.
        run_dir = Path(model.trainer.save_dir)
        model = YOLO(str(run_dir / "weights" / "best.pt"))

    # ---- validate and export predictions ----------------------------------
    print("\n[eval] running validation with save_json")
    metrics = model.val(
        data=cfg["data"],
        split=cfg["split"],
        imgsz=cfg["imgsz"],
        max_det=cfg["max_det"],
        save_json=True,
        name=f"{run_name}_val",
        exist_ok=True,
    )
    pred_json = Path(metrics.save_dir) / "predictions.json"
    if not pred_json.exists():
        sys.exit(f"[ERROR] predictions.json not produced at {pred_json}")

    # ---- COCO stratified evaluation ---------------------------------------
    out_dir = RESULTS_DIR / "coco"
    gt_path, id_map = build_gt(cfg["split"], DATASET_ROOT, out_dir)
    pred_remapped = remap_predictions(pred_json, id_map, out_dir / f"{run_name}_pred.json")

    print("\n[eval] COCO scale-stratified")
    coco_stats = evaluate(gt_path, pred_remapped, cfg["max_det"], quiet=True)
    for k in ("AP", "AP50", "AP_small", "AP_medium", "AP_large"):
        print(f"  {k:10s}: {coco_stats[k]:.4f}")

    print("\n[eval] density-stratified")
    counts, _ = load_counts(gt_path)
    arr = np.array(sorted(counts.values()))
    with contextlib.redirect_stdout(io.StringIO()):
        bins = make_bins(counts, arr)
    density = evaluate_bins(gt_path, pred_remapped, bins, cfg["max_det"])

    print("[eval] per-class")
    per_class = per_class_ap(gt_path, pred_remapped, cfg["max_det"])
    for name, v in per_class.items():
        print(f"  {name:18s}: {v:.4f}")

    print("\n[eval] latency (batch=1)")
    lat = measure_latency(model, cfg["imgsz"])
    print(f"  {lat['latency_ms_mean']:.2f} +/- {lat['latency_ms_std']:.2f} ms  ({lat['fps']:.1f} FPS)")

    # ---- record -----------------------------------------------------------
    n_params = sum(p.numel() for p in model.model.parameters())
    record = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "tag": args.tag,
        "seed": args.seed,
        "smoke": args.smoke,
        "config": cfg,
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "train_seconds": train_seconds,
        "epochs_completed": getattr(getattr(model, "trainer", None), "epoch", None),
        "params": n_params,
        "pretrained_transfer": pretrained_summary,
        "ultralytics_metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        },
        "coco": coco_stats,
        "density": density,
        "per_class_AP": per_class,
        "latency": lat,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rec_path = RESULTS_DIR / f"{run_name}.json"
    if rec_path.exists():
        rec_path = RESULTS_DIR / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    rec_path.write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 70)
    print(f"DONE — {rec_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
