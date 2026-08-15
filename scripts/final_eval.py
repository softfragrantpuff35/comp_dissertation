"""
final_eval.py — held-out evaluation and controlled latency benchmarking
=======================================================================

Two corrections that need existing checkpoints but no retraining.

1. TEST-DEV EVALUATION (examiner Q13)
   Checkpoints were selected on validation fitness and results were reported on
   the same split. The optimism this introduces is modest but avoidable. This
   evaluates every completed run on VisDrone test-dev, held out throughout.

   The density bin boundaries are those derived from val and are applied
   UNCHANGED (examiner Q44). Re-deriving tertiles on test-dev would make "high
   density" denote a different absolute object count on the two splits, and the
   two would not be comparable. Unbalanced bins under a fixed definition are
   preferable to balanced bins under a moving one.

2. CONTROLLED LATENCY (examiner Q21, Q24)
   Latency measured during the original run queue varied by up to 91% between
   seeds of the same configuration — thermal drift across an eight-hour session,
   not model behaviour. Those numbers are unusable.

   This re-measures all checkpoints in one session in RANDOMISED ORDER, so any
   residual thermal drift is spread across configurations rather than confounded
   with them, and reports forward-pass and end-to-end latency separately.

PRE-REGISTRATION (examiner Q55)
-------------------------------
The claims to be checked on test-dev are fixed before it is run: the mAP50,
mAP50-95 and AP75 differences against the val-derived variance band; the sign of
the per-seed paired differences; the per-class pattern; and the scale profile.
No new thresholds or subgroup definitions are to be introduced after seeing the
result.

USAGE
-----
    python scripts/final_eval.py --latency        # ~30 min, run on a cold GPU
    python scripts/final_eval.py --testdev        # ~1 h
    python scripts/final_eval.py --latency --testdev
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import DATASET_ROOT, VISDRONE_NAMES, build_gt, evaluate, remap_predictions  # noqa: E402
from density_analysis import evaluate_bins  # noqa: E402

RUNS = Path("runs/detect")
RESULTS = Path("results")
IMGSZ, MAX_DET = 640, 1000

# Frozen from the val split. Applied unchanged to test-dev — see module docstring.
DENSITY_EDGES = (48.0, 83.0)


def completed_runs() -> list[Path]:
    """Every run directory holding a usable best.pt, excluding smoke tests."""
    out = []
    for d in sorted(RUNS.glob("*/")):
        if d.name.endswith("_val") or "smoke" in d.name or d.name.startswith("_"):
            continue
        if (d / "weights" / "best.pt").exists():
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def measure(model, warmup: int = 50, iters: int = 200) -> dict:
    """Forward-pass and end-to-end latency at batch size 1, FP32."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = model.model.float().eval().to(dev)
    dummy = torch.rand(1, 3, IMGSZ, IMGSZ, device=dev)
    img = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)

    def timed(fn, n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return np.array(ts)

    with torch.no_grad():
        timed(lambda: net(dummy), warmup)
        fwd = timed(lambda: net(dummy), iters)

    # End-to-end includes preprocessing, decoding and the max_det cap, which is
    # part of deployment cost and is not captured by the forward pass alone.
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(10):
            model.predict(img, imgsz=IMGSZ, max_det=MAX_DET, verbose=False)
        e2e = timed(
            lambda: model.predict(img, imgsz=IMGSZ, max_det=MAX_DET, verbose=False),
            max(50, iters // 4),
        )

    temp = None
    try:
        import subprocess
        temp = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0])
    except Exception:
        pass

    f = lambda t: {"mean": float(t.mean()), "std": float(t.std()),
                   "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
                   "fps": float(1000 / t.mean())}
    return {"forward": f(fwd), "end_to_end": f(e2e), "gpu_temp_c": temp,
            "precision": "fp32", "batch_size": 1,
            "warmup_iters": warmup, "timed_iters": iters}


def run_latency(runs: list[Path]) -> dict:
    from ultralytics import YOLO
    from attention import register as reg
    reg()

    order = list(runs)
    random.Random(0).shuffle(order)   # decorrelate thermal drift from configuration
    print("=" * 84)
    print(f"LATENCY — {len(order)} checkpoints, randomised order, single session")
    print("=" * 84)
    print(f"{'run':32s}{'fwd_ms':>9s}{'fwd_std':>9s}{'e2e_ms':>9s}{'fps_e2e':>9s}{'temp':>7s}")
    print("-" * 84)

    out = {}
    for d in order:
        m = YOLO(str(d / "weights" / "best.pt"))
        r = measure(m)
        out[d.name] = r
        print(f"{d.name:32s}{r['forward']['mean']:9.2f}{r['forward']['std']:9.2f}"
              f"{r['end_to_end']['mean']:9.2f}{r['end_to_end']['fps']:9.1f}"
              f"{str(r['gpu_temp_c'] or '-'):>7s}")
        del m
        torch.cuda.empty_cache()

    temps = [v["gpu_temp_c"] for v in out.values() if v["gpu_temp_c"]]
    if temps:
        print("-" * 84)
        print(f"  GPU temperature over the session: {min(temps)}–{max(temps)} °C")
        if max(temps) - min(temps) > 15:
            print("  [WARN] wide temperature range; randomised order spreads the effect")
            print("         across configurations but the residual should be reported.")
    return out


# ---------------------------------------------------------------------------
# Test-dev
# ---------------------------------------------------------------------------
def fixed_bins(gt_path: Path) -> dict:
    """Density bins at the frozen val-derived absolute edges."""
    gt = json.loads(gt_path.read_text())
    counts = {im["id"]: 0 for im in gt["images"]}
    for a in gt["annotations"]:
        counts[a["image_id"]] += 1
    lo, hi = DENSITY_EDGES
    bins = {"low": {"range": [0, lo], "image_ids": []},
            "medium": {"range": [lo, hi], "image_ids": []},
            "high": {"range": [hi, 1e9], "image_ids": []}}
    for i, n in counts.items():
        bins["low" if n <= lo else ("medium" if n <= hi else "high")]["image_ids"].append(i)
    print(f"  density bins at frozen val edges (<={lo:.0f}, {lo:.0f}-{hi:.0f}, >{hi:.0f}): "
          + ", ".join(f"{k} {len(v['image_ids'])}" for k, v in bins.items()))
    return bins


def run_testdev(runs: list[Path]) -> dict:
    from ultralytics import YOLO
    from attention import register as reg
    reg()

    print("\n" + "=" * 84)
    print(f"TEST-DEV EVALUATION — {len(runs)} checkpoints")
    print("=" * 84)

    out_dir = RESULTS / "coco"
    gt_path, id_map = build_gt("test", DATASET_ROOT, out_dir)
    bins = fixed_bins(gt_path)
    print()

    out = {}
    for d in runs:
        print(f"  {d.name} ...", end=" ", flush=True)
        m = YOLO(str(d / "weights" / "best.pt"))
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = m.val(data="VisDrone.yaml", split="test", imgsz=IMGSZ,
                            max_det=MAX_DET, save_json=True,
                            name=f"{d.name}_testdev", exist_ok=True)
            pred = Path(metrics.save_dir) / "predictions.json"
            remapped = remap_predictions(pred, id_map, out_dir / f"{d.name}_testdev_pred.json")
            coco = evaluate(gt_path, remapped, MAX_DET, quiet=True)
            dens = evaluate_bins(gt_path, remapped, bins, MAX_DET)
        out[d.name] = {
            "ultralytics": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)},
            "coco": coco, "density": dens,
        }
        print(f"mAP50 {metrics.box.map50:.4f}  mAP50-95 {metrics.box.map:.4f}  "
              f"AP75 {coco['AP75']:.4f}")
        del m
        torch.cuda.empty_cache()
    return out


def compare(res: dict) -> None:
    """Check the pre-registered claims against the test-dev results."""
    def grp(prefix):
        v = [res[k] for k in res if k.startswith(prefix) and k.endswith(("seed0", "seed1", "seed2"))]
        return v

    a, b = grp("yolov10n_seed"), grp("yolo26n_seed")
    if len(a) != 3 or len(b) != 3:
        print("\n  baselines not all present; skipping the pre-registered comparison")
        return

    print("\n" + "=" * 84)
    print("PRE-REGISTERED CHECKS ON TEST-DEV")
    print("=" * 84)
    print(f"{'metric':12s}{'YOLOv10n':>18s}{'YOLO26n':>18s}{'diff':>10s}{'2sd':>9s}")
    for label, path in (("mAP50", ("ultralytics", "mAP50")),
                        ("mAP50-95", ("ultralytics", "mAP50_95")),
                        ("AP75", ("coco", "AP75")),
                        ("AP_small", ("coco", "AP_small")),
                        ("AP_medium", ("coco", "AP_medium"))):
        x = np.array([r[path[0]][path[1]] for r in a])
        y = np.array([r[path[0]][path[1]] for r in b])
        sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        print(f"{label:12s}{x.mean():11.4f}±{x.std(ddof=1):6.4f}"
              f"{y.mean():11.4f}±{y.std(ddof=1):6.4f}{y.mean() - x.mean():+10.4f}{2 * sd:9.4f}")

    paired = [b[i]["ultralytics"]["mAP50_95"] - a[i]["ultralytics"]["mAP50_95"] for i in range(3)]
    print(f"\n  paired per-seed differences (26n − v10n): "
          + ", ".join(f"{p:+.4f}" for p in paired))
    print(f"  all same sign: {all(p < 0 for p in paired) or all(p > 0 for p in paired)}")
    print("\n  Compare against the val-derived conclusion. Do not introduce new")
    print("  thresholds or subgroups now.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--testdev", action="store_true")
    args = ap.parse_args()
    if not (args.latency or args.testdev):
        ap.error("choose --latency and/or --testdev")

    runs = completed_runs()
    if not runs:
        sys.exit("[ERROR] no completed runs found under runs/detect")
    print(f"found {len(runs)} completed runs\n")

    out = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_runs": len(runs)}
    if args.latency:
        out["latency"] = run_latency(runs)
    if args.testdev:
        out["testdev"] = run_testdev(runs)
        compare(out["testdev"])

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"final_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {p}")
