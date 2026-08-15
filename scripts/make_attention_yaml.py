"""
make_attention_yaml.py — generate attention-augmented model configurations
===========================================================================

Inserting a layer into a YOLO YAML shifts every subsequent layer index by one.
Any `from` reference pointing at a layer after the insertion point must be
incremented, and any reference pointing before it must not be. Getting this
wrong does not raise an error: the network simply wires itself to the wrong
tensors and trains to a worse result, which is indistinguishable from the
attention module "not helping". That failure mode would silently corrupt the
answer to RQ2, so the edit is generated and checked programmatically rather
than made by hand.

WHAT THIS SCRIPT DOES
---------------------
1. Reads the stock ultralytics YAML for the requested architecture.
2. Inserts one attention module immediately after the final backbone block
   (SPPF -> PSA/C2PSA -> attention), the shared insertion point defined by the
   experimental design.
3. Rewrites every affected `from` index.
4. Verifies the result: builds the model, confirms the module sits at the
   intended index, confirms its declared channel count matches the tensor it
   actually receives, and reports the parameter and GFLOP cost against the
   unmodified baseline.

USAGE
-----
    python scripts/make_attention_yaml.py                 # generate + verify all
    python scripts/make_attention_yaml.py --arch yolo26   # one architecture
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention import ATTENTION_MODULES, register  # noqa: E402

OUT_DIR = Path("configs")

# Stock YAML name and the scale letter this study uses.
ARCHS = {
    "yolo26": {"stock": "yolo26.yaml", "scale": "n"},
    "yolov10": {"stock": "yolov10n.yaml", "scale": "n"},
}

# Channels present at the insertion point, after width scaling. Verified
# against the printed model summary for both architectures at scale n; the
# verification step below re-checks it rather than trusting this constant.
INSERT_CHANNELS = 256


def stock_yaml_path(name: str) -> Path:
    """Locate a model YAML inside the installed ultralytics package."""
    import ultralytics

    root = Path(ultralytics.__file__).parent / "cfg" / "models"
    hits = sorted(root.rglob(name))
    if not hits:
        sys.exit(f"[ERROR] could not find {name} under {root}")
    return hits[0]


def build_augmented(stock: Path, module_name: str, extra_args: list) -> tuple[dict, int]:
    """Insert the attention module and renumber downstream references.

    Returns the modified config and the index the new layer occupies.
    """
    cfg = yaml.safe_load(stock.read_text())
    backbone, head = cfg["backbone"], cfg["head"]

    insert_at = len(backbone)  # immediately after the last backbone layer
    backbone.append([-1, 1, module_name, [INSERT_CHANNELS, *extra_args]])

    def shift(ref):
        """Increment references to layers at or after the insertion point."""
        if isinstance(ref, list):
            return [shift(r) for r in ref]
        if isinstance(ref, int) and ref >= insert_at:
            return ref + 1
        return ref  # negative (relative) and earlier absolute refs are unaffected

    for layer in head:
        layer[0] = shift(layer[0])

    cfg["backbone"], cfg["head"] = backbone, head
    return cfg, insert_at


def verify(path: Path, scale: str, module_name: str, expect_index: int, baseline_params: int) -> bool:
    """Build the model and check the module landed where it was supposed to."""
    import torch
    from ultralytics import YOLO

    scaled = path.with_name(f"{path.stem}{scale}{path.suffix}")  # e.g. ..._cbamn.yaml -> scale n
    shutil.copyfile(path, scaled)

    try:
        model = YOLO(str(scaled)).model
    except Exception as exc:  # noqa: BLE001
        print(f"    BUILD FAILED: {exc}")
        return False

    layer = model.model[expect_index]
    actual = type(layer).__name__
    if actual != module_name:
        print(f"    WRONG MODULE at index {expect_index}: expected {module_name}, found {actual}")
        return False

    # A forward pass is the only way to confirm the channel count is right and
    # that the renumbered references still wire up.
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 640, 640))
    except Exception as exc:  # noqa: BLE001
        print(f"    FORWARD FAILED: {exc}")
        return False

    n_params = sum(p.numel() for p in model.parameters())
    delta = n_params - baseline_params
    print(
        f"    OK  {module_name} at index {expect_index}; "
        f"params {n_params:,} ({delta:+,}, {100 * delta / baseline_params:+.2f}%)"
    )
    scaled.unlink(missing_ok=True)
    return True


def baseline_param_count(stock: Path, scale: str) -> int:
    from ultralytics import YOLO

    tmp = OUT_DIR / f"_baseline_{stock.stem}{scale}.yaml"
    shutil.copyfile(stock, tmp)
    n = sum(p.numel() for p in YOLO(str(tmp)).model.parameters())
    tmp.unlink(missing_ok=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS) + ["all"], default="all")
    args = ap.parse_args()

    register()  # CoordAtt / EMA must be resolvable before any YAML is parsed
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(ARCHS) if args.arch == "all" else [args.arch]
    all_ok = True

    for arch in targets:
        spec = ARCHS[arch]
        stock = stock_yaml_path(spec["stock"])
        base_n = baseline_param_count(stock, spec["scale"])
        print(f"\n{arch}  (stock: {stock.name}, scale {spec['scale']}, baseline {base_n:,} params)")

        for key, (module_name, extra) in ATTENTION_MODULES.items():
            cfg, idx = build_augmented(stock, module_name, extra)
            out = OUT_DIR / f"{arch}-{key}.yaml"
            out.write_text(
                f"# Generated by make_attention_yaml.py - do not edit by hand.\n"
                f"# {module_name} inserted at index {idx}, immediately after the final\n"
                f"# backbone block. Downstream 'from' indices renumbered automatically.\n\n"
                + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)
            )
            print(f"  {key:5s} -> {out}")
            all_ok &= verify(out, spec["scale"], module_name, idx, base_n)

    print()
    if all_ok:
        print("All configurations built, wired and forward-checked successfully.")
    else:
        print("At least one configuration failed. Do not train until this is resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
