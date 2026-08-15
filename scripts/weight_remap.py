"""
weight_remap.py — index-aware pretrained weight transfer
=========================================================

THE PROBLEM
-----------
Ultralytics transfers pretrained weights by matching parameter *names*:

    intersect_dicts(checkpoint_state_dict, model_state_dict)

Parameter names encode the layer's position in the network, e.g.
`model.13.cv1.conv.weight`. Inserting a module at index 11 renumbers every
later layer, so `model.13.*` in the model corresponds to `model.12.*` in the
checkpoint. None of those names match, and the entire neck and detection head
are silently left at random initialisation.

Measured on this project's configuration: inserting CBAM at index 11 reduced
the transfer from 708/708 to 240/711, with every layer from 11 onward matching
zero tensors — including all 240 tensors of the detection head.

Nothing raises. Training proceeds and converges to a plausible-looking number.
The resulting run is not "baseline plus attention"; it is "backbone pretrained,
everything downstream from scratch". Compared against a fully pretrained
baseline it would show a large deficit, and that deficit is indistinguishable
from the attention module being harmful. Left unfixed it would have produced a
confident and completely wrong answer to RQ2.

THE FIX
-------
Shift the layer indices in the checkpoint's parameter names to match the
modified architecture before transferring. Layers before the insertion point
keep their names; layers at or after it are incremented by the number of
modules inserted.

After remapping, transfer should account for every tensor except those
belonging to the newly inserted module itself, which has no pretrained
counterpart and must be randomly initialised. That residue is the intended
experimental condition (plan §7.2), and `verify_transfer` reports it explicitly
rather than leaving it to be assumed.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

_LAYER_RE = re.compile(r"^(model\.)(\d+)(\..*)$")


def shift_state_dict(state: dict[str, torch.Tensor], insert_at: int, n_inserted: int = 1) -> dict:
    """Renumber `model.<i>.*` keys so index i >= insert_at becomes i + n_inserted."""
    out = {}
    for k, v in state.items():
        m = _LAYER_RE.match(k)
        if m:
            idx = int(m.group(2))
            if idx >= insert_at:
                k = f"{m.group(1)}{idx + n_inserted}{m.group(3)}"
        out[k] = v
    return out


def _checkpoint_state(weights: str | Path) -> dict[str, torch.Tensor]:
    from ultralytics.nn.tasks import load_checkpoint

    model, _ = load_checkpoint(str(weights))
    return model.float().state_dict()


def load_pretrained_shifted(
    model, weights: str | Path, insert_at: int, n_inserted: int = 1, verbose: bool = True
) -> dict[str, int]:
    """Transfer pretrained weights into a model whose layers were renumbered.

    `model` is an ultralytics YOLO wrapper. Returns a summary of what was
    transferred so the caller can record it and assert on it.
    """
    net = model.model if hasattr(model, "model") else model
    tgt = net.state_dict()
    src = shift_state_dict(_checkpoint_state(weights), insert_at, n_inserted)

    transferred, skipped_shape, absent = {}, [], []
    for k, v in tgt.items():
        if k not in src:
            absent.append(k)
        elif src[k].shape != v.shape:
            skipped_shape.append(k)
        else:
            transferred[k] = src[k]

    net.load_state_dict(transferred, strict=False)

    summary = {
        "total": len(tgt),
        "transferred": len(transferred),
        "shape_mismatch": len(skipped_shape),
        "no_counterpart": len(absent),
    }
    if verbose:
        print(
            f"[weights] transferred {summary['transferred']}/{summary['total']} tensors "
            f"(shape mismatch {summary['shape_mismatch']}, no counterpart {summary['no_counterpart']})"
        )
        # The detection head is the part most likely to be silently dropped, so
        # name it explicitly rather than trusting the aggregate count.
        head = [k for k in tgt if k.startswith(f"model.{max(int(m.group(2)) for m in (_LAYER_RE.match(x) for x in tgt) if m)}.")]
        got = sum(1 for k in head if k in transferred)
        print(f"[weights] detection head: {got}/{len(head)} tensors transferred")
        if absent and len(absent) <= 12:
            print(f"[weights] no counterpart: {absent}")
    return summary


def verify_transfer(summary: dict[str, int], expected_new_tensors: int, tolerance: int = 0) -> None:
    """Fail loudly if the transfer did not account for everything it should.

    Only the freshly inserted module may lack a pretrained counterpart. Anything
    beyond that indicates the remapping is wrong, which is exactly the failure
    this module exists to prevent.
    """
    unexplained = summary["no_counterpart"] + summary["shape_mismatch"] - expected_new_tensors
    if unexplained > tolerance:
        raise RuntimeError(
            f"Weight transfer left {unexplained} tensors unexplained "
            f"(no counterpart {summary['no_counterpart']}, shape mismatch {summary['shape_mismatch']}, "
            f"expected {expected_new_tensors} from the inserted module). "
            f"The index remapping is incorrect - do not train on this model."
        )
