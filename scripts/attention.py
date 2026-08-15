"""
attention.py — attention modules for the RQ2 comparison
========================================================

CBAM is already provided by ultralytics (`ultralytics.nn.modules.CBAM`), so
only Coordinate Attention and EMA are implemented here, following the original
papers:

  CA   Hou, Zhou & Feng (2021), "Coordinate Attention for Efficient Mobile
       Network Design", CVPR.
  EMA  Ouyang et al. (2023), "Efficient Multi-Scale Attention Module with
       Cross-Spatial Learning", ICASSP.

REGISTRATION
------------
`parse_model` in ultralytics/nn/tasks.py resolves module names from that
module's own global namespace:

    m = getattr(torch.nn, m[3:]) if m.startswith("nn.") else globals()[m]

so a custom module becomes usable in a YAML only once it is injected there.
`register()` performs that injection and must be called before any model is
built from a YAML that names these modules.

CHANNEL ARGUMENT
----------------
None of these three modules appears in parse_model's `base_modules` set. Two
consequences follow, and both matter:

  1. parse_model does NOT prepend the input channel count to the argument list,
     so the channel count must be stated explicitly in the YAML.
  2. parse_model does NOT scale the argument by the model's width multiplier,
     so the value in the YAML is the post-scaling channel count, not the
     nominal one.

At the insertion point used in this study (immediately after the backbone's
final block) both YOLOv10n and YOLO26n carry 256 channels. A mismatch here
would not raise at construction time — it would surface as a shape error deep
in the forward pass, or silently misbehave — so every module below asserts its
channel count on the first forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ChannelChecked(nn.Module):
    """Base class that verifies the declared channel count against real input.

    The YAML author states the channel count by hand (see module docstring).
    Checking it on the first forward turns a silent configuration error into an
    immediate, legible failure.
    """

    c1: int

    def _check(self, x: torch.Tensor) -> None:
        if x.shape[1] != self.c1:
            raise ValueError(
                f"{type(self).__name__} was constructed for {self.c1} channels but received "
                f"{x.shape[1]}. The channel count in the model YAML does not match the network "
                f"at this insertion point."
            )


class CoordAtt(_ChannelChecked):
    """Coordinate Attention (Hou et al., 2021).

    Factorises global pooling into two 1-D poolings, one per spatial axis, so
    that long-range dependencies are captured along one direction while precise
    positional information is retained along the other. The motivation for
    including it here is that positional precision is exactly what the
    baseline comparison identified as the axis of difference between the two
    detectors.
    """

    def __init__(self, c1: int, reduction: int = 32):
        super().__init__()
        self.c1 = c1
        mid = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mid, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, c1, kernel_size=1)
        self.conv_w = nn.Conv2d(mid, c1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        _, _, h, w = x.shape

        x_h = x.mean(dim=3, keepdim=True)                       # (n, c, h, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)   # (n, c, w, 1)

        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        return x * self.conv_h(y_h).sigmoid() * self.conv_w(y_w).sigmoid()


class EMA(_ChannelChecked):
    """Efficient Multi-Scale Attention (Ouyang et al., 2023).

    Splits channels into groups and combines a 1x1 branch carrying directional
    pooled context with a 3x3 branch carrying local multi-scale context, then
    fuses the two by cross-spatial matrix multiplication. Included here because
    its multi-scale interaction is the mechanism most often claimed to help
    small, densely packed objects.
    """

    def __init__(self, c1: int, factor: int = 8):
        super().__init__()
        self.c1 = c1
        self.groups = factor
        cg = c1 // self.groups
        if cg <= 0:
            raise ValueError(f"EMA: {c1} channels cannot be split into {factor} groups")

        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(cg, cg)
        self.conv1x1 = nn.Conv2d(cg, cg, kernel_size=1)
        self.conv3x3 = nn.Conv2d(cg, cg, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        b, c, h, w = x.shape
        g = self.groups
        cg = c // g
        xg = x.reshape(b * g, cg, h, w)

        x_h = self.pool_h(xg)                                   # (bg, cg, h, 1)
        x_w = self.pool_w(xg).permute(0, 1, 3, 2)               # (bg, cg, w, 1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        a_h, a_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(xg * a_h.sigmoid() * a_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(xg)

        w1 = self.softmax(self.agp(x1).reshape(b * g, -1, 1).permute(0, 2, 1))
        w2 = self.softmax(self.agp(x2).reshape(b * g, -1, 1).permute(0, 2, 1))
        weights = (
            torch.matmul(w1, x2.reshape(b * g, cg, -1))
            + torch.matmul(w2, x1.reshape(b * g, cg, -1))
        ).reshape(b * g, 1, h, w)

        return (xg * weights.sigmoid()).reshape(b, c, h, w)


class GatedAttention(nn.Module):
    """Wraps an attention module so that it starts as the identity function.

    WHY THIS EXISTS
    ---------------
    All three modules here are multiplicative gates: their output is the input
    scaled elementwise by one or more sigmoids. At random initialisation those
    sigmoids sit near 0.5, so the module attenuates its input by roughly a
    constant factor before anything has been learned. The layers downstream are
    pretrained and expect the original feature scale, so inserting such a module
    perturbs the pretrained network from the first forward pass.

    That perturbation is confounded with the question being asked. A drop in
    accuracy after inserting an attention module could mean the module is
    unhelpful, or merely that inserting a randomly initialised gate damaged the
    pretrained features. The two are indistinguishable in the standard setup
    used throughout the literature.

    This wrapper separates them:

        out = x + gamma * (attn(x) - x),    gamma initialised to 0

    At initialisation gamma is zero, so the module is exactly the identity and
    the pretrained network is untouched. gamma is a learned parameter, so
    training decides whether the module contributes at all and how strongly.
    Comparing a gated variant against its ungated counterpart isolates the
    module's own effect from the cost of disturbing the initialisation.

    The same construction appears as zero-initialised residual scaling in
    ResNet training, as the zero-initialised projection in LoRA, and as the
    zero convolution in ControlNet.

    The learned gamma is itself informative: a value that stays near zero says
    the network declined to use the module.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.attn = module
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * (self.attn(x) - x)


class CBAMGated(GatedAttention):
    """Identity-initialised CBAM."""

    def __init__(self, c1: int, kernel_size: int = 7):
        from ultralytics.nn.modules import CBAM

        super().__init__(CBAM(c1, kernel_size))


class CoordAttGated(GatedAttention):
    """Identity-initialised Coordinate Attention."""

    def __init__(self, c1: int, reduction: int = 32):
        super().__init__(CoordAtt(c1, reduction))


class EMAGated(GatedAttention):
    """Identity-initialised EMA."""

    def __init__(self, c1: int, factor: int = 8):
        super().__init__(EMA(c1, factor))


def register() -> None:
    """Make every attention module resolvable by name inside parse_model.

    CBAM ships with ultralytics but is not imported into tasks.py's namespace,
    so naming it in a YAML raises KeyError('CBAM') at parse time despite the
    class existing. It therefore needs the same injection as the modules
    defined here.
    """
    from ultralytics.nn import tasks
    from ultralytics.nn.modules import CBAM

    for mod in (CBAM, CoordAtt, EMA, CBAMGated, CoordAttGated, EMAGated):
        setattr(tasks, mod.__name__, mod)


# Names usable in a model YAML once register() has been called. The three
# plain modules follow the standard practice used throughout the literature;
# the three gated variants are the identity-initialised controls that isolate
# the module's own effect from the cost of perturbing the pretrained weights.
ATTENTION_MODULES = {
    "cbam": ("CBAM", [7]),               # CBAM(c1, kernel_size)
    "ca": ("CoordAtt", [32]),            # CoordAtt(c1, reduction)
    "ema": ("EMA", [8]),                 # EMA(c1, factor)
    "cbam-gated": ("CBAMGated", [7]),
    "ca-gated": ("CoordAttGated", [32]),
    "ema-gated": ("EMAGated", [8]),
}
