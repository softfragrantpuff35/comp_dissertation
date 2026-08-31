# Attention modules and NMS-free detectors on dense aerial imagery

Code and results for an MSc dissertation comparing YOLOv10n and YOLO26n on
VisDrone2019-DET, and testing whether three attention modules produce a change
exceeding between-seed variability.

School of Computer Science, University of Leeds, 2026.

## What is here

    configs/     model configurations with the attention modules inserted
    scripts/     training, evaluation, collection and figure scripts
    results/     collected metrics, per-run JSON, and the figures
    _orig/       original VisDrone validation annotations (see below)
    *.bat        the batch files used to queue the training runs

## Reproducing the results

Every table and figure in Chapter 4 of the dissertation is produced from the
collected results by a script here.

    python scripts/collect_metrics.py       # aggregate the per-run JSON
    python scripts/collect_iou_sweep.py     # AP at each IoU threshold
    python scripts/collect_class_iou.py     # AP per class per threshold
    python scripts/make_figures.py          # Figures 4.1 to 4.3

Training runs are queued by `run_baselines.bat` and `run_attention.bat`.

## Licence

This project builds on the ultralytics package, which is distributed under the
GNU Affero General Public License v3.0. This repository is therefore
distributed under the same licence. See LICENSE.

## Dataset

VisDrone2019-DET is not redistributed here. It is downloaded automatically by
the training scripts through the framework's dataset configuration. The dataset
is described in:

> Zhu, P., Wen, L., Du, D., Bian, X., Fan, H., Hu, Q. and Ling, H. (2022)
> 'Detection and tracking meet drones challenge', *IEEE Transactions on Pattern
> Analysis and Machine Intelligence*, 44(11).

### The `_orig/` directory

This holds the original VisDrone validation annotations, retained so that the
conversion check reported in Section 3.1 of the dissertation can be reproduced:
twenty images were sampled at random and compared box by box against these
files. The annotations are the property of the dataset's authors and are
included here for that purpose only. The full dataset is available from
https://github.com/VisDrone/VisDrone-Dataset

## Not included

Trained checkpoints and the per-run training artefacts (`runs/`) are not in
version control. The metrics computed from them are in `results/`.
