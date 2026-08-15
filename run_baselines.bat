@echo off
call conda activate yolov10
cd /d C:\comp_dissertation

python scripts\run_experiment.py --model yolo26n --seed 1
python scripts\run_experiment.py --model yolo26n --seed 2
python scripts\run_experiment.py --model yolov10n --seed 0
python scripts\run_experiment.py --model yolov10n --seed 1
python scripts\run_experiment.py --model yolov10n --seed 2

echo.
echo ALL BASELINE RUNS COMPLETE
pause