@echo off
call conda activate yolov10
cd /d C:\comp_dissertation

python scripts\run_experiment.py --model configs\yolo26-cbam.yaml --weights yolo26n.pt --tag cbam --seed 0
python scripts\run_experiment.py --model configs\yolo26-cbam.yaml --weights yolo26n.pt --tag cbam --seed 1
python scripts\run_experiment.py --model configs\yolo26-cbam.yaml --weights yolo26n.pt --tag cbam --seed 2
python scripts\run_experiment.py --model configs\yolo26-ca.yaml --weights yolo26n.pt --tag ca --seed 0
python scripts\run_experiment.py --model configs\yolo26-ca.yaml --weights yolo26n.pt --tag ca --seed 1
python scripts\run_experiment.py --model configs\yolo26-ca.yaml --weights yolo26n.pt --tag ca --seed 2

echo.
echo ALL DONE
pause