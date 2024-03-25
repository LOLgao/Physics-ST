# Physics-ST
Source code for paper "Physics-Informed Spatio-Temporal Model for Human Mobility Prediction".
## Requirements
We build this project by Python 3.8 with the following packages:

```
torch==1.12.0
numpy==1.23.3
```
## Generate Dataset
```
python generate_datasets.py
```
## Model training and Evaluation
```
python Run.py --data_dir='data/ode_data/D1' --lr_init=0.003 --ms=4 --weights=0.1 --batch_size=16
```
## Acknowledgement
This repo is modified from [AGCRN](https://github.com/LeiBAI/AGCRN) and [ST-SSL](https://github.com/Echo-Ji/ST-SSL).
