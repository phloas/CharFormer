# CharFormer

Source codes for CharFormer


### Running Procedures:

- train CharFormer model

> python train.py --dataset word --epochs 90 --mask local --recall --sub

- optional arguments:

| -dataset | dataset name which is under folder ./folder/  |
|---|---|
| --nt  | # of training samples  |
| --nq  | # of query items |
| --nv  | # of valid items  |
| --epochs  | # number of epochs  |
| --save-split  | save split data folder  |
| --recall  | print recall  |
| --sub  | sub-string  |
| --mask | # mask strategies  |
