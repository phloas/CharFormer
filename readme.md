# CharFormer

Source codes for CharFormer


### Running Procedures:

1. Unzip the preprocess folder, you can get the word dataset and Uniref dataset.

2. Run 'train.py' to train the model.

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