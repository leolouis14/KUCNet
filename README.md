# KUCNet
The code for our paper "Knowledge-Enhanced Recommendation with User-Centric Subgraph Network".



## Instructions

A quick instruction is given for readers to reproduce the whole process.



## Environment Requirements 

- pytorch  == 1.12.1

- torch_scatter == 2.0.9

- torchdrug == 0.2.0.post1

- numpy == 1.22.4

- tqdm == 4.62.3

  

## Run the Codes

For traditional recommendation :

    python train.py --data_path=data/last-fm/

For new item recommendation :

```
python train.py --data_path=data/new_last-fm/
```

For disease-gene prediction task (new item setting) :

```
python train.py --data_path=data/Dis_5fold_item/
```

For disease-gene prediction task (new user setting) :

```
python train.py --data_path=data/Dis_5fold_user/
```

The results will be displayed in the folder 'results'.
