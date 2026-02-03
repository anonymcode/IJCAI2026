# Position-Aware Sequential Attention for Accurate Next Item Recommendations

This repository contains the implementation of Position-Aware Sequential Attention (PAA) for sequential recommendation tasks. The code includes the PAA model along with baseline implementations (NoPE, Classic, CAPE, RoPE) for comparison.

Currently available datasets: `ml-1m`, `beauty` and `yelp`.

## Requirements installation

```bash
conda env create -f environment.yml
conda activate pasrec
```

## Training

to train and evaluate our model:

```bash
python train_with_best_config.py \
--name test_run --type_of_model PAA --study_name texc11 --type_of_custom_A 5 --type_of_trinagularity la --type_of_connection 0 --dataset ml-1m \
--num_epochs 300
```

to train and evaluate baseline models:

```bash
python train_with_best_config.py \
--name test_run --type_of_model empty --study_name texc11 --dataset ml-1m \
--num_epochs 300
```

```bash
--type_of_model empty # for NoPE
--type_of_model normal # for Classic
--type_of_model CAPE # for CAPE
--type_of_model RoPE # for RoPE
```

