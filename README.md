# SCD-MoE

This repository contains the official compact implementation of SCD-MoE,
including the model definition, training script, and evaluation script used in
the paper experiments.

## Contents

- `changedetection/`: model, dataset, config, and metric code required for inference.
- `classification/`: VSSM backbone support code.
- `kernels/`: selective scan CUDA kernel sources used by the backbone.
- `tools/infer_scd_moe.py`: unified evaluation entrypoint.
- `tools/train_scd_moe.py`: unified training entrypoint.

The released model path is intentionally compact. `changedetection/models/SCDMoE.py`
contains the SCD-MoE network, `changedetection/models/decoder.py` contains the
unified multi-scale decoder, and `changedetection/models/moes/dense_MoE.py`
contains only the IA-MoE and TD-MoE modules used by the released checkpoints.

## Environment

Create a Python environment with PyTorch, CUDA support, and the dependencies in
`requirements.txt`.

```bash
pip install -r requirements.txt
```

The VSSM backbone uses the selective scan CUDA extension under
`kernels/selective_scan`. Build it if your environment does not already provide
the required operator.

## Checkpoints

Download the released model weights and place them as follows:

```text
pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth
checkpoints/SECOND/scd_moe_second.pth
checkpoints/JL1/scd_moe_jl1.pth
checkpoints/Landsat/scd_moe_landsat.pth
```

## Default Dataset Paths

The evaluation script uses the following local dataset paths by default:

- SECOND: `/mnt/data1/hq/SECOND/SECOND/SECOND/test`, list `/mnt/data1/hq/SECOND/SECOND/SECOND/test.txt`
- JL1: `/mnt/data1/hq/JL1/JL1/test`, list `/mnt/data1/hq/JL1/JL1/list/test.txt`
- Landsat-SCD: `/mnt/data1/hq/Landsat/Landsat-SCD`, list `/mnt/data1/hq/Landsat/Landsat-SCD/test_list_old.txt`

Override them with `--test_dataset_path` and `--test_data_list_path` if needed.

## Evaluation

```bash
python tools/infer_scd_moe.py --dataset all
```

Run a single dataset:

```bash
python tools/infer_scd_moe.py --dataset SECOND
python tools/infer_scd_moe.py --dataset JL1
python tools/infer_scd_moe.py --dataset Landsat
```

Metrics are written to `results/eval_metrics.csv`.

## Training

The training script uses dataset-specific defaults matching the packaged
SCD-MoE setting. Checkpoints are saved under `saved_models/` by default.

```bash
python tools/train_scd_moe.py --dataset SECOND
python tools/train_scd_moe.py --dataset JL1
python tools/train_scd_moe.py --dataset Landsat
```

Useful overrides:

```bash
python tools/train_scd_moe.py \
  --dataset SECOND \
  --train_dataset_path /path/to/train \
  --train_data_list_path /path/to/train.txt \
  --test_dataset_path /path/to/test \
  --test_data_list_path /path/to/test.txt \
  --batch_size 2 \
  --crop_size 512 \
  --learning_rate 1e-4
```

For a quick environment check:

```bash
python tools/train_scd_moe.py \
  --dataset SECOND --max_iters 1 --num_workers 0 --val_interval 0
```
