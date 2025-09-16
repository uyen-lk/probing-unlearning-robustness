#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=SCT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


source ~/miniconda3/etc/profile.d/conda.sh
conda activate probe-unlearn

nvidia-smi || true
python - << 'PY'
import torch
print("cuda.is_available:", torch.cuda.is_available(), "dev_count:", torch.cuda.device_count())
PY


srun python src/eval.py --input output/all_inference.csv --output output/all_eval.csv --batch_size 64
