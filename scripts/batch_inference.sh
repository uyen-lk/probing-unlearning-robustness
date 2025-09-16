#!/usr/bin/env bash
#SBATCH --job-name=inference
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

INPUT="output/all_attack.csv"
OUTDIR="output"
mkdir -p "$OUTDIR" logs

# Common flags for all runs
COMMON="--input $INPUT --output $OUTDIR --max_new_tokens 128 --batch_size 16 --do_sample false --use_chat_template true --padding_side left --dtype auto --seed 0"

# LLaMA variants
for MODEL in \
  <huggingface_model_id_or_path>
  <huggingface_model_id_or_path>
  <huggingface_model_id_or_path>
do
  echo "[INFO] $MODEL"
  srun python src/inference.py --model "$MODEL" $COMMON
done

# Phi variants
for MODEL in \
  <huggingface_model_id_or_path>
  <huggingface_model_id_or_path>
  <huggingface_model_id_or_path>
do
  echo "[INFO] $MODEL"
  srun python src/inference.py --model "$MODEL" $COMMON
done
