## Probing the Robustness of Unlearning: Recovering Forgotten Knowledge in LLMs with Adversarial Prompts


## ⚡ Quickstart
```
#Environment setup
conda create -n probe-unlearn python=3.11
conda activate probe-unlearn
pip install -r requirements.txt

mkdir logs
```

## 💻 Black-box Attacks
We provide 5 jailbreaking black-box attacks: 

1. **Do Anything Now (DAN)**: Instruct the model to act as a “do anything now” persona that ignores guidelines/safeguards.

2. **Prefix injection**: Force responses to begin with the prefix "Sure! Here's what I found."

3. **Pretending**: Role-play as an acquisitions editor performing internal author profiling.

4. **Refusal suppression**: Instruct the model to avoid refusals (“sorry”, “cannot”, etc.) and answer directly.

5. **Step jailbreaking**: Provide step-wise instructions nudging the model to ignore safety rules.


Example: Generate DAN attack prompts

``
python attackers/DAN.py --input data/dataset.csv --output output/DAN_prompts.csv
``


## 🚀 Orchestrate Black-box Attacks 

![Black-box attack prompt templates](assets/prompt_template.png)


Use src/orchestrate_attackers.py to run multiple attack generators and merge results into one CSV. Individual attacker outputs are also saved in a separate folder.

```
python src/orchestrate_attackers.py \
  --input data/dataset.csv \
  --output output/all_attacks.csv \
  --tmp_dir output/single_attacks \
  --attackers DAN,pretending,refusal_suppression,prefix_injection,step_jailbreaking
```

- `output` - Path to the merged CSV of all attack prompts.
- `tmp_dir`- Directory to save per-attacker CSVs.
- `attackers`- List of attackers to run (args: DAN,pretending,refusal_suppression,prefix_injection,step_jailbreaking)

## ✨ Inference
Generate model responses for all attack prompts using your unlearned model. 

Note: Modify your HUGGINGFACE_HUB_TOKEN in the inference.py repo if your model is private.

```
srun --mem=4G --cpus-per-task=4 --gres=gpu:1 python src/inference.py \
    --model <huggingface_model_id_or_path>" \
    --input output/all_attack.csv \
    --output output/all_inference.csv \
    --max_new_tokens 128 \
    --batch_size 16 \
    --do_sample false \
    --use_chat_template true \
    --padding_side left \
    --dtype auto \
    --seed 0
```

## 📊 Evaluation
Compute text similarity and semantic similarity metrics to quantify unlearned knowledge recovered (higher similarity ⇒ more leakage ⇒ more effective attacks)

```
srun --mem=4G --cpus-per-task=4 --gres=gpu:1 python src/eval.py \
    --input output/all_inference.csv \
    --output output/all_eval.csv \
    --batch_size 64
```

Text similarity metrics include ROUGE-L Recall, BLEU, METEOR. Semantic similarity metrics include BERTScore, USECS, SentenceBERT. 

Optional: To skip any metrics, add arg (eg: --skip-bert-score) into the command. 

```
srun --mem=4G --cpus-per-task=4 --gres=gpu:1 python src/eval.py \
    --input output/all_inference.csv \
    --output output/all_eval.csv \
    --batch_size 64 \
    --skip-bert-score
```

## 📍 Running Experiments
Submit the provided scripts to run inference and evaluation at scale.

Note: Users need to specify the HuggingFace unlearned model path in the batch_inference.sh script.

```
sbatch scripts/batch_inference.sh 
sbatch scripts/batch_eval.sh 
```


## 📝 Citing Our Work

If you find our codebase and dataset beneficial, please cite our work:
```
```

