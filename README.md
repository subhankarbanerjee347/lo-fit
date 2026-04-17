# LoFiT: Localized Fine-tuning on LLM Representations

This paper provides code for the paper [LoFiT: Localized Fine-tuning on LLM Representations](https://arxiv.org/abs/2406.01563). In this work, we introduce LoFiT, a two-step localized fine-tuning method for LLMs that selects a subset of attention heads and learns task-specific offset vectors to be added to the hidden representations of the targeted attention heads. We show the strong downstream performance of LoFiT on tasks involving truthfulness and reasoning, outperforming representation intervention methods (ITI and RepE) and matching strong PEFT methods (LoRA) with fewer learned parameters.

## Abstract
> Recent work in interpretability shows that large language models (LLMs) can be adapted for new tasks in a learning-free way: it is possible to intervene on LLM representations to elicit desired behaviors for alignment. For instance, adding certain bias vectors to the outputs of certain attention heads is reported to boost the truthfulness of models. In this work, we show that localized fine-tuning serves as an effective alternative to such representation intervention methods. We introduce a framework called Localized Fine-Tuning on LLM Representations LoFiT, which identifies a subset of attention heads that are most important for learning a specific task, then trains offset vectors to add to the model's hidden representations at those selected heads. LoFiT localizes to a sparse set of heads (3%) and learns the offset vectors from limited training data, comparable to the settings used for representation intervention. For truthfulness and reasoning tasks, we find that LoFiT's intervention vectors are more effective for LLM adaptation than vectors from representation intervention methods such as Inference-time Intervention. We also find that the localization step is important: selecting a task-specific set of attention heads can lead to higher performance than intervening on heads selected for a different task. Finally, for the tasks we study, LoFiT achieves comparable performance to other parameter-efficient fine-tuning methods such as LoRA, despite modifying 20x-200x fewer parameters than these methods.

## Table of Contents
1. [Installation](#installation)
2. [Data](#data)
3. [Train and Evaluate](#train-and-evaluate)
4. [LoFiT-RoPE Extension](#lofit-rope-extension)
5. [How to Cite](#how-to-cite)

## Installation
We have tested using Python 3.8.10. Before building the environment, please install the appropriate PyTorch version that corresponds to the hardware configurations (especially GPUs) of your machine here: https://pytorch.org/get-started/locally/
(Note: If you encounter errors like  ```RuntimeError: CUDA error: device kernel image is invalid``` at inference time when doing the evaluation, please check the PyTorch and CUDA driver version. We have tested on a single NVIDIA RTX A6000 GPU with 48G memory using PyTorch 2.2.2+cu121)

Then, run the following.
```
# Setup virtual environmnet
python3.8 -m venv lofit
source lofit/bin/activate
# install requirements
pip install -r requirements.txt
# Create the directory to store the finetuned checkpoints and evaluation outputs
mkdir finetuned_checkpoints
mkdir finetuned_outputs
mkdir tqa_results
# Create the directory to store important attention heads selected by LoFiT 
mkdir top_heads
```
## Data
We use TruthfulQA, MQuAKE, and CLUTRR to evaluate LoFiT. The pre-processing of these datasets is described in Section 4 of our paper. We include the train/val/test splits we used for each dataset under ```datasets```. You can use 
## Train and Evaluate
### Setting up the models
We currently support Gemma 7B, Llama 2-7B, and Llama 2-13B as base models to fine-tune. Paths to the huggingface checkpoints of these models should be defined in ```models_map``` in ```lofit_trainer.py``` before running any training or evaluation script. All fine-tuning experiments can be run on a single 48G GPU.

We modified the above models with additional parameters as mentioned in the paper in ```models/modeling_llama.py``` and  ```models/modeling_gemma.py```. To initialize the modified model or load a trained checkpooint, we need to use the method ```LlamaForCausalLM.custom_from_pretrained(...)``` or  ```GemmaForCausalLM.custom_from_pretrained(...)``` in these two files to properly load the additional parameters.

### End-to-end training
We provide an end-to-end script to run head selection, bias tuning, and final evaluation of LoFiT for each dataset in one line as ```train_script_{task}.sh```. For example, to fine-tune Llama 2-7B-base on TruthfulQA with LoFiT, please run the following:
```
bash train_script_truthfulqa.sh
```
Specific hyperparameter configurations might be needed for different models and different datasets. Please refer to our paper for details.
### Evaluation
We integrate the evaluation step and the training step into the end-to-end script mentioned above. The codes to evaluate LoFiT on TruthfulQA are adapted from the codebase of [Inference-time Intervention](https://github.com/likenneth/honest_llama). Details of evaluating LoFiT on MQuAKE and CLUTRR can be found in ```utils/evaluate.py```.

**[Updated 01/15/2025]** We release the weights of fine-tuned models that integrate the tuned biases here:

https://huggingface.co/fcyin/llama2_7B_base_lofit_mquake

https://huggingface.co/fcyin/llama2_7B_base_lofit_truthfulqa

You can use these weights and the code snippets included in the hugging face repo to run evaluations.

## LoFiT-RoPE Extension

This fork extends LoFiT with **LoFiT-RoPE**, which adds per-head positional intervention via learned RoPE
(Rotary Position Embedding) modifications. Inspired by [LongRoPE](https://arxiv.org/abs/2402.13753), LoFiT-RoPE
learns task-specific frequency rescale factors and phase offsets at a sparse set of attention heads selected by
LoFiT's existing head selection step.

### How It Works

LoFiT-RoPE follows the same two-step framework as LoFiT:

**Step 1 — Head Selection (unchanged):** Learn scaling factors with L1 regularization, select the top-K heads.

**Step 2 — Joint Bias + RoPE Tuning (new):** At each selected head, learn three sets of parameters:
- **Bias vectors** `v` (original LoFiT) — additive offsets on attention head outputs
- **RoPE rescale factors** `λ` (new) — per-dimension frequency scaling (λ > 1 = attend closer, λ < 1 = attend further)
- **RoPE phase offsets** `φ` (new) — constant angular shifts to reposition attention peaks

### New `lofit_component` Options

| Value | What it trains | Parameters per head | Use case |
|-------|---------------|-------------------|----------|
| `A` | Scaling factors (Step 1) | 128 | Head selection (unchanged) |
| `v` | Bias vectors only (Step 2) | 128 | Original LoFiT |
| `rope` | RoPE rescale + phase only (Step 2) | 128 (64+64) | Positional intervention only |
| `v_rope` | Bias + RoPE rescale + phase (Step 2) | 256 (128+64+64) | Combined content + positional |

### New Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rope_reg_alpha` | 0.01 | L2 regularization for rescale factors toward 1 (identity) |
| `--rope_reg_beta` | 0.01 | L2 regularization for phase offsets toward 0 (no shift) |

### Running LoFiT-RoPE

**Prerequisites:** Same as the base LoFiT setup (see [Installation](#installation) above). Requires a GPU with
48GB memory (e.g., NVIDIA A6000 or A100). You must configure model paths in `models_map` in `lofit_trainer.py`
and have access to Llama-2 weights via HuggingFace.

**Full pipeline on TruthfulQA (Llama 2-7B):**
```
bash train_script_truthfulqa_rope.sh
```

This runs Step 1 (head selection), then three Step 2 variants: rope-only, combined v_rope, and original v-only
for comparison.

**Individual steps:**

```bash
# Step 1: Head selection (same as original LoFiT)
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task truthfulqa --base_model_name llama2_7B \
    --ft_method lofit --lofit_component A \
    --use_topk_heads 160 --tqa_fold_num 0 \
    --lr 5e-3 --train_batch 8 --num_epoch 5 \
    --output_dir ./finetuned_checkpoints/truthfulqa/step1 \
    --run_mode train --output_file_name ./finetuned_outputs/step1 \
    --applied_module attention --save_strategy no \
    --l1_lambda 5e-4 --eval_batch 32 --seed 42

# Step 2: RoPE-only intervention
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task truthfulqa --base_model_name llama2_7B \
    --ft_method lofit --lofit_component rope \
    --use_topk_heads 32 --tqa_fold_num 0 \
    --lofit_heads ./top_heads/llama2_7B_truthfulqa_Aonly_top160heads_42.npy \
    --lr 1e-2 --train_batch 8 --num_epoch 5 \
    --output_dir ./finetuned_checkpoints/truthfulqa/rope \
    --run_mode train --output_file_name llama2_7B_truthfulqa_rope_42 \
    --applied_module attention --save_strategy no \
    --l1_lambda 0 --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 \
    --eval_batch 32 --seed 42

# Step 2: Combined bias + RoPE intervention
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task truthfulqa --base_model_name llama2_7B \
    --ft_method lofit --lofit_component v_rope \
    --use_topk_heads 32 --tqa_fold_num 0 \
    --lofit_heads ./top_heads/llama2_7B_truthfulqa_Aonly_top160heads_42.npy \
    --lr 1e-2 --train_batch 8 --num_epoch 5 \
    --output_dir ./finetuned_checkpoints/truthfulqa/v_rope \
    --run_mode train --output_file_name llama2_7B_truthfulqa_v_rope_42 \
    --applied_module attention --save_strategy no \
    --l1_lambda 0 --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 \
    --eval_batch 32 --seed 42
```

### Evaluation Metrics

| Task | Metric | Output Location |
|------|--------|----------------|
| TruthfulQA | MC1 (single-answer accuracy), MC2 (multi-answer accuracy) | `./tqa_results/summary_dump/` |
| MQuAKE | Exact Match (EM) | `<output_dir>/outputs.json` |
| CLUTRR | Exact Match (EM) | `<output_dir>/outputs.json` |

### Key Hyperparameters to Tune

- `--lr`: Learning rate for Step 2. Start with 1e-2, also try 5e-3 and 1e-3.
- `--rope_reg_alpha`: Higher values keep rescale factors closer to 1 (conservative). Range: 1e-3 to 1e-1.
- `--rope_reg_beta`: Higher values keep phase offsets closer to 0 (conservative). Range: 1e-3 to 1e-1.
- `--use_topk_heads`: Number of heads to intervene on. Paper uses 3% (K=32 for Llama 2-7B).

## How to Cite
If you have any question regarding the code and our work, please feel free to reach out to Fangcong Yin (fangcongyin@utexas.edu).

If you find our work useful, please consider citing us with the following format:
```
@inproceedings{
      yin2024lofit,
      title={LoFiT: Localized Fine-tuning on {LLM} Representations},
      author={Fangcong Yin and Xi Ye and Greg Durrett},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024},
      url={https://openreview.net/forum?id=dfiXFbECSZ}
}
```
