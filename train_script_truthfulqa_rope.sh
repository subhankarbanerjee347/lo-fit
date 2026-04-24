#!/bin/bash
# LoFiT-RoPE training script for TruthfulQA on Llama2-7B
# Runs three variants: rope-only, v_rope (combined), and original lofit (v-only) for comparison
#
# Pipeline:
#   Step 1 (A): Head selection — identical to original LoFiT
#   Step 2 (rope): RoPE-only intervention at selected heads
#   Step 2 (v_rope): Combined bias + RoPE intervention at selected heads
#   Step 2 (v): Original LoFiT bias-only (for comparison)

model_name="llama2_7B";
task="truthfulqa";
seed=42;
echo "LoFiT-RoPE: ${model_name} on ${task}";

### Use cached model files only — avoids SSL errors behind corporate proxies
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

###############################################################################
# Step 1: Head Selection (shared across all Step 2 variants)
# This is identical to the original LoFiT head selection step.
###############################################################################

### Fold 0
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --ft_method lofit \
    --lofit_component A \
    --use_topk_heads 160 \
    --tqa_fold_num 0 \
    --lr 5e-3 \
    --train_batch 4 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}" \
    --run_mode train \
    --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}" \
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 5e-4 \
    --eval_batch 16 \
    --seed $seed;

### Fold 1
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --ft_method lofit \
    --lofit_component A \
    --use_topk_heads 160 \
    --tqa_fold_num 1 \
    --lr 5e-3 \
    --train_batch 4 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}" \
    --run_mode train \
    --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}" \
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 5e-4 \
    --eval_batch 16 \
    --seed $seed;

###############################################################################
# Step 2a: RoPE-only (positional intervention, no bias vectors)
###############################################################################

echo "=== Step 2a: RoPE-only ===";

### Fold 0
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --tqa_fold_num 0 \
    --ft_method lofit \
    --lofit_component rope \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 \
    --train_batch 4 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_rope_seed${seed}" \
    --run_mode train \
    --output_file_name "${model_name}_${task}_rope_seed${seed}" \
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --rope_reg_alpha 1e-2 \
    --rope_reg_beta 1e-2 \
    --eval_batch 16 \
    --seed $seed;

### Fold 1
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --tqa_fold_num 1 \
    --ft_method lofit \
    --lofit_component rope \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 \
    --train_batch 4 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_rope_seed${seed}" \
    --run_mode train \
    --output_file_name "${model_name}_${task}_rope_seed${seed}" \
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --rope_reg_alpha 1e-2 \
    --rope_reg_beta 1e-2 \
    --eval_batch 16 \
    --seed $seed;

###############################################################################
# Step 2b: Combined v + RoPE (bias vectors + positional intervention)
###############################################################################

echo "=== Step 2b: v_rope (combined) ===";

### Fold 0
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --tqa_fold_num 0 \
    --ft_method lofit \
    --lofit_component v_rope \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 \
    --train_batch 4 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_rope_seed${seed}" \
    --run_mode train \
    --output_file_name "${model_name}_${task}_v_rope_seed${seed}" \
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --rope_reg_alpha 1e-2 \
    --rope_reg_beta 1e-2 \
    --eval_batch 16 \
    --seed $seed;

### Fold 1
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --tqa_fold_num 1 \
    --ft_method lofit \
    --lofit_component v_rope \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 \
    --train_batch 4 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_rope_seed${seed}" \
    --run_mode train \
    --output_file_name "${model_name}_${task}_v_rope_seed${seed}" \
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --rope_reg_alpha 1e-2 \
    --rope_reg_beta 1e-2 \
    --eval_batch 16 \
    --seed $seed;

###############################################################################
# Step 2c: Original LoFiT v-only (bias vectors only, for comparison)
###############################################################################

echo "=== Step 2c: v-only (original LoFiT) ===";

### Fold 0
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --tqa_fold_num 0 \
    --ft_method lofit \
    --lofit_component v \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 \
    --train_batch 4 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_seed${seed}" \
    --run_mode train \
    --output_file_name "${model_name}_${task}_v_seed${seed}" \
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --eval_batch 16 \
    --seed $seed;

### Fold 1
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --tqa_fold_num 1 \
    --ft_method lofit \
    --lofit_component v \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 \
    --train_batch 4 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_seed${seed}" \
    --run_mode train \
    --output_file_name "${model_name}_${task}_v_seed${seed}" \
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --eval_batch 16 \
    --seed $seed;
