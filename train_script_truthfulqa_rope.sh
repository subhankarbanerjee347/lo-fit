#!/bin/bash
# LoFiT-RoPE training script for TruthfulQA
# Runs Step 1 (head selection) + Step 2 variants (rope, v_rope, v) for comparison
# Supports: llama2_7B, llama2_13B, gemma_7b
# NOTE: Gemma requires modeling_gemma.py to be updated with RoPE support first
#
# Usage: Uncomment the model section you want to run

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

###############################################################################
# MODEL: llama2_7B
###############################################################################
model_name="llama2_7B";
task="truthfulqa";
seed=42;
echo "LoFiT-RoPE: ${model_name} on ${task}";

# --- Step 1: Head Selection (fold 0 + fold 1) ---
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task --base_model_name $model_name --apply_chat_template False \
    --ft_method lofit --lofit_component A --use_topk_heads 160 \
    --tqa_fold_num 0 --lr 5e-3 --train_batch 4 --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}" \
    --run_mode train --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}" \
    --applied_module attention --save_strategy no --l1_lambda 5e-4 --eval_batch 16 --seed $seed;

CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task --base_model_name $model_name --apply_chat_template False \
    --ft_method lofit --lofit_component A --use_topk_heads 160 \
    --tqa_fold_num 1 --lr 5e-3 --train_batch 4 --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}" \
    --run_mode train --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}" \
    --applied_module attention --save_strategy no --l1_lambda 5e-4 --eval_batch 16 --seed $seed;

# --- Step 2a: RoPE-only (fold 0 + fold 1) ---
echo "=== ${model_name}: Step 2a RoPE-only ===";
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task --base_model_name $model_name --apply_chat_template False \
    --tqa_fold_num 0 --ft_method lofit --lofit_component rope --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 --train_batch 4 --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_rope_seed${seed}" \
    --run_mode train --output_file_name "${model_name}_${task}_rope_seed${seed}" \
    --applied_module attention --save_strategy no --l1_lambda 0 \
    --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 16 --seed $seed;

CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task --base_model_name $model_name --apply_chat_template False \
    --tqa_fold_num 1 --ft_method lofit --lofit_component rope --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 --train_batch 4 --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_rope_seed${seed}" \
    --run_mode train --output_file_name "${model_name}_${task}_rope_seed${seed}" \
    --applied_module attention --save_strategy no --l1_lambda 0 \
    --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 16 --seed $seed;

# --- Step 2b: v_rope combined (fold 0 + fold 1) ---
echo "=== ${model_name}: Step 2b v_rope ===";
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task --base_model_name $model_name --apply_chat_template False \
    --tqa_fold_num 0 --ft_method lofit --lofit_component v_rope --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 --train_batch 4 --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_rope_seed${seed}" \
    --run_mode train --output_file_name "${model_name}_${task}_v_rope_seed${seed}" \
    --applied_module attention --save_strategy no --l1_lambda 0 \
    --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 16 --seed $seed;

CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task --base_model_name $model_name --apply_chat_template False \
    --tqa_fold_num 1 --ft_method lofit --lofit_component v_rope --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 --train_batch 4 --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_rope_seed${seed}" \
    --run_mode train --output_file_name "${model_name}_${task}_v_rope_seed${seed}" \
    --applied_module attention --save_strategy no --l1_lambda 0 \
    --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 16 --seed $seed;

# --- Step 2c: v-only original LoFiT (fold 0 + fold 1) ---
echo "=== ${model_name}: Step 2c v-only ===";
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task --base_model_name $model_name --apply_chat_template False \
    --tqa_fold_num 0 --ft_method lofit --lofit_component v --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 --train_batch 4 --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_seed${seed}" \
    --run_mode train --output_file_name "${model_name}_${task}_v_seed${seed}" \
    --applied_module attention --save_strategy no --l1_lambda 0 --eval_batch 16 --seed $seed;

CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task --base_model_name $model_name --apply_chat_template False \
    --tqa_fold_num 1 --ft_method lofit --lofit_component v --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
    --lr 1e-2 --train_batch 4 --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_seed${seed}" \
    --run_mode train --output_file_name "${model_name}_${task}_v_seed${seed}" \
    --applied_module attention --save_strategy no --l1_lambda 0 --eval_batch 16 --seed $seed;

###############################################################################
# MODEL: llama2_13B (uncomment to run — needs ~40GB GPU, use train_batch 2)
###############################################################################
# model_name="llama2_13B";
# task="truthfulqa";
# seed=42;
# echo "LoFiT-RoPE: ${model_name} on ${task}";
#
# # --- Step 1: Head Selection (fold 0 + fold 1) ---
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --ft_method lofit --lofit_component A --use_topk_heads 160 \
#     --tqa_fold_num 0 --lr 1e-3 --train_batch 2 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}" \
#     --run_mode train --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 1e-3 --eval_batch 8 --seed $seed;
#
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --ft_method lofit --lofit_component A --use_topk_heads 160 \
#     --tqa_fold_num 1 --lr 1e-3 --train_batch 2 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}" \
#     --run_mode train --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 1e-3 --eval_batch 8 --seed $seed;
#
# # --- Step 2a: RoPE-only (fold 0 + fold 1) ---
# echo "=== ${model_name}: Step 2a RoPE-only ===";
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 0 --ft_method lofit --lofit_component rope --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 2 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_rope_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_rope_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 \
#     --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 8 --seed $seed;
#
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 1 --ft_method lofit --lofit_component rope --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 2 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_rope_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_rope_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 \
#     --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 8 --seed $seed;
#
# # --- Step 2b: v_rope combined (fold 0 + fold 1) ---
# echo "=== ${model_name}: Step 2b v_rope ===";
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 0 --ft_method lofit --lofit_component v_rope --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 2 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_rope_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_v_rope_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 \
#     --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 8 --seed $seed;
#
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 1 --ft_method lofit --lofit_component v_rope --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 2 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_rope_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_v_rope_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 \
#     --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 8 --seed $seed;
#
# # --- Step 2c: v-only original LoFiT (fold 0 + fold 1) ---
# echo "=== ${model_name}: Step 2c v-only ===";
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 0 --ft_method lofit --lofit_component v --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 2 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_v_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 --eval_batch 8 --seed $seed;
#
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 1 --ft_method lofit --lofit_component v --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 2 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_v_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 --eval_batch 8 --seed $seed;

###############################################################################
# MODEL: gemma_7b (uncomment to run — requires modeling_gemma.py RoPE update)
###############################################################################
# model_name="gemma_7b";
# task="truthfulqa";
# seed=42;
# echo "LoFiT-RoPE: ${model_name} on ${task}";
#
# # --- Step 1: Head Selection (fold 0 + fold 1) ---
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --ft_method lofit --lofit_component A --use_topk_heads 160 \
#     --tqa_fold_num 0 --lr 5e-3 --train_batch 4 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}" \
#     --run_mode train --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 5e-4 --eval_batch 16 --seed $seed;
#
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --ft_method lofit --lofit_component A --use_topk_heads 160 \
#     --tqa_fold_num 1 --lr 5e-3 --train_batch 4 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}" \
#     --run_mode train --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 5e-4 --eval_batch 16 --seed $seed;
#
# # --- Step 2a: RoPE-only (fold 0 + fold 1) ---
# echo "=== ${model_name}: Step 2a RoPE-only ===";
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 0 --ft_method lofit --lofit_component rope --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 4 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_rope_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_rope_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 \
#     --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 16 --seed $seed;
#
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 1 --ft_method lofit --lofit_component rope --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 4 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_rope_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_rope_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 \
#     --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 16 --seed $seed;
#
# # --- Step 2b: v_rope combined (fold 0 + fold 1) ---
# echo "=== ${model_name}: Step 2b v_rope ===";
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 0 --ft_method lofit --lofit_component v_rope --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 4 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_rope_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_v_rope_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 \
#     --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 16 --seed $seed;
#
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 1 --ft_method lofit --lofit_component v_rope --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 4 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_rope_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_v_rope_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 \
#     --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 --eval_batch 16 --seed $seed;
#
# # --- Step 2c: v-only original LoFiT (fold 0 + fold 1) ---
# echo "=== ${model_name}: Step 2c v-only ===";
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 0 --ft_method lofit --lofit_component v --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 4 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_v_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 --eval_batch 16 --seed $seed;
#
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task --base_model_name $model_name --apply_chat_template False \
#     --tqa_fold_num 1 --ft_method lofit --lofit_component v --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy" \
#     --lr 2e-2 --train_batch 4 --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_v_seed${seed}" \
#     --run_mode train --output_file_name "${model_name}_${task}_v_seed${seed}" \
#     --applied_module attention --save_strategy no --l1_lambda 0 --eval_batch 16 --seed $seed;
