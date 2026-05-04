#!/bin/bash
# Master script: Run LoFiT-RoPE experiments across all tasks and models
#
# Total experiments: 3 tasks x 2 models x 4 steps = 24 training runs
# (Gemma excluded — needs modeling_gemma.py RoPE update first)
#
# Estimated time on A6000 48GB:
#   llama2_7B:  ~12 min per run x 12 runs = ~2.5 hours per task
#   llama2_13B: ~20 min per run x 12 runs = ~4 hours per task
#   Total: ~15-20 hours
#
# Usage:
#   bash run_all_rope.sh              # run everything
#   bash run_all_rope.sh llama2_7B    # run only llama2_7B on all tasks
#   bash run_all_rope.sh llama2_7B truthfulqa  # run only llama2_7B on truthfulqa

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Parse optional arguments
TARGET_MODEL="${1:-all}"
TARGET_TASK="${2:-all}"

###############################################################################
# Hyperparameter lookup
###############################################################################
# Usage: get_hparams MODEL TASK
# Sets: STEP1_LR, STEP1_L1, STEP2_LR, TOPK, TRAIN_BATCH, EVAL_BATCH, EXTRA_ARGS
get_hparams() {
    local model=$1
    local task=$2

    # Defaults
    EXTRA_ARGS=""
    TRAIN_BATCH=4
    EVAL_BATCH=16

    # Model-specific batch sizes
    if [ "$model" = "llama2_13B" ]; then
        TRAIN_BATCH=2
        EVAL_BATCH=8
    fi

    # Task + model specific hyperparameters (from the original LoFiT paper)
    case "${task}_${model}" in
        # TruthfulQA
        truthfulqa_llama2_7B)   STEP1_LR=5e-3; STEP1_L1=5e-4; STEP2_LR=1e-2; TOPK=32;;
        truthfulqa_llama2_13B)  STEP1_LR=1e-3; STEP1_L1=1e-3; STEP2_LR=2e-2; TOPK=48;;
        truthfulqa_gemma_7b)    STEP1_LR=5e-3; STEP1_L1=5e-4; STEP2_LR=2e-2; TOPK=16;;
        # MQuAKE
        mquake_llama2_7B)       STEP1_LR=5e-3; STEP1_L1=1e-3; STEP2_LR=1e-2; TOPK=32;;
        mquake_llama2_13B)      STEP1_LR=1e-3; STEP1_L1=1e-3; STEP2_LR=8e-3; TOPK=48;;
        mquake_gemma_7b)        STEP1_LR=5e-3; STEP1_L1=5e-4; STEP2_LR=8e-3; TOPK=16;;
        # CLUTRR
        clutrr_llama2_7B)       STEP1_LR=5e-4; STEP1_L1=5e-3; STEP2_LR=1e-2; TOPK=32; EXTRA_ARGS="--train_size 300";;
        clutrr_llama2_13B)      STEP1_LR=1e-3; STEP1_L1=1e-3; STEP2_LR=1e-2; TOPK=48; EXTRA_ARGS="--train_size 300";;
        clutrr_gemma_7b)        STEP1_LR=5e-4; STEP1_L1=5e-3; STEP2_LR=1e-2; TOPK=16; EXTRA_ARGS="--train_size 300";;
        *) echo "ERROR: Unknown combination ${task}_${model}"; exit 1;;
    esac
}

###############################################################################
# Run one complete experiment (Step 1 + Step 2 variants)
###############################################################################
run_experiment() {
    local model=$1
    local task=$2
    local seed=42

    get_hparams "$model" "$task"

    echo ""
    echo "################################################################"
    echo "# ${model} on ${task}"
    echo "# Step1: lr=${STEP1_LR} l1=${STEP1_L1} | Step2: lr=${STEP2_LR} topk=${TOPK}"
    echo "################################################################"

    # TruthfulQA uses 2-fold cross-validation
    if [ "$task" = "truthfulqa" ]; then
        FOLDS="0 1"
    else
        FOLDS="0"
    fi

    # ---- Step 1: Head Selection ----
    echo "=== ${model}/${task}: Step 1 - Head Selection ===";
    for fold in $FOLDS; do
        FOLD_ARGS=""
        if [ "$task" = "truthfulqa" ]; then
            FOLD_ARGS="--tqa_fold_num $fold"
        fi

        CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
            --task $task --base_model_name $model --apply_chat_template False \
            --ft_method lofit --lofit_component A --use_topk_heads 160 \
            $FOLD_ARGS --lr $STEP1_LR --train_batch $TRAIN_BATCH --num_epoch 5 \
            --output_dir "./finetuned_checkpoints/${task}/${model}_${task}_Aonly_seed${seed}" \
            --run_mode train --output_file_name "./finetuned_outputs/${task}/${model}_${task}_Aonly_seed${seed}" \
            --applied_module attention --save_strategy no --l1_lambda $STEP1_L1 \
            --eval_batch $EVAL_BATCH --seed $seed $EXTRA_ARGS;
    done

    # ---- Step 2a: RoPE-only ----
    echo "=== ${model}/${task}: Step 2a - RoPE-only ===";
    for fold in $FOLDS; do
        FOLD_ARGS=""
        if [ "$task" = "truthfulqa" ]; then
            FOLD_ARGS="--tqa_fold_num $fold"
        fi

        CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
            --task $task --base_model_name $model --apply_chat_template False \
            $FOLD_ARGS --ft_method lofit --lofit_component rope --use_topk_heads $TOPK \
            --lofit_heads "./top_heads/${model}_${task}_Aonly_top160heads_${seed}.npy" \
            --lr $STEP2_LR --train_batch $TRAIN_BATCH --num_epoch 5 \
            --output_dir "./finetuned_checkpoints/${task}/${model}_${task}_rope_seed${seed}" \
            --run_mode train --output_file_name "${model}_${task}_rope_seed${seed}" \
            --applied_module attention --save_strategy no --l1_lambda 0 \
            --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 \
            --eval_batch $EVAL_BATCH --seed $seed $EXTRA_ARGS;
    done

    # ---- Step 2b: v_rope combined ----
    echo "=== ${model}/${task}: Step 2b - v_rope ===";
    for fold in $FOLDS; do
        FOLD_ARGS=""
        if [ "$task" = "truthfulqa" ]; then
            FOLD_ARGS="--tqa_fold_num $fold"
        fi

        CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
            --task $task --base_model_name $model --apply_chat_template False \
            $FOLD_ARGS --ft_method lofit --lofit_component v_rope --use_topk_heads $TOPK \
            --lofit_heads "./top_heads/${model}_${task}_Aonly_top160heads_${seed}.npy" \
            --lr $STEP2_LR --train_batch $TRAIN_BATCH --num_epoch 5 \
            --output_dir "./finetuned_checkpoints/${task}/${model}_${task}_v_rope_seed${seed}" \
            --run_mode train --output_file_name "${model}_${task}_v_rope_seed${seed}" \
            --applied_module attention --save_strategy no --l1_lambda 0 \
            --rope_reg_alpha 1e-2 --rope_reg_beta 1e-2 \
            --eval_batch $EVAL_BATCH --seed $seed $EXTRA_ARGS;
    done

    # ---- Step 2c: v-only (original LoFiT) ----
    echo "=== ${model}/${task}: Step 2c - v-only ===";
    for fold in $FOLDS; do
        FOLD_ARGS=""
        if [ "$task" = "truthfulqa" ]; then
            FOLD_ARGS="--tqa_fold_num $fold"
        fi

        CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
            --task $task --base_model_name $model --apply_chat_template False \
            $FOLD_ARGS --ft_method lofit --lofit_component v --use_topk_heads $TOPK \
            --lofit_heads "./top_heads/${model}_${task}_Aonly_top160heads_${seed}.npy" \
            --lr $STEP2_LR --train_batch $TRAIN_BATCH --num_epoch 5 \
            --output_dir "./finetuned_checkpoints/${task}/${model}_${task}_v_seed${seed}" \
            --run_mode train --output_file_name "${model}_${task}_v_seed${seed}" \
            --applied_module attention --save_strategy no --l1_lambda 0 \
            --eval_batch $EVAL_BATCH --seed $seed $EXTRA_ARGS;
    done

    echo "=== ${model}/${task}: DONE ==="
}

###############################################################################
# Main: Run all combinations
###############################################################################

# Models to run (Llama models first for partial results, Gemma last)
MODELS="llama2_7B llama2_13B gemma_7b"
TASKS="truthfulqa mquake clutrr"

# Apply filters if provided
if [ "$TARGET_MODEL" != "all" ]; then
    MODELS="$TARGET_MODEL"
fi
if [ "$TARGET_TASK" != "all" ]; then
    TASKS="$TARGET_TASK"
fi

echo "============================================================"
echo "LoFiT-RoPE: Running experiments"
echo "Models: $MODELS"
echo "Tasks:  $TASKS"
echo "============================================================"

for task in $TASKS; do
    for model in $MODELS; do
        run_experiment "$model" "$task"
    done
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Check results:"
echo "  TruthfulQA: python compute_scores.py"
echo "  MQuAKE:     cat ./finetuned_outputs/mquake/*/outputs.json"
echo "  CLUTRR:     cat ./finetuned_outputs/clutrr/*/outputs.json"
