#!/bin/bash
# CUDA_VISIBLE_DEVICES=2,3 ray start --head
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/visual7w_train.parquet \
    data.val_files=./data/visual7w_val.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=4 \
    data.max_length=8192 \
    model.partial_pretrain=Qwen/Qwen2.5-VL-3B-Instruct \
    +model.fsdp_config.param_offload=True \
    +model.fsdp_config.optimizer_offload=True \
    ++optim.lr=1e-5 \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=test \
    trainer.experiment_name=test-sft-qwen-2.5-vl-3b-instruct-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=10 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 