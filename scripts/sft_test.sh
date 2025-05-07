#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3 ray start --head
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
    data.micro_batch_size=4 \
    data.max_length=8192 \
    model.partial_pretrain=Qwen/Qwen2.5-VL-3B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=test \
    trainer.experiment_name=test-sft-qwen-2.5-vl-3b-instruct-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.total_training_steps=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 