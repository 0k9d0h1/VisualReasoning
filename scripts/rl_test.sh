
set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3 python3 -m verl.trainer.main_ppo \
    --config-path="/home/kdh0901/Desktop/Train_VisualReasoning/config/" \
    --config-name='multiturn_grpo_test' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=4 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.need_tools_kwargs=True \
    actor_rollout_ref.model.path=/home/kdh0901/Desktop/cache_dir/kdh0901/checkpoints/global_step_260 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='gsm8k_async_rl' \
    trainer.experiment_name='qwen2.5-3b_function_rm-gsm8k-async-sgl-multi-w-tool-verify-n16' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=/home/kdh0901/Desktop/cache_dir/kdh0901/verl/dataset/visual7w/train.parquet \
    data.val_files=/home/kdh0901/Desktop/cache_dir/kdh0901/verl/dataset/visual7w/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="/home/kdh0901/Desktop/Train_VisualReasoning/tool_config/execution_tool_config.yaml" \
    trainer.total_epochs=15 $@

