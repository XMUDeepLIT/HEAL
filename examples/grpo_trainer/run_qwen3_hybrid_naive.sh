set -x

DATA_DIR=./data
MODEL_DIR=
project_name='Qwen3-Few-Shot-VeRL'
exp_name='Qwen3-8B-Base-Math-Hybrid-grpo'
CHECKPOINTS_DIR_PREFIX=
CHECKPOINTS_DIR=${CHECKPOINTS_DIR_PREFIX}/${project_name}/${exp_name}
train_file_name=hybrid_deepscaler_train.parquet
val_file_name=deepscaler_val.parquet
SANDBOX_URL=""
export WANDB_MODE="offline"


python3 -m verl.trainer.main_ppo \
    reward_model.sandbox_fusion.url=$SANDBOX_URL \
    reward_model.sandbox_fusion.max_concurrent=128 \
    reward_model.reward_manager=prime \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train/$train_file_name \
    data.val_files=$DATA_DIR/val/$val_file_name \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_DIR/Qwen3-8B-Base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.default_local_dir=$CHECKPOINTS_DIR \
    trainer.val_before_train=False \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=50 $@

