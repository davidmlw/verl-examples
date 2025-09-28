set -x
ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping


HF_MODEL_PATH=${HF_MODEL_PATH:-"/data/Qwen2.5-VL-72B-Instruct"}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.0}

verl_version=$(git rev-parse --short=4 HEAD)
NNODES=${NNODES:-8}
TURN=${TURN:-1}
experiment_name=${experiment_name:-72b-mcore-n${NNODES}-${verl_version}-${TURN}}

train_path=$HOME/data/geo3k/train.parquet
test_path=$HOME/data/geo3k/test.parquet

verl_args=(
    --config-path=config
    --config-name='ppo_megatron_trainer.yaml'
    algorithm.adv_estimator=grpo
    data.train_files="$train_path"
    data.val_files="$test_path"
    data.train_batch_size=512
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.dataloader_num_workers=0
    data.truncation='error'
    actor_rollout_ref.model.path=$HF_MODEL_PATH
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=32
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=8
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20
    actor_rollout_ref.rollout.tensor_model_parallel_size=8
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480
    actor_rollout_ref.rollout.name=$ENGINE
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=32
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=8
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    #+actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=3
    actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD}
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name='verl_grpo_example_geo3k'
    trainer.experiment_name=$experiment_name
    trainer.n_gpus_per_node=8
    trainer.nnodes=$NNODES
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
)

python3 -m verl.trainer.main_ppo ${verl_args[@]} $@ 2>&1 | tee -a ${experiment_name}.log
