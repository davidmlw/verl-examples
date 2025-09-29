set -x
ENGINE=${1:-vllm}

project_name=${project_name:-verl-qwen2.5-vl}
MSIZE=${MSIZE:-7b}
NNODES=${NNODES:-4}
DYNAMIC_BSZ=${DYNAMIC_BSZ:-False}
FREEZE_VISION=${FREEZE_VISION:-False}
verl_version=$(git rev-parse --short=4 HEAD)
TURN=${TURN:-1}


verl_args=()
if [ $MSIZE == "7b" ]; then
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.model.path=/data/Qwen2.5-VL-7B-Instruct
    )
elif [ $MSIZE == "32b" ]; then
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.model.path=/data/Qwen2.5-VL-32B-Instruct
    )
elif [ $MSIZE == "72b" ]; then
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.model.path=/data/Qwen2.5-VL-72B-Instruct
    )
fi

if [ $DYNAMIC_BSZ == "True" ]; then
    DYNAMIC_BSZ_FLAG="-db"
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144
        actor_rollout_ref.rollout.free_cache_engine=False
    )
else
    DYNAMIC_BSZ_FLAG=""
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.actor.use_dynamic_bsz=False ## default from actor.yaml
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 ## default from actor.yaml
        actor_rollout_ref.rollout.free_cache_engine=True ## default from rollout.yaml
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 ## default from actor.yaml
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20
    )
fi

if [ $FREEZE_VISION == "True" ]; then
    FREEZE_VISION_FLAG="-fv"
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.actor.freeze_vision_tower=True
    )
else
    FREEZE_VISION_FLAG=""
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.actor.freeze_vision_tower=False
    )
fi

if [ $DYNAMIC_BSZ == "True" ] or [ $FREEZE_VISION == "True" ]; then
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.model.use_fused_kernels=False ## default from hf_model.yaml
    )
else
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.model.use_fused_kernels=True
    )
fi

experiment_name=${experiment_name:-${MSIZE}-fsdp-vllm-n${NNODES}${DYNAMIC_BSZ_FLAG}${FREEZE_VISION_FLAG}-${verl_version}-${TURN}}

verl_args=(
    ${freeze_vision_args[@]}
    algorithm.adv_estimator=grpo
    data.train_files=$HOME/data/geo3k/train.parquet
    data.val_files=$HOME/data/geo3k/test.parquet
    data.train_batch_size=512
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.image_key=images
    data.dataloader_num_workers=0
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    #actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.actor.ppo_mini_batch_size=128
    #actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=$ENGINE
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=8
    trainer.nnodes=${NNODES}
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
)

python3 -m verl.trainer.main_ppo ${verl_args[@]} $@ 2>&1 | tee -a ${experiment_name}-${TURN}.log