
import sys
import hydra
from dataclasses import dataclass
from hydra import compose, initialize, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from copy import deepcopy
import verl
from verl.trainer.main_ppo import run_ppo
from typing import Optional
import subprocess

@dataclass
class ParallelArgs:
    pp: Optional[int] = None
    tp: Optional[int] = None

@dataclass
class MainArgs:
    project: str = "verl-qwen2.5-vl"
    msize: str = "7b"
    trainer: str = "fsdp"
    rollout: str = "vllm"
    nnodes: int = 4
    trainer_parallel: ParallelArgs = ParallelArgs(2, 4)
    rollout_parallel: ParallelArgs = ParallelArgs(None, 4)
    dynamic_bsz: bool = False
    freeze_vision: bool = False
    n_gpus_per_node: int = 8
    offload: bool = False
    turn: int = 2
    verl_version: Optional[str] = None
    experiment: Optional[str] = None

cs = ConfigStore.instance()
cs.store(name="main_args", node=MainArgs)

def get_verl_version():
    return subprocess.check_output(["git", "rev-parse", "--short=4", "HEAD"]).decode("utf-8").strip()

@hydra.main(config_name="main_args", config_path=None,  version_base=None)
def main(cfg: DictConfig):
    db_flag = "-db" if cfg.dynamic_bsz else ""
    fv_flag = "-fv" if cfg.freeze_vision else ""
    pptp_flag = f"-tp{cfg.trainer_parallel.tp}-pp{cfg.trainer_parallel.pp}" if cfg.trainer == "megatron" else ""
    offload_flag = "-ol" if cfg.offload else ""
    rollout_tp_flag = f"-rtp{cfg.rollout_parallel.tp}" if cfg.rollout_parallel.tp is not None else ""
    version_flag = f"-{get_verl_version()}" if cfg.verl_version is None else cfg.verl_version
    cfg.experiment = f"{cfg.msize}-{cfg.trainer}-{cfg.rollout}-n{cfg.nnodes}{pptp_flag}{rollout_tp_flag}{db_flag}{fv_flag}{offload_flag}{version_flag}-{cfg.turn}"

    verl_args = [
        "algorithm.adv_estimator=grpo",
        "data.train_files=/data/geo3k/train.parquet",
        "data.val_files=/data/geo3k/test.parquet",
        "data.train_batch_size=512",
        "data.max_prompt_length=1024",
        "data.max_response_length=2048",
        "data.filter_overlong_prompts=True",
        "data.truncation='error'",
        "data.image_key=images",
        "data.dataloader_num_workers=0",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.rollout_parallel.tp}",
        f"actor_rollout_ref.rollout.name={cfg.rollout}",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=[\"console\",\"wandb\"]",
        f"trainer.project_name={cfg.project}",
        f"trainer.experiment_name={cfg.experiment}",
        f"trainer.n_gpus_per_node={cfg.n_gpus_per_node}",
        f"trainer.nnodes={cfg.nnodes}",
        "trainer.save_freq=20",
        "trainer.test_freq=5",
        "trainer.total_epochs=15",
    ]

    if cfg.msize == "7b":
        verl_args.append("actor_rollout_ref.model.path=/data/Qwen2.5-VL-7B-Instruct")
    elif cfg.msize == "32b":
        verl_args.append("actor_rollout_ref.model.path=/data/Qwen2.5-VL-32B-Instruct")
    elif cfg.msize == "72b":
        verl_args.append("actor_rollout_ref.model.path=/data/Qwen2.5-VL-72B-Instruct")

    if cfg.trainer == "fsdp":
        verl_args.extend([
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.actor.ppo_mini_batch_size=128",
            f"actor_rollout_ref.actor.fsdp_config.param_offload={cfg.offload}",
            f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={cfg.offload}",
            f"actor_rollout_ref.ref.fsdp_config.param_offload={cfg.offload}"])
    elif cfg.trainer == "megatron":
        verl_args.extend([
            "actor_rollout_ref.actor.megatron.use_mbridge=True",
            "actor_rollout_ref.actor.ppo_mini_batch_size=128",
            f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={cfg.trainer_parallel.pp}",
            f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={cfg.trainer_parallel.tp}",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
            f"actor_rollout_ref.actor.megatron.param_offload={cfg.offload}",
            f"actor_rollout_ref.actor.megatron.optimizer_offload={cfg.offload}",
            f"actor_rollout_ref.actor.megatron.grad_offload={cfg.offload}",
            f"actor_rollout_ref.ref.megatron.param_offload={cfg.offload} ",
            f"+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0",
            f"+actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True",
            f"+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True",
            f"+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True",
        ])

    if cfg.dynamic_bsz:
        verl_args.extend([
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            "actor_rollout_ref.rollout.free_cache_engine=False"])
        if cfg.trainer == "fsdp":
            verl_args.extend([
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144"])
        elif cfg.trainer == "megatron":
            verl_args.extend([
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120"])
    elif not cfg.dynamic_bsz:
        verl_args.extend([
            "actor_rollout_ref.actor.use_dynamic_bsz=False",
            "actor_rollout_ref.rollout.free_cache_engine=True",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True"])
        if cfg.trainer == "fsdp":
            verl_args.extend([
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20"])
        elif cfg.trainer == "megatron":
            verl_args.extend([
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20"])

    if cfg.freeze_vision:
        verl_args.extend(["actor_rollout_ref.actor.freeze_vision_tower=True"])
    else:
        verl_args.extend(["actor_rollout_ref.actor.freeze_vision_tower=False"])

    if cfg.dynamic_bsz or cfg.freeze_vision:
        verl_args.extend(["actor_rollout_ref.model.use_fused_kernels=False"])
    else:
        verl_args.extend(["actor_rollout_ref.model.use_fused_kernels=True"])

    GlobalHydra.instance().clear()
    config_name = "ppo_megatron_trainer" if cfg.trainer == "megatron" else "ppo_trainer"
    config_path = "verl.trainer.config"
    with initialize_config_module(config_module=config_path, version_base=None):
        with open(f"bench/{cfg.experiment}.log", "w") as f:
            for arg in verl_args:
                f.write(arg + "\n")
        cfg = compose(config_name=config_name, overrides=verl_args)
        verl.trainer.main_ppo.run_ppo(cfg)

if __name__ == '__main__':
    main()