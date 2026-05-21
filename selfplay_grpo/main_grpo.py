# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GRPO entrypoint for the AZR-GRPO fork.
"""
import ray
import hydra
from pathlib import Path
from pprint import pprint

from omegaconf import OmegaConf
try:
    from verl.utils.fs import copy_local_path_from_hdfs
except ImportError:
    from selfplay_grpo.utils.fs_compat import copy_local_path_from_hdfs
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

from selfplay_grpo.rewards.grpo_reward_manager import (
    GRPORewardManager,
    INTRINSIC_SELF_CONSISTENCY_REWARD_MODE,
    canonicalize_solver_reward_mode,
    CodeIORewardManager,
)
from selfplay_grpo.trainer.grpo.grpo_trainer import SelfPlayGRPORayTrainer


@hydra.main(config_path="configs", config_name="grpo_trainer", version_base=None)
def main(config):
    run_grpo(config)


def run_grpo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            }},
            num_cpus=config.ray_init.num_cpus,
        )

    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        self._hydrate_compat_config(config)

        if config.trainer.debug:
            import debugpy
            debugpy.listen(("0.0.0.0", config.trainer.debug_port))
            print(f"Debugger listening on port {config.trainer.debug_port}")
            debugpy.wait_for_client()
            print("Debugger attached!")

        batch_multiplier = 2 if config.selfplay.train_propose else 1
        config.actor_rollout_ref.actor.ppo_mini_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n * batch_multiplier
        )
        pprint(f"auto setting ppo_mini_batch_size: {config.actor_rollout_ref.actor.ppo_mini_batch_size}")
        config.selfplay.data_selection_strategy.data_len = config.data.train_batch_size * config.selfplay.data_selection_strategy.update_iteration
        pprint(f"auto setting data_len: {config.selfplay.data_selection_strategy.data_len}")

        task_suffix = config.selfplay.task.replace("pred_", "")
        config.trainer.default_local_dir = (
            Path(config.trainer.default_local_dir)
            / task_suffix
            / config.actor_rollout_ref.model.path.split("/")[-1]
            / config.reward_fn.extraction_type
        ).as_posix()

        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        if config.actor_rollout_ref.model.pretrained_tokenizer:
            tokenizer.chat_template = "{%- for message in messages -%}{{- '\\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("GRPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        grounded_kwargs = dict(
            tokenizer=tokenizer,
            num_examine=0,
            reward_fn_extraction_type=config.reward_fn.extraction_type,
            math_metric=config.reward_fn.math_metric,
            split="train",
            splitter=config.reward_fn.splitter,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            generation_reward_config=config.selfplay.reward.generation_reward_config,
            valid_program_filter=config.selfplay.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
            extract_code_block=config.selfplay.reward.extract_code_block,
            code_f_reward_type=config.selfplay.reward.code_f_reward_type,
            boxed_retry=config.reward_fn.boxed_retry,
            program_validity_config=config.selfplay.program_validity,
        )
        growth_reward_fn = GRPORewardManager(reward_mode=config.selfplay.proposer_reward_mode, **grounded_kwargs)
        reward_fn = GRPORewardManager(reward_mode=config.selfplay.solver_reward_mode, **grounded_kwargs)
        val_reward_fn = GRPORewardManager(reward_mode="grounded", split="test", **{k: v for k, v in grounded_kwargs.items() if k != "split"})
        intrinsic_val_reward_fn = None
        if config.selfplay.solver_reward_mode == INTRINSIC_SELF_CONSISTENCY_REWARD_MODE:
            intrinsic_val_reward_fn = GRPORewardManager(
                reward_mode=INTRINSIC_SELF_CONSISTENCY_REWARD_MODE,
                split="test",
                **{k: v for k, v in grounded_kwargs.items() if k != "split"},
            )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        wandb_tags = [
            "azr-grpo",
            config.selfplay.task,
            "solver-" + config.selfplay.solver_reward_mode,
            "proposer-" + config.selfplay.proposer_reward_mode,
            "questioner-solver" if config.selfplay.train_propose else "solver-only",
            config.selfplay.pred_data_mix_strategy,
            "executor-" + config.selfplay.executor,
            config.selfplay.data_selection_strategy.valid_program_filter,
            config.selfplay.gen_data_probabilities_strategy,
        ]
        if config.trainer.wandb_tags is not None:
            config.trainer.wandb_tags = wandb_tags + config.trainer.wandb_tags.split(",")
        else:
            config.trainer.wandb_tags = wandb_tags

        trainer = SelfPlayGRPORayTrainer(
            past_epoch_window=config.selfplay.past_epoch_window,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            growth_reward_fn=growth_reward_fn,
            intrinsic_val_reward_fn=intrinsic_val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()

    @staticmethod
    def _hydrate_compat_config(config):
        task_to_problem_type = {
            "pred_code_o": "code_o",
            "pred_code_f": "code_f",
            "pred_dsl_o": "dsl_o",
        }
        if config.selfplay.task not in task_to_problem_type:
            raise ValueError(f"Unsupported selfplay.task: {config.selfplay.task}")

        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False
        config.actor_rollout_ref.rollout.n = int(config.actor_rollout_ref.rollout.n)
        config.actor_rollout_ref.actor.use_kl_loss = True
        config.actor_rollout_ref.actor.kl_loss_coef = float(config.actor_rollout_ref.actor.kl_loss_coef)

        # Keep these hidden compatibility fields so we can reuse upstream helpers.
        config.selfplay.problem_types = [task_to_problem_type[config.selfplay.task]]
        solver_reward_mode = OmegaConf.select(config, "selfplay.solver_reward_mode")
        if solver_reward_mode is None:
            solver_reward_mode = OmegaConf.select(config, "selfplay.reward_mode", default="grounded")
        solver_reward_mode = canonicalize_solver_reward_mode(solver_reward_mode)
        proposer_reward_mode = OmegaConf.select(config, "selfplay.proposer_reward_mode", default="grounded")
        program_validity_mode = CodeIORewardManager._get_program_validity_mode(
            OmegaConf.select(config, "selfplay.program_validity", default={})
        )
        store_execution_output = bool(
            OmegaConf.select(config, "selfplay.program_validity.store_execution_output", default=True)
        )
        if solver_reward_mode not in {"grounded", INTRINSIC_SELF_CONSISTENCY_REWARD_MODE}:
            raise ValueError(f"Unsupported selfplay.solver_reward_mode: {solver_reward_mode}")
        if proposer_reward_mode not in {"grounded", "intrinsic", "intrinsic_vote"}:
            raise ValueError(f"Unsupported selfplay.proposer_reward_mode: {proposer_reward_mode}")
        if proposer_reward_mode == "grounded" and not store_execution_output:
            raise ValueError(
                "proposer_reward_mode=grounded requires "
                "selfplay.program_validity.store_execution_output=True."
            )
        difficulty_mode = str(
            OmegaConf.select(
                config,
                "selfplay.reward.generation_reward_config.difficulty_reward.mode",
                default="none",
            )
        ).strip().lower()
        if config.selfplay.train_propose and difficulty_mode == "none":
            print(
                "[selfplay-grpo] Disabling proposer training because "
                "difficulty_reward.mode=none implies no proposer optimization signal."
            )
            config.selfplay.train_propose = False
        config.selfplay.solver_reward_mode = solver_reward_mode
        config.selfplay.proposer_reward_mode = proposer_reward_mode
        config.selfplay.program_validity.mode = program_validity_mode
        config.selfplay.reward_mode = solver_reward_mode
        config.selfplay.pretrain_pred_steps = -1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(0)
    except Exception:
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)
