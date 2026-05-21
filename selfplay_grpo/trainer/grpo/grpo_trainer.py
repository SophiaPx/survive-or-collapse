import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import compute_advantage, reduce_metrics, compute_timing_metrics
from verl.utils.debug import marked_timer

from selfplay_grpo.trainer.base.code_io_trainer import (
    CodeIORayPPOTrainer,
    compute_data_metrics,
    process_elements,
)
from selfplay_grpo.utils.logging_utils.stdout import PrettyPrinter
from selfplay_grpo.utils.tracking import ReasonRLTracking


class SelfPlayGRPORayTrainer(CodeIORayPPOTrainer):
    def __init__(
        self,
        growth_reward_fn,
        intrinsic_val_reward_fn=None,
        *args,
        **kwargs,
    ):
        self.growth_reward_fn = growth_reward_fn
        self.intrinsic_val_reward_fn = intrinsic_val_reward_fn
        super().__init__(*args, **kwargs)
        self.public_task = self.config.selfplay.task
        self.problem_type = self.public_task.replace("pred_", "")
        self.pred_problem_type = self.public_task
        self.gen_problem_type = f"gen_{self.problem_type}"

    def _get_code_f_uniqueness_gate_config(self):
        gate_cfg = OmegaConf.select(self.config, "selfplay.program_validity.code_f_uniqueness_gate", default={})
        return gate_cfg or {}

    def _use_code_f_uniqueness_gate(self, problem_type: str) -> bool:
        return problem_type == "gen_code_f" and bool(self._get_code_f_uniqueness_gate_config().get("enabled", False))

    @staticmethod
    def _extract_code_f_candidate_functions(entries):
        candidates = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            snippet = str(entry.get("snippet", "")).strip()
            if not snippet:
                continue
            imports = entry.get("imports", [])
            if isinstance(imports, np.ndarray):
                imports = imports.tolist()
            if imports is None:
                imports = []
            imports = [str(imp) for imp in imports]
            key = (snippet, tuple(sorted(imports)))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "snippet": snippet,
                    "original_snippet": str(entry.get("original_snippet", snippet)).strip(),
                    "imports": imports,
                }
            )
        return candidates

    def _build_code_f_candidate_functions(self):
        gate_cfg = self._get_code_f_uniqueness_gate_config()
        sources = str(gate_cfg.get("candidate_sources", "seed_and_problem")).strip().lower()
        max_seed_candidates = gate_cfg.get("max_seed_candidates")
        max_problem_candidates = gate_cfg.get("max_problem_candidates")

        source_to_dataset = {
            "seed": ("seed",),
            "problem": ("problem",),
            "seed_and_problem": ("seed", "problem"),
        }
        dataset_names = source_to_dataset.get(sources)
        if dataset_names is None:
            raise ValueError(f"Invalid code_f uniqueness candidate_sources: {sources}")

        requested = []
        if "seed" in dataset_names:
            requested.append(self.dataset_manager.get_dataset.remote("seed"))
        if "problem" in dataset_names:
            requested.append(self.dataset_manager.get_dataset.remote("problem"))
        fetched = ray.get(requested) if requested else []

        merged_candidates = []
        fetch_idx = 0
        if "seed" in dataset_names:
            seed_entries = fetched[fetch_idx]
            fetch_idx += 1
            if max_seed_candidates is not None:
                seed_entries = seed_entries[-int(max_seed_candidates):]
            merged_candidates.extend(self._extract_code_f_candidate_functions(seed_entries))
        if "problem" in dataset_names:
            problem_entries = fetched[fetch_idx]
            if max_problem_candidates is not None:
                problem_entries = problem_entries[-int(max_problem_candidates):]
            merged_candidates.extend(self._extract_code_f_candidate_functions(problem_entries))

        return self._extract_code_f_candidate_functions(merged_candidates)

    def fit(self):
        logger = ReasonRLTracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            tags=self.config.trainer.wandb_tags,
            resume="must" if self.config.trainer.resume_mode == "auto" and self.config.trainer.wandb_run_id is not None else False,
            run_id=self.config.trainer.wandb_run_id if self.config.trainer.wandb_run_id is not None else None,
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.config.actor_rollout_ref.model.pretrained_tokenizer:
            self.tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"

        if self.val_reward_fn is not None and self.config.trainer.get("val_only", False):
            val_metrics = self._validate(force_intrinsic=True)
            val_metrics.update(self._validate_codegen())
            logger.log(data=val_metrics, step=self.global_steps)
            return

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True) and self.global_steps == 0:
            val_metrics = self._validate()
            val_metrics.update(self._validate_codegen())
            logger.log(data=val_metrics, step=self.global_steps)

        self._initialize_task_datasets()
        self.global_steps += 1

        while self.global_steps < self.total_training_steps:
            metrics = {}
            timing_raw = {}

            with marked_timer("step", timing_raw):
                if self.global_steps - self._last_cleanup_step >= self._cleanup_frequency:
                    with marked_timer("cleanup", timing_raw):
                        self.cleanup()
                    self._last_cleanup_step = self.global_steps

                growth_batch = self._run_online_growth(
                    metrics,
                    timing_raw,
                    collect_batch=self.config.selfplay.train_propose,
                )
                pred_batch = self._build_grpo_pred_batch(metrics, timing_raw)
                batch_parts = [pred_batch]
                if growth_batch is not None:
                    batch_parts.insert(0, growth_batch)
                batch = DataProto.concat(batch_parts) if len(batch_parts) > 1 else pred_batch

                with marked_timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    with marked_timer("testing", timing_raw):
                        metrics.update(self._validate())

                codegen_freq = int(self.config.trainer.get('codegen_val_freq', 0) or 0)
                if codegen_freq > 0 and self.global_steps % codegen_freq == 0:
                    with marked_timer("codegen_testing", timing_raw):
                        metrics.update(self._validate_codegen())

                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with marked_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            metrics.update({
                "training/global_step": self.global_steps,
                "training/task": self.public_task,
                "training/train_propose": float(self.config.selfplay.train_propose),
            })
            if growth_batch is not None:
                growth_metrics = compute_data_metrics(batch=growth_batch, use_critic=False, tokenizer=self.tokenizer)
                growth_metrics = {f"{self.gen_problem_type}/{k}": v for k, v in growth_metrics.items()}
                metrics.update(growth_metrics)

            pred_metrics = compute_data_metrics(batch=pred_batch, use_critic=False, tokenizer=self.tokenizer)
            pred_metrics = {f"{self.pred_problem_type}/{k}": v for k, v in pred_metrics.items()}
            metrics.update(pred_metrics)
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            logger.log(data=metrics, step=self.global_steps)

            self.global_steps += 1

        final_val = self._validate(force_intrinsic=True)
        final_val.update(self._validate_codegen())
        logger.log(data=final_val, step=self.global_steps)
        if self.config.trainer.save_freq > 0:
            self._save_checkpoint()

    def _initialize_task_datasets(self):
        if self.loaded_datasets:
            return

        seed_dataset = []
        code_f_dataset = []

        if self.config.selfplay.seed_dataset is not None:
            with open(self.config.selfplay.seed_dataset, "r") as file:
                seed_dataset = [json.loads(line) for line in file]
            seed_dataset = seed_dataset[:self.config.selfplay.data_selection_strategy.data_len * self.config.selfplay.data_selection_strategy.seed_batch_factor]
            if self.problem_type == "code_f":
                ray.get(self.dataset_manager.update_seed.remote(seed_dataset))

        if self.problem_type == "code_f" and self.config.selfplay.code_f_seed_dataset is not None:
            with open(self.config.selfplay.code_f_seed_dataset, "r") as file:
                code_f_dataset = [json.loads(line) for line in file]
            code_f_dataset = code_f_dataset[:self.config.selfplay.data_selection_strategy.data_len * self.config.selfplay.data_selection_strategy.seed_batch_factor]

        need_seed_dataset = len(seed_dataset) == 0
        need_code_f_dataset = self.problem_type == "code_f" and len(code_f_dataset) == 0
        if need_seed_dataset or need_code_f_dataset:
            generated_seed, _, generated_code_f = self._init_seed_dataset(problem_types=self.config.selfplay.problem_types)
            if need_seed_dataset:
                seed_dataset = generated_seed
            if need_code_f_dataset:
                code_f_dataset = generated_code_f

        if self.problem_type in ("code_o", "dsl_o"):
            for item in seed_dataset:
                item['_is_seed'] = True
            processed_seed_dataset = process_elements(seed_dataset)
            ray.get(self.dataset_manager.add_output_batch.remote(processed_seed_dataset, self.global_steps))
        elif self.problem_type == "code_f":
            processed_code_f_dataset = process_elements(code_f_dataset)
            ray.get(self.dataset_manager.add_problem_batch.remote(processed_code_f_dataset, self.global_steps))
        else:
            raise ValueError(f"Unsupported task: {self.public_task}")

    def _run_online_growth(self, metrics: dict, timing_raw: dict, collect_batch: bool = False):
        data_len = self.config.data.train_batch_size * self.config.selfplay.data_selection_strategy.update_iteration
        growth_dataloader = self._create_train_code_gen_dataloader(problem_type=self.problem_type, data_len=data_len)
        if growth_dataloader is None:
            metrics[f"gen_{self.problem_type}/skipped_empty_dataset"] = 1.0
            return None

        batch_dict = next(growth_dataloader)
        batch = DataProto.from_single_dict(batch_dict)
        growth_batch, growth_metrics = self._run_reward_batch(
            batch=batch,
            metrics=metrics,
            timing_raw=timing_raw,
            problem_type=self.gen_problem_type,
            reward_fn=self.growth_reward_fn,
            update_dataset=True,
            compute_advantages=collect_batch,
        )
        metrics.update(growth_metrics)
        return growth_batch if collect_batch else None

    def _build_grpo_pred_batch(self, metrics: dict, timing_raw: dict) -> DataProto:
        data_len = self.config.data.train_batch_size * self.config.selfplay.data_selection_strategy.update_iteration
        pred_dataloader = self._create_train_code_pred_dataloader(problem_type=self.problem_type, data_len=data_len)
        batch_dict = next(pred_dataloader)
        batch = DataProto.from_single_dict(batch_dict)
        pred_batch, pred_metrics = self._run_reward_batch(
            batch=batch,
            metrics=metrics,
            timing_raw=timing_raw,
            problem_type=self.pred_problem_type,
            reward_fn=self.reward_fn,
            update_dataset=False,
            compute_advantages=True,
        )
        metrics.update(pred_metrics)
        return pred_batch

    def _run_reward_batch(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
        problem_type: str,
        reward_fn,
        update_dataset: bool,
        compute_advantages: bool,
    ):
        gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": False,
            "n": self.config.actor_rollout_ref.rollout.n,
        }
        with marked_timer(f"gen/{problem_type}", timing_raw):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        batch.batch["response_mask"] = batch.batch["attention_mask"][:, -batch.batch["responses"].size(1):]

        if getattr(self.config.trainer, "balance_batch", False):
            self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        if compute_advantages:
            with marked_timer(f"old_log_prob/{problem_type}", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                with marked_timer(f"ref/{problem_type}", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch) if not self.ref_in_actor else self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

        reward_kwargs = {
            "data": batch,
            "problem_type": problem_type,
            "executor": self._executor,
        }
        if problem_type.startswith("gen"):
            input_type_counters, output_type_counters, error_type_counters = None, None, None
            if problem_type == "gen_code_i":
                input_type_counters = ray.get(self.dataset_manager.get_type_counter.remote("input"))
            elif problem_type == "gen_code_o":
                output_type_counters = ray.get(self.dataset_manager.get_type_counter.remote("output"))
            elif problem_type == "gen_code_e":
                error_type_counters = ray.get(self.dataset_manager.get_type_counter.remote("error"))
            elif problem_type == "gen_code_f":
                input_type_counters = ray.get(self.dataset_manager.get_type_counter.remote("input"))
                output_type_counters = ray.get(self.dataset_manager.get_type_counter.remote("output"))
            elif problem_type == "gen_dsl_o":
                output_type_counters = ray.get(self.dataset_manager.get_type_counter.remote("output"))

            reward_kwargs.update({
                "rollout_actor_wg": self.actor_rollout_wg,
                "banned_words": self.config.selfplay.data_selection_strategy.banned_words,
                "n_samples": self.config.selfplay.reward.n_samples,
                "input_type_counters": input_type_counters,
                "output_type_counters": output_type_counters,
                "error_type_counters": error_type_counters,
            })
            if self._use_code_f_uniqueness_gate(problem_type):
                reward_kwargs["code_f_candidate_functions"] = self._build_code_f_candidate_functions()

        with marked_timer(f"reward_fn/{problem_type}", timing_raw):
            reward_tensor, train_metrics, valid_programs, _ = reward_fn(**reward_kwargs)

        train_metrics = {f"{problem_type}/{k}": np.mean(v) if isinstance(v, list) else v for k, v in train_metrics.items()}
        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = reward_tensor

        if update_dataset and valid_programs:
            processed_programs = process_elements(valid_programs)
            if problem_type.endswith("code_o") or problem_type.endswith("dsl_o"):
                ray.get(self.dataset_manager.add_output_batch.remote(processed_programs, self.global_steps))
            elif problem_type.endswith("code_f"):
                ray.get(self.dataset_manager.add_problem_batch.remote(processed_programs, self.global_steps))
            train_metrics[f"{problem_type}/num_valid_programs"] = len(valid_programs)

        if compute_advantages:
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
            )
        return batch, train_metrics

    def _should_run_intrinsic_validation(self, force: bool = False) -> bool:
        if self.intrinsic_val_reward_fn is None:
            return False

        intrinsic_val_freq = int(self.config.eval.get("intrinsic_val_freq", 1))
        if intrinsic_val_freq <= 0:
            return False

        if force or self.global_steps == 0:
            return True

        test_freq = max(int(self.config.trainer.test_freq), 1)
        return self.global_steps % (test_freq * intrinsic_val_freq) == 0

    def _validate(self, force_intrinsic: bool = False):
        metrics = {}
        metrics.update(self._run_validation_pass(reward_fn=self.val_reward_fn, metric_prefix="val/grounded", do_sample=False, n=1, alias_primary=True))
        if self._should_run_intrinsic_validation(force=force_intrinsic):
            metrics.update(
                self._run_validation_pass(
                    reward_fn=self.intrinsic_val_reward_fn,
                    metric_prefix="val/intrinsic",
                    do_sample=True,
                    n=self.config.eval.intrinsic_val_n,
                    alias_primary=False,
                )
            )
        return metrics

    def _validate_codegen(self):
        """Periodic HumanEvalPlus / MbppPlus validation against the live policy.
        No-op if data.codegen_val_files is unset.
        """
        if getattr(self, 'codegen_val_dataloader', None) is None:
            return {}
        return self._run_validation_pass(
            reward_fn=self.val_reward_fn,
            metric_prefix="val/codegen",
            do_sample=False,
            n=1,
            alias_primary=False,
            dataloader=self.codegen_val_dataloader,
        )

    def _get_validation_chunk_size(self, n: int) -> int:
        max_sequences_per_chunk = max(
            self.config.data.train_batch_size * max(1, self.config.actor_rollout_ref.rollout.n),
            self.actor_rollout_wg.world_size,
        )
        return max(1, max_sequences_per_chunk // max(1, n))

    def _run_validation_pass(self, reward_fn, metric_prefix: str, do_sample: bool, n: int, alias_primary: bool, dataloader=None):
        reward_tensor_lst = []
        data_source_lst = []
        all_eval_metrics = defaultdict(list)

        dl = dataloader if dataloader is not None else self.val_dataloader
        for test_data in dl:
            full_test_batch = DataProto.from_single_dict(test_data)
            if self.config.reward_model.enable and full_test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Avoid carrying mutable per-generation metadata such as timing across chunks.
            full_test_batch.meta_info = {}
            full_test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(full_test_batch.batch))], dtype=object)
            validation_chunk_size = self._get_validation_chunk_size(n=n)
            num_chunks = (len(full_test_batch) + validation_chunk_size - 1) // validation_chunk_size

            for chunk_idx, start in enumerate(range(0, len(full_test_batch), validation_chunk_size), start=1):
                end = min(start + validation_chunk_size, len(full_test_batch))
                test_batch = full_test_batch[start:end]
                test_batch.meta_info = {}
                test_gen_batch = test_batch.pop(["input_ids", "attention_mask", "position_ids"])
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": do_sample,
                    "validate": True,
                    "n": n,
                }

                # vLLM validation forces n=1 internally, so repeat the prompts up front
                # when we need multiple samples for intrinsic majority-vote evaluation.
                if n > 1:
                    test_batch = test_batch.repeat(repeat_times=n, interleave=True)
                    test_gen_batch = test_gen_batch.repeat(repeat_times=n, interleave=True)

                print(
                    f"[validation] generation_start prefix={metric_prefix} "
                    f"chunk={chunk_idx}/{num_chunks} prompts={end - start} expanded_batch={len(test_gen_batch)} "
                    f"do_sample={do_sample} n={n}"
                )
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                test_batch = test_batch.union(test_output_gen_batch)
                print(
                    f"[validation] reward_fn_start prefix={metric_prefix} "
                    f"chunk={chunk_idx}/{num_chunks} batch_size={len(test_batch)} do_sample={do_sample} n={n}"
                )

                reward_tensor, eval_metrics, _, _ = reward_fn(
                    test_batch,
                    problem_type=None,
                    executor=self._executor,
                )
                print(
                    f"[validation] reward_fn_done prefix={metric_prefix} "
                    f"chunk={chunk_idx}/{num_chunks} batch_size={len(test_batch)} reward_shape={tuple(reward_tensor.shape)}"
                )
                chunk_size = reward_tensor.shape[0]
                for k, v in eval_metrics.items():
                    metric_value = np.mean(v) if isinstance(v, list) else float(v)
                    all_eval_metrics[k].append((metric_value, chunk_size))

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        metric_dict = {}
        grouped = defaultdict(list)
        for i in range(reward_tensor.shape[0]):
            grouped[data_sources[i]].append(reward_tensor[i].item())
        for data_source, rewards in grouped.items():
            value = np.mean(rewards)
            if alias_primary and metric_prefix == "val/grounded":
                metric_dict[f"val/test_score/{data_source}"] = value
            else:
                metric_dict[f"{metric_prefix}/test_score/{data_source}"] = value

        for k, v in all_eval_metrics.items():
            total_weight = sum(weight for _, weight in v)
            metric_dict[f"{metric_prefix}/{k}"] = sum(value * weight for value, weight in v) / max(total_weight, 1)
        return metric_dict

    def _save_checkpoint(self):
        super()._save_checkpoint()
        self._save_datasets(Path(self.config.trainer.default_local_dir) / "datasets")
        PrettyPrinter.status("SAVE", f"Saved checkpoint to {self.config.trainer.default_local_dir}", "success")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps == 0:
            PrettyPrinter.section_header("Training from scratch")
        else:
            PrettyPrinter.section_header(f"Resuming training from checkpoint, step {self.global_steps}")

        code_dir = Path(self.config.trainer.default_local_dir) / "code"
        self._code_dir = code_dir
        self.loaded_datasets = False
        dataset_state = os.path.join(self.config.trainer.default_local_dir, "datasets", "datasets.pkl")
        if self.config.trainer.resume_mode == "auto" and os.path.exists(dataset_state):
            self._load_datasets(self.config.trainer.default_local_dir)
        elif self.config.trainer.resume_mode == "disable":
            if code_dir.exists():
                for file in code_dir.glob("**/*"):
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        file.rmdir()
            PrettyPrinter.status("Directory", f"Cleaned existing code directory at {code_dir}", "info")
        elif not code_dir.exists():
            code_dir.mkdir(parents=True, exist_ok=True)
            PrettyPrinter.status("Directory", f"Created new code directory at {code_dir}", "info")
