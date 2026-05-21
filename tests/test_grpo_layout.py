from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_files_exist():
    assert (ROOT / "selfplay_grpo" / "main_grpo.py").is_file()
    assert (ROOT / "selfplay_grpo" / "configs" / "grpo_trainer.yaml").is_file()
    assert (ROOT / "selfplay_grpo" / "trainer" / "grpo" / "grpo_trainer.py").is_file()
    assert (ROOT / "selfplay_grpo" / "rewards" / "grpo_reward_manager.py").is_file()


def test_config_defaults_present():
    config_text = (ROOT / "selfplay_grpo" / "configs" / "grpo_trainer.yaml").read_text()
    assert "task: pred_code_o" in config_text
    assert "reward_mode: grounded" in config_text
    assert "solver_reward_mode:" in config_text
    assert "intrinsic_self_consistency" in config_text
    assert "difficulty_reward:" in config_text
    assert "label_source:" not in config_text
    assert "mode: none" in config_text
    assert "none disables proposer training" in config_text
    assert "proposer_reward_mode: grounded" in config_text
    assert "train_propose: True" in config_text
    assert "program_validity:" in config_text
    assert "mode: execution" in config_text
    assert "store_execution_output: True" in config_text
    assert "require_execution_output_for_dataset: True" in config_text
    assert "n: 16" in config_text
    assert "adv_estimator: grpo" in config_text
    assert "intrinsic_val_n:" in config_text



def test_run_script_exposes_memory_controls():
    run_script = (ROOT / "scripts" / "run_grpo.sh").read_text()
    assert "ACTOR_USE_DYNAMIC_BSZ" in run_script
    assert "PPO_MAX_TOKEN_LEN_PER_GPU" in run_script
    assert "ACTOR_PARAM_OFFLOAD" in run_script
    assert "ACTOR_OPTIMIZER_OFFLOAD" in run_script
    assert "TRAIN_PROPOSE=${TRAIN_PROPOSE:-True}" in run_script
    assert 'SOLVER_REWARD_MODE=${SOLVER_REWARD_MODE:-${REWARD_MODE}}' in run_script
    assert 'PROPOSER_REWARD_MODE=${PROPOSER_REWARD_MODE:-grounded}' in run_script
    assert 'PROPOSER_INTRINSIC_MODE=${PROPOSER_INTRINSIC_MODE:-none}' in run_script
    assert 'PROGRAM_VALIDITY_MODE=${PROGRAM_VALIDITY_MODE:-execution}' in run_script
    assert 'PROGRAM_VALIDITY_CHECK_DETERMINISM=${PROGRAM_VALIDITY_CHECK_DETERMINISM:-True}' in run_script
    assert 'PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT=${PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT:-True}' in run_script
    assert 'PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT=${PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT:-True}' in run_script
    assert 'selfplay.train_propose="${TRAIN_PROPOSE}"' in run_script
    assert 'selfplay.solver_reward_mode="${SOLVER_REWARD_MODE}"' in run_script
    assert 'selfplay.proposer_reward_mode="${PROPOSER_REWARD_MODE}"' in run_script
    assert 'selfplay.reward.generation_reward_config.difficulty_reward.mode="${PROPOSER_INTRINSIC_MODE}"' in run_script
    assert 'selfplay.program_validity.mode="${PROGRAM_VALIDITY_MODE}"' in run_script
    assert 'selfplay.program_validity.check_determinism="${PROGRAM_VALIDITY_CHECK_DETERMINISM}"' in run_script
    assert 'selfplay.program_validity.store_execution_output="${PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT}"' in run_script
    assert 'selfplay.program_validity.require_execution_output_for_dataset="${PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT}"' in run_script
    assert "VERL_SENTINEL=verl/trainer/ppo/ray_trainer.py" in run_script
    assert "Using installed verl" in run_script
    assert "EXTRACTION_TYPE=${EXTRACTION_TYPE:-none}" in run_script


def test_qwen3_scripts_use_conservative_defaults():
    selfplay_script = (ROOT / "scripts" / "selfplay" / "qwen3_4b.sh").read_text()
    seeding_script = (ROOT / "scripts" / "seeding" / "qwen3_4b.sh").read_text()

    for script_text in (selfplay_script, seeding_script):
        assert "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}" in script_text
        assert "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}" in script_text
        assert "PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}" in script_text
        assert "ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}" in script_text
        assert "PROPOSER_INTRINSIC_MODE=${PROPOSER_INTRINSIC_MODE:-none}" in script_text

    assert "TRAIN_PROPOSE=${TRAIN_PROPOSE:-False}" not in seeding_script
    assert "TRAIN_PROPOSE=${TRAIN_PROPOSE:-False}" not in selfplay_script



def test_fs_compat_exists():
    compat_file = ROOT / "selfplay_grpo" / "utils" / "fs_compat.py"
    assert compat_file.is_file()
    compat_text = compat_file.read_text()
    assert "copy_local_path_from_hdfs" in compat_text


def test_grpo_uses_task_native_code_parsing_defaults():
    config_text = (ROOT / "selfplay_grpo" / "configs" / "grpo_trainer.yaml").read_text()
    reward_manager_text = (ROOT / "selfplay_grpo" / "rewards" / "grpo_reward_manager.py").read_text()

    assert "extraction_type: none" in config_text
    assert "def _parse_prediction_answer" in reward_manager_text
    assert "def _canonicalize_output_answer" in reward_manager_text
    assert "reward_tensor[i, valid_response_length - 1] = 0.0" in reward_manager_text


def test_grpo_can_optionally_train_proposer():
    main_text = (ROOT / "selfplay_grpo" / "main_grpo.py").read_text()
    trainer_text = (ROOT / "selfplay_grpo" / "trainer" / "grpo" / "grpo_trainer.py").read_text()
    reward_manager_text = (ROOT / "selfplay_grpo" / "rewards" / "grpo_reward_manager.py").read_text()

    assert "difficulty_reward.mode=none implies no proposer optimization signal" in main_text
    assert "config.selfplay.train_propose = False" in main_text
    assert "INTRINSIC_SELF_CONSISTENCY_REWARD_MODE" in main_text
    assert "batch_multiplier = 2 if config.selfplay.train_propose else 1" in main_text
    assert 'config.selfplay.proposer_reward_mode' in main_text
    assert 'config.selfplay.solver_reward_mode' in main_text
    assert 'config.selfplay.program_validity' in main_text
    assert 'store_execution_output=True' in main_text
    assert "collect_batch=self.config.selfplay.train_propose" in trainer_text
    assert "DataProto.concat(batch_parts)" in trainer_text
    assert "intrinsic_self_consistency" in reward_manager_text
    assert "def _get_program_validity_mode" in reward_manager_text
    assert "def _run_generation_execution_check" in reward_manager_text
    assert "def _is_generation_sample_reward_eligible" in reward_manager_text
    assert "solver_accuracy" in reward_manager_text
    assert "difficulty" in reward_manager_text
    assert "difficulty_reward" in reward_manager_text
    assert "execution_output_available" in reward_manager_text
    assert "dataset_eligibility" in reward_manager_text
    assert "self_output_matches_execution" in reward_manager_text
    assert "grounded_accuracy" in reward_manager_text
    assert "vote_share_grounded_gap" in reward_manager_text
    assert "vote_share_grounded_abs_gap" in reward_manager_text
    assert 'Unsupported proposer reward mode' in reward_manager_text
