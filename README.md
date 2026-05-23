# Survive or Collapse

Code release for **"Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL"**.

> Self-play RL on code reasoning is governed by two distinct levers: a data-level **gate** that decides which proposer-generated tasks enter the training pool, and the **reward signal** that updates the policy on tasks already admitted. We show these levers are asymmetric — a strict gate is sufficient for stability under every reward variant; no reward variant is sufficient once the gate is removed.

Built on top of [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) and [verl](https://github.com/volcengine/verl). The trainer is GRPO-only (`selfplay_grpo/main_grpo.py`).

## What's here

```
selfplay_grpo/         Core package (trainer, rewards, executors)
  configs/grpo_trainer.yaml Hydra config (Appendix A hyperparameters)
  rewards/                      Reward managers — gate F_eps, grounded/intrinsic, codegen scorer
  trainer/grpo/                 GRPO Ray trainer with the asymmetric self-play loop
  utils/code_utils/             pebble-isolated Python sandbox
  utils/dsl_utils/              15-operator DSL interpreter (Appendix B)

scripts/
  run_grpo.sh                   Main entry point (all Hydra overrides documented inline)
  dsl/                          DSL configurations matching Table 1
  phase_diagram/                Continuous gate-leak sweep (Section 4)
  seeding/                      Generate seed pools for new base models
  build_codegen_val_parquet.py  Build HumanEvalPlus/MbppPlus in-loop val parquets
  eval_dsl_val_stratified.py    Offline DSL stratified eval

data/
  qwen3_4b_seed_io.jsonl        Coding-task seed pool (32 programs)
  dsl_seed_hard.jsonl           DSL-task seed pool (24 expressions)
  dsl_val_stratified.parquet    DSL validation set (300 expressions, depth 1-6)
  # Full codegen val parquets are not checked in (~175MB). Build them once after install:
  #   python scripts/build_codegen_val_parquet.py

evaluation/code_eval/data/      Raw evalplus jsonl source (for rebuilding the parquets)
tests/                          Sandbox + codegen-scorer unit tests
```

## Requirements

- **Hardware**: 8× A100 40GB (single node) for the headline runs. Smaller models / smaller batches will fit on less.
- **Software**: Python 3.10+, PyTorch, vLLM, Ray. Exact versions are pinned in `environment.yml`.

## Install

```bash
git clone <this-repo-url>
cd survive-or-collapse
conda env create -f environment.yml
conda activate azr-grpo

```

## Quickstart: reproduce GG+exec on the coding task (Table 1, row 1)

```bash
bash scripts/run_grpo.sh \
  selfplay.solver_reward_mode=grounded \
  selfplay.proposer_reward_mode=grounded \
  selfplay.program_validity.mode=execution
```

Logs and checkpoints land under `checkpoints/code_io/<project_name>/<experiment_name>/`. Validation accuracy on the in-domain probe shows up in wandb as `val/test_score/code_o`.

## Reproducing the paper

### Table 1 — self-play configurations

**Coding task (Python output prediction):** the cells differ only in three Hydra keys:
`selfplay.proposer_reward_mode ∈ {grounded, intrinsic}`,
`selfplay.solver_reward_mode ∈ {grounded, intrinsic_self_consistency}`,
`selfplay.program_validity.mode ∈ {execution, off}`.

**DSL task (controlled twin, deterministic 15-operator language):**
```bash
bash scripts/dsl/dsl_A_grounded_grounded_execution.sh   # GG+exec
```

### Section 4 — phase transitions under continuous gate relaxation

```bash
bash scripts/phase_diagram/phase_eps_0.05.sh
bash scripts/phase_diagram/phase_eps_0.10.sh
bash scripts/phase_diagram/phase_eps_0.20.sh
bash scripts/phase_diagram/phase_eps_0.40.sh
bash scripts/phase_diagram/phase_eps_0.70.sh
# eps=0.00 and eps=1.00 = the binary endpoints already covered by II+exec and II+off.
```

Adaptive schedule (Figure 3):
```bash
# After a stable strict-gate run reaches step 150, point CKPT in
# scripts/phase_diagram/run_p0ext_serial.sh at that checkpoint, then:
bash scripts/phase_diagram/run_p0ext_serial.sh
```

## In-loop HumanEvalPlus / MbppPlus validation

We package a live-policy codegen evaluator that runs HumanEvalPlus (164 problems) and MbppPlus (374 problems) periodically during training, against the in-memory policy — no checkpoint reload, no separate vLLM. Two config keys drive it:

```yaml
data:
  codegen_val_files:
    - data/codegen_val_humaneval_plus.parquet
    - data/codegen_val_mbpp_plus.parquet
trainer:
  codegen_val_freq: 50    # set 0 to disable
```

Both keys have sensible defaults in `selfplay_grpo/configs/grpo_trainer.yaml`. Metrics surface in wandb as `val/codegen/test_score/{humaneval_plus,mbpp_plus}` (headline = plus_pass binary, matching evalplus's stricter convention) plus auxiliary `val/codegen/codegen_base_pass` / `codegen_plus_pass` for the base/plus breakdown.

**Build the parquets** :
```bash
python scripts/build_codegen_val_parquet.py
# → data/codegen_val_humaneval_plus.parquet  (164 problems)
# → data/codegen_val_mbpp_plus.parquet       (374 problems)
```


## Generating seed pools for new base models

```bash
bash scripts/seeding/qwen3_4b.sh
```

## 📌 Citation
```
@misc{pu2026survivecollapseasymmetricroles,
      title={Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL}, 
      author={Sophia Xiao Pu and Zhaotian Weng and Chengzhi Liu and Jayanth Srinivasa and Gaowen Liu and William Yang Wang and Xin Eric Wang},
      year={2026},
      eprint={2605.22217},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.22217}, 
}
```
## License

[MIT](LICENSE). This project builds on [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) and [verl](https://github.com/volcengine/verl); their original licenses still apply to their respective source files.
