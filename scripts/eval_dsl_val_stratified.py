"""
Run offline evaluation for DSL training checkpoints on the stratified
validation set and report per-depth accuracy.

Workflow:
  1. For each requested checkpoint step, merge the FSDP shards into HF weights
     with verl.model_merger.
  2. Run batch inference with vLLM on dsl_val_stratified.parquet using greedy
     decoding.
  3. Parse the integer answer from each model output and compare it with the
     gold label.
  4. Group results by depth, compute accuracy, and append them to --out-csv.

Example usage for a single run with selected steps:
    python scripts/eval_dsl_val_stratified.py \
        --actor-dir artifacts/runs/selfplay_grpo/pred_dsl_o__solver-grounded__proposer-grounded-hard__program-execution__qwen3-4b-base/selfplay_grpo_qwen3_4b.../checkpoints/dsl_o/Qwen3-4B-Base/none \
        --run-name DSL-A \
        --steps 0,10,50,100,200,500,1000,2000 \
        --val-parquet data/dsl_val_stratified.parquet \
        --out-csv artifacts/dsl_stratified_eval.csv

step=0 uses the base model (Qwen/Qwen3-4B-Base) without loading any
checkpoint weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Answer parsing ───────────────────────────────────────────────────────────

_OUTPUT_BLOCK_RE = re.compile(r'```output\s*\n?(-?\d+)\n?```', re.IGNORECASE)
_BARE_INT_RE     = re.compile(r'(?<!\d)(-?\d{1,8})(?!\d)')


def extract_integer(text: str) -> Optional[int]:
    """Extract an integer answer from model output text.

    Prefer an ```output``` block; otherwise fall back to the last integer in
    the text.
    """
    m = _OUTPUT_BLOCK_RE.search(text)
    if m:
        return int(m.group(1))
    ints = _BARE_INT_RE.findall(text)
    if ints:
        return int(ints[-1])
    return None


# ── FSDP merge ───────────────────────────────────────────────────────────────

def merge_fsdp(actor_step_dir: Path, target_dir: Path) -> None:
    """Merge FSDP shards from global_step_X/actor/ into HF weights."""
    import subprocess
    cmd = [
        sys.executable, '-m', 'verl.model_merger', 'merge',
        '--backend', 'fsdp',
        '--local_dir', str(actor_step_dir),
        '--target_dir', str(target_dir),
    ]
    print(f"  [merge] {actor_step_dir.name} → {target_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"model_merger failed:\n{result.stderr}")


# ── vLLM inference ───────────────────────────────────────────────────────────

def run_inference(model_path: str, prompts: list[str],
                  max_new_tokens: int = 512,
                  temperature: float = 0.0) -> list[str]:
    """Run greedy decoding with vLLM and return the generated texts."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype='bfloat16', max_model_len=2048,
              gpu_memory_utilization=0.85, trust_remote_code=True,
              tensor_parallel_size=8)
    sampling = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outputs  = llm.generate(prompts, sampling)
    texts    = [o.outputs[0].text for o in outputs]
    del llm
    return texts


# ── Single-step evaluation ───────────────────────────────────────────────────

def evaluate_step(
    step: int,
    actor_dir: Path,
    base_model: str,
    val_df: pd.DataFrame,
    run_name: str,
    tmp_root: Path,
    keep_merged: bool = False,
) -> list[dict]:
    """Merge the checkpoint or use the base model, then return per-depth results."""

    if step == 0:
        model_path = base_model
        print(f"\n[step=0] using base model: {base_model}")
    else:
        step_dir    = actor_dir / f'global_step_{step}' / 'actor'
        merged_dir  = tmp_root / f'merged_step_{step}'
        if not merged_dir.exists():
            merged_dir.mkdir(parents=True)
            merge_fsdp(step_dir, merged_dir)
        else:
            print(f"  [merge] reusing cached {merged_dir}")
        model_path = str(merged_dir)

    # Build prompts in chat format.
    prompts = [row['prompt'][0]['content'] for _, row in val_df.iterrows()]

    print(f"  running inference on {len(prompts)} val items ...")
    generations = run_inference(model_path, prompts)

    # Evaluate generations against gold answers.
    records = []
    for (_, row), gen in zip(val_df.iterrows(), generations):
        gold  = int(row['extra_info']['output'])
        depth = int(row['extra_info']['depth'])
        pred  = extract_integer(gen)
        correct = (pred == gold) if pred is not None else False
        records.append({'run': run_name, 'step': step, 'depth': depth,
                        'correct': int(correct), 'pred': pred, 'gold': gold})

    if not keep_merged and step != 0:
        shutil.rmtree(merged_dir, ignore_errors=True)

    return records


# ── Aggregate and write CSV ──────────────────────────────────────────────────

def summarise(records: list[dict]) -> list[dict]:
    """Aggregate accuracy by (run, step, depth)."""
    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for r in records:
        buckets[(r['run'], r['step'], r['depth'])].append(r['correct'])
    rows = []
    for (run, step, depth), corrects in sorted(buckets.items()):
        rows.append({'run': run, 'step': step, 'depth': depth,
                     'n': len(corrects), 'accuracy': sum(corrects) / len(corrects)})
    return rows


def append_csv(path: Path, rows: list[dict]) -> None:
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'step', 'depth', 'n', 'accuracy'])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"  → appended {len(rows)} rows to {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor-dir', required=True,
                        help='Path to the checkpoints/dsl_o/Qwen3-4B-Base/none/ directory')
    parser.add_argument('--run-name', required=True,
                        help='Run identifier (for example DSL-A) written to the CSV run column')
    parser.add_argument('--steps', required=True,
                        help='Comma-separated list of steps to evaluate (0 = base model)')
    parser.add_argument('--val-parquet', default=str(ROOT / 'data/dsl_val_stratified.parquet'))
    parser.add_argument('--base-model', default='Qwen/Qwen3-4B-Base')
    parser.add_argument('--out-csv', default=str(ROOT / 'artifacts/dsl_stratified_eval.csv'))
    parser.add_argument('--tmp-dir', default=None,
                        help='Temporary directory for merged HF weights (default: system tmpdir)')
    parser.add_argument('--keep-merged', action='store_true',
                        help='Keep merged HF weights on disk for reuse')
    args = parser.parse_args()

    actor_dir = Path(args.actor_dir)
    out_csv   = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(s.strip()) for s in args.steps.split(',')]
    val_df = pd.read_parquet(args.val_parquet)
    print(f"Val set: {len(val_df)} samples, depths={sorted(val_df['extra_info'].apply(lambda x: x['depth']).unique())}")

    tmp_root = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix='dsl_ckpt_'))
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"Tmp dir: {tmp_root}")

    all_records = []
    for step in steps:
        print(f"\n{'='*60}")
        print(f"Evaluating step={step}  run={args.run_name}")
        records = evaluate_step(
            step=step,
            actor_dir=actor_dir,
            base_model=args.base_model,
            val_df=val_df,
            run_name=args.run_name,
            tmp_root=tmp_root,
            keep_merged=args.keep_merged,
        )
        all_records.extend(records)
        # Append after each step to avoid losing results on interruption.
        summary = summarise(records)
        append_csv(out_csv, summary)

    print(f"\n[done] total {len(all_records)} sample-level results")
    print(f"Results in: {out_csv}")


if __name__ == '__main__':
    main()
