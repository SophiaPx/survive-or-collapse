"""Offline evaluation for code_f-trained checkpoints.

Reads one or more val parquets (mixing problem types: code_f, code_o, code_i),
runs vLLM greedy inference per checkpoint, and scores each row by its declared
`metric` field (pred_code_f, pred_code_o, pred_code_i). Writes a per-step,
per-data_source accuracy summary to CSV. Designed to be run AFTER training
(no live trainer state needed).

Why three metric types: a code_f-trained model is the trained-on task, but we
also want to probe whether code reasoning ability on cruxeval_o / cruxeval_i /
livecodebench (all carried by data/code_reason/test_answer.parquet) is
preserved or collapses under each gate configuration. This is the cross-
training-task × cross-evaluation-task probe (paper §3).

Scoring rules:
  - pred_code_o / pred_code_i: extract <answer>…</answer>, string-compare to
    extra_info['output'] / extra_info['input'] (or reward_model.ground_truth).
  - pred_code_f: extract `def f(...)`, exec it, call f(*hidden_input_args),
    compare to hidden_outputs. Pass = ALL hidden cases match (binary).

Greedy decode (temperature=0) for offline eval — matches HumanEval/MBPP convention.

Usage example::
    python scripts/eval_code_f_offline.py \
        --actor-dir artifacts/runs/selfplay_grpo/pred_code_f__solver-grounded__.../checkpoints/code_f/Qwen3-4B-Base/none \
        --run-name code_f_A_GG_exec \
        --steps 0,200,400,600 \
        --val-parquets data/code_f_val_humaneval_mbpp.parquet,data/code_reason/test_answer.parquet \
        --out-csv artifacts/code_f_offline_eval.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Output extraction ─────────────────────────────────────────────────────────

ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
PYTHON_BLOCK_RE = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
INPUT_BLOCK_RE = re.compile(r'```input\s*\n(.*?)```', re.DOTALL)
OUTPUT_BLOCK_RE = re.compile(r'```output\s*\n(.*?)```', re.DOTALL)


def extract_answer_string(text: str) -> Optional[str]:
    """For code_o / code_i: pull the answer span from the model output."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: try a fenced ```output / ```input block
    m = OUTPUT_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    m = INPUT_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def extract_code_block(text: str) -> Optional[str]:
    """For code_f: pull the Python code block from the model output."""
    m = PYTHON_BLOCK_RE.search(text)
    if m:
        return m.group(1)
    # Fallback: anything between <answer>...</answer> if it looks like code
    m = ANSWER_RE.search(text)
    if m and 'def ' in m.group(1):
        return m.group(1)
    return None


# ── code_f scoring (binary pass on all hidden cases) ─────────────────────────


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _disabled_input(*args, **kwargs):
    raise RuntimeError('interactive input disabled during offline eval')


def _safe_call(snippet: str, fn_name: str, args, timeout: int = 2) -> Optional[str]:
    ns: dict = {'input': _disabled_input}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(snippet, ns)
        fn = ns.get(fn_name)
        if fn is None:
            return None
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return repr(fn(*args))
    except Exception:
        return None
    finally:
        signal.alarm(0)


def _detect_fn_name(code: str) -> Optional[str]:
    m = re.search(r'def\s+(\w+)\s*\(', code)
    return m.group(1) if m else None


def score_code_f(generation: str, hidden_inputs: list[str], hidden_outputs: list[str]) -> int:
    """Returns 1 if ALL hidden test cases pass, else 0."""
    code = extract_code_block(generation)
    if code is None:
        return 0
    fn_name = _detect_fn_name(code)
    if fn_name is None:
        return 0
    for in_str, expected in zip(hidden_inputs, hidden_outputs):
        try:
            args = eval(f'[{in_str}]')
        except Exception:
            return 0
        got = _safe_call(code, fn_name, args)
        if got is None or got != expected:
            return 0
    return 1


def score_code_io(generation: str, gold: str) -> int:
    """Generic exact-match scoring for code_i / code_o."""
    pred = extract_answer_string(generation)
    if pred is None:
        return 0
    return int(pred.strip() == str(gold).strip())


def score_row(row: pd.Series, generation: str) -> int:
    ext = row['extra_info']
    metric = ext.get('metric')
    if metric == 'pred_code_f':
        hi = list(ext.get('hidden_inputs', []))
        ho = list(ext.get('hidden_outputs', []))
        return score_code_f(generation, hi, ho)
    if metric == 'pred_code_o':
        return score_code_io(generation, ext.get('output'))
    if metric == 'pred_code_i':
        return score_code_io(generation, ext.get('input'))
    return 0


# ── FSDP merge ───────────────────────────────────────────────────────────────


def merge_fsdp(actor_step_dir: Path, target_dir: Path) -> None:
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


def run_inference(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int = 2048,
    max_model_len: int = 32768,
) -> list[str]:
    # Greedy offline eval does not benefit from flashinfer sampling, and
    # disabling it avoids JIT-time ninja/toolchain issues on some nodes.
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    os.environ['PATH'] = f"{Path(sys.executable).resolve().parent}:{os.environ.get('PATH', '')}"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype='bfloat16',
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        tensor_parallel_size=8,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling)
    texts = [o.outputs[0].text for o in outputs]
    del llm
    return texts


# ── per-step evaluation ──────────────────────────────────────────────────────


def evaluate_step(
    step: int,
    actor_dir: Path,
    base_model: str,
    val_df: pd.DataFrame,
    run_name: str,
    tmp_root: Path,
    keep_merged: bool,
    max_model_len: int,
) -> list[dict]:
    if step == 0:
        model_path = base_model
        print(f"\n[step=0] base model: {base_model}")
    else:
        step_dir = actor_dir / f'global_step_{step}' / 'actor'
        merged_dir = tmp_root / f'merged_step_{step}'
        if not merged_dir.exists():
            merged_dir.mkdir(parents=True)
            merge_fsdp(step_dir, merged_dir)
        else:
            print(f"  [merge] reusing cached {merged_dir}")
        model_path = str(merged_dir)

    prompts = [row['prompt'][0]['content'] for _, row in val_df.iterrows()]
    print(f"  running inference on {len(prompts)} val items ...")
    generations = run_inference(model_path, prompts, max_model_len=max_model_len)

    records = []
    for (_, row), gen in zip(val_df.iterrows(), generations):
        correct = score_row(row, gen)
        records.append({
            'run': run_name,
            'step': step,
            'data_source': row['data_source'],
            'metric': row['extra_info'].get('metric'),
            'correct': correct,
        })

    if not keep_merged and step != 0:
        shutil.rmtree(tmp_root / f'merged_step_{step}', ignore_errors=True)
    return records


# ── summarisation ────────────────────────────────────────────────────────────


def summarise(records: list[dict]) -> list[dict]:
    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for r in records:
        buckets[(r['run'], r['step'], r['data_source'], r['metric'])].append(r['correct'])
    rows = []
    for (run, step, ds, metric), corrects in sorted(buckets.items()):
        rows.append({
            'run': run,
            'step': step,
            'data_source': ds,
            'metric': metric,
            'n': len(corrects),
            'accuracy': sum(corrects) / len(corrects),
        })
    return rows


def filter_overlong_rows(
    val_df: pd.DataFrame,
    base_model: str,
    max_model_len: int,
) -> pd.DataFrame:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    keep_rows = []
    skipped_rows = []
    for idx, row in val_df.iterrows():
        prompt = row['prompt'][0]['content']
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count > max_model_len:
            skipped_rows.append((idx, row['data_source'], token_count))
            continue
        keep_rows.append(idx)

    if skipped_rows:
        skipped_counts = Counter(ds for _, ds, _ in skipped_rows)
        print(
            f"Skipping {len(skipped_rows)} overlong prompts "
            f"(max_model_len={max_model_len}): {dict(skipped_counts)}"
        )
        for idx, ds, token_count in sorted(skipped_rows, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  skipped idx={idx} data_source={ds} prompt_tokens={token_count}")
    else:
        print(f"No overlong prompts detected for max_model_len={max_model_len}")

    return val_df.loc[keep_rows].reset_index(drop=True)


def append_csv(path: Path, rows: list[dict]) -> None:
    fields = ['run', 'step', 'data_source', 'metric', 'n', 'accuracy']
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"  → appended {len(rows)} summary rows to {path}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--actor-dir', required=True,
                   help='ckpt dir whose children are global_step_N/actor/...')
    p.add_argument('--run-name', required=True)
    p.add_argument('--steps', required=True, help='comma-separated, 0 = base')
    p.add_argument('--val-parquets', required=True,
                   help='comma-separated parquet paths; rows will be concatenated')
    p.add_argument('--base-model', default='Qwen/Qwen3-4B-Base')
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--out-csv', default=str(ROOT / 'artifacts/code_f_offline_eval.csv'))
    p.add_argument('--tmp-dir', default=None)
    p.add_argument('--keep-merged', action='store_true')
    args = p.parse_args()

    actor_dir = Path(args.actor_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(s.strip()) for s in args.steps.split(',')]

    parquet_paths = [Path(s.strip()) for s in args.val_parquets.split(',')]
    dfs = [pd.read_parquet(p) for p in parquet_paths]
    val_df = pd.concat(dfs, ignore_index=True)
    print(f"Val pool: {len(val_df)} rows from {len(parquet_paths)} parquet(s)")
    src_counts = val_df['data_source'].value_counts().to_dict()
    print(f"  per-source: {src_counts}")
    val_df = filter_overlong_rows(
        val_df=val_df,
        base_model=args.base_model,
        max_model_len=args.max_model_len,
    )
    filtered_src_counts = val_df['data_source'].value_counts().to_dict()
    print(f"Filtered val pool: {len(val_df)} rows")
    print(f"  per-source after filter: {filtered_src_counts}")

    tmp_root = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix='code_f_eval_'))
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"Tmp dir: {tmp_root}")

    for step in steps:
        print(f"\n{'='*60}\nEvaluating step={step}  run={args.run_name}")
        records = evaluate_step(
            step=step,
            actor_dir=actor_dir,
            base_model=args.base_model,
            val_df=val_df,
            run_name=args.run_name,
            tmp_root=tmp_root,
            keep_merged=args.keep_merged,
            max_model_len=args.max_model_len,
        )
        summary = summarise(records)
        append_csv(out_csv, summary)
        for r in summary:
            print(f"    {r['data_source']:>20s} ({r['metric']}): n={r['n']} acc={r['accuracy']:.3f}")

    print(f"\n[done] results: {out_csv}")


if __name__ == '__main__':
    main()
