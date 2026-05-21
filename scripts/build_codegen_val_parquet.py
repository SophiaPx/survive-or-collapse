"""Build code-generation validation parquets from HumanEvalPlus and MbppPlus.

Outputs (default):
  data/codegen_val_humaneval_plus.parquet   (164 rows)
  data/codegen_val_mbpp_plus.parquet        (378 rows)
  data/codegen_val_humaneval_plus_smoke.parquet  (first 8 rows, for fast tests)

Each row carries pre-computed gold outputs for base_input and plus_input
(run the canonical_solution at build time once, so the in-loop scorer only
has to execute the model's generated function once per test). Schema mirrors
code_f_val_humaneval_mbpp.parquet plus four extra keys in extra_info:
entry_point / base_inputs / base_outputs / plus_inputs / plus_outputs / atol.

The in-loop scorer (grpo_reward_manager codegen branch) reads the parquet,
extracts the model's generated function via parse_code_function, then for each
(arg-tuple, gold-repr) pair executes `snippet + f"\\nf = {entry_point}\\nrepr(f({args}))"`
in the pebble-isolated PythonExecutor and compares.

Run:
    python scripts/build_codegen_val_parquet.py
"""

import argparse
import json
import signal
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / 'evaluation' / 'code_eval' / 'data'
OUT_DIR = REPO / 'data'

EXEC_TIMEOUT_SEC = 5

CODEGEN_PROMPT_TEMPLATE = """You are given the start of a Python function. Complete the function so that it satisfies its docstring. Return ONLY the full function definition (including the signature you were given), wrapped in a ```python``` code block.

```python
{prompt}
```
"""


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def safe_exec_call(snippet: str, fn_name: str, args):
    """Exec `snippet`, then call `fn_name(*args)`. Returns repr(result) or None on error/timeout."""
    ns: dict = {}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(EXEC_TIMEOUT_SEC)
    try:
        exec(snippet, ns)
        fn = ns.get(fn_name)
        if fn is None:
            return None
        result = fn(*args)
        return repr(result)
    except Exception:
        return None
    finally:
        signal.alarm(0)


def _compute_outputs(snippet: str, fn_name: str, inputs):
    """Returns (input_reprs, output_reprs, n_skipped).

    Stores each input as `'arg1, arg2, ...'` (comma-joined reprs of positional
    args) and each output as repr(result). Matches the string-list schema used
    by data/code_f_val_humaneval_mbpp.parquet so pyarrow can serialize cleanly.
    Skips inputs where the canonical solution raises or times out.
    """
    kept_in, kept_out, skipped = [], [], 0
    for raw in inputs:
        args = raw if isinstance(raw, list) else [raw]
        out = safe_exec_call(snippet, fn_name, args)
        if out is None:
            skipped += 1
            continue
        kept_in.append(', '.join(repr(a) for a in args))
        kept_out.append(out)
    return kept_in, kept_out, skipped


def build_row(eval_row, data_source, idx):
    snippet = eval_row['prompt'] + eval_row['canonical_solution']
    entry_point = eval_row['entry_point']

    base_inputs, base_outputs, base_skip = _compute_outputs(snippet, entry_point, eval_row['base_input'])
    plus_inputs, plus_outputs, plus_skip = _compute_outputs(snippet, entry_point, eval_row.get('plus_input', []))

    if not base_inputs:
        return None, (base_skip, plus_skip)

    user_msg = CODEGEN_PROMPT_TEMPLATE.format(prompt=eval_row['prompt'])
    return {
        'data_source': data_source,
        'prompt': [{'role': 'user', 'content': user_msg}],
        'problem': eval_row['prompt'],
        'ability': 'code',
        'reward_model': {'style': 'rule', 'ground_truth': snippet},
        'extra_info': {
            'split': 'test',
            'index': idx,
            'metric': 'pred_codegen',
            'problem_type': 'codegen',
            'task_id': eval_row['task_id'],
            'entry_point': entry_point,
            'base_inputs': base_inputs,
            'base_outputs': base_outputs,
            'plus_inputs': plus_inputs,
            'plus_outputs': plus_outputs,
            'atol': float(eval_row.get('atol', 0) or 0),
            'imports': [],
        },
    }, (base_skip, plus_skip)


def build_parquet(jsonl_path: Path, data_source: str, out_path: Path):
    rows = []
    total_base_skip = total_plus_skip = total_problems = dropped = 0
    for line in open(jsonl_path):
        eval_row = json.loads(line)
        row, (bs, ps) = build_row(eval_row, data_source, idx=total_problems)
        total_problems += 1
        total_base_skip += bs
        total_plus_skip += ps
        if row is None:
            dropped += 1
            continue
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(
        f'  {data_source}: wrote {len(df)} rows to {out_path.name} '
        f'(dropped {dropped} problems, skipped {total_base_skip}/{total_plus_skip} base/plus tests)'
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--humaneval-jsonl', default=str(DATA_DIR / 'HumanEvalPlus.jsonl'))
    parser.add_argument('--mbpp-jsonl', default=str(DATA_DIR / 'MbppPlus.jsonl'))
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    parser.add_argument('--smoke-rows', type=int, default=8,
                        help='Also emit a smoke parquet with the first N HumanEval rows.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print('=== Building codegen val parquets ===')

    df_he = build_parquet(Path(args.humaneval_jsonl), 'humaneval_plus',
                          out_dir / 'codegen_val_humaneval_plus.parquet')
    df_mb = build_parquet(Path(args.mbpp_jsonl), 'mbpp_plus',
                          out_dir / 'codegen_val_mbpp_plus.parquet')

    if args.smoke_rows > 0:
        smoke_path = out_dir / 'codegen_val_humaneval_plus_smoke.parquet'
        df_he.head(args.smoke_rows).to_parquet(smoke_path)
        print(f'  smoke: wrote {min(args.smoke_rows, len(df_he))} rows to {smoke_path.name}')


if __name__ == '__main__':
    main()
