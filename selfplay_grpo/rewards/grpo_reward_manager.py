import ast
import os
import math
import time
from functools import partial
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
import re
import uuid

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.protocol import DataProtoItem
from verl.utils.dataset.rl_dataset import collate_fn
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

import selfplay_grpo.rewards.custom_evaluate as custom_evaluate
from selfplay_grpo.rewards.code_reward import (
    parse_code_input_output,
    parse_inputs_message,
    parse_code_function,
    ast_edit_distance,
    get_code_complexity_reward,
    get_halstead_reward,
    get_type_counts_reward,
)
from selfplay_grpo.rewards.custom_evaluate import get_format_reward, extract_answer, extract_thought
from selfplay_grpo.data_construction.process_data import boxed_instruction, instruction_following
from selfplay_grpo.data_construction.constructor import get_code_problem_predictor_prompt
from selfplay_grpo.utils.dataset.rl_dataset import RLHFDataset
from selfplay_grpo.utils.logging_utils.stdout import PrettyPrinter
from selfplay_grpo.utils.code_utils.checks import check_composite_function, check_no_definitions


INTRINSIC_SELF_CONSISTENCY_REWARD_MODE = "intrinsic_self_consistency"
LEGACY_INTRINSIC_MAJORITY_VOTE_REWARD_MODE = "intrinsic_majority_vote"


def canonicalize_solver_reward_mode(reward_mode: str) -> str:
    if reward_mode == LEGACY_INTRINSIC_MAJORITY_VOTE_REWARD_MODE:
        return INTRINSIC_SELF_CONSISTENCY_REWARD_MODE
    return reward_mode


def _codegen_approx_eq(a, b, atol: float) -> bool:
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - float(b)) <= atol
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return all(_codegen_approx_eq(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        return all(_codegen_approx_eq(a[k], b[k], atol) for k in a)
    return a == b


def _codegen_outputs_equal(got_repr: str, gold_repr: str, atol: float) -> bool:
    if got_repr == gold_repr:
        return True
    if atol > 0:
        try:
            return _codegen_approx_eq(ast.literal_eval(got_repr), ast.literal_eval(gold_repr), atol)
        except Exception:
            pass
    return False


def run_codegen_tests(executor, snippet: str, *, entry_point: str, inputs, expected, atol: float) -> bool:
    """True iff `snippet` defines `entry_point` and that function reproduces `expected[i]`
    for every `inputs[i]` (each input is a comma-joined repr string like "1, [2, 3]").

    Uses executor.batch_apply so all tests for one problem dispatch through the
    pebble ProcessPool in parallel; perf scales with the executor's max_workers.
    """
    if not inputs:
        return True
    snippets = [f"{snippet}\nf = {entry_point}\nrepr(f({args_str}))" for args_str in inputs]
    results = executor.batch_apply(snippets)
    for (out, status), gold in zip(results, expected):
        if 'error' in status.lower():
            return False
        if not _codegen_outputs_equal(out, gold, atol):
            return False
    return True


class CodeIORewardManager():
    """The reward manager."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int,
        split: str,
        reward_fn_extraction_type: str,
        math_metric: str,
        splitter: str,
        output_path: str,
        generation_reward_config: Dict[str, Any],
        debug: bool = False,
        max_prompt_length: int = 8192,
        valid_program_filter: str = 'all',
        batched_estimate: bool = False,
        extract_code_block: bool = True,
        num_inputs: int = 10,
        code_f_reward_type: str = 'accuracy',
        boxed_retry: bool = False,
        program_validity_config: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = partial(custom_evaluate.get_reward, math_metric=math_metric, boxed_retry=boxed_retry)
        self.reward_fn_extraction_type = reward_fn_extraction_type
        self.split = split
        self.splitter = splitter
        self.output_path = output_path
        self.max_prompt_length = max_prompt_length
        self.generation_reward_config = generation_reward_config
        self.valid_program_filter = valid_program_filter
        self.batched_estimate = batched_estimate
        self.debug = debug
        self.extract_code_block = extract_code_block
        self.use_original_code_as_ref = generation_reward_config.use_original_code_as_ref
        self.num_inputs = num_inputs
        self.code_f_reward_type = code_f_reward_type
        self.boxed_retry = boxed_retry
        self.program_validity_config = program_validity_config or {}
        leak_seed = int(self.program_validity_config.get("seed", 42))
        self._leak_rng = np.random.default_rng(leak_seed)

    def _use_self_output_for_generation_difficulty(self, problem_type: str) -> bool:
        return problem_type.endswith('code_o') and getattr(self, "reward_mode", "grounded") == "intrinsic"

    def _use_majority_vote_for_generation_difficulty(self) -> bool:
        return getattr(self, "reward_mode", "grounded") == "intrinsic_vote"

    @staticmethod
    def _get_program_validity_mode(program_validity_config: Optional[Dict[str, Any]]) -> str:
        if program_validity_config is None:
            return "execution"
        mode = str(program_validity_config.get("mode", "execution")).strip().lower()
        if mode not in {"execution", "syntax", "off", "execution_noisy"}:
            raise ValueError(f"Invalid program validity mode: {mode}")
        return mode

    def _get_effective_program_validity_mode(self, problem_type: Optional[str]) -> str:
        if problem_type is None or problem_type.endswith("code_f"):
            return "execution"
        return self._get_program_validity_mode(self.program_validity_config)

    def _get_leak_rate(self) -> float:
        return float(self.program_validity_config.get("leak_rate", 0.0))

    def _should_store_execution_output(self, problem_type: Optional[str]) -> bool:
        if problem_type is None or problem_type.endswith("code_f"):
            return True
        if self._get_effective_program_validity_mode(problem_type) == "execution":
            return True
        return bool(self.program_validity_config.get("store_execution_output", True))

    def _should_check_determinism(self, problem_type: Optional[str]) -> bool:
        if problem_type is None or problem_type.endswith("code_f"):
            return True
        return bool(self.program_validity_config.get("check_determinism", True))

    def _require_execution_output_for_dataset(self, problem_type: Optional[str]) -> bool:
        if problem_type is None or problem_type.endswith("code_f"):
            return True
        return bool(self.program_validity_config.get("require_execution_output_for_dataset", True))

    def _get_code_f_uniqueness_gate_config(self) -> Dict[str, Any]:
        gate_cfg = self.program_validity_config.get("code_f_uniqueness_gate", {})
        return gate_cfg or {}

    def _use_code_f_uniqueness_gate(self, problem_type: Optional[str]) -> bool:
        if problem_type != "gen_code_f":
            return False
        return bool(self._get_code_f_uniqueness_gate_config().get("enabled", False))

    @staticmethod
    def _normalize_imports(imports: Any) -> List[str]:
        if imports is None:
            return []
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if isinstance(imports, (list, tuple)):
            return [str(imp) for imp in imports]
        return [str(imports)]

    @staticmethod
    def _build_code_f_candidate_key(snippet: Any, imports: Any) -> Tuple[str, Tuple[str, ...]]:
        normalized_snippet = str(snippet).strip()
        normalized_imports = tuple(sorted(CodeIORewardManager._normalize_imports(imports)))
        return normalized_snippet, normalized_imports

    @staticmethod
    def _split_code_f_examples(inputs: List[Any], outputs: List[Any]) -> Dict[str, List[Any]]:
        n = min(len(inputs), len(outputs))
        mid = n // 2
        inputs = list(inputs[:n])
        outputs = list(outputs[:n])
        return {
            "given_inputs": inputs[:mid],
            "given_outputs": outputs[:mid],
            "hidden_inputs": inputs[mid:],
            "hidden_outputs": outputs[mid:],
        }

    def _prepare_code_f_candidate_functions(
        self,
        gold_snippet: str,
        gold_imports: List[str],
        candidate_functions: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        deduped_candidates: List[Dict[str, Any]] = []
        seen = set()

        def add_candidate(snippet: Any, imports: Any, original_snippet: Any = None) -> None:
            key = self._build_code_f_candidate_key(snippet, imports)
            if not key[0] or key in seen:
                return
            seen.add(key)
            deduped_candidates.append(
                {
                    "snippet": key[0],
                    "imports": list(key[1]),
                    "original_snippet": str(original_snippet).strip() if original_snippet is not None else key[0],
                }
            )

        add_candidate(gold_snippet, gold_imports, gold_snippet)
        for candidate in candidate_functions or []:
            if not isinstance(candidate, dict):
                continue
            add_candidate(
                candidate.get("snippet", ""),
                candidate.get("imports", []),
                candidate.get("original_snippet", candidate.get("snippet", "")),
            )
        return deduped_candidates

    def _evaluate_code_f_uniqueness_gate(
        self,
        executor,
        gold_snippet: str,
        gold_imports: List[str],
        inputs: List[Any],
        outputs: List[Any],
        candidate_functions: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        split_examples = self._split_code_f_examples(inputs, outputs)
        given_inputs = split_examples["given_inputs"]
        given_outputs = split_examples["given_outputs"]
        gate_cfg = self._get_code_f_uniqueness_gate_config()
        max_matching_candidates = int(gate_cfg.get("max_matching_candidates", 1))
        candidates = self._prepare_code_f_candidate_functions(
            gold_snippet=gold_snippet,
            gold_imports=gold_imports,
            candidate_functions=candidate_functions,
        )

        if not given_inputs or not given_outputs:
            return {
                "passed": False,
                "candidate_pool_size": len(candidates),
                "checked_candidates": 0,
                "matching_candidates": 0,
                "matching_candidates_capped": 0,
                "gold_in_candidates": 1.0,
                "failure_reason": "empty_given_io",
            }

        matching_candidates = 0
        checked_candidates = 0
        for candidate in candidates:
            checked_candidates += 1
            candidate_matches = True
            for inpt, gold_output in zip(given_inputs, given_outputs):
                execution_validity, candidate_output = executor.check_all(
                    code=candidate["snippet"],
                    inputs=inpt,
                    banned_keywords=[],
                    check_determinism=True,
                    imports=candidate["imports"],
                    check_error=False,
                    banned_keywords_for_errors_and_exceptions=[],
                )
                if not execution_validity:
                    candidate_matches = False
                    break
                if self._canonicalize_output_answer(candidate_output) != self._canonicalize_output_answer(gold_output):
                    candidate_matches = False
                    break
            if candidate_matches:
                matching_candidates += 1
                if matching_candidates > max_matching_candidates:
                    return {
                        "passed": False,
                        "candidate_pool_size": len(candidates),
                        "checked_candidates": checked_candidates,
                        "matching_candidates": matching_candidates,
                        "matching_candidates_capped": matching_candidates,
                        "gold_in_candidates": 1.0,
                        "failure_reason": f"multiple_matching_candidates>{max_matching_candidates}",
                    }

        return {
            "passed": matching_candidates <= max_matching_candidates,
            "candidate_pool_size": len(candidates),
            "checked_candidates": checked_candidates,
            "matching_candidates": matching_candidates,
            "matching_candidates_capped": matching_candidates,
            "gold_in_candidates": 1.0,
            "failure_reason": "" if matching_candidates <= max_matching_candidates else f"multiple_matching_candidates>{max_matching_candidates}",
        }

    def _run_generation_execution_check(
        self,
        problem_type: str,
        executor,
        result: Dict[str, Any],
        banned_words: List[str],
        banned_assertion_keywords: List[str],
    ) -> Tuple[bool, Any]:
        return executor.check_all(
            code=result['code'],
            inputs=result['input'],
            banned_keywords=banned_words,
            check_determinism=self._should_check_determinism(problem_type),
            imports=list(set(result['imports'])),
            check_error=problem_type == 'gen_code_e',
            banned_keywords_for_errors_and_exceptions=banned_assertion_keywords,
        )

    @staticmethod
    def _get_execution_output(answer: Dict[str, Any]):
        return answer.get('execution_output', answer.get('output'))

    def _get_generation_reward_output(self, answer: Dict[str, Any], use_self_output_label: bool):
        return answer.get('self_output') if use_self_output_label else self._get_execution_output(answer)

    @staticmethod
    def _sanitize_text(text: Any) -> str:
        return str(text).replace('\x00', '')

    def _is_generation_sample_reward_eligible(
        self,
        problem_type: str,
        answer: Optional[Dict[str, Any]],
        use_self_output_label: bool,
    ) -> bool:
        if answer is None:
            return False
        if problem_type.endswith('code_f'):
            return True
        # Majority vote mode needs no gold label — only valid code + input for prompting.
        if self._use_majority_vote_for_generation_difficulty():
            if self._require_execution_output_for_dataset(problem_type) and self._get_execution_output(answer) is None:
                return False
            return True
        reward_output = self._get_generation_reward_output(answer, use_self_output_label)
        if reward_output is None:
            return False
        if self._require_execution_output_for_dataset(problem_type) and self._get_execution_output(answer) is None:
            return False
        return True

    @staticmethod
    def extract_input_output(extracted_content: str, return_input: bool = True, return_output: bool = False) -> Optional[str]:
        input_pattern = r"```input\s*\n?(.*?)\n?```"
        output_pattern = r"```output\s*\n?(.*?)\n?```"
        assert not (return_input and return_output), "Cannot return both input and output"
        assert return_input or return_output, "Must return at least one of input or output"
        extracted_content = CodeIORewardManager._sanitize_text(extracted_content)

        # Use flags for case-insensitive matching and dotall
        flags = re.DOTALL | re.IGNORECASE
        if return_input:
            input_matches = list(re.finditer(input_pattern, extracted_content, flags))
            if not input_matches:
                # Try alternative pattern without explicit input block
                input_matches = list(re.finditer(r"# Input:\s*(.*?)(?=\n```|$)", extracted_content, flags))
            if not input_matches:
                # Match input() function call and preserve quotes
                input_matches = list(re.finditer(r'input\s*\((.*?)\)', extracted_content, flags))
            if not input_matches:
                # Match <input> tag with optional closing tag, strip spaces
                input_matches = list(re.finditer(r"<input>\s*(.*?)(?:</input>|\s*$)", extracted_content, flags))
            if not input_matches:
                # Match "The input is" pattern case-insensitively
                input_matches = list(re.finditer(r"the input is\s*(.*?)\.?$", extracted_content, flags))
            # if still no input matches, use the extracted answer as the input
            # Don't strip() here to preserve quotes
            input_snippet = input_matches[-1].group(1) if input_matches else extracted_content
            return input_snippet

        if return_output:
            output_matches = list(re.finditer(output_pattern, extracted_content, flags))
            if not output_matches:
                # Try alternative pattern without explicit output block
                output_matches = list(re.finditer(r"# Output:\s*(.*?)(?=\n```|$)", extracted_content, flags))
            if not output_matches:
                # Match output() function call and preserve quotes
                output_matches = list(re.finditer(r'output\s*\((.*?)\)', extracted_content, flags))
            if not output_matches:
                # Match <output> tag with optional closing tag, strip spaces
                output_matches = list(re.finditer(r"<output>\s*(.*?)(?:</output>|\s*$)", extracted_content, flags))
            if not output_matches:
                # Match "The output is" pattern case-insensitively, strip space after "is" and period at end
                output_matches = list(re.finditer(r"the output is\s*(.*?)\.?$", extracted_content, flags))
            # if still no output matches, use the extracted answer as the output
            output_snippet = output_matches[-1].group(1) if output_matches else extracted_content
            return output_snippet

    @staticmethod
    def _last_non_empty_line(text: str) -> Optional[str]:
        lines = [line.strip().strip('`') for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else None

    @staticmethod
    def _candidate_strings(text: str) -> List[str]:
        stripped = CodeIORewardManager._sanitize_text(text).strip().strip('`')
        if not stripped:
            return []

        candidates: List[str] = []

        def add(candidate: Optional[str]) -> None:
            if candidate is None:
                return
            candidate = str(candidate).strip().strip('`')
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add(stripped)
        add(CodeIORewardManager._last_non_empty_line(stripped))

        if ':' in stripped:
            add(stripped.rsplit(':', 1)[-1])

        generic_code_blocks = re.findall(r"```(?:[\w+-]+)?\s*\n?(.*?)\n?```", stripped, re.DOTALL | re.IGNORECASE)
        if generic_code_blocks:
            add(generic_code_blocks[-1])

        return candidates

    @staticmethod
    def _decode_generation(tokenizer: AutoTokenizer, valid_response_ids: torch.Tensor) -> str:
        return tokenizer.decode(valid_response_ids, skip_special_tokens=True).strip().strip('\"\'')

    def _extract_generation_content(self, generation: str) -> str:
        if self.reward_fn_extraction_type.startswith('none'):
            return generation
        return extract_answer(generation, self.reward_fn_extraction_type, boxed_retry=self.boxed_retry)

    @staticmethod
    def _normalize_input_candidate(text: str) -> Optional[str]:
        for candidate in CodeIORewardManager._candidate_strings(text):
            try:
                ast.parse(f'f({candidate})')
                return candidate
            except Exception:
                continue
        return None

    @staticmethod
    def _is_safe_output_expression(candidate: str) -> bool:
        try:
            tree = ast.parse(candidate, mode='eval')
        except (SyntaxError, TypeError, ValueError):
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                return False
            if isinstance(node, ast.Name) and node.id not in {'True', 'False', 'None'}:
                return False
        return True

    @staticmethod
    def _normalize_output_candidate(text: str) -> Optional[str]:
        for candidate in CodeIORewardManager._candidate_strings(text):
            try:
                return repr(ast.literal_eval(candidate))
            except Exception:
                if CodeIORewardManager._is_safe_output_expression(candidate):
                    try:
                        value = eval(compile(ast.parse(candidate, mode='eval'), '<output>', 'eval'), {'__builtins__': {}}, {})
                        return repr(value)
                    except Exception:
                        return candidate
        return None

    @staticmethod
    def _normalize_error_candidate(text: str) -> Optional[str]:
        for candidate in CodeIORewardManager._candidate_strings(text):
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', candidate):
                return candidate
        return None

    def _parse_prediction_answer(self, problem_type: str, extracted_content: str) -> Tuple[bool, Any]:
        extracted_content = self._sanitize_text(extracted_content)
        if problem_type.endswith('code_i'):
            raw_input = self.extract_input_output(extracted_content, return_input=True, return_output=False)
            if raw_input and raw_input != extracted_content and self.extract_code_block:
                answer = self._sanitize_text(raw_input).strip()
            else:
                answer = self._normalize_input_candidate(extracted_content)
            return answer is not None and str(answer).strip() != '', answer

        if problem_type.endswith('code_o'):
            raw_output = self.extract_input_output(extracted_content, return_input=False, return_output=True)
            if raw_output and raw_output != extracted_content:
                answer = self._sanitize_text(raw_output).strip()
            else:
                answer = self._normalize_output_candidate(extracted_content)
            return answer is not None and str(answer).strip() != '', answer

        if problem_type.endswith('code_e'):
            raw_output = self.extract_input_output(extracted_content, return_input=False, return_output=True)
            if raw_output and raw_output != extracted_content:
                answer = self._sanitize_text(raw_output).strip()
            else:
                answer = self._normalize_error_candidate(extracted_content)
            return answer is not None and str(answer).strip() != '', answer

        if problem_type.endswith('code_f'):
            success, code_snippet = parse_code_function(extracted_content)
            return success, code_snippet if success else None

        if problem_type.endswith('codegen'):
            success, code_snippet = parse_code_function(extracted_content)
            return success, code_snippet if success else None

        if problem_type.endswith('dsl_o'):
            raw_output = self.extract_input_output(extracted_content, return_input=False, return_output=True)
            if raw_output and raw_output != extracted_content:
                answer = self._sanitize_text(raw_output).strip()
            else:
                answer = self._normalize_output_candidate(extracted_content)
            return answer is not None and str(answer).strip() != '', answer

        raise ValueError(f'Invalid problem type: {problem_type}')

    @staticmethod
    def _canonicalize_output_answer(answer: Any) -> Optional[str]:
        return CodeIORewardManager._normalize_output_candidate(str(answer))

    @staticmethod
    def _canonicalize_input_answer(answer: Any) -> Optional[str]:
        return CodeIORewardManager._normalize_input_candidate(str(answer))

    def _get_data_dict(
        self,
        data_item: DataProtoItem,
        problem_type: str,
        executor,
        banned_words: List[str],
        uid: str,
        banned_assertion_keywords: List[str],
        code_f_candidate_functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)

        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']
        extra_info = data_item.non_tensor_batch['extra_info']
        non_special_tokens_sequences_str = self.tokenizer.decode(self.tokenizer.encode(sequences_str), skip_special_tokens=True)

        generation = self._decode_generation(self.tokenizer, valid_response_ids)
        extracted_content = self._extract_generation_content(generation)
        thought = extract_thought(generation)

        data_dict = {
            'generation': generation,
            'data_source': data_source,
            'ground_truth': ground_truth,
            'extra_info': extra_info,
            'non_special_tokens_sequences_str': non_special_tokens_sequences_str,
            'valid_response_length': valid_response_length,
            'extracted_content': extracted_content,
            'thought': thought,
            'uid': uid,
        }
        if problem_type.startswith('gen'):
            data_dict['references'] = [ref['snippet'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
            if problem_type != 'gen_code_f':
                data_dict['composite_functions'] = data_item.non_tensor_batch['extra_info']['composite_functions'].tolist()
            else:
                data_dict['imports'] = [ref['imports'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
            if self.use_original_code_as_ref:
                data_dict['original_references'] = [ref['original_snippet'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
        elif problem_type.startswith('pred') and 'code_f' not in problem_type and 'codegen' not in problem_type:
            format_score, answer = self._parse_prediction_answer(problem_type, extracted_content)
            data_dict['format_score'] = bool(format_score)
            data_dict['answer'] = answer
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['input'] = data_item.non_tensor_batch['extra_info']['input']
            data_dict['output'] = data_item.non_tensor_batch['extra_info']['output']
            data_dict['imports'] = data_item.non_tensor_batch['extra_info'].get('imports', [])
        elif problem_type.startswith('pred') and 'code_f' in problem_type:
            format_score, answer = self._parse_prediction_answer(problem_type, extracted_content)
            data_dict['format_score'] = bool(format_score)
            data_dict['answer'] = answer
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['given_inputs'] = data_item.non_tensor_batch['extra_info']['given_inputs']
            data_dict['given_outputs'] = data_item.non_tensor_batch['extra_info']['given_outputs']
            data_dict['hidden_inputs'] = data_item.non_tensor_batch['extra_info']['hidden_inputs']
            data_dict['hidden_outputs'] = data_item.non_tensor_batch['extra_info']['hidden_outputs']
            data_dict['message'] = data_item.non_tensor_batch['extra_info']['message']
            data_dict['imports'] = data_item.non_tensor_batch['extra_info'].get('imports', [])
        elif problem_type.startswith('pred') and 'codegen' in problem_type:
            format_score, answer = self._parse_prediction_answer(problem_type, extracted_content)
            data_dict['format_score'] = bool(format_score)
            data_dict['answer'] = answer
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['entry_point'] = data_item.non_tensor_batch['extra_info']['entry_point']
            data_dict['base_inputs'] = list(data_item.non_tensor_batch['extra_info']['base_inputs'])
            data_dict['base_outputs'] = list(data_item.non_tensor_batch['extra_info']['base_outputs'])
            data_dict['plus_inputs'] = list(data_item.non_tensor_batch['extra_info']['plus_inputs'])
            data_dict['plus_outputs'] = list(data_item.non_tensor_batch['extra_info']['plus_outputs'])
            data_dict['atol'] = float(data_item.non_tensor_batch['extra_info'].get('atol', 0.0))
            data_dict['imports'] = list(data_item.non_tensor_batch['extra_info'].get('imports', []))

        # if QA task, we only need to check the format
        if problem_type is None:
            if self.reward_fn_extraction_type.startswith('none'):
                format_score = 1. if generation else 0.
            else:
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
            data_dict['format_score'] = format_score
            return data_dict
        # DSL generation branch
        elif problem_type == 'gen_dsl_o':
            from selfplay_grpo.utils.dsl_utils.dsl_executor import (
                parse_dsl_input_output, parse_dsl, measure_depth,
            )
            success, result = parse_dsl_input_output(extracted_content)
            if success:
                execution_validity, execution_output = executor.check_all(result['snippet'], result['input'])
                code_validity = bool(execution_validity)
                if not code_validity:
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict
                try:
                    depth = measure_depth(parse_dsl(result['snippet']))
                except Exception:
                    depth = 0
                data_dict['answer'] = {
                    'snippet': result['snippet'],
                    'original_snippet': result['snippet'],
                    'input': result['input'],
                    'output': execution_output,
                    'execution_output': execution_output,
                    'execution_validity': 1.0,
                    'execution_output_available': float(execution_output is not None),
                    'self_output': None,
                    'imports': [],
                    'thought': thought,
                    'composite_functions': [],
                    'program_validity_mode': 'execution',
                    'depth': depth,
                }
                data_dict['format_score'] = 1.0
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        # first go through, we only checking the format
        elif problem_type.startswith('gen') and 'code_f' not in problem_type:
            parse_self_output = self._use_self_output_for_generation_difficulty(problem_type)
            program_validity_mode = self._get_effective_program_validity_mode(problem_type)
            success, result = parse_code_input_output(
                extracted_content,
                parse_output=parse_self_output,
                remove_after_return=self.generation_reward_config.remove_after_return and self.split == 'train',
                remove_comments=self.generation_reward_config.remove_comments and self.split == 'train',
                remove_print=self.generation_reward_config.remove_print and self.split == 'train',
                reject_multiple_functions=self.generation_reward_config.reject_multiple_functions,
                f_replace_location=self.generation_reward_config.f_replace_location,
                reject_test_input_in_code=self.generation_reward_config.reject_test_input_in_code,
                code_location=self.generation_reward_config.code_location,
            )
            if len(data_dict['composite_functions']) > 0 and success:
                # first, check if the composite function names are redefined in the code, which we do not allow
                success = check_no_definitions(result['code'], [f'g_{i}' for i in range(len(data_dict['composite_functions']))])
                if not success: # if the composite function names are redefined, we do not allow the code
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict

                composite_imports = '\n'.join(
                    '\n'.join(list(d['imports'])) if list(d['imports']) else '' for d in data_dict['composite_functions']
                ).strip()

                composite_snippets = '\n\n'.join(d['snippet'] for d in data_dict['composite_functions']).strip()

                # cache the original code
                result['original_code'] = result['code']

                result['code'] = f"{composite_imports}\n\n{composite_snippets}\n\n{result['code']}".strip()
                # TODO: composite function check
                success = check_composite_function(
                    code = result['code'],
                    composite_functions = [d['snippet'] for d in data_dict['composite_functions']],
                )
            if success:
                self_output = None
                if parse_self_output:
                    self_output = self._canonicalize_output_answer(result.get('output', ''))
                    if self_output is None:
                        data_dict['code_validity'] = False
                        data_dict['format_score'] = 0.
                        return data_dict
                execution_output = None
                execution_validity = None
                if self._should_store_execution_output(problem_type):
                    execution_validity, execution_output = self._run_generation_execution_check(
                        problem_type=problem_type,
                        executor=executor,
                        result=result,
                        banned_words=banned_words,
                        banned_assertion_keywords=banned_assertion_keywords,
                    )

                if program_validity_mode == "execution_noisy":
                    if bool(execution_validity):
                        code_validity = True
                    else:
                        code_validity = self._leak_rng.random() < self._get_leak_rate()
                else:
                    code_validity = bool(execution_validity) if program_validity_mode == "execution" else True
                if not code_validity:
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict
                # means the code is valid, we append any good programs, but we eval format separately
                data_dict['answer'] = {
                    'snippet': result['code'],
                    'original_snippet': result['original_code'] if 'original_code' in result else result['code'],
                    'input': result['input'],
                    'output': execution_output,
                    'execution_output': execution_output,
                    'execution_validity': float(bool(execution_validity)) if execution_validity is not None else 0.0,
                    'execution_output_available': float(execution_output is not None),
                    'self_output': self_output,
                    'imports': result['imports'],
                    'thought': thought,
                    'composite_functions': data_dict['composite_functions'],
                    'program_validity_mode': program_validity_mode,
                }
                if self_output is not None and execution_output is not None:
                    canonical_execution_output = self._canonicalize_output_answer(execution_output)
                    data_dict['answer']['self_output_matches_execution'] = float(self_output == canonical_execution_output)
                data_dict['format_score'] = 1.0
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        elif problem_type == 'gen_code_f':
            success, result = parse_inputs_message(
                extracted_content,
                num_inputs=self.num_inputs,
            )
            if success and len(result['inputs']) == self.num_inputs: # for code_f, we need to ensure the number of inputs is correct
                outputs = []
                for inpt in result['inputs']:
                    code_validity, output = executor.check_all(
                        code=data_dict['references'][0],
                        inputs=inpt,
                        banned_keywords=[],
                        check_determinism=True,
                        imports=data_dict['imports'][0],
                        check_error=False,
                        banned_keywords_for_errors_and_exceptions=[],
                    )
                    if not code_validity:
                        data_dict['code_validity'] = False
                        data_dict['format_score'] = 0.
                        return data_dict
                    outputs.append(output)
                uniqueness_gate_stats = {
                    "passed": True,
                    "candidate_pool_size": 0,
                    "checked_candidates": 0,
                    "matching_candidates": 0,
                    "matching_candidates_capped": 0,
                    "gold_in_candidates": 0.0,
                    "failure_reason": "",
                }
                if self._use_code_f_uniqueness_gate(problem_type):
                    uniqueness_gate_stats = self._evaluate_code_f_uniqueness_gate(
                        executor=executor,
                        gold_snippet=data_dict['references'][0],
                        gold_imports=data_dict['imports'][0],
                        inputs=result['inputs'],
                        outputs=outputs,
                        candidate_functions=code_f_candidate_functions,
                    )
                    if not uniqueness_gate_stats["passed"]:
                        data_dict['uniqueness_gate'] = uniqueness_gate_stats
                        data_dict['code_validity'] = False
                        data_dict['format_score'] = 0.
                        return data_dict
                split_examples = self._split_code_f_examples(result['inputs'], outputs)
                data_dict['answer'] = {
                    'snippet': data_dict['references'][0],
                    'inputs': result['inputs'],
                    'outputs': outputs,
                    'message': result['message'],
                    'imports': data_dict['imports'][0],
                    'thought': thought,
                    'given_inputs': split_examples['given_inputs'],
                    'given_outputs': split_examples['given_outputs'],
                    'hidden_inputs': split_examples['hidden_inputs'],
                    'hidden_outputs': split_examples['hidden_outputs'],
                    'uniqueness_gate_passed': float(uniqueness_gate_stats["passed"]),
                    'uniqueness_candidate_pool_size': float(uniqueness_gate_stats["candidate_pool_size"]),
                    'uniqueness_checked_candidates': float(uniqueness_gate_stats["checked_candidates"]),
                    'uniqueness_matching_candidates': float(uniqueness_gate_stats["matching_candidates"]),
                    'uniqueness_gold_in_candidates': float(uniqueness_gate_stats["gold_in_candidates"]),
                    'uniqueness_failure_reason': uniqueness_gate_stats["failure_reason"],
                }
                data_dict['uniqueness_gate'] = uniqueness_gate_stats
                data_dict['format_score'] = 1.0
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        # if prediction is the task
        elif problem_type.startswith('pred'):
            # Check required blocks
            if problem_type.endswith('code_i'): # parse input
                success, input_snippet = self._parse_prediction_answer(problem_type=problem_type, extracted_content=extracted_content)
                if not success:
                    data_dict['format_score'] = 0.
                    return data_dict
                data_dict['format_score'] = 1.0
                data_dict['answer'] = input_snippet
                return data_dict
            elif problem_type.endswith('code_o') or problem_type.endswith('code_e') or problem_type.endswith('dsl_o'): #  parse output, code_e format is same as code_o
                success, output_snippet = self._parse_prediction_answer(problem_type=problem_type, extracted_content=extracted_content)
                if not success:
                    data_dict['format_score'] = 0.
                    return data_dict
                data_dict['format_score'] = 1.0
                data_dict['answer'] = output_snippet
                return data_dict
            elif problem_type.endswith('code_f'):
                success, code_snippet = self._parse_prediction_answer(problem_type=problem_type, extracted_content=extracted_content)
                if not success:
                    data_dict['format_score'] = 0.
                    return data_dict
                data_dict['format_score'] = 1.0
                data_dict['answer'] = {
                    'snippet': code_snippet,
                    'given_inputs': data_dict['given_inputs'],
                    'given_outputs': data_dict['given_outputs'],
                    'hidden_inputs': data_dict['hidden_inputs'],
                    'hidden_outputs': data_dict['hidden_outputs'],
                    'message': data_dict['message'],
                    'imports': data_dict['imports'],
                    'thought': thought,
                    'gold_program': data_dict['program'],
                }
                return data_dict
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")

    def __call__(
        self,
        data: DataProto,
        problem_type: str = None,
        executor = None,
        rollout_actor_wg = None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],
        n_samples: int = 1,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
        code_f_candidate_functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = defaultdict(list)
        data_dicts = []
        valid_programs = [] # for gen tasks, we need to store the valid programs for later use, ignore this if prediction task
        correct_predictions = []
        uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)
        if problem_type is None:
            problem_types = [d.non_tensor_batch['extra_info']['metric'] for d in data]
            problem_type = 'pred' # dummy set
        else:
            problem_types = [problem_type] * len(data)
        PrettyPrinter.section_header("Getting Data Dicts")
        for i in range(len(data)): # get format score
            data_dict = self._get_data_dict(
                data[i],
                problem_types[i],
                executor,
                banned_words,
                uids[i],
                banned_assertion_keywords,
                code_f_candidate_functions=code_f_candidate_functions if problem_types[i] == "gen_code_f" else None,
            )
            data_dicts.append(data_dict)

        if problem_type.startswith('gen') and rollout_actor_wg is not None: # get generation rewards
            return self._compute_generation_task_rewards(
                data=data,
                data_dicts=data_dicts,
                problem_type=problem_type,
                executor=executor,
                rollout_actor_wg=rollout_actor_wg,
                n_samples=n_samples,
                input_type_counters=input_type_counters,
                output_type_counters=output_type_counters,
                error_type_counters=error_type_counters,
            )
        elif problem_type.startswith('pred'): # get prediction rewards
            PrettyPrinter.section_header("Getting Prediction Rewards")
            all_scores['none_count'] = 0
            acc_rewards = []
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                imports = data_dict['imports']
                # Pull branch-specific fields using the per-row type (not the global dummy `problem_type`).
                pt = problem_types[i]
                answer = gold_input = gold_output = program = None
                hidden_inputs = hidden_outputs = None
                if pt.endswith('codegen'):
                    pass  # codegen branch reads entry_point/base_inputs/... from data_dict directly
                elif pt.endswith('code_f'):
                    hidden_inputs = data_dict['hidden_inputs']
                    hidden_outputs = data_dict['hidden_outputs']
                else:
                    answer = data_dict['answer']
                    gold_input = data_dict['input']
                    gold_output = data_dict['output']
                    program = data_dict['program']
                if not data_dicts[i]['format_score']: # early stop if the format is not correct
                    acc_reward = 0.
                elif problem_types[i].endswith('code_i'):
                    acc_reward = executor.eval_input_prediction(code=program, gold_output=gold_output, agent_input=answer, imports=list(set(imports)))
                    # problematic, but we did not encounter too much of this
                    if acc_reward is None:
                        all_scores['none_count'] += 1
                        acc_reward = 0.
                        print(f"error in pred_code_i, not in [0, 1], acc_reward={acc_reward}\nprogram:\n{program}\n---\nanswer:\n{answer}\n---\nimports:\n{imports}\n---\n")
                    if acc_reward > 0.0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('code_o'):
                    acc_reward = executor.eval_output_prediction(code=program, gold_output=gold_output, agent_output=answer, imports=list(set(imports)))
                    # problematic, but we did not encounter too much of this
                    if acc_reward is None:
                        all_scores['none_count'] += 1
                        acc_reward = 0.
                        print(f"error in pred_code_o, not in [0, 1], acc_reward={acc_reward}\nprogram:\n{program}\n---\nanswer:\n{answer}\n---\nimports:\n{imports}\n---\n")
                    if acc_reward > 0.0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('dsl_o'):
                    acc_reward = executor.eval_output_prediction(code=program, gold_output=gold_output, agent_output=answer, imports=[])
                    if acc_reward is None:
                        all_scores['none_count'] += 1
                        acc_reward = 0.
                    if acc_reward > 0.0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('code_e'): # string matching for errors
                    answer = answer.split(' ')[0].split(':')[0]
                    if answer.lower() == gold_output.lower():
                        acc_reward = 1.0
                        correct_predictions.append(data_dict)
                    else:
                        acc_reward = 0.0
                elif problem_types[i].endswith('code_f'):
                    input_output_accs = []
                    program = data_dict['answer']['snippet']
                    for inpt, outpt in zip(hidden_inputs, hidden_outputs):
                        input_output_acc = executor.eval_input_prediction(
                            code=program,
                            gold_output=outpt,
                            agent_input=inpt,
                            imports=list(set(imports)),
                        )
                        if input_output_acc is not None:
                            input_output_accs.append(input_output_acc)
                    acc_reward = np.mean(input_output_accs) if input_output_accs else 0.0
                    if self.code_f_reward_type == 'binary':
                        acc_reward = 1.0 if acc_reward == 1.0 else 0.0
                    elif self.code_f_reward_type == 'if_one_correct':
                        acc_reward = 1.0 if acc_reward > 0 else 0.0
                    # note that if code_f_reward_type==accuracy, it is already handled in the above
                    if acc_reward > 0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('codegen'):
                    snippet = data_dict.get('answer')
                    if not snippet:
                        acc_reward = 0.0
                        all_scores.setdefault('codegen_parse_fail', 0)
                        all_scores['codegen_parse_fail'] += 1
                    else:
                        base_pass = run_codegen_tests(
                            executor, snippet,
                            entry_point=data_dict['entry_point'],
                            inputs=data_dict['base_inputs'],
                            expected=data_dict['base_outputs'],
                            atol=data_dict['atol'],
                        )
                        plus_pass = base_pass and run_codegen_tests(
                            executor, snippet,
                            entry_point=data_dict['entry_point'],
                            inputs=data_dict['plus_inputs'],
                            expected=data_dict['plus_outputs'],
                            atol=data_dict['atol'],
                        )
                        acc_reward = 1.0 if plus_pass else 0.0
                        all_scores.setdefault('codegen_base_pass', []).append(1.0 if base_pass else 0.0)
                        all_scores.setdefault('codegen_plus_pass', []).append(1.0 if plus_pass else 0.0)
                    if acc_reward > 0:
                        correct_predictions.append(data_dict)
                else:
                    raise ValueError(f"Invalid problem type: {problem_types[i]}")

                if self.split == 'train':
                    if data_dicts[i]['format_score'] > 0:
                        reward_tensor[i, valid_response_length - 1] = acc_reward if acc_reward > 0 else 0.0
                    else:
                        reward_tensor[i, valid_response_length - 1] = 0.0
                elif self.split == 'test': # only acc reward for eval
                    if acc_reward > 0:
                        reward_tensor[i, valid_response_length - 1] = 1.0
                    else:
                        reward_tensor[i, valid_response_length - 1] = 0.0
                acc_rewards.append(acc_reward)
            all_scores['accuracy'] = acc_rewards
            all_scores['format_score'] = [data_dicts[i]['format_score'] for i in range(len(data))]
            all_scores['none_ratio'] = all_scores['none_count'] / len(data)
        return reward_tensor, all_scores, valid_programs, correct_predictions

    def _get_problem_generator_rewards_and_valid_programs(
        self,
        data_dicts: List[Dict],
        problem_type: str,
        n_samples: int,
        rollout_actor_wg,
        executor,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
        """This function uses samples to estimate the accuracy reward for each program, also computes the code complexity and mean edit distance of generated programs.
            Also returns the valid programs using filters.
            Args:
                data_dicts: List[Dict]: A list of data dictionaries.
                problem_type: str: The type of problem.
                n_samples: int: The number of samples to use.
                rollout_actor_wg: RolloutActorWG: The rollout actor.
                executor: PythonExecutor/CodeBoxExecutor: The executor.
                type_counters: Dict[str, Dict[str, int]]: The type counters.
            Returns:
               rewards: Dict[str, Dict[str, float]]: A dictionary of rewards for each program.
               valid_programs: List[Dict[str, str]]: A list of valid programs.
        """
        if problem_type.endswith('code_i'):
            type_counters = input_type_counters
        elif problem_type.endswith('code_o') or problem_type.endswith('dsl_o'):
            type_counters = output_type_counters
        elif problem_type.endswith('code_e'):
            type_counters = error_type_counters
        difficulty_cfg = self.generation_reward_config.get('difficulty_reward', None)
        use_self_output_label = self._use_self_output_for_generation_difficulty(problem_type)
        rewardable_valid_data_dicts = [
            data_dict for data_dict in data_dicts
            if data_dict['code_validity']
            and self._is_generation_sample_reward_eligible(problem_type, data_dict.get('answer'), use_self_output_label)
        ]
        uid2valid_dict_idx = {data_dict['uid']: i for i, data_dict in enumerate(rewardable_valid_data_dicts)}
        valid_uids = [data_dict['uid'] for data_dict in rewardable_valid_data_dicts]
        ineligible_valid_uids = [
            data_dict['uid'] for data_dict in data_dicts
            if data_dict['code_validity']
            and data_dict['uid'] not in uid2valid_dict_idx
        ]
        invalid_uids = [data_dict['uid'] for data_dict in data_dicts if not data_dict['code_validity']]
        assert len(valid_uids) + len(ineligible_valid_uids) + len(invalid_uids) == len(data_dicts)
        accuracies = {uid: 1.0 for uid in invalid_uids + ineligible_valid_uids} # samples without reward labels get no difficulty reward
        rewards = defaultdict(dict)
        valid_programs = []
        use_majority_vote = self._use_majority_vote_for_generation_difficulty()
        if len(valid_uids) > 0:
            if self.reward_fn_extraction_type.startswith('boxed'):
                instruction_template = boxed_instruction
            elif self.reward_fn_extraction_type.startswith('answer'):
                instruction_template = instruction_following
            elif self.reward_fn_extraction_type.startswith('none'):
                instruction_template = '{}'
            else:
                raise ValueError(f"Invalid instruction type: {self.reward_fn_extraction_type}")
            prompts = []
            if problem_type.endswith('code_i'):
                pt = 'code_i'
            elif problem_type.endswith('code_o'):
                pt = 'code_o'
            elif problem_type.endswith('code_e'):
                pt = 'code_e'
            elif problem_type.endswith('code_f'):
                pt = 'code_f'
            elif problem_type.endswith('dsl_o'):
                pt = 'dsl_o'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in rewardable_valid_data_dicts:
                if pt == 'code_f':
                    num_given_inputs = len(data_dict['answer']['inputs']) // 2
                    num_given_outputs = len(data_dict['answer']['outputs']) // 2
                    data_dict['answer']['given_inputs'] = data_dict['answer']['inputs'][:num_given_inputs]
                    data_dict['answer']['given_outputs'] = data_dict['answer']['outputs'][:num_given_outputs]
                    data_dict['answer']['hidden_inputs'] = data_dict['answer']['inputs'][num_given_inputs:]
                    data_dict['answer']['hidden_outputs'] = data_dict['answer']['outputs'][num_given_outputs:]
                    io_prompt = instruction_template.format(
                        get_code_problem_predictor_prompt(
                            problem_type=problem_type,
                            snippet=data_dict['answer']['snippet'],
                            message=data_dict['answer']['message'],
                            input_output_pairs=zip(data_dict['answer']['given_inputs'], data_dict['answer']['given_outputs']),
                        )
                    )
                else:
                    reward_output = self._get_generation_reward_output(data_dict['answer'], use_self_output_label)
                    io_prompt = instruction_template.format(
                        get_code_problem_predictor_prompt(
                            problem_type=pt,
                            snippet=data_dict['answer']['snippet'],
                            input_args=data_dict['answer']['input'],
                            output=reward_output,
                        )
                    )
                prompts_dict = {
                    'prompt': [{'role': 'user', 'content': io_prompt}],
                    'uid': data_dict['uid'],
                    'problem': data_dict['answer'],
                    'data_source': data_dict['data_source'],
                    'ground_truth': reward_output if pt != 'code_f' else data_dict['answer']['snippet'],
                    'extra_info': data_dict['extra_info'],
                    'program': data_dict['answer']['snippet'],
                    'imports': data_dict['answer']['imports'],
                    'references': data_dict['references'],
                }
                if pt == 'code_f':
                    prompts_dict.update({
                        'given_inputs': data_dict['answer']['given_inputs'],
                        'given_outputs': data_dict['answer']['given_outputs'],
                        'hidden_inputs': data_dict['answer']['hidden_inputs'],
                        'hidden_outputs': data_dict['answer']['hidden_outputs'],
                        'message': data_dict['answer']['message'],
                    })
                else:
                    prompts_dict.update({
                        'input': data_dict['answer']['input'],
                        'output': reward_output,
                        'execution_output': data_dict['answer']['output'],
                        'self_output': data_dict['answer'].get('self_output'),
                        'original_program': data_dict['answer']['original_snippet'],
                        'composite_functions': data_dict['answer']['composite_functions'],
                    })
                prompts.append(prompts_dict)

            # sampling to estimate the accuracy
            PrettyPrinter.section_header("Sampling to Estimate Accuracy")
            pd.DataFrame(prompts).to_parquet(f'{self.output_path}/temp.parquet') # RLHFDataset expects parquet
            temp_data = RLHFDataset(
                parquet_files=f'{self.output_path}/temp.parquet',
                tokenizer=self.tokenizer,
                prompt_key='prompt',
                max_prompt_length=self.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error'
            )
            os.remove(f'{self.output_path}/temp.parquet') # we do not need this file after we load in the dataset
            sampler = torch.utils.data.SequentialSampler(data_source=temp_data)

            dataloader = torch.utils.data.DataLoader(
                dataset=temp_data,
                batch_size=len(temp_data),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            assert len(dataloader) == 1
            data = next(iter(dataloader))
            batch = DataProto.from_single_dict(data)
            gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': True,
                'validate': False,
                'n': n_samples,
            }
            # pad to be divisible by dp_size
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)

            padded_prompt_count = gen_batch_padded.batch.batch_size[0]
            padded_response_count = output_gen_batch_padded.batch.batch_size[0]
            assert padded_response_count % padded_prompt_count == 0, (
                f"Expected generated batch size to be a multiple of prompt batch size, "
                f"got {padded_response_count} and {padded_prompt_count}"
            )
            actual_n = padded_response_count // padded_prompt_count

            # Some rollout backends ignore per-call `meta_info['n']` and fall back to the
            # worker default. Align unpadding and source-batch repetition to the actual
            # number of returned samples so online-growth reward estimation stays well-formed.
            pad_size *= actual_n
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            if actual_n > 1:
                batch = batch.repeat(repeat_times=actual_n, interleave=True)
            batch = batch.union(output_gen_batch)
            batched_responses = []
            for b in batch:
                batch_dict = {
                        'extracted_answers': self._extract_generation_content(
                            self._decode_generation(self.tokenizer, b.batch['responses'])
                        ),
                        'uid': b.non_tensor_batch['uid'],
                        'problem': b.non_tensor_batch['problem'],
                        'data_source': b.non_tensor_batch['data_source'],
                        'extra_info': b.non_tensor_batch['extra_info'],
                        'program': b.non_tensor_batch['program'],
                        'references': b.non_tensor_batch['references'],
                        'imports': b.non_tensor_batch['imports'],
                    }
                if pt == 'code_f':
                    batch_dict.update({
                        'given_inputs': b.non_tensor_batch['given_inputs'],
                        'given_outputs': b.non_tensor_batch['given_outputs'],
                        'hidden_inputs': b.non_tensor_batch['hidden_inputs'],
                        'hidden_outputs': b.non_tensor_batch['hidden_outputs'],
                        'message': b.non_tensor_batch['message'],
                    })
                else:
                    batch_dict.update({
                        'input': b.non_tensor_batch['input'],
                        'output': b.non_tensor_batch['output'],
                        'execution_output': b.non_tensor_batch.get('execution_output'),
                        'self_output': b.non_tensor_batch.get('self_output'),
                        'original_program': b.non_tensor_batch['original_program'],
                        'composite_functions': b.non_tensor_batch['composite_functions'].tolist(),
                    })
                batched_responses.append(batch_dict)
            df = pd.DataFrame(batched_responses)

            # estimating accuracy using python executor
            PrettyPrinter.section_header(
                "Estimating Accuracy Using Majority Vote" if use_majority_vote
                else "Estimating Accuracy Using Python Executor"
            )
            for valid_uid in valid_uids:
                df_valid = df[df['uid'] == valid_uid]
                if df_valid.empty: # the prompt got filtered out TODO: check
                    accuracies[valid_uid] = 0.0
                    continue
                answers = []
                for answer in df_valid['extracted_answers'].tolist():
                    success, parsed_answer = self._parse_prediction_answer(problem_type=problem_type, extracted_content=answer)
                    if pt != 'code_f':
                        answers.append(parsed_answer if success else '')
                    else:
                        answers.append((success, parsed_answer if success else ''))

                if use_majority_vote and pt != 'code_f':
                    # Majority vote: accuracy = vote share of most common canonical answer.
                    # No gold label needed — purely measures solver consensus (R-Zero style).
                    canonical_answers = []
                    for answer in answers:
                        if not answer:
                            canonical_answers.append(None)
                        elif pt in ('code_o',):
                            canonical_answers.append(self._canonicalize_output_answer(answer))
                        elif pt in ('code_i',):
                            canonical_answers.append(self._canonicalize_input_answer(answer))
                        elif pt in ('code_e',):
                            canonical_answers.append(str(answer).split(' ')[0].split(':')[0].lower())
                        else:
                            canonical_answers.append(str(answer))
                    valid_canonical = [c for c in canonical_answers if c is not None]
                    if valid_canonical:
                        counts = Counter(valid_canonical)
                        majority_count = counts.most_common(1)[0][1]
                        accuracies[valid_uid] = majority_count / len(answers)
                    else:
                        accuracies[valid_uid] = 0.0
                else:
                    # Gold-label comparison: compare each solver answer against gold_output.
                    answer_cache = {} # for the same uid, the answer is the same and the program is assumed to be deterministic, therefore we cache the answer -> accuracy mapping
                    if pt == 'code_f':
                        hidden_outputs = df_valid['hidden_outputs'].tolist()[0].tolist()
                        hidden_inputs = df_valid['hidden_inputs'].tolist()[0].tolist()
                    else:
                        gold_output = df_valid['output'].tolist()[0]
                        program = df_valid['program'].tolist()[0]
                        # gold_input = df_valid['input'].tolist()[0]
                    imports = df_valid['imports'].tolist()[0]
                    problem_accuracies = []
                    if problem_type.endswith('code_i'):
                        if self.batched_estimate:
                            problem_accuracies = executor.eval_k_input_prediction(code=program, gold_output=gold_output, k_agent_inputs=answers, imports=list(set(imports)))
                        else:
                            for answer in answers:
                                if answer in answer_cache:
                                    problem_accuracies.append(answer_cache[answer])
                                    continue
                                acc_reward = executor.eval_input_prediction(code=program, gold_output=gold_output, agent_input=answer, imports=list(set(imports)))
                                if acc_reward is not None:
                                    problem_accuracies.append(acc_reward)
                                    answer_cache[answer] = acc_reward
                            # if self.debug:
                            #     batched_problem_accuracies = executor.eval_k_input_prediction(code=program, gold_output=gold_output, k_agent_inputs=answers, imports=list(set(imports)))
                            #     assert np.mean(batched_problem_accuracies) == np.mean(problem_accuracies), f"Gen I batch accuracy: {np.mean(batched_problem_accuracies)}, Single accuracy: {np.mean(problem_accuracies)}"
                    elif problem_type.endswith('code_o'):
                        if self.batched_estimate:
                            problem_accuracies = executor.eval_k_output_prediction(code=program, gold_output=gold_output, k_agent_outputs=answers, imports=list(set(imports)))
                        else:
                            for answer in answers:
                                if answer in answer_cache:
                                    problem_accuracies.append(answer_cache[answer])
                                    continue
                                acc_reward = executor.eval_output_prediction(code=program, gold_output=gold_output, agent_output=answer, imports=list(set(imports)))
                                if acc_reward is not None:
                                    problem_accuracies.append(acc_reward)
                                    answer_cache[answer] = acc_reward
                            # if self.debug:
                            #     batched_problem_accuracies = executor.eval_k_output_prediction(code=program, gold_output=gold_output, k_agent_outputs=answers, imports=list(set(imports)))
                            #     assert np.mean(batched_problem_accuracies) == np.mean(problem_accuracies), f"Gen O batch accuracy: {np.mean(batched_problem_accuracies)}, Single accuracy: {np.mean(problem_accuracies)}"
                    elif problem_type.endswith('code_e'): # string matching for errors
                        for answer in answers:
                            answer = answer.split(' ')[0].split(':')[0]
                            if answer.lower() == gold_output.lower():
                                problem_accuracies.append(1.0)
                            else:
                                problem_accuracies.append(0.0)
                    elif problem_type.endswith('code_f'):
                        for parsed, answer in answers: # for each input/output set, we sampled n codes to estimate the accuracy
                            if not parsed: # the code answer is not parsed, we assume the code is not valid
                                problem_accuracies.append(0.0)
                                continue
                            code_accuracies = []
                            for inpt, outpt in zip(hidden_inputs, hidden_outputs):
                                code_accuracies.append(executor.eval_input_prediction(code=answer, gold_output=outpt, agent_input=inpt, imports=list(set(imports))))
                            answer_acc = np.mean([a for a in code_accuracies if a is not None]) if code_accuracies else 0.0
                            if self.code_f_reward_type == 'binary':
                                problem_accuracies.append(1.0 if answer_acc == 1.0 else 0.0)
                            elif self.code_f_reward_type == 'if_one_correct':
                                problem_accuracies.append(1.0 if answer_acc > 0 else 0.0)
                            elif self.code_f_reward_type == 'accuracy':
                                problem_accuracies.append(answer_acc)
                            else:
                                raise ValueError(f"Invalid code_f_reward_type: {self.code_f_reward_type}")
                    accuracies[valid_uid] = sum(problem_accuracies) / len(problem_accuracies) if problem_accuracies else 0.0

                # filtering valid programs
                if self.valid_program_filter == 'all':
                    valid_programs.append(rewardable_valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_one':
                    if accuracies[valid_uid] < 1.0:
                        valid_programs.append(rewardable_valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_extremes':
                    if accuracies[valid_uid] > 0.0 and accuracies[valid_uid] < 1.0:
                        valid_programs.append(rewardable_valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                else:
                    raise ValueError(f"Invalid valid program filter: {self.valid_program_filter}")

        # getting other rewards
        PrettyPrinter.section_header("Getting Other Rewards")
        # outputting rewards
        for d in data_dicts:
            uid = d['uid']
            rewards[uid]['solver_accuracy'] = float(accuracies[uid])
            rewards[uid]['difficulty'] = self._compute_generation_difficulty_score(
                accuracy=accuracies[uid],
                difficulty_cfg=difficulty_cfg,
            )
            rewards[uid]['difficulty_uses_self_output'] = float(use_self_output_label)
            rewards[uid]['difficulty_uses_majority_vote'] = float(use_majority_vote)
            rewards[uid]['self_output_available'] = 0.0
            rewards[uid]['self_output_matches_execution'] = 0.0
            rewards[uid]['code_validity'] = float(d.get('code_validity', False))
            rewards[uid]['reward_label_available'] = 0.0
            rewards[uid]['dataset_eligibility'] = float(uid in uid2valid_dict_idx)
            rewards[uid]['execution_output_available'] = 0.0
            rewards[uid]['execution_validity'] = 0.0
            rewards[uid]['program_validity_is_execution'] = float(
                self._get_effective_program_validity_mode(problem_type) == 'execution'
            )
            uniqueness_stats = d.get('uniqueness_gate', {})
            rewards[uid]['uniqueness_gate_enabled'] = float(self._use_code_f_uniqueness_gate(problem_type))
            rewards[uid]['uniqueness_gate_passed'] = float(uniqueness_stats.get('passed', not self._use_code_f_uniqueness_gate(problem_type)))
            rewards[uid]['uniqueness_candidate_pool_size'] = float(uniqueness_stats.get('candidate_pool_size', 0.0))
            rewards[uid]['uniqueness_checked_candidates'] = float(uniqueness_stats.get('checked_candidates', 0.0))
            rewards[uid]['uniqueness_matching_candidates'] = float(uniqueness_stats.get('matching_candidates', 0.0))
            rewards[uid]['uniqueness_gold_in_candidates'] = float(uniqueness_stats.get('gold_in_candidates', 0.0))
            if problem_type.endswith('code_o') and d.get('answer'):
                self_output = d['answer'].get('self_output')
                execution_output = self._get_execution_output(d['answer'])
                reward_output = self._get_generation_reward_output(d['answer'], use_self_output_label)
                rewards[uid]['self_output_available'] = float(self_output is not None)
                rewards[uid]['reward_label_available'] = float(reward_output is not None)
                rewards[uid]['execution_output_available'] = float(execution_output is not None)
                rewards[uid]['execution_validity'] = float(d['answer'].get('execution_validity', execution_output is not None))
                if self_output is not None:
                    rewards[uid]['self_output_matches_execution'] = float(self_output == self._canonicalize_output_answer(execution_output))
            elif d.get('answer'):
                execution_output = self._get_execution_output(d['answer'])
                reward_output = self._get_generation_reward_output(d['answer'], use_self_output_label)
                rewards[uid]['reward_label_available'] = float(reward_output is not None)
                rewards[uid]['execution_output_available'] = float(execution_output is not None)
                rewards[uid]['execution_validity'] = float(d['answer'].get('execution_validity', execution_output is not None))
            rewards[uid]['accuracy'] = float(accuracies[uid])

        if problem_type.endswith('dsl_o'):
            # DSL has no Python AST metrics; set structural reward keys to 0.0
            # and compute DSL-specific depth metric.
            for data_dict in data_dicts:
                uid = data_dict['uid']
                rewards[uid]['complexity'] = 0.0
                rewards[uid]['mean_edit_distance'] = 0.0
                rewards[uid]['halstead'] = 0.0
                rewards[uid]['type_counts'] = 0.0
                rewards[uid]['depth'] = float(
                    data_dict['answer']['depth'] if 'answer' in data_dict and 'depth' in data_dict.get('answer', {}) else 0
                )
        elif not problem_type.endswith('code_f'):
            code_key = 'original_snippet' if self.use_original_code_as_ref else 'snippet'
            reference_key = 'original_references' if self.use_original_code_as_ref else 'references'
            complexity_enabled = bool(self.generation_reward_config.complexity_reward.enabled)
            mean_edit_distance_enabled = bool(self.generation_reward_config.mean_edit_distance_reward.enabled)
            halstead_enabled = bool(self.generation_reward_config.halstead_reward.enabled)
            answer_diversity_enabled = bool(self.generation_reward_config.answer_diversity_reward.enabled)
            if problem_type.endswith('code_i'):
                type_counter_key = 'input'
            elif problem_type.endswith('code_o'):
                type_counter_key = 'output'
            elif problem_type.endswith('code_e'):
                type_counter_key = 'error'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['complexity'] = (
                    get_code_complexity_reward(data_dict['answer'][code_key])
                    if complexity_enabled and 'answer' in data_dict else 0.0
                )
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['mean_edit_distance'] = (
                    np.mean([ast_edit_distance(data_dict['answer'][code_key], ref) for ref in data_dict[reference_key]])
                    if mean_edit_distance_enabled and 'answer' in data_dict else 0.0
                )
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['halstead'] = (
                    get_halstead_reward(data_dict['answer'][code_key])
                    if halstead_enabled and 'answer' in data_dict else 0.0
                )
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['type_counts'] = (
                    get_type_counts_reward(
                        data_dict['answer'][type_counter_key],
                        type_counters,
                        hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                    )
                    if answer_diversity_enabled and 'answer' in data_dict else 0.0
                )
            if self.debug:
                for data_dict in data_dicts:
                    if 'answer' in data_dict:
                        continue
        else:
            input_diversity_enabled = bool(self.generation_reward_config.f_input_answer_diversity_reward.enabled)
            output_diversity_enabled = bool(self.generation_reward_config.f_output_answer_diversity_reward.enabled)
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['input_type_counts'] = []
                rewards[data_dict['uid']]['output_type_counts'] = []
                if 'answer' in data_dict:
                    if input_diversity_enabled or output_diversity_enabled:
                        for inpt, outpt in zip(data_dict['answer']['inputs'], data_dict['answer']['outputs']):
                            if input_diversity_enabled:
                                rewards[data_dict['uid']]['input_type_counts'].append(get_type_counts_reward(
                                    inpt,
                                    input_type_counters,
                                    hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                                ))
                            if output_diversity_enabled:
                                rewards[data_dict['uid']]['output_type_counts'].append(get_type_counts_reward(
                                    outpt,
                                    output_type_counters,
                                    hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                                ))
                    rewards[data_dict['uid']]['input_type_counts'] = (
                        np.mean(rewards[data_dict['uid']]['input_type_counts'])
                        if rewards[data_dict['uid']]['input_type_counts'] else 0.0
                    )
                    rewards[data_dict['uid']]['output_type_counts'] = (
                        np.mean(rewards[data_dict['uid']]['output_type_counts'])
                        if rewards[data_dict['uid']]['output_type_counts'] else 0.0
                    )
                else:
                    rewards[data_dict['uid']]['input_type_counts'] = 0.0
                    rewards[data_dict['uid']]['output_type_counts'] = 0.0

        # turn into normal dict
        rewards = dict(rewards)
        return rewards, valid_programs

    def _get_generation_intrinsic_components(
        self,
        problem_type: str,
        rewards: Dict[str, Dict[str, float]],
        uid: str,
        include_difficulty: bool = False,
    ) -> List[float]:
        intrinsic_reward_components = []
        difficulty_cfg = self.generation_reward_config.get('difficulty_reward', None)
        difficulty_mode = self._get_difficulty_reward_mode(difficulty_cfg)
        if include_difficulty and difficulty_mode != 'none':
            intrinsic_reward_components.append(min(
                float(difficulty_cfg.get('coef', 1.0)) * rewards[uid].get('difficulty', 0.0),
                float(difficulty_cfg.get('max', 1.0)),
            ))
        if problem_type.endswith('code_f'):
            if self.generation_reward_config.f_input_answer_diversity_reward.enabled:
                intrinsic_reward_components.append(min(
                    self.generation_reward_config.f_input_answer_diversity_reward.coef * rewards[uid]['input_type_counts'],
                    self.generation_reward_config.f_input_answer_diversity_reward.max,
                ))
            if self.generation_reward_config.f_output_answer_diversity_reward.enabled:
                intrinsic_reward_components.append(min(
                    self.generation_reward_config.f_output_answer_diversity_reward.coef * rewards[uid]['output_type_counts'],
                    self.generation_reward_config.f_output_answer_diversity_reward.max,
                ))
        else:
            if self.generation_reward_config.complexity_reward.enabled:
                intrinsic_reward_components.append(min(
                    self.generation_reward_config.complexity_reward.coef * rewards[uid]['complexity'],
                    self.generation_reward_config.complexity_reward.max,
                ))
            if self.generation_reward_config.mean_edit_distance_reward.enabled:
                intrinsic_reward_components.append(min(
                    self.generation_reward_config.mean_edit_distance_reward.coef * rewards[uid]['mean_edit_distance'],
                    self.generation_reward_config.mean_edit_distance_reward.max,
                ))
            if self.generation_reward_config.halstead_reward.enabled:
                intrinsic_reward_components.append(min(
                    self.generation_reward_config.halstead_reward.coef * rewards[uid]['halstead'],
                    self.generation_reward_config.halstead_reward.max,
                ))
            if self.generation_reward_config.answer_diversity_reward.enabled:
                intrinsic_reward_components.append(min(
                    self.generation_reward_config.answer_diversity_reward.coef * rewards[uid]['type_counts'],
                    self.generation_reward_config.answer_diversity_reward.max,
                ))
        return intrinsic_reward_components

    @staticmethod
    def _get_difficulty_reward_mode(difficulty_cfg: Optional[Dict[str, Any]]) -> str:
        if difficulty_cfg is None:
            return 'none'
        mode = difficulty_cfg.get('mode', None)
        if mode is not None:
            mode = str(mode).strip().lower()
            if mode in {'none', 'hard', 'medium'}:
                return mode
            raise ValueError(f"Invalid difficulty reward mode: {mode}")
        # Backward compatibility for earlier boolean-style configs.
        return 'medium' if difficulty_cfg.get('enabled', False) else 'none'

    @staticmethod
    def _compute_generation_difficulty_score(accuracy: float, difficulty_cfg: Optional[Dict[str, Any]]) -> float:
        if difficulty_cfg is None:
            return 0.0
        mode = CodeIORewardManager._get_difficulty_reward_mode(difficulty_cfg)
        clamped_accuracy = min(max(float(accuracy), 0.0), 1.0)
        if mode == 'none':
            return 0.0
        if mode == 'hard':
            return 1.0 - clamped_accuracy
        target_accuracy = float(difficulty_cfg.get('target_accuracy', 0.5))
        width = max(float(difficulty_cfg.get('width', 0.5)), 1e-8)
        return max(0.0, 1.0 - abs(clamped_accuracy - target_accuracy) / width)

    @staticmethod
    def _get_capped_generation_difficulty_reward(difficulty: float, difficulty_cfg: Optional[Dict[str, Any]]) -> float:
        if difficulty_cfg is None:
            return 0.0
        mode = CodeIORewardManager._get_difficulty_reward_mode(difficulty_cfg)
        if mode == 'none':
            return 0.0
        return min(
            float(difficulty_cfg.get('coef', 1.0)) * float(difficulty),
            float(difficulty_cfg.get('max', 1.0)),
        )

    @staticmethod
    def _combine_generation_rewards(base_reward: Optional[float], intrinsic_components: List[float], method: str) -> float:
        components = [c for c in intrinsic_components if c is not None]
        if base_reward is None:
            if not components:
                return 0.0
            if method in {'sum', 'multiply_sum'}:
                return float(sum(components))
            if method in {'multiply', 'sum_multiply'}:
                return float(np.prod(components))
            raise ValueError(f"Unknown combination method: {method}")

        if method == 'sum':
            return base_reward + sum(components) if components else base_reward
        if method == 'multiply':
            return base_reward * np.prod(components) if components else base_reward
        if method == 'sum_multiply':
            return base_reward + np.prod(components) if components else base_reward
        if method == 'multiply_sum':
            return base_reward * sum(components) if components else base_reward
        raise ValueError(f"Unknown combination method: {method}")

    def _compute_generation_task_rewards(
        self,
        data: DataProto,
        data_dicts: List[Dict],
        problem_type: str,
        executor,
        rollout_actor_wg,
        n_samples: int,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = defaultdict(list)
        difficulty_cfg = self.generation_reward_config.get('difficulty_reward', None)

        PrettyPrinter.section_header("Generating Rewards for Generation Tasks")
        rewards, valid_programs = self._get_problem_generator_rewards_and_valid_programs(
            data_dicts=data_dicts,
            problem_type=problem_type,
            n_samples=n_samples,
            rollout_actor_wg=rollout_actor_wg,
            executor=executor,
            input_type_counters=input_type_counters,
            output_type_counters=output_type_counters,
            error_type_counters=error_type_counters,
        )
        PrettyPrinter.section_header("Combining Rewards for Generation Tasks")
        for i in range(len(data_dicts)):
            uid = data_dicts[i]['uid']
            valid_response_length = data_dicts[i]['valid_response_length']
            if valid_response_length <= 0:
                continue

            format_reward = data_dicts[i]['format_score']
            difficulty_reward = self._get_capped_generation_difficulty_reward(
                difficulty=rewards[uid]['difficulty'],
                difficulty_cfg=difficulty_cfg,
            )

            final_reward = 0.0
            if format_reward > 0:
                # Proposer reward is defined purely by difficulty. Other generation
                # statistics remain logged for analysis but do not affect optimization.
                final_reward = difficulty_reward

            reward_tensor[i, valid_response_length - 1] = final_reward

        all_scores['accuracy'] = [rewards[uid]['accuracy'] for uid in rewards]
        all_scores['solver_accuracy'] = [rewards[uid]['solver_accuracy'] for uid in rewards]
        all_scores['difficulty'] = [rewards[uid]['difficulty'] for uid in rewards]
        all_scores['difficulty_reward'] = [
            self._get_capped_generation_difficulty_reward(rewards[uid]['difficulty'], difficulty_cfg)
            for uid in rewards
        ]
        all_scores['code_validity'] = [rewards[uid]['code_validity'] for uid in rewards]
        all_scores['reward_label_available'] = [rewards[uid]['reward_label_available'] for uid in rewards]
        all_scores['dataset_eligibility'] = [rewards[uid]['dataset_eligibility'] for uid in rewards]
        all_scores['execution_output_available'] = [rewards[uid]['execution_output_available'] for uid in rewards]
        all_scores['execution_validity'] = [rewards[uid]['execution_validity'] for uid in rewards]
        all_scores['program_validity_is_execution'] = [rewards[uid]['program_validity_is_execution'] for uid in rewards]
        all_scores['uniqueness_gate_enabled'] = [rewards[uid]['uniqueness_gate_enabled'] for uid in rewards]
        all_scores['uniqueness_gate_passed'] = [rewards[uid]['uniqueness_gate_passed'] for uid in rewards]
        all_scores['uniqueness_candidate_pool_size'] = [rewards[uid]['uniqueness_candidate_pool_size'] for uid in rewards]
        all_scores['uniqueness_checked_candidates'] = [rewards[uid]['uniqueness_checked_candidates'] for uid in rewards]
        all_scores['uniqueness_matching_candidates'] = [rewards[uid]['uniqueness_matching_candidates'] for uid in rewards]
        all_scores['uniqueness_gold_in_candidates'] = [rewards[uid]['uniqueness_gold_in_candidates'] for uid in rewards]
        all_scores['difficulty_uses_self_output'] = [rewards[uid]['difficulty_uses_self_output'] for uid in rewards]
        all_scores['difficulty_uses_majority_vote'] = [rewards[uid]['difficulty_uses_majority_vote'] for uid in rewards]
        all_scores['self_output_available'] = [rewards[uid]['self_output_available'] for uid in rewards]
        all_scores['self_output_matches_execution'] = [rewards[uid]['self_output_matches_execution'] for uid in rewards]
        all_scores['format_score'] = [data_dicts[i]['format_score'] for i in range(len(data))]
        if problem_type.endswith('dsl_o'):
            all_scores['depth'] = [rewards[uid]['depth'] for uid in rewards]
        elif 'code_f' not in problem_type:
            all_scores['answer_diversity'] = [rewards[uid]['type_counts'] for uid in rewards]
            all_scores['complexity'] = [rewards[uid]['complexity'] for uid in rewards]
            all_scores['mean_edit_distance'] = [rewards[uid]['mean_edit_distance'] for uid in rewards]
            all_scores['halstead'] = [rewards[uid]['halstead'] for uid in rewards]
        else:
            all_scores['input_answer_diversity'] = [rewards[uid]['input_type_counts'] for uid in rewards]
            all_scores['output_answer_diversity'] = [rewards[uid]['output_type_counts'] for uid in rewards]
        return reward_tensor, all_scores, valid_programs, []


class GRPORewardManager(CodeIORewardManager):
    def __init__(self, reward_mode: str = "grounded", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_mode = canonicalize_solver_reward_mode(reward_mode)

    def __call__(
        self,
        data: DataProto,
        problem_type: str = None,
        executor=None,
        rollout_actor_wg=None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],
        n_samples: int = 1,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
        code_f_candidate_functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        print(
            f"[grpo_reward_manager] enter mode={self.reward_mode} "
            f"problem_type={problem_type} batch_size={len(data)}"
        )
        if problem_type is not None and problem_type.startswith("gen"):
            if self.reward_mode in {"grounded", "intrinsic", "intrinsic_vote"}:
                data_dicts = []
                uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)
                for i in range(len(data)):
                    data_dicts.append(
                        self._get_data_dict(
                            data[i],
                            problem_type,
                            executor,
                            banned_words,
                            uids[i],
                            banned_assertion_keywords,
                            code_f_candidate_functions=code_f_candidate_functions if problem_type == "gen_code_f" else None,
                        )
                    )
                return self._compute_generation_task_rewards(
                    data=data,
                    data_dicts=data_dicts,
                    problem_type=problem_type,
                    executor=executor,
                    rollout_actor_wg=rollout_actor_wg,
                    n_samples=n_samples,
                    input_type_counters=input_type_counters,
                    output_type_counters=output_type_counters,
                    error_type_counters=error_type_counters,
                )
            raise ValueError(f"Unsupported proposer reward mode: {self.reward_mode}")

        if self.reward_mode == "grounded":
            return super().__call__(
                data=data,
                problem_type=problem_type,
                executor=executor,
                rollout_actor_wg=rollout_actor_wg,
                banned_words=banned_words,
                banned_assertion_keywords=banned_assertion_keywords,
                n_samples=n_samples,
                input_type_counters=input_type_counters,
                output_type_counters=output_type_counters,
                error_type_counters=error_type_counters,
                code_f_candidate_functions=code_f_candidate_functions,
            )
        if self.reward_mode != INTRINSIC_SELF_CONSISTENCY_REWARD_MODE:
            raise ValueError(f"Unsupported solver reward mode: {self.reward_mode}")
        return self._compute_intrinsic_self_consistency(data=data, problem_type=problem_type, executor=executor)

    def _compute_intrinsic_self_consistency(self, data: DataProto, problem_type: str, executor) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        t_start = time.time()
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        all_scores = defaultdict(list)
        problem_types = self._resolve_problem_types(data=data, problem_type=problem_type)
        print(f"[intrinsic_self_consistency] start batch_size={len(data)} problem_type={problem_type}")

        uids = data.non_tensor_batch.get("uid")
        if uids is None:
            uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)

        data_dicts = []
        for i in range(len(data)):
            data_dicts.append(
                self._get_data_dict(
                    data_item=data[i],
                    problem_type=problem_types[i],
                    executor=executor,
                    banned_words=[],
                    uid=uids[i],
                    banned_assertion_keywords=[],
                )
            )
            if (i + 1) % 128 == 0 or (i + 1) == len(data):
                print(
                    f"[intrinsic_self_consistency] parsed {i + 1}/{len(data)} samples "
                    f"elapsed={time.time() - t_start:.1f}s"
                )

        grounded_none_count = [0]
        grounded_cache = {}
        grounded_accuracies = [
            self._compute_grounded_prediction_accuracy(
                problem_type=problem_types[i],
                data_dict=data_dicts[i],
                executor=executor,
                none_count_ref=grounded_none_count,
                cache=grounded_cache,
            )
            for i in range(len(data_dicts))
        ]

        grouped = defaultdict(list)
        for idx, data_dict in enumerate(data_dicts):
            grouped[data_dict["uid"]].append((idx, data_dict))
        print(
            f"[intrinsic_self_consistency] grouped into {len(grouped)} prompts "
            f"after_parse_elapsed={time.time() - t_start:.1f}s"
        )

        majority_sizes = []
        unique_counts = []
        valid_total = 0

        for group_idx, items in enumerate(grouped.values(), start=1):
            key_counts = defaultdict(int)
            resolved = {}
            group_size = len(items)
            for idx, data_dict in items:
                key = self._intrinsic_key(problem_type=problem_types[idx], data_dict=data_dict, executor=executor)
                resolved[idx] = key
                if key is not None:
                    key_counts[key] += 1
                    valid_total += 1

            majority_size = max(key_counts.values()) if key_counts else 0
            unique_count = len(key_counts)
            majority_sizes.append(float(majority_size))
            unique_counts.append(float(unique_count))

            for idx, data_dict in items:
                valid_response_length = int(data_dict["valid_response_length"])
                if valid_response_length <= 0:
                    continue
                key = resolved[idx]
                vote_share = float(key_counts[key]) / float(group_size) if key is not None else 0.0
                grounded_accuracy = grounded_accuracies[idx]
                reward_tensor[idx, valid_response_length - 1] = vote_share
                all_scores["vote_share"].append(vote_share)
                all_scores["grounded_accuracy"].append(grounded_accuracy)
                all_scores["vote_share_grounded_gap"].append(vote_share - grounded_accuracy)
                all_scores["vote_share_grounded_abs_gap"].append(abs(vote_share - grounded_accuracy))

            if group_idx % 32 == 0 or group_idx == len(grouped):
                print(
                    f"[intrinsic_self_consistency] resolved {group_idx}/{len(grouped)} prompt groups "
                    f"elapsed={time.time() - t_start:.1f}s"
                )

        sample_count = max(len(data_dicts), 1)
        all_scores["valid_ratio"] = valid_total / sample_count
        all_scores["grounded_none_ratio"] = grounded_none_count[0] / sample_count
        all_scores["majority_size_mean"] = np.mean(majority_sizes) if majority_sizes else 0.0
        all_scores["unique_answer_count_mean"] = np.mean(unique_counts) if unique_counts else 0.0
        all_scores["vote_share_mean"] = float(reward_tensor.sum(dim=-1).mean().item()) if len(data_dicts) > 0 else 0.0
        print(f"[intrinsic_self_consistency] done elapsed={time.time() - t_start:.1f}s")
        return reward_tensor, all_scores, [], []

    @staticmethod
    def _resolve_problem_types(data: DataProto, problem_type: Optional[str]) -> List[str]:
        if problem_type is not None:
            return [problem_type] * len(data)
        return [d.non_tensor_batch["extra_info"]["metric"] for d in data]

    def _compute_grounded_prediction_accuracy(
        self,
        problem_type: str,
        data_dict: Dict,
        executor,
        none_count_ref: Optional[List[int]] = None,
        cache: Optional[Dict[Tuple[Any, ...], float]] = None,
    ) -> float:
        if not data_dict.get("format_score"):
            return 0.0

        imports = list(set(data_dict.get("imports", [])))
        imports_key = tuple(sorted(imports))

        def _record_none() -> float:
            if none_count_ref is not None:
                none_count_ref[0] += 1
            return 0.0

        if problem_type.endswith("code_i"):
            answer = data_dict.get("answer")
            if answer is None:
                return 0.0
            cache_key = (
                "code_i",
                data_dict.get("program"),
                data_dict.get("output"),
                self._canonicalize_input_answer(answer),
                imports_key,
            )
            if cache is not None and cache_key in cache:
                return cache[cache_key]
            acc_reward = executor.eval_input_prediction(
                code=data_dict["program"],
                gold_output=data_dict["output"],
                agent_input=answer,
                imports=imports,
            )
            if acc_reward is None:
                return _record_none()
            acc_reward = float(acc_reward)
            if cache is not None:
                cache[cache_key] = acc_reward
            return acc_reward

        if problem_type.endswith("code_o"):
            answer = data_dict.get("answer")
            if answer is None:
                return 0.0
            cache_key = (
                "code_o",
                data_dict.get("program"),
                data_dict.get("output"),
                self._canonicalize_output_answer(answer),
                imports_key,
            )
            if cache is not None and cache_key in cache:
                return cache[cache_key]
            acc_reward = executor.eval_output_prediction(
                code=data_dict["program"],
                gold_output=data_dict["output"],
                agent_output=answer,
                imports=imports,
            )
            if acc_reward is None:
                return _record_none()
            acc_reward = float(acc_reward)
            if cache is not None:
                cache[cache_key] = acc_reward
            return acc_reward

        if problem_type.endswith("code_e"):
            answer = data_dict.get("answer")
            if answer is None:
                return 0.0
            normalized_answer = str(answer).split(" ")[0].split(":")[0].lower()
            gold_output = str(data_dict.get("output", "")).lower()
            return 1.0 if normalized_answer == gold_output else 0.0

        if problem_type.endswith("code_f"):
            answer = data_dict.get("answer")
            if answer is None:
                return 0.0
            hidden_inputs = data_dict.get("hidden_inputs", [])
            hidden_outputs = data_dict.get("hidden_outputs", [])
            cache_key = (
                "code_f",
                answer.get("snippet"),
                tuple(hidden_inputs),
                tuple(hidden_outputs),
                imports_key,
            )
            if cache is not None and cache_key in cache:
                return cache[cache_key]
            input_output_accs = []
            program = answer["snippet"]
            for inpt, outpt in zip(hidden_inputs, hidden_outputs):
                input_output_acc = executor.eval_input_prediction(
                    code=program,
                    gold_output=outpt,
                    agent_input=inpt,
                    imports=imports,
                )
                if input_output_acc is not None:
                    input_output_accs.append(input_output_acc)
            acc_reward = np.mean(input_output_accs) if input_output_accs else 0.0
            if self.code_f_reward_type == "binary":
                acc_reward = 1.0 if acc_reward == 1.0 else 0.0
            elif self.code_f_reward_type == "if_one_correct":
                acc_reward = 1.0 if acc_reward > 0 else 0.0
            acc_reward = float(acc_reward)
            if cache is not None:
                cache[cache_key] = acc_reward
            return acc_reward

        if problem_type.endswith("dsl_o"):
            answer = data_dict.get("answer")
            if answer is None:
                return 0.0
            acc_reward = executor.eval_output_prediction(
                code=data_dict["program"],
                gold_output=data_dict["output"],
                agent_output=answer,
                imports=[],
            )
            if acc_reward is None:
                return _record_none()
            return float(acc_reward)

        raise ValueError(f"Intrinsic reward does not support problem type: {problem_type}")

    def _intrinsic_key(self, problem_type: str, data_dict: Dict, executor) -> Optional[Tuple]:
        if not data_dict.get("format_score"):
            return None
        if problem_type.endswith("code_i"):
            answer = data_dict.get("answer")
            if answer is None:
                return None
            canonical_answer = self._canonicalize_input_answer(answer)
            return ("input", canonical_answer) if canonical_answer else None
        if problem_type.endswith("code_o"):
            answer = data_dict.get("answer")
            if answer is None:
                return None
            canonical_answer = self._canonicalize_output_answer(answer)
            return ("output", canonical_answer) if canonical_answer else None
        if problem_type.endswith("code_f"):
            answer = data_dict.get("answer")
            if answer is None:
                return None
            outputs = []
            for hidden_input in answer["hidden_inputs"]:
                code_validity, output = executor.check_all(
                    code=answer["snippet"],
                    inputs=hidden_input,
                    banned_keywords=[],
                    check_determinism=True,
                    imports=list(set(answer["imports"])),
                    check_error=False,
                    banned_keywords_for_errors_and_exceptions=[],
                )
                if not code_validity:
                    return None
                outputs.append(output)
            return ("code_f", tuple(outputs))
        if problem_type.endswith("dsl_o"):
            answer = data_dict.get("answer")
            if answer is None:
                return None
            canonical_answer = self._canonicalize_output_answer(answer)
            return ("output", canonical_answer) if canonical_answer else None
        raise ValueError(f"Intrinsic reward does not support problem type: {problem_type}")
