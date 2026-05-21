"""
DSL Executor for the pred_dsl_o / gen_dsl_o task.

Language specification
----------------------
Binary operators:  ADD(a,b)=a+b  SUB(a,b)=a-b  MUL(a,b)=a*b
                   DIV(a,b)=a//b  MOD(a,b)=a%b  MAX(a,b)  MIN(a,b)
Unary operators:   NEG(a)=-a  ABS(a)=|a|
Conditional:       ITE(cond, then_expr, else_expr)
Conditions:        GT(a,b)=a>b  LT(a,b)=a<b  EQ(a,b)=a==b
                   GEQ(a,b)=a>=b  LEQ(a,b)=a<=b
Variables:         x, y  (integers, typically -10 .. 10)
Literals:          any signed integer
Output:            always a single integer

Grammar (EBNF)
--------------
  expr      := atom
             | unary_op  '(' expr ')'
             | binary_op '(' expr ',' expr ')'
             | ITE        '(' cond ',' expr ',' expr ')'
  cond      := cond_op '(' expr ',' expr ')'
  atom      := INT | 'x' | 'y'
  binary_op := ADD | SUB | MUL | DIV | MOD | MAX | MIN
  unary_op  := NEG | ABS
  cond_op   := GT | LT | EQ | GEQ | LEQ
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Operator sets
# ---------------------------------------------------------------------------

_BINARY_OPS = frozenset({'ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'MAX', 'MIN'})
_UNARY_OPS  = frozenset({'NEG', 'ABS'})
_COND_OPS   = frozenset({'GT', 'LT', 'EQ', 'GEQ', 'LEQ'})
_ITE        = 'ITE'

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'[A-Z]+|-?\d+|[xy]|[(),]')


def _tokenize(text: str) -> List[str]:
    """Split a DSL expression string into tokens, ignoring whitespace."""
    return _TOKEN_RE.findall(text.replace(' ', '').replace('\n', ''))


# ---------------------------------------------------------------------------
# Recursive-descent parser  →  nested-tuple AST
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: List[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: Optional[str] = None) -> str:
        tok = self.tokens[self.pos]
        if expected is not None and tok != expected:
            raise SyntaxError(
                f"Expected '{expected}', got '{tok}' at position {self.pos}"
            )
        self.pos += 1
        return tok

    def parse_expr(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")

        if tok in _BINARY_OPS:
            op = self._consume()
            self._consume('(')
            a = self.parse_expr()
            self._consume(',')
            b = self.parse_expr()
            self._consume(')')
            return (op, a, b)

        if tok in _UNARY_OPS:
            op = self._consume()
            self._consume('(')
            a = self.parse_expr()
            self._consume(')')
            return (op, a)

        if tok == _ITE:
            self._consume()
            self._consume('(')
            cond = self.parse_cond()
            self._consume(',')
            then_e = self.parse_expr()
            self._consume(',')
            else_e = self.parse_expr()
            self._consume(')')
            return (_ITE, cond, then_e, else_e)

        # atom: variable or integer literal
        if tok in ('x', 'y'):
            return self._consume()

        # signed integer (the regex may produce "-3" as one token)
        try:
            self._consume()
            return int(tok)
        except ValueError:
            raise SyntaxError(f"Unexpected token '{tok}' at position {self.pos}")

    def parse_cond(self) -> Any:
        tok = self._peek()
        if tok not in _COND_OPS:
            raise SyntaxError(f"Expected condition operator, got '{tok}'")
        op = self._consume()
        self._consume('(')
        a = self.parse_expr()
        self._consume(',')
        b = self.parse_expr()
        self._consume(')')
        return (op, a, b)

    def parse_top(self) -> Any:
        ast = self.parse_expr()
        if self.pos != len(self.tokens):
            raise SyntaxError(
                f"Unexpected trailing tokens: {self.tokens[self.pos:]}"
            )
        return ast


def parse_dsl(expression: str) -> Any:
    """
    Parse a DSL expression string into a nested-tuple AST.
    Raises SyntaxError on malformed input.
    """
    tokens = _tokenize(expression)
    if not tokens:
        raise SyntaxError("Empty expression")
    return _Parser(tokens).parse_top()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def eval_dsl_ast(ast: Any, env: Dict[str, int]) -> int:
    """
    Evaluate a parsed DSL AST with variable bindings.
    env must contain 'x' and 'y' as integers.
    Always returns int.  Raises ZeroDivisionError on DIV/MOD by zero.
    """
    if isinstance(ast, int):
        return ast

    if isinstance(ast, str):
        if ast in env:
            return env[ast]
        raise NameError(f"Unbound variable '{ast}'")

    if not isinstance(ast, tuple):
        raise TypeError(f"Invalid AST node: {ast!r}")

    op = ast[0]

    if op in _BINARY_OPS:
        _, a, b = ast
        va = eval_dsl_ast(a, env)
        vb = eval_dsl_ast(b, env)
        if op == 'ADD': return va + vb
        if op == 'SUB': return va - vb
        if op == 'MUL': return va * vb
        if op == 'DIV':
            if vb == 0:
                raise ZeroDivisionError("DIV by zero")
            return va // vb
        if op == 'MOD':
            if vb == 0:
                raise ZeroDivisionError("MOD by zero")
            return va % vb
        if op == 'MAX': return max(va, vb)
        if op == 'MIN': return min(va, vb)

    if op in _UNARY_OPS:
        _, a = ast
        va = eval_dsl_ast(a, env)
        if op == 'NEG': return -va
        if op == 'ABS': return abs(va)

    if op == _ITE:
        _, cond, then_e, else_e = ast
        cond_val = _eval_cond(cond, env)
        return eval_dsl_ast(then_e if cond_val else else_e, env)

    raise ValueError(f"Unknown operator '{op}'")


def _eval_cond(cond_ast: Any, env: Dict[str, int]) -> bool:
    op, a, b = cond_ast
    va = eval_dsl_ast(a, env)
    vb = eval_dsl_ast(b, env)
    if op == 'GT':  return va > vb
    if op == 'LT':  return va < vb
    if op == 'EQ':  return va == vb
    if op == 'GEQ': return va >= vb
    if op == 'LEQ': return va <= vb
    raise ValueError(f"Unknown condition operator '{op}'")


def evaluate_dsl(expression: str, x: int, y: int) -> Optional[int]:
    """
    High-level entry point: parse and evaluate the expression with x, y.
    Returns the integer result, or None on any parse / runtime error.
    """
    try:
        ast = parse_dsl(expression)
        return eval_dsl_ast(ast, {'x': x, 'y': y})
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Depth measurement
# ---------------------------------------------------------------------------

def measure_depth(ast: Any) -> int:
    """Return the nesting depth of a parsed DSL AST."""
    if isinstance(ast, (int, str)):
        return 0
    if not isinstance(ast, tuple):
        return 0
    op = ast[0]
    if op in _UNARY_OPS:
        return 1 + measure_depth(ast[1])
    if op in _BINARY_OPS:
        return 1 + max(measure_depth(ast[1]), measure_depth(ast[2]))
    if op == _ITE:
        _, cond, then_e, else_e = ast
        return 1 + max(
            _cond_depth(cond),
            measure_depth(then_e),
            measure_depth(else_e),
        )
    return 0


def _cond_depth(cond_ast: Any) -> int:
    _, a, b = cond_ast
    return 1 + max(measure_depth(a), measure_depth(b))


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_dsl_expression(expression: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a DSL expression string.
    Returns (True, None) if valid, (False, error_message) otherwise.
    """
    try:
        ast = parse_dsl(expression)
    except SyntaxError as e:
        return False, f"Parse error: {e}"
    except Exception as e:
        return False, f"Unexpected parse error: {e}"

    # Probe a small grid of inputs to catch systematic evaluation failures
    errors = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            try:
                result = eval_dsl_ast(ast, {'x': x, 'y': y})
                if not isinstance(result, int):
                    return False, f"Non-integer result for x={x}, y={y}: {result!r}"
            except ZeroDivisionError:
                pass  # acceptable — caller must choose safe inputs
            except Exception as e:
                errors.append(str(e))

    if len(errors) == 25:  # failed on every probe point
        return False, f"Evaluation failed on all probe inputs: {errors[0]}"

    return True, None


def find_valid_inputs(
    expression: str,
    n_samples: int = 20,
    x_range: Tuple[int, int] = (-10, 10),
    y_range: Tuple[int, int] = (-10, 10),
) -> List[Tuple[int, int]]:
    """
    Return up to n_samples (x, y) pairs for which the expression evaluates
    without error.
    """
    try:
        ast = parse_dsl(expression)
    except Exception:
        return []

    candidates = [
        (x, y)
        for x in range(x_range[0], x_range[1] + 1)
        for y in range(y_range[0], y_range[1] + 1)
    ]
    random.shuffle(candidates)
    valid: List[Tuple[int, int]] = []
    for x, y in candidates:
        try:
            result = eval_dsl_ast(ast, {'x': x, 'y': y})
            if isinstance(result, int):
                valid.append((x, y))
        except Exception:
            continue
        if len(valid) >= n_samples:
            break
    return valid


# ---------------------------------------------------------------------------
# LLM response parser  (proposer output)
# ---------------------------------------------------------------------------

_DSL_BLOCK_RE   = re.compile(r'```dsl\s*\n?(.*?)\n?```',   re.DOTALL | re.IGNORECASE)
_INPUT_BLOCK_RE = re.compile(r'```input\s*\n?(.*?)\n?```', re.DOTALL | re.IGNORECASE)
_OUTPUT_BLOCK_RE= re.compile(r'```output\s*\n?(.*?)\n?```',re.DOTALL | re.IGNORECASE)


def parse_dsl_input_output(
    extracted_content: str,
    parse_output: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Parse an LLM-generated DSL proposer response.

    Expected format::

        ```dsl
        ADD(x, MUL(2, y))
        ```
        ```input
        x=3, y=-1
        ```
        (optionally)
        ```output
        1
        ```

    Returns ``(True, result_dict)`` on success, ``(False, None)`` on failure.
    result_dict keys: snippet, input, imports, composite_functions,
                      and optionally output (if parse_output=True).
    """
    content = extracted_content.strip()

    dsl_match = _DSL_BLOCK_RE.search(content)
    if not dsl_match:
        return False, None
    snippet = dsl_match.group(1).strip()

    input_match = _INPUT_BLOCK_RE.search(content)
    if not input_match:
        return False, None
    input_str = input_match.group(1).strip()

    result: Dict[str, Any] = {
        'snippet': snippet,
        'input': input_str,
        'imports': [],
        'composite_functions': [],
    }

    if parse_output:
        output_match = _OUTPUT_BLOCK_RE.search(content)
        result['output'] = output_match.group(1).strip() if output_match else None

    return True, result


# ---------------------------------------------------------------------------
# DSLExecutor  — drop-in interface replacement for PythonExecutor
# ---------------------------------------------------------------------------

class DSLExecutor:
    """
    Synchronous DSL evaluator that mirrors the PythonExecutor public interface
    used throughout grpo_reward_manager.py.

    The 'code' argument in all methods is a DSL expression string.
    The 'inputs' argument is an "x=<int>, y=<int>" string.
    All 'imports' arguments are accepted but ignored (DSL has no imports).
    """

    def __init__(
        self,
        max_workers: int = 1,      # kept for interface compatibility
        ast_check: bool = False,   # kept for interface compatibility
        timeout_length: int = 10,  # kept for interface compatibility
    ) -> None:
        pass

    def cleanup(self) -> None:
        """No-op: DSLExecutor has no background processes."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_inputs(inputs_str: str) -> Optional[Dict[str, int]]:
        """
        Parse the inputs field into a variable environment.

        Accepted formats:
          * ``"x=3, y=-1"``   (named)
          * ``"3, -1"``        (positional: first → x, second → y)
        """
        s = inputs_str.strip()
        # named form
        m = re.fullmatch(r'x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)', s)
        if m:
            return {'x': int(m.group(1)), 'y': int(m.group(2))}
        # positional form
        m = re.fullmatch(r'(-?\d+)\s*,\s*(-?\d+)', s)
        if m:
            return {'x': int(m.group(1)), 'y': int(m.group(2))}
        return None

    @staticmethod
    def _to_int(s: str) -> Optional[int]:
        """Parse an output string to int. Returns None on failure."""
        try:
            return int(str(s).strip())
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Core evaluation — mirrors PythonExecutor.check_all return semantics
    # ------------------------------------------------------------------

    def check_all(
        self,
        code: str,
        inputs: str,
        banned_keywords: List[str] = [],
        check_determinism: bool = True,  # always True for DSL; arg kept for compat
        imports: List[str] = [],
        check_error: bool = False,
        banned_keywords_for_errors_and_exceptions: List[str] = [],
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a DSL expression and evaluate it for the given inputs.

        Returns:
          (True,  str(result))  — valid expression, execution succeeded
          (False, None)         — parse error, runtime error, or bad inputs
        """
        valid, _ = validate_dsl_expression(code)
        if not valid:
            return False, None

        env = self.parse_inputs(inputs)
        if env is None:
            return False, None

        try:
            ast = parse_dsl(code)
            result = eval_dsl_ast(ast, env)
            if not isinstance(result, int):
                return False, None
            return True, str(result)
        except ZeroDivisionError:
            return False, None
        except Exception:
            return False, None

    def run_code(
        self,
        code: str,
        inputs: str,
        imports: List[str] = [],
    ) -> Tuple[str, str]:
        """
        Evaluate a DSL expression and return (repr(result), status).
        Mirrors PythonExecutor.run_code return convention:
          - ('', 'error')  on failure
          - (str, 'done')  on success
        """
        env = self.parse_inputs(inputs)
        if env is None:
            return '', 'error: invalid inputs format'
        result = evaluate_dsl(code, env['x'], env['y'])
        if result is None:
            return '', 'error'
        return str(result), 'done'

    # ------------------------------------------------------------------
    # Prediction evaluation
    # ------------------------------------------------------------------

    def eval_output_prediction(
        self,
        code: str,
        gold_output: str,
        agent_output: str,
        imports: List[str] = [],
    ) -> float:
        """
        Compare the agent's predicted integer output to the gold output.
        Returns 1.0 on match, 0.0 otherwise.
        """
        gold_int  = self._to_int(gold_output)
        agent_int = self._to_int(agent_output)
        if gold_int is None or agent_int is None:
            return 0.0
        return 1.0 if gold_int == agent_int else 0.0

    def eval_k_output_prediction(
        self,
        code: str,
        gold_output: str,
        k_agent_outputs: List[str],
        imports: List[str] = [],
    ) -> List[float]:
        """Vectorised version of eval_output_prediction."""
        gold_int = self._to_int(gold_output)
        if gold_int is None:
            return [0.0] * len(k_agent_outputs)
        return [
            1.0 if self._to_int(agent) == gold_int else 0.0
            for agent in k_agent_outputs
        ]

    def eval_input_prediction(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError(
            "DSLExecutor does not support input prediction (pred_dsl_i)."
        )
