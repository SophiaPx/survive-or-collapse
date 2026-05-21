import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfplay_grpo.data_construction.prompts import get_code_problem_generator_prompt


def test_code_o_prompt_can_require_self_output():
    prompt = get_code_problem_generator_prompt(
        problem_type="code_o",
        reference_snippets=[],
        banned_keywords=[],
        banned_assertion_keywords=[],
        require_self_output=True,
    )
    assert "Format your self-output with" in prompt
    assert "```output" in prompt


def test_intrinsic_proposer_mode_controls_self_output_path():
    reward_manager_text = (ROOT / "selfplay_grpo" / "rewards" / "grpo_reward_manager.py").read_text()
    assert "def _use_self_output_for_generation_difficulty" in reward_manager_text
    assert 'getattr(self, "reward_mode", "grounded") == "intrinsic"' in reward_manager_text
