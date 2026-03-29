"""
Task environments for the Reflexion memory study.

Three domains:
- CodeEnvironment: HumanEval code generation benchmark (164 problems, pass@k)
- ReasoningEnvironment: HotpotQA multi-step reasoning (distractor set, exact/partial match)
- ToolEnvironment: ToolBench G1 single-tool tasks (LLM-as-judge pass rate)
"""

from environments.base_env import BaseEnvironment
from environments.code_env import CodeEnvironment
from environments.reasoning_env import ReasoningEnvironment
from environments.tool_env import ToolEnvironment

__all__ = ["BaseEnvironment", "CodeEnvironment", "ReasoningEnvironment", "ToolEnvironment"]
