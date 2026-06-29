"""Error trajectory analysis for self-distillation.

Generates micro-reflective corrections from agent failures, following the
TAPO (Trajectory-Augmented Policy Optimization) paradigm:
https://arxiv.org/abs/2606.18844v1

The key insight: when an agent fails on a task, we can construct a learning
trajectory that shows:
1. The agent's erroneous reasoning (preserving the on-policy prefix)
2. A natural-language diagnosis of why it failed
3. Corrected reasoning guided by a reference solution

This provides fine-grained corrective signal beyond simple distributional
alignment, enabling the agent to learn from specific mistakes rather than
just imitating a privileged distribution.
"""

import dspy
from dataclasses import dataclass, field
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class ReflectiveTrajectory:
    """A micro-reflective correction trajectory.

    Contains the failed agent response, a diagnosis of the error, and
    corrected reasoning that can be used for self-distillation.
    """
    task_input: str  # Original task
    agent_output: str  # What the agent produced (failed)
    expected_behavior: str  # What a good response should look like
    error_diagnosis: str = ""  # LLM-generated diagnosis of why it failed
    corrected_reasoning: str = ""  # Corrected approach/solution
    skill_text: str = ""  # The skill/instructions the agent was following

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "agent_output": self.agent_output,
            "expected_behavior": self.expected_behavior,
            "error_diagnosis": self.error_diagnosis,
            "corrected_reasoning": self.corrected_reasoning,
            "skill_text": self.skill_text,
        }

    def to_training_example(self) -> dict:
        """Format as a training example for self-distillation.

        The trajectory shows the transition from failure to correction,
        preserving the agent's own reasoning prefix.
        """
        return {
            "task_input": self.task_input,
            "failed_response": self.agent_output,
            "diagnosis": self.error_diagnosis,
            "corrected_response": self.corrected_reasoning,
        }


class ErrorTrajectoryAnalyzer(dspy.Module):
    """Analyzes failures and generates micro-reflective corrections.

    Uses LLM-as-judge to:
    1. Understand why the agent's response failed
    2. Diagnose the specific error
    3. Provide corrected reasoning that addresses the failure
    """

    class DiagnosisSignature(dspy.Signature):
        """Diagnose why an agent's response is incorrect and provide corrected reasoning.

        Analyze the agent's response against the expected behavior:
        1. Identify where the reasoning went wrong
        2. Explain the specific error or misconception
        3. Provide corrected reasoning that addresses the failure

        The corrected reasoning should preserve the overall approach while
        fixing the specific error.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        agent_output: str = dspy.InputField(desc="The agent's incorrect response")
        expected_behavior: str = dspy.InputField(desc="What a correct response should achieve")
        skill_text: str = dspy.InputField(desc="The skill instructions the agent was following", optional=True)
        error_diagnosis: str = dspy.OutputField(desc="Explanation of why the agent's response is incorrect")
        corrected_reasoning: str = dspy.OutputField(desc="Corrected reasoning that addresses the failure")

    def __init__(self, config: EvolutionConfig):
        super().__init__()
        self.config = config
        self.diagnoser = dspy.ChainOfThought(self.DiagnosisSignature)

    def forward(self, task_input: str, agent_output: str, expected_behavior: str, skill_text: str = "") -> dspy.Prediction:
        """Analyze a failure and generate reflective correction."""
        result = self.diagnoser(
            task_input=task_input,
            agent_output=agent_output,
            expected_behavior=expected_behavior,
            skill_text=skill_text or "N/A",
        )
        return dspy.Prediction(
            error_diagnosis=result.error_diagnosis,
            corrected_reasoning=result.corrected_reasoning,
        )


def analyze_failure_trajectory(
    task_input: str,
    agent_output: str,
    expected_behavior: str,
    skill_text: str = "",
    config: Optional[EvolutionConfig] = None,
) -> ReflectiveTrajectory:
    """Generate a micro-reflective trajectory from a single failure.

    Args:
        task_input: The task the agent was given
        agent_output: The agent's incorrect response
        expected_behavior: What a correct response should achieve
        skill_text: The skill instructions the agent was following
        config: Evolution config (uses default if None)

    Returns:
        ReflectiveTrajectory with diagnosis and corrected reasoning
    """
    from evolution.core.config import EvolutionConfig

    if config is None:
        config = EvolutionConfig()

    analyzer = ErrorTrajectoryAnalyzer(config)
    lm = dspy.LM(config.eval_model)

    with dspy.context(lm=lm):
        result = analyzer(
            task_input=task_input,
            agent_output=agent_output,
            expected_behavior=expected_behavior,
            skill_text=skill_text,
        )

    return ReflectiveTrajectory(
        task_input=task_input,
        agent_output=agent_output,
        expected_behavior=expected_behavior,
        error_diagnosis=str(getattr(result, "error_diagnosis", "")),
        corrected_reasoning=str(getattr(result, "corrected_reasoning", "")),
        skill_text=skill_text,
    )


def extract_failures_from_holdout(
    holdout_examples: list,
    baseline_scores: list[float],
    evolved_scores: list[float],
    baseline_preds: list,
    evolved_preds: list,
    threshold: float = 0.5,
) -> list[dict]:
    """Extract failure examples from holdout evaluation.

    Identifies examples where:
    - The evolved skill performed poorly (score < threshold)
    - The evolved skill did worse than baseline
    - Either baseline or evolved failed

    Args:
        holdout_examples: List of DSPy Examples from holdout set
        baseline_scores: Scores for baseline predictions
        evolved_scores: Scores for evolved predictions
        baseline_preds: Baseline prediction objects
        evolved_preds: Evolved prediction objects
        threshold: Score threshold for considering a response a failure

    Returns:
        List of failure dicts with task_input, outputs, scores, etc.
    """
    failures = []

    for i, (ex, base_score, evo_score, base_pred, evo_pred) in enumerate(
        zip(holdout_examples, baseline_scores, evolved_scores, baseline_preds, evolved_preds)
    ):
        # Extract fields
        task = getattr(ex, "task_input", "")
        expected = getattr(ex, "expected_behavior", "")
        base_out = str(getattr(base_pred, "output", ""))
        evo_out = str(getattr(evo_pred, "output", ""))

        # Failure conditions
        is_evolved_failure = evo_score < threshold
        is_regression = evo_score < base_score
        is_baseline_failure = base_score < threshold

        if is_evolved_failure or is_regression or is_baseline_failure:
            failures.append({
                "index": i,
                "task_input": task,
                "expected_behavior": expected,
                "baseline_output": base_out,
                "evolved_output": evo_out,
                "baseline_score": base_score,
                "evolved_score": evo_score,
                "failure_type": "evolved_failure" if is_evolved_failure else
                               "regression" if is_regression else
                               "baseline_failure",
            })

    return failures
