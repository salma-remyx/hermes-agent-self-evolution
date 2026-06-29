"""Tests for error trajectory analysis and reflective corrections."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import dspy
import pytest

from evolution.skills.error_trajectory import (
    ReflectiveTrajectory,
    analyze_failure_trajectory,
    extract_failures_from_holdout,
)


class TestReflectiveTrajectory:
    def test_to_dict(self):
        trajectory = ReflectiveTrajectory(
            task_input="Test task",
            agent_output="Wrong answer",
            expected_behavior="Correct answer",
            error_diagnosis="Misunderstood the question",
            corrected_reasoning="Here's the right approach...",
        )

        result = trajectory.to_dict()

        assert result["task_input"] == "Test task"
        assert result["agent_output"] == "Wrong answer"
        assert result["expected_behavior"] == "Correct answer"
        assert result["error_diagnosis"] == "Misunderstood the question"

    def test_to_training_example(self):
        trajectory = ReflectiveTrajectory(
            task_input="Test task",
            agent_output="Wrong answer",
            expected_behavior="Correct answer",
            error_diagnosis="Misunderstood the question",
            corrected_reasoning="Here's the right approach...",
        )

        result = trajectory.to_training_example()

        assert "task_input" in result
        assert "failed_response" in result
        assert "diagnosis" in result
        assert "corrected_response" in result
        assert result["failed_response"] == "Wrong answer"


class TestExtractFailures:
    def test_extract_evolved_failures(self):
        """Test extraction of examples where evolved skill performed poorly."""
        # Create mock examples
        examples = [
            Mock(task_input="task1", expected_behavior="expected1"),
            Mock(task_input="task2", expected_behavior="expected2"),
        ]
        baseline_scores = [0.8, 0.7]
        evolved_scores = [0.3, 0.9]  # First is failure (< 0.5)
        baseline_preds = [Mock(output="base1"), Mock(output="base2")]
        evolved_preds = [Mock(output="evo1"), Mock(output="evo2")]

        failures = extract_failures_from_holdout(
            examples, baseline_scores, evolved_scores, baseline_preds, evolved_preds
        )

        assert len(failures) == 1
        assert failures[0]["failure_type"] == "evolved_failure"
        assert failures[0]["evolved_score"] == 0.3

    def test_extract_regressions(self):
        """Test extraction of examples where evolved did worse than baseline."""
        examples = [Mock(task_input="task1", expected_behavior="expected1")]
        baseline_scores = [0.7]
        evolved_scores = [0.5]  # Worse than baseline
        baseline_preds = [Mock(output="base1")]
        evolved_preds = [Mock(output="evo1")]

        failures = extract_failures_from_holdout(
            examples, baseline_scores, evolved_scores, baseline_preds, evolved_preds
        )

        assert len(failures) == 1
        assert failures[0]["failure_type"] == "regression"

    def test_no_failures_when_all_good(self):
        """Test that no failures are extracted when all scores are good."""
        examples = [Mock(task_input="task1", expected_behavior="expected1")]
        baseline_scores = [0.8]
        evolved_scores = [0.9]  # Both above threshold, evolved improved
        baseline_preds = [Mock(output="base1")]
        evolved_preds = [Mock(output="evo1")]

        failures = extract_failures_from_holdout(
            examples, baseline_scores, evolved_scores, baseline_preds, evolved_preds
        )

        assert len(failures) == 0


class TestAnalyzeFailureTrajectory:
    def test_trajectory_structure(self):
        """Test that ReflectiveTrajectory has expected structure."""
        trajectory = ReflectiveTrajectory(
            task_input="Test task",
            agent_output="Wrong answer",
            expected_behavior="Correct answer",
        )

        # Verify all required fields exist
        assert trajectory.task_input == "Test task"
        assert trajectory.agent_output == "Wrong answer"
        assert trajectory.expected_behavior == "Correct answer"

        # Verify optional fields default to empty strings
        assert trajectory.error_diagnosis == ""
        assert trajectory.corrected_reasoning == ""
        assert trajectory.skill_text == ""
