"""
DSPy Optimization Utilities

Generic wrappers for DSPy optimization strategies.
"""

from typing import Callable, Any, Optional
import dspy
from dspy.teleprompt import BootstrapFewShot, COPRO


def run_bootstrap_optimization(
    student_module: dspy.Module,
    trainset: list,
    metric: Callable,
    max_bootstrapped_demos: int = 16,
    max_labeled_demos: int = 16,
    max_rounds: int = 4,
    max_errors: int = 5
) -> dspy.Module:
    """
    Run BootstrapFewShot optimization.

    Args:
        student_module: DSPy module to optimize
        trainset: Training examples
        metric: Validation metric function(example, pred, trace) -> float
        max_bootstrapped_demos: Max number of bootstrapped demonstrations
        max_labeled_demos: Max number of labeled demonstrations
        max_rounds: Max optimization rounds
        max_errors: Max allowed errors before stopping

    Returns:
        Optimized DSPy module
    """
    teleprompter = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        max_errors=max_errors
    )

    return teleprompter.compile(student_module, trainset=trainset)


def run_two_stage_optimization(
    student_module: dspy.Module,
    trainset: list,
    metric: Callable,
    bootstrap_config: Optional[dict] = None,
    copro_config: Optional[dict] = None
) -> dspy.Module:
    """
    Run two-stage optimization: BootstrapFewShot followed by COPRO.

    Args:
        student_module: DSPy module to optimize
        trainset: Training examples
        metric: Validation metric function
        bootstrap_config: Config dict for BootstrapFewShot. Keys:
            - max_bootstrapped_demos (default: 20)
            - max_labeled_demos (default: 20)
            - max_rounds (default: 6)
            - max_errors (default: 10)
        copro_config: Config dict for COPRO. Keys:
            - breadth (default: 10)
            - depth (default: 3)
            - init_temperature (default: 1.4)

    Returns:
        Optimized DSPy module
    """
    # Default configs
    if bootstrap_config is None:
        bootstrap_config = {}
    if copro_config is None:
        copro_config = {}

    bootstrap_defaults = {
        'max_bootstrapped_demos': 20,
        'max_labeled_demos': 20,
        'max_rounds': 6,
        'max_errors': 10
    }
    copro_defaults = {
        'breadth': 10,
        'depth': 3,
        'init_temperature': 1.4
    }

    bootstrap_config = {**bootstrap_defaults, **bootstrap_config}
    copro_config = {**copro_defaults, **copro_config}

    # Stage 1: BootstrapFewShot
    bootstrap_teleprompter = BootstrapFewShot(
        metric=metric,
        **bootstrap_config
    )
    stage1_optimized = bootstrap_teleprompter.compile(student_module, trainset=trainset)

    # Stage 2: COPRO
    try:
        copro_teleprompter = COPRO(
            metric=metric,
            **copro_config
        )
        final_optimized = copro_teleprompter.compile(stage1_optimized, trainset=trainset)
        return final_optimized
    except Exception:
        # Fall back to stage 1 if COPRO fails
        return stage1_optimized
