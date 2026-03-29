"""SkyDiscover evolution backends for ClawLoop (enterprise-only).

Usage — synchronous (blocks during evolution):

    from enterprise.evolution.backends import SkyDiscoverAdaEvolve

    evolver = SkyDiscoverAdaEvolve(
        adapter=my_adapter,
        tasks=task_list,
        agent_state_factory=my_factory,
        iterations=20,
        model="claude-sonnet-4-6",
    )
    harness = Harness(evolver=evolver, ...)

Usage — async (returns immediately, poll for results):

    from enterprise.evolution.backends import CloudAdaEvolve

    evolver = CloudAdaEvolve(
        adapter=my_adapter,
        tasks=task_list,
        agent_state_factory=my_factory,
        iterations=20,
        max_concurrent=1,
    )
    harness = Harness(evolver=evolver, ...)

    # After forward_backward, check FBResult.info["status"] == "running"
    # Poll via harness.evolution_summary(run_id)
    # Retrieve via evolver.get_result(run_id)
"""

from enterprise.evolution.backends.skydiscover_adaevolve import SkyDiscoverAdaEvolve
from enterprise.evolution.backends.skydiscover_cloud import CloudAdaEvolve
from enterprise.evolution.backends.skydiscover_evaluator import (
    AgentStateFactory,
    ClawLoopEvaluator,
)

__all__ = [
    "SkyDiscoverAdaEvolve",
    "CloudAdaEvolve",
    "ClawLoopEvaluator",
    "AgentStateFactory",
]
