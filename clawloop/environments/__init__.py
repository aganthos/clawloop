"""Environments — built-in task environments and benchmark adapters."""

from clawloop.environments.base import EnvAdapter
from clawloop.environments.harbor import HarborAdapter, HarborTaskEnvironment
from clawloop.environments.enterpriseops_gym import (
    EnterpriseOpsGymAdapter,
    EnterpriseOpsGymEnvironment,
    build_adapter_from_hf,
)


def __getattr__(name: str):
    if name == "TauBenchAdapter":
        from clawloop.environments.taubench import TauBenchAdapter
        return TauBenchAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnvAdapter",
    "HarborAdapter",
    "HarborTaskEnvironment",
    "TauBenchAdapter",
    "EnterpriseOpsGymAdapter",
    "EnterpriseOpsGymEnvironment",
    "build_adapter_from_hf",
]
