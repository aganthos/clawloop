"""Environments — built-in task environments and benchmark adapters."""

from clawloop.environments.base import EnvAdapter

_LAZY_IMPORTS = {
    "HarborAdapter": "clawloop.environments.harbor",
    "HarborTaskEnvironment": "clawloop.environments.harbor",
    "EnterpriseOpsGymAdapter": "clawloop.environments.enterpriseops_gym",
    "EnterpriseOpsGymEnvironment": "clawloop.environments.enterpriseops_gym",
    "build_adapter_from_hf": "clawloop.environments.enterpriseops_gym",
    "TauBenchAdapter": "clawloop.environments.taubench",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(mod, name)
        globals()[name] = value
        return value
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
