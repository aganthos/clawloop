# Third-Party Notices — Enterprise Evolution Backends

## SkyDiscover

- **Repository**: https://github.com/skydiscover-ai/skydiscover
- **License**: Apache License 2.0
- **Copyright**: Copyright 2025 SkyDiscover Team (UC Berkeley Sky Computing Lab)

The `enterprise/evolution/backends/skydiscover_*.py` modules integrate with
SkyDiscover's `run_discovery()` API to provide AdaEvolve multi-island adaptive
search as a ClawLoop Evolver backend. No SkyDiscover source code is copied or
redistributed — these modules are original ClawLoop code that calls SkyDiscover
as an optional runtime dependency (`pip install skydiscover`).

SkyDiscover is part of the Berkeley Sky Computing ecosystem alongside SkyPilot
and SkyRL.
