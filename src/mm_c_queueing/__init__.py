"""M/M/c queueing model for AI task scheduling in cloud data centers.

Provides analytical Erlang-C solutions and an event-driven discrete-event
simulator, with a CLI to compare the two side by side.
"""

from .analytical import mmc_metrics, mmc_p0, mmc_p_wait, mmc_rho
from .simulation import simulate_mmc

__all__ = [
    "mmc_rho",
    "mmc_p0",
    "mmc_p_wait",
    "mmc_metrics",
    "simulate_mmc",
]

__version__ = "0.1.0"
