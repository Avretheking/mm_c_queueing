"""M/M/c queueing model for AI task scheduling in cloud data centers.

Provides analytical Erlang-C solutions and an event-driven discrete-event
simulator, with a CLI to compare the two side by side.
"""

from .analytical import mmc_metrics, mmc_p0, mmc_p_wait, mmc_rho
from .simulation import simulate_mmc


def compare(lam, mu, c, n_tasks=150_000, warmup=5_000, seed=None):
    """Run analytical (Erlang-C) and simulation side by side.

    Returns a dict with rho, stability flag, and per-measure analytical vs
    simulation values (plus relative error where computable). Returns None
    when the system is unstable (rho >= 1).
    """
    ana = mmc_metrics(lam, mu, c)
    if ana is None:
        return None

    sim = simulate_mmc(lam, mu, c, n_tasks=n_tasks, warmup=warmup, seed=seed)
    result = {"rho": ana["rho"], "stable": ana["rho"] < 1.0}
    for key in ("p_wait", "Lq", "L", "Wq", "W"):
        a = ana[key]
        s = sim[key] if sim else None
        entry = {"analytical": a, "simulation": s}
        if a is not None and s is not None and a != 0:
            entry["relative_error"] = abs(s - a) / a
        result[key] = entry
    return result


__all__ = [
    "mmc_rho",
    "mmc_p0",
    "mmc_p_wait",
    "mmc_metrics",
    "compare",
    "simulate_mmc",
]

__version__ = "0.1.0"
