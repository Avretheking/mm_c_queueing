"""Analytical M/M/c queueing model (Erlang-C).

Closed-form steady-state solution for an M/M/c queue:

- Poisson arrivals at rate lambda (tasks / time unit)
- c homogeneous servers, each with exponential service rate mu
- FCFS discipline, infinite waiting room

Stability requires rho = lambda / (c * mu) < 1.

State probabilities:
    p_0 = [ sum_{n=0}^{c-1} a^n / n! + (a^c / c!) / (1 - rho) ]^{-1}
    p_n = p_0 * a^n / n!                          for n <= c
    p_n = p_0 * (a^c / c!) * (a / c)^{n-c}        for n > c

where a = lambda / mu (offered load) and rho = a / c.

Erlang-C (probability an arriving task must wait):
    C(c, a) = (a^c / c!) * (1 / (1 - rho)) * p_0

Performance measures (Little's law: L = lambda * W, Lq = lambda * Wq):
    L_q = C(c, a) * rho / (1 - rho)
    L   = L_q + a
    W_q = L_q / lambda
    W   = W_q + 1 / mu
"""

import math


def mmc_rho(lam: float, mu: float, c: int):
    """Return (is_stable, rho) for the M/M/c queue.

    rho = lambda / (c * mu).  The system is stable iff rho < 1.
    """
    rho = lam / (c * mu)
    return rho < 1.0, rho


def mmc_p0(lam: float, mu: float, c: int) -> float:
    """Normalizing probability p_0 (probability the system is empty)."""
    stable, rho = mmc_rho(lam, mu, c)
    if not stable:
        return 0.0

    a = lam / mu
    finite_sum = sum(a**n / math.factorial(n) for n in range(c))
    tail = (a**c / math.factorial(c)) / (1.0 - rho)
    return 1.0 / (finite_sum + tail)


def mmc_p_wait(lam: float, mu: float, c: int) -> float:
    """Erlang-C: probability an arriving task must wait (all servers busy)."""
    stable, rho = mmc_rho(lam, mu, c)
    if not stable:
        return 0.0

    a = lam / mu
    p0 = mmc_p0(lam, mu, c)
    return (a**c / math.factorial(c)) * (1.0 / (1.0 - rho)) * p0


def mmc_metrics(lam: float, mu: float, c: int):
    """Return dict of steady-state performance measures, or None if unstable.

    Keys: rho, p0, p_wait, L, Lq, W, Wq
    """
    stable, rho = mmc_rho(lam, mu, c)
    if not stable:
        return None

    a = lam / mu
    p0 = mmc_p0(lam, mu, c)
    p_wait = mmc_p_wait(lam, mu, c)

    lq = p_wait * rho / (1.0 - rho)
    l = lq + a
    wq = lq / lam
    w = wq + 1.0 / mu

    return {
        "rho": rho,
        "p0": p0,
        "p_wait": p_wait,
        "L": l,
        "Lq": lq,
        "W": w,
        "Wq": wq,
    }
