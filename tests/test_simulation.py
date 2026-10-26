"""Tests for the discrete-event M/M/c simulator.

TDD discipline: each test was written before the implementation it drives.
Run with:  pytest tests/test_simulation.py -v
"""

import pytest

from mm_c_queueing.simulation import simulate_mmc


# --------------------------------------------------------------------------- #
# Basic correctness vs analytical
# --------------------------------------------------------------------------- #
def test_simulator_matches_analytical_low_load():
    """At light load the simulator must track the analytical values closely."""
    lam, mu, c = 0.5, 1.0, 2
    sim = simulate_mmc(lam, mu, c, n_tasks=100_000, warmup=5_000, seed=42)

    assert sim is not None
    assert sim["rho"] == pytest.approx(lam / (c * mu), rel=0.02)
    assert sim["Lq"] == pytest.approx(1.0 / 30.0, rel=0.10)   # analytical 0.0333
    assert sim["Wq"] == pytest.approx(1.0 / 15.0, rel=0.10)   # analytical 0.0666


def test_simulator_littles_law():
    """Simulated L = lambda * W and Lq = lambda * Wq (within sampling error)."""
    lam, mu, c = 1.0, 1.0, 3
    sim = simulate_mmc(lam, mu, c, n_tasks=150_000, warmup=5_000, seed=7)

    assert sim is not None
    assert sim["L"] == pytest.approx(lam * sim["W"], rel=0.05)
    assert sim["Lq"] == pytest.approx(lam * sim["Wq"], rel=0.05)


def test_simulator_stability_rejects_rho_ge_1():
    """lambda >= c * mu must return None (no steady state)."""
    assert simulate_mmc(2.0, 1.0, 1, n_tasks=10_000, seed=1) is None
    assert simulate_mmc(4.0, 1.0, 4, n_tasks=10_000, seed=1) is None


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_simulator_is_deterministic_with_seed():
    """Same seed must reproduce identical results."""
    lam, mu, c = 1.0, 1.0, 3
    s1 = simulate_mmc(lam, mu, c, n_tasks=50_000, warmup=2_000, seed=123)
    s2 = simulate_mmc(lam, mu, c, n_tasks=50_000, warmup=2_000, seed=123)
    assert s1 == s2


def test_simulator_different_seed_different_result():
    """Different seeds should (with overwhelming probability) differ."""
    lam, mu, c = 1.0, 1.0, 3
    s1 = simulate_mmc(lam, mu, c, n_tasks=50_000, warmup=2_000, seed=123)
    s2 = simulate_mmc(lam, mu, c, n_tasks=50_000, warmup=2_000, seed=456)
    assert s1 != s2


# --------------------------------------------------------------------------- #
# Integration: simulator vs analytical across utilizations
# --------------------------------------------------------------------------- #
@pytest.mark.integration
@pytest.mark.parametrize(
    "lam,mu,c",
    [
        (0.1, 1.0, 4),   # rho = 0.025
        (0.5, 1.0, 4),   # rho = 0.125
        (1.0, 1.0, 4),   # rho = 0.25
        (2.0, 1.0, 4),   # rho = 0.50
        (3.0, 1.0, 4),   # rho = 0.75
    ],
)
def test_agreement_across_utilizations(lam, mu, c):
    """Simulation Wq must be within 12% of analytical Wq for rho < 1."""
    from mm_c_queueing.analytical import mmc_metrics as analytical

    ana = analytical(lam, mu, c)
    sim = simulate_mmc(lam, mu, c, n_tasks=200_000, warmup=10_000, seed=99)

    assert ana is not None
    assert sim is not None
    rel_err = abs(sim["Wq"] - ana["Wq"]) if ana["Wq"] > 0 else abs(sim["Wq"])
    abs_tol = max(0.12 * ana["Wq"], 1e-4)
    assert rel_err < abs_tol, f"Wq mismatch at rho={ana['rho']:.3f}: sim={sim['Wq']:.4f}, ana={ana['Wq']:.4f}"
