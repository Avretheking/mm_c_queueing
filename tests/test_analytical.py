"""Tests for the analytical M/M/c (Erlang-C) module.

TDD discipline: each test was written before the implementation it drives.
Run with:  pytest tests/test_analytical.py -v
"""

import math
import pytest

from mm_c_queueing.analytical import mmc_metrics, mmc_p0, mmc_p_wait, mmc_rho


# --------------------------------------------------------------------------- #
# Utilization and stability
# --------------------------------------------------------------------------- #
def test_utilization_formula():
    """rho = lambda / (c * mu)."""
    _, rho = mmc_rho(0.5, 1.0, 2)
    assert rho == pytest.approx(0.25)
    _, rho = mmc_rho(1.0, 1.0, 4)
    assert rho == pytest.approx(0.25)
    _, rho = mmc_rho(3.0, 1.0, 4)
    assert rho == pytest.approx(0.75)


def test_stability_returns_false_at_or_above_one():
    """rho >= 1 means the system is unstable."""
    stable, rho = mmc_rho(1.0, 1.0, 1)
    assert rho == pytest.approx(1.0)
    assert stable is False

    stable, rho = mmc_rho(2.0, 1.0, 1)
    assert rho == pytest.approx(2.0)
    assert stable is False

    stable, rho = mmc_rho(4.0, 1.0, 4)
    assert rho == pytest.approx(1.0)
    assert stable is False


def test_stability_returns_true_below_one():
    """rho < 1 means the system is stable."""
    stable, rho = mmc_rho(0.5, 1.0, 2)
    assert rho == pytest.approx(0.25)
    assert stable is True


# --------------------------------------------------------------------------- #
# Normalizing probability p0
# --------------------------------------------------------------------------- #
def test_p0_simple_case():
    """lambda=0.5, mu=1, c=2 -> a=0.5, rho=0.25.
    p0 = 1 / (sum_{n=0}^{1} 0.5^n/n! + (0.5^2/2!)/(1-0.25))
       = 1 / (1 + 0.5 + (0.25/2)/0.75)
       = 1 / (1.5 + 0.166666...)
       = 1 / 1.666666...
       = 0.6
    """
    p0 = mmc_p0(0.5, 1.0, 2)
    assert p0 == pytest.approx(0.6, abs=1e-9)


def test_p0_single_server_reduces_to_1_minus_rho():
    """For c=1, p0 = 1 - rho = 1 - lambda/mu."""
    lam, mu, c = 0.3, 1.0, 1
    p0 = mmc_p0(lam, mu, c)
    assert p0 == pytest.approx(1.0 - lam / mu, abs=1e-9)


def test_p0_probabilities_sum_to_one():
    """State probabilities derived from p0 must sum to 1."""
    lam, mu, c = 1.0, 1.0, 3
    rho = lam / (c * mu)
    a = lam / mu
    p0 = mmc_p0(lam, mu, c)

    total = 0.0
    for n in range(c):
        total += a**n / math.factorial(n) * p0
    for n in range(c, c + 50):
        total += (a**c / math.factorial(c)) * (a / c) ** (n - c) * p0

    assert total == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Erlang-C waiting probability C(c, a)
# --------------------------------------------------------------------------- #
def test_p_wait_in_unit_interval():
    """C(c, a) must always be between 0 and 1 (inclusive)."""
    for lam, mu, c in [(0.1, 1.0, 1), (0.5, 1.0, 2), (2.5, 1.0, 4)]:
        pw = mmc_p_wait(lam, mu, c)
        assert 0.0 <= pw <= 1.0


def test_p_wait_single_server_equals_rho():
    """For c=1, C(1, a) = rho = lambda/mu."""
    lam, mu, c = 0.4, 1.0, 1
    pw = mmc_p_wait(lam, mu, c)
    assert pw == pytest.approx(lam / mu, abs=1e-9)


def test_p_wait_matches_hand_computation():
    """lambda=0.5, mu=1, c=2 -> a=0.5, rho=0.25, p0=0.6.
    C = (a^c / c!) * (1/(1-rho)) * p0
      = (0.25/2) * (1/0.75) * 0.6
      = 0.125 * 1.333... * 0.6
      = 0.1
    """
    pw = mmc_p_wait(0.5, 1.0, 2)
    assert pw == pytest.approx(0.1, abs=1e-9)


# --------------------------------------------------------------------------- #
# Performance measures L, Lq, W, Wq
# --------------------------------------------------------------------------- #
def test_performance_measures_satisfy_littles_law():
    """L = lambda * W and Lq = lambda * Wq."""
    lam, mu, c = 1.0, 1.0, 3
    m = mmc_metrics(lam, mu, c)
    assert m["L"] == pytest.approx(lam * m["W"], abs=1e-9)
    assert m["Lq"] == pytest.approx(lam * m["Wq"], abs=1e-9)


def test_performance_relationships_hold():
    """L = Lq + a, W = Wq + 1/mu."""
    lam, mu, c = 1.0, 1.0, 3
    m = mmc_metrics(lam, mu, c)
    a = lam / mu
    assert m["L"] == pytest.approx(m["Lq"] + a, abs=1e-9)
    assert m["W"] == pytest.approx(m["Wq"] + 1.0 / mu, abs=1e-9)


def test_metrics_known_case():
    """lambda=0.5, mu=1, c=2 -> rho=0.25, p0=0.6, C=0.1.
    Lq = C * rho / (1 - rho) = 0.1 * 0.25 / 0.75 = 0.0333...
    L = Lq + a = 0.0333... + 0.5 = 0.5333...
    Wq = Lq / lambda = 0.0333... / 0.5 = 0.0666...
    W = Wq + 1 = 1.0666...
    """
    m = mmc_metrics(0.5, 1.0, 2)
    assert m["rho"] == pytest.approx(0.25)
    assert m["p0"] == pytest.approx(0.6)
    assert m["p_wait"] == pytest.approx(0.1)
    assert m["Lq"] == pytest.approx(1.0 / 30.0, abs=1e-9)  # 0.0333...
    assert m["L"] == pytest.approx(8.0 / 15.0, abs=1e-9)    # 0.5333...
    assert m["Wq"] == pytest.approx(1.0 / 15.0, abs=1e-9)   # 0.0666...
    assert m["W"] == pytest.approx(16.0 / 15.0, abs=1e-9)   # 1.0666...


def test_metrics_returns_none_when_unstable():
    """rho >= 1 yields None (no steady state)."""
    assert mmc_metrics(2.0, 1.0, 1) is None
    assert mmc_metrics(4.0, 1.0, 4) is None
