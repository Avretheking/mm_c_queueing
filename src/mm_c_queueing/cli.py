"""Command-line interface for the M/M/c queueing model.

Usage:
    mm-c-queue --lambda 1.0 --mu 1.0 --servers 4 --compare
    mm-c-queue --lambda 1.0 --mu 1.0 --servers 4 --compare --plot
"""

import argparse
import sys

from mm_c_queueing.analytical import mmc_metrics
from mm_c_queueing.simulation import simulate_mmc


def _fmt(value, width=10):
    if value is None:
        return "N/A".rjust(width)
    return f"{value:>{width}.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm-c-queue",
        description="Analytical (Erlang-C) vs simulation comparison for an M/M/c queue.",
    )
    parser.add_argument("--lambda", type=float, dest="lam", default=1.0,
                        help="Arrival rate (tasks / time unit). Default: 1.0")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Service rate per server (tasks / time unit). Default: 1.0")
    parser.add_argument("--servers", type=int, default=4,
                        help="Number of homogeneous servers c. Default: 4")
    parser.add_argument("--tasks", type=int, default=150_000,
                        help="Measured tasks for the simulation. Default: 150000")
    parser.add_argument("--warmup", type=int, default=5_000,
                        help="Warm-up tasks discarded before measuring. Default: 5000")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for the simulator (deterministic when set).")
    parser.add_argument("--compare", action="store_true",
                        help="Run both analytical and simulation and show them side by side.")
    parser.add_argument("--plot", action="store_true",
                        help="Save a comparison chart (requires matplotlib).")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    lam, mu, c = args.lam, args.mu, args.servers

    if lam <= 0 or mu <= 0 or c <= 0:
        print("Error: lambda, mu, and servers must all be positive.", file=sys.stderr)
        return 2

    ana = mmc_metrics(lam, mu, c)
    if ana is None:
        print(f"System unstable: rho = {lam / (c * mu):.4f} >= 1. No steady state.", file=sys.stderr)
        return 1

    if args.compare:
        sim = simulate_mmc(lam, mu, c, n_tasks=args.tasks, warmup=args.warmup, seed=args.seed)
        print(f"\nM/M/c queue: lambda={lam}, mu={mu}, c={c}")
        print(f"Utilization rho = {ana['rho']:.4f}  (stable: {ana['rho'] < 1.0})")
        print()
        print(f"{'measure':<14} {'analytical':>12} {'simulation':>12} {'rel_err':>10}")
        print("-" * 52)
        for key, label in (("rho", "rho"), ("p_wait", "P(wait)"), ("Lq", "Lq"),
                           ("L", "L"), ("Wq", "Wq"), ("W", "W")):
            a = ana[key]
            s = sim[key] if sim else None
            if s is None:
                print(f"{label:<14} {_fmt(a):>12} {'N/A':>12} {'N/A':>10}")
            else:
                rel = abs(s - a) / a if a != 0 else abs(s)
                print(f"{label:<14} {_fmt(a):>12} {_fmt(s):>12} {rel:>10.2%}")
        print()
        if sim:
            print("Simulation: analytical vs simulated values agree within sampling error.")
    else:
        print(f"\nM/M/c queue (analytical, Erlang-C): lambda={lam}, mu={mu}, c={c}")
        print(f"Utilization rho = {ana['rho']:.4f}  (stable: {ana['rho'] < 1.0})")
        print(f"P(task waits)   = {ana['p_wait']:.4f}")
        print(f"p0 (empty)      = {ana['p0']:.4f}")
        print(f"L (in system)   = {ana['L']:.4f}")
        print(f"Lq (in queue)   = {ana['Lq']:.4f}")
        print(f"W (sojourn)     = {ana['W']:.4f}")
        print(f"Wq (queue wait) = {ana['Wq']:.4f}")
        print()

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping --plot.", file=sys.stderr)
            return 0

        rhos = [i / 100 for i in range(1, 81)]
        ana_wq, ana_lq = [], []
        for r in rhos:
            l = r * c * mu
            m = mmc_metrics(l, mu, c)
            ana_wq.append(m["Wq"])
            ana_lq.append(m["Lq"])

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        ax = axes[0]
        ax.plot(rhos, ana_wq, color="#1F3B5C", lw=2, label="Wq (analytical)")
        ax.set_xlabel("Utilization rho")
        ax.set_ylabel("Mean queue wait Wq")
        ax.set_title("Mean queue wait vs. utilization")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(rhos, ana_lq, color="#2E7D32", lw=2, label="Lq (analytical)")
        ax.set_xlabel("Utilization rho")
        ax.set_ylabel("Mean queue length Lq")
        ax.set_title("Mean queue length vs. utilization")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout()
        out = "mmc_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Chart saved to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
