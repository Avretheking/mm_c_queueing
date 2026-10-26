"""Event-driven discrete-event simulator for the M/M/c queue.

FCFS discipline, Poisson arrivals (rate lambda), exponential service times
(rate mu per server), c homogeneous servers, infinite waiting room.

The simulator is independent of the analytical module — it drives a pure
event clock with heapq and records queue length, waiting time, and sojourn
time over a measured horizon after an initial warm-up period.
"""

import heapq
import random


def simulate_mmc(lam: float, mu: float, c: int, n_tasks: int = 150_000,
                 warmup: int = 5_000, seed: int | None = None):
    """Run an event-driven M/M/c FCFS simulation.

    Returns a dict with keys rho, L, Lq, W, Wq, or None if the system is
    unstable (lambda >= c * mu).
    """
    if lam <= 0 or mu <= 0 or c <= 0:
        return None
    if lam >= c * mu:
        return None

    if seed is not None:
        random.seed(seed)

    servers_busy = 0
    queue = []            # arrival times of tasks waiting for a server
    total_wait = 0.0      # sum of waiting times (for measured tasks)
    total_sojourn = 0.0   # sum of sojourn times (for measured tasks)
    q_len_area = 0.0      # area under the queue-length curve
    busy_area = 0.0       # area under the busy-server-count curve
    tasks_completed = 0   # measured-task completions (excludes warm-up)
    arrivals_measured = 0 # arrivals after warm-up (for p_wait)
    tasks_waited = 0      # arrivals after warm-up that found all servers busy

    measuring = False  # becomes True once warm-up tasks are completed

    t = 0.0
    last_t = 0.0
    seq = 0
    events = []  # (time, seq, kind, payload)

    seq += 1
    heapq.heappush(events, (random.expovariate(lam), seq, "arr", None))

    while tasks_completed < n_tasks + warmup:
        t, _, kind, payload = heapq.heappop(events)
        dt = t - last_t
        q_len_area += len(queue) * dt
        busy_area += servers_busy * dt
        last_t = t

        if kind == "arr":
            seq += 1
            heapq.heappush(events, (t + random.expovariate(lam), seq, "arr", None))
            if measuring:
                arrivals_measured += 1
            if servers_busy < c:
                servers_busy += 1
                st = random.expovariate(mu)
                seq += 1
                heapq.heappush(events, (t + st, seq, "dep", st))
            else:
                if measuring:
                    tasks_waited += 1
                queue.append(t)
        else:  # departure
            st = payload
            tasks_completed += 1
            if tasks_completed >= warmup and not measuring:
                measuring = True
            total_sojourn += st  # service portion of sojourn

            if queue:
                arr_t = queue.pop(0)
                wait = t - arr_t
                total_wait += wait
                total_sojourn += wait  # waiting portion of sojourn
                st2 = random.expovariate(mu)
                seq += 1
                heapq.heappush(events, (t + st2, seq, "dep", st2))
            else:
                servers_busy -= 1

    t_final = t
    lq = q_len_area / t_final
    l = (busy_area / t_final) + lq
    # Little's law with effective arrival rate lambda (stable system)
    wq = lq / lam
    w = l / lam
    util = busy_area / (c * t_final)
    p_wait_sim = (tasks_waited / arrivals_measured) if arrivals_measured > 0 else 0.0

    return {
        "rho": util,
        "p_wait": p_wait_sim,
        "L": l,
        "Lq": lq,
        "W": w,
        "Wq": wq,
    }
