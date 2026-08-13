"""Ground-truth-injectable order-flow simulator.

The simulator exists to VERIFY the statistics pipeline, so its ground truth must
be exact and its recovery must be free of errors-in-variables (EIV) bias — a
biased simulator would make the recovery gate lie. This module achieves that by
keeping the *order flow* and the *efficient price* linked ONLY through an
EXOGENOUS driver, and by exposing the price on a CONTINUOUS scale so that
tick-quantization can never attenuate the recovered slope.

Why the earlier draft was biased (diagnosis, kept here as a warning):

  1. It drove the latent price by ``beta * (book-derived OFI e_n)``, and e_n
     includes the price-change term (an indicator times the whole best-queue).
     That makes the driver ENDOGENOUS to price — a runaway feedback loop that
     overflows for larger beta — and huge/noisy besides.
  2. It regressed onto forward returns computed from the TICK-QUANTIZED book
     mid. With per-event drift smaller than a tick, the quantized mid is a
     coarsely rounded version of the true drift — classic EIV ATTENUATION
     (recovered slope ~0.45x beta, verified empirically).

The design here fixes both:

1. **Order flow** — a Poisson-baseline mix of limit submissions, cancellations,
   and marketable executions, with a latent AR(1) buy/sell pressure ``z_n`` that
   gives the flow realistic short memory. It produces a self-consistent book (no
   orphan cancels, no crossed quotes, no negative sizes) emitted in the canonical
   :mod:`src.schema` event format.

2. **An EXOGENOUS driver** ``g_n`` — a signed order-flow-imbalance quantity built
   purely from the flow we draw (side, action, size): ``+size`` for adds on the
   pressure side, ``-size`` for cancels/executions on the pressure side, mirrored
   for the opposite side, scaled to O(1). Crucially ``g_n`` does NOT depend on the
   price-change term, so there is NO price -> driver feedback.

3. **A CONTINUOUS efficient log-price** ``lp`` driven by the injected relation
   ``lp_{n+1} = lp_n + beta * g_n + sigma * eps_n``, ``eps ~ N(0, 1)``. The
   exposed recovery target is the CONTINUOUS mid ``exp(lp)`` (NOT the book mid),
   so ``log(mid_{n+1}) - log(mid_n) = beta * g_n + sigma * eps_n`` holds EXACTLY.
   OLS of that forward return on ``g_n`` recovers ``beta`` with no attenuation;
   ``beta = 0`` is therefore a true null (the placebo mode).

The emitted book/quotes still track ``exp(lp)`` on the tick grid (for realism and
for the reconstruction differential), but the recovery gate consumes the exposed
continuous mid and driver, so quantization is irrelevant to the estimate.

The injected ``beta`` is expressed in **log-return per unit of the (scaled)
signed-flow driver**. Everything is deterministic in the seed (numpy
``default_rng`` seeded from :data:`config.registry.MASTER_SEED`). See
docs/DESIGN.md for the modelling rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import registry as R

from ..schema import (
    EVENT_COLUMNS,
    EVENT_DTYPES,
    EventType,
    validate_event_frame,
)


@dataclass(slots=True)
class SimConfig:
    """Simulator parameters (all documented; defaults give a large-tick, ~\\$30
    stock with dense best-level queues, calibrated so a modest injected ``beta``
    moves the continuous mid measurably while the book stays valid)."""

    beta: float = 0.0  # injected driver -> forward-log-return coefficient
    sigma: float = 8e-5  # idiosyncratic per-event log-price noise (sd)
    p0: int = 300_000  # initial price in 1/10000 dollar ($30.00)
    tick: int = 100  # tick size in 1/10000 dollar ($0.01)
    ar_phi: float = 0.5  # AR(1) persistence of latent buy/sell pressure
    ar_sigma: float = 1.0  # innovation sd of the latent pressure
    base_queue: int = 200  # target resting size at each best quote (shares)
    exec_prob: float = 0.25  # P(event is a marketable execution)
    cancel_prob: float = 0.35  # P(event is a cancellation) (rest are submissions)
    mean_order: int = 100  # mean order size (shares); also the driver scale
    seed_offset: int = 0  # added to MASTER_SEED so distinct streams differ


@dataclass(slots=True)
class SimGroundTruth:
    """The exact ground truth exposed for the recovery/placebo gates.

    ``driver`` is the exogenous per-event signal ``g_n`` that drives the price;
    ``mid_continuous`` is the CONTINUOUS efficient mid ``exp(lp_n)`` (never
    tick-quantized) after event ``n``; ``fwd_ret_1`` is the next-event forward
    log-return of that continuous mid (NaN at the last event). By construction
    ``fwd_ret_1[n] == beta * driver[n] + sigma * eps_n`` exactly, so an OLS of
    ``fwd_ret_1`` on ``driver`` recovers ``beta`` without EIV attenuation."""

    beta: float
    driver: np.ndarray  # exogenous signed-flow driver g_n (all finite)
    mid_continuous: np.ndarray  # continuous efficient mid exp(lp_n) after event n
    fwd_ret_1: np.ndarray  # next-event continuous forward log-return (NaN at end)
    extra: dict = field(default_factory=dict)


class OrderFlowSimulator:
    """Implements ``interfaces.OrderFlowSimulator``: :meth:`generate` returns a
    canonical event frame; :meth:`generate_with_truth` also returns the injected
    ground truth for verification."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()

    # -- contract ----------------------------------------------------------- #

    def generate(self, n_events: int, seed: int) -> pd.DataFrame:
        """Return a canonical, schema-valid event frame of ``n_events`` rows,
        deterministic in ``seed``."""
        frame, _ = self._run(n_events, seed)
        return frame

    def generate_with_truth(self, n_events: int, seed: int) -> tuple[pd.DataFrame, SimGroundTruth]:
        """As :meth:`generate`, plus the exact :class:`SimGroundTruth`."""
        return self._run(n_events, seed)

    # -- engine ------------------------------------------------------------- #

    def _run(self, n_events: int, seed: int) -> tuple[pd.DataFrame, SimGroundTruth]:
        if n_events < 2:
            raise ValueError("n_events must be >= 2")
        cfg = self.config
        rng = np.random.default_rng(R.MASTER_SEED + cfg.seed_offset + int(seed))
        scale = float(max(cfg.mean_order, 1))  # keeps the driver O(1)

        # Latent efficient log-price and AR(1) buy/sell pressure.
        lp = float(np.log(cfg.p0))
        z = 0.0

        # Best quotes (integer, on the tick grid); the book always shows a
        # one-tick spread straddling exp(lp). Sizes evolve purely from the flow,
        # so they are independent of the price path (no feedback into the driver).
        bid_px = self._round_tick(np.exp(lp) - cfg.tick / 2.0)
        ask_px = bid_px + cfg.tick
        bid_sz = cfg.base_queue
        ask_sz = cfg.base_queue

        # One aggregate resting order per side keeps cancels/executions valid and
        # the reconstructed best level exactly controllable.
        next_oid = 1
        bid_oid, ask_oid = next_oid, next_oid + 1
        next_oid += 2

        times = np.empty(n_events, dtype="float64")
        etypes = np.empty(n_events, dtype="int64")
        oids = np.empty(n_events, dtype="int64")
        sizes = np.empty(n_events, dtype="int64")
        prices = np.empty(n_events, dtype="int64")
        dirs = np.empty(n_events, dtype="int64")

        driver = np.empty(n_events, dtype="float64")  # exogenous g_n
        mid_c = np.empty(n_events, dtype="float64")  # continuous mid exp(lp_n)

        t = 34200.0  # market open in seconds after midnight, for realism

        for n in range(n_events):
            # 1) advance latent pressure (AR(1)) and inter-arrival time (Poisson).
            z = cfg.ar_phi * z + cfg.ar_sigma * rng.standard_normal()
            t += rng.exponential(1e-3)  # ~1ms mean inter-arrival
            buy = rng.random() < _logistic(z)  # pressure side
            side = 1 if buy else -1

            # 2) re-center the tick-grid quotes on the current efficient price.
            bid_px = self._round_tick(np.exp(lp) - cfg.tick / 2.0)
            ask_px = bid_px + cfg.tick

            # 3) choose an action and emit ONE message that keeps the book valid.
            #    ``add`` marks a liquidity-adding submission (vs a cancel/execute
            #    removal); ``qty`` is the realized size that hits the book.
            u = rng.random()
            osize = 1 + int(rng.geometric(1.0 / scale))
            if u < cfg.exec_prob:
                # Marketable execution consuming the opposite best queue.
                add = False
                etype = int(EventType.EXECUTE_VISIBLE)
                if buy:  # buy trade lifts the ask (resting sell order hit)
                    qty = min(osize, ask_sz)
                    oid, px, d = ask_oid, ask_px, -1
                    ask_sz -= qty
                    if ask_sz <= 0:  # queue emptied -> refill next event
                        ask_sz = cfg.base_queue
                        ask_oid = next_oid
                        next_oid += 1
                else:  # sell trade hits the bid
                    qty = min(osize, bid_sz)
                    oid, px, d = bid_oid, bid_px, 1
                    bid_sz -= qty
                    if bid_sz <= 0:
                        bid_sz = cfg.base_queue
                        bid_oid = next_oid
                        next_oid += 1
            elif u < cfg.exec_prob + cfg.cancel_prob:
                # Cancellation reducing the pressure-side best queue (leave >= 1
                # share so the level never vanishes and the book stays valid).
                add = False
                etype = int(EventType.CANCEL_PARTIAL)
                if buy:
                    qty = min(osize, max(bid_sz - 1, 0))
                    oid, px, d = bid_oid, bid_px, 1
                    bid_sz -= qty
                else:
                    qty = min(osize, max(ask_sz - 1, 0))
                    oid, px, d = ask_oid, ask_px, -1
                    ask_sz -= qty
            else:
                # New limit submission adding depth on the pressure side.
                add = True
                etype = int(EventType.SUBMIT)
                qty = osize
                if buy:
                    bid_sz += osize
                    oid, px, d = bid_oid, bid_px, 1
                else:
                    ask_sz += osize
                    oid, px, d = ask_oid, ask_px, -1

            times[n] = t
            etypes[n] = etype
            oids[n] = oid
            sizes[n] = max(qty, 0)
            prices[n] = px
            dirs[n] = d

            # 4) EXOGENOUS driver g_n: signed by (pressure side) x (add vs remove),
            #    scaled to O(1). Built only from the flow we drew — never from the
            #    price-change term — so there is no price -> driver feedback.
            g_n = side * (1.0 if add else -1.0) * qty / scale
            driver[n] = g_n

            # 5) expose the CONTINUOUS mid, then inject the signal into the NEXT
            #    increment: lp_{n+1} = lp_n + beta * g_n + sigma * eps_n. Because
            #    fwd_ret_1[n] = lp_{n+1} - lp_n exactly, recovery is unbiased and
            #    quantization-proof (the book mid is never used for recovery).
            mid_c[n] = np.exp(lp)
            lp = lp + cfg.beta * g_n + cfg.sigma * rng.standard_normal()

        frame = pd.DataFrame(
            {
                "time_s": times,
                "event_type": etypes,
                "order_id": oids,
                "size": sizes,
                "price": prices,
                "direction": dirs,
            }
        ).astype(EVENT_DTYPES)[list(EVENT_COLUMNS)]
        validate_event_frame(frame)

        # Forward return of the CONTINUOUS mid: exact one-step drift + noise.
        fwd_ret_1 = np.full(n_events, np.nan, dtype="float64")
        log_mid = np.log(mid_c)
        fwd_ret_1[:-1] = log_mid[1:] - log_mid[:-1]
        truth = SimGroundTruth(
            beta=cfg.beta,
            driver=driver,
            mid_continuous=mid_c,
            fwd_ret_1=fwd_ret_1,
            extra={"n_events": n_events, "seed": int(seed), "driver_scale": scale},
        )
        return frame, truth

    def _round_tick(self, price: float) -> int:
        tick = self.config.tick
        return int(round(price / tick) * tick)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
