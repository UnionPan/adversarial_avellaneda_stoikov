"""
Robustness sweep for the cross-layer Avellaneda--Stoikov game (Reviewer 1, comment 5).
=====================================================================================

Kraken only exposes a 7-day tick window, so genuinely independent calm/volatile/
stressed *historical* windows are not available. Instead we calibrate the
regime-switching jump-diffusion ONCE from the existing data (loaded from
experiments/results/regime_parameters.csv) and probe robustness by sweeping the
*process parameters* that define a market condition:

    * volatility ratio  sigma_volatile / sigma_stable    (regime severity)
    * switching speed: both calibrated rates lambda01/lambda10 scaled by a common
      factor (persistence) -- this preserves the stationary volatile occupancy
    * predator strength  xi*gamma                         (adversarial intensity)

The transition chain is ASYMMETRIC and calibrated (lambda01 ~ 1.49, lambda10 ~ 6.17
per day), so the stationary volatile occupancy is lambda01/(lambda01+lambda10) ~ 19%,
matching the ~21% observed in the data (not the 50% of a symmetric chain).

For every parameter regime we run a Monte-Carlo counterfactual of Vanilla AS vs
Equilibrium AS (effective volatility sigma_eff^2 = sigma^2 + xi*gamma) facing the
same strategic predator w*(q) = -xi*gamma*q, and report aggregated PnL/Sharpe and
the equilibrium-over-vanilla improvement. This is the structural robustness check
that replaces (unavailable) multi-window data.

Reuses the exact formulas from demo_counterfactual_simulation.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Calibrated base parameters (loaded from regime_parameters.csv at runtime;
# the literals below are only a fallback if the artifact is missing).
# ---------------------------------------------------------------------------

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class Scenario:
    name: str
    sigma_stable: float = 0.231437   # canonical fallback (regime_parameters.csv is loaded at runtime)
    sigma_volatile: float = 0.704989 # canonical fallback: 70.50% annualized
    lam01: float = 1.488372    # stable->volatile transition rate, per day (canonical)
    lam10: float = 6.171429    # volatile->stable transition rate, per day (canonical)
    xi: float = 10.0           # predator cost coefficient (xi*gamma = 0.2 baseline)
    gamma: float = 0.02        # MM risk aversion
    lambda0: float = 250_000.0 # order-arrival base intensity, per year
    kappa: float = 10.0        # spread sensitivity
    avg_trade: float = 0.05    # BTC per fill
    q_max: float = 10.0
    horizon_hours: float = 12.0
    dt_seconds: float = 15.0
    s0: float = 90_863.90
    regime0: int = 0

    @property
    def n_steps(self) -> int:
        return int(self.horizon_hours * 3600 / self.dt_seconds)

    @property
    def dt_annual(self) -> float:
        return self.dt_seconds / SECONDS_PER_YEAR

    @property
    def mu_bar(self) -> float:          # mean transition rate, per day
        return 0.5 * (self.lam01 + self.lam10)

    @property
    def occ_vol(self) -> float:         # stationary volatile occupancy of the 2-state chain
        return self.lam01 / (self.lam01 + self.lam10)


def load_canonical():
    """Load the canonical calibration from regime_parameters.csv (relative to THIS file).
    Returns (overrides_dict, source_str); overrides is empty if the artifact is missing."""
    here = Path(__file__).resolve().parent
    for p in (here.parent / "results" / "regime_parameters.csv",
              here.parent.parent / "results" / "regime_parameters.csv"):
        try:
            df = pd.read_csv(p)
            return {
                "sigma_stable": float(df["SIGMA_STABLE"].iloc[0]),
                "sigma_volatile": float(df["SIGMA_VOLATILE"].iloc[0]),
                "lam01": float(df["TRANSITION_RATE_01"].iloc[0]),
                "lam10": float(df["TRANSITION_RATE_10"].iloc[0]),
            }, str(p)
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
            continue
    return {}, "fallback-defaults"


# ---------------------------------------------------------------------------
# Strategy / predator formulas (identical to demo_counterfactual_simulation.py)
# ---------------------------------------------------------------------------

def vanilla_as_spread(q, sigma, gamma, kappa):
    reservation = (gamma * sigma**2) / (2 * kappa)
    skew = (gamma * sigma**2 * q) / (2 * kappa)
    premium = np.log(1 + gamma / kappa) / kappa
    return np.clip(reservation + premium + skew, 1e-4, 0.06)


def equilibrium_as_spread(q, sigma, gamma, kappa, xi):
    sigma_eff_sq = sigma**2 + xi * gamma          # sigma_eff^2 = sigma^2 + xi*gamma
    reservation = (gamma * sigma_eff_sq) / (2 * kappa)
    skew = (gamma * sigma_eff_sq * q) / (2 * kappa)
    premium = np.log(1 + gamma / kappa) / kappa
    return np.clip(reservation + premium + skew, 1e-4, 0.06)


def predator_drift(q, gamma, xi):
    return np.clip(-xi * gamma * q, -0.2, 0.2)     # w*(q) = -xi*gamma*q, bounded +/-20%


# ---------------------------------------------------------------------------
# Single trajectory.  Common random numbers (regime path + Brownian shocks) are
# shared across the two strategies for paired variance reduction; order arrivals
# use a separate reproducible stream.
# ---------------------------------------------------------------------------

def run_trajectory(sc: Scenario, strategy: str, regimes, dW, rng_arr):
    n = sc.n_steps
    S = sc.s0
    q = 0.0
    cash = 0.0
    spread_sum = 0.0
    drift_abs_sum = 0.0
    for i in range(1, n):
        regime = regimes[i - 1]
        sigma = sc.sigma_volatile if regime == 1 else sc.sigma_stable

        w = predator_drift(q, sc.gamma, sc.xi)
        drift_abs_sum += abs(w)
        dS = S * (w * sc.dt_annual + sigma * dW[i])
        S = max(S + dS, 1.0)

        if strategy == "vanilla":
            delta = vanilla_as_spread(q, sigma, sc.gamma, sc.kappa)
        else:
            delta = equilibrium_as_spread(q, sigma, sc.gamma, sc.kappa, sc.xi)
        spread_sum += delta

        bid = S * (1 - delta)
        ask = S * (1 + delta)
        lam = sc.lambda0 * np.exp(-sc.kappa * delta) * sc.dt_annual
        n_buy = rng_arr.poisson(lam)    # customer buys -> MM sells -> inventory down
        n_sell = rng_arr.poisson(lam)   # customer sells -> MM buys  -> inventory up
        dq = (n_sell - n_buy) * sc.avg_trade
        q_new = float(np.clip(q + dq, -sc.q_max, sc.q_max))
        actual = q_new - q
        if actual > 0:
            cash -= actual * bid
        elif actual < 0:
            cash += (-actual) * ask
        q = q_new

    liq_penalty = 0.001
    terminal_pnl = cash + q * S * (1 - liq_penalty * np.sign(q))
    return terminal_pnl, spread_sum / (n - 1) * 1e4, drift_abs_sum / (n - 1)


def gen_regimes(sc: Scenario, rng):
    """Asymmetric two-state chain: from stable use lambda01, from volatile use lambda10."""
    n = sc.n_steps
    p01 = 1.0 - np.exp(-sc.lam01 * sc.dt_seconds / 86400.0)  # stable -> volatile per step
    p10 = 1.0 - np.exp(-sc.lam10 * sc.dt_seconds / 86400.0)  # volatile -> stable per step
    regimes = np.empty(n, dtype=int)
    regimes[0] = sc.regime0
    u = rng.random(n)
    for i in range(1, n):
        prev = regimes[i - 1]
        thr = p01 if prev == 0 else p10
        regimes[i] = (1 - prev) if u[i] < thr else prev
    return regimes


def evaluate(sc: Scenario, n_mc: int, base_seed: int = 7):
    pnl_by = {}
    out = {}
    for strat in ("vanilla", "equilibrium"):
        pnls = np.empty(n_mc)
        spreads = np.empty(n_mc)
        drifts = np.empty(n_mc)
        for k in range(n_mc):
            # Common random numbers: the SAME regime path + Brownian shocks are used for
            # both strategies on path k (seed depends only on k), so the PnL difference is
            # a paired statistic with reduced variance.
            rng_path = np.random.default_rng(base_seed * 100_003 + k)
            rng_arr = np.random.default_rng(base_seed * 100_003 + k + 50_000_000)
            regimes = gen_regimes(sc, rng_path)
            dW = rng_path.normal(0.0, np.sqrt(sc.dt_annual), sc.n_steps)
            pnls[k], spreads[k], drifts[k] = run_trajectory(sc, strat, regimes, dW, rng_arr)
        pnl_by[strat] = pnls
        mean, std = float(pnls.mean()), float(pnls.std())
        out[strat] = {
            "mean_pnl": mean,
            "std_pnl": std,
            "sharpe": mean / std if std > 0 else 0.0,   # Sharpe-LIKE: mean/std of terminal PnL
            "avg_spread_bps": float(spreads.mean()),
            "avg_abs_drift": float(drifts.mean()),
        }
    v, e = out["vanilla"], out["equilibrium"]
    # Paired (common-random-number) PnL difference and its 95% CI.
    d = pnl_by["equilibrium"] - pnl_by["vanilla"]
    d_mean = float(d.mean())
    d_ci95 = float(1.96 * d.std(ddof=1) / np.sqrt(n_mc))
    out["d_pnl_abs_mean"] = d_mean
    out["d_pnl_abs_ci95"] = d_ci95
    out["d_pnl_pct"] = (e["mean_pnl"] / v["mean_pnl"] - 1) * 100 if v["mean_pnl"] != 0 else float("nan")
    out["d_pnl_ci95_pct"] = d_ci95 / v["mean_pnl"] * 100 if v["mean_pnl"] != 0 else float("nan")
    out["d_sharpe_pct"] = (e["sharpe"] / v["sharpe"] - 1) * 100 if v["sharpe"] != 0 else float("nan")
    out["d_spread_pct"] = (e["avg_spread_bps"] / v["avg_spread_bps"] - 1) * 100
    out["frac_paths_eq_wins"] = float((d > 0).mean())
    return out


# ---------------------------------------------------------------------------
# Sweep design
# ---------------------------------------------------------------------------

def main():
    n_mc = int(os.environ.get("N_MC", "800"))
    seed = int(os.environ.get("SEED", "7"))

    canon, canon_src = load_canonical()
    base = Scenario("Base (calibrated)", **canon)
    print(f"[base] from {canon_src}: sigma_s={base.sigma_stable:.4f} sigma_v={base.sigma_volatile:.4f} "
          f"lam01={base.lam01:.3f} lam10={base.lam10:.3f} mu_bar={base.mu_bar:.2f}/day "
          f"vol-occupancy={base.occ_vol*100:.1f}%")

    def scale_rates(sc: Scenario, s: float) -> Scenario:
        # Scale both transition rates by a common factor (persistence axis):
        # changes switching speed while preserving the stationary occupancy.
        return replace(sc, lam01=base.lam01 * s, lam10=base.lam10 * s)

    # (1) Named market-condition scenarios (process parameters varied jointly).
    scenarios = [
        replace(scale_rates(base, 0.5), name="Calm",     sigma_volatile=1.5 * base.sigma_stable, xi=5.0),
        base,
        replace(scale_rates(base, 2.0), name="Volatile", sigma_volatile=1.00, xi=10.0),
        replace(scale_rates(base, 4.0), name="Stressed", sigma_volatile=1.40, xi=20.0),
    ]

    print(f"[scenarios] N_MC={n_mc} seed={seed}")
    rows = []
    for sc in scenarios:
        r = evaluate(sc, n_mc, seed)
        rows.append({
            "Scenario": sc.name,
            "sigma_v/sigma_s": round(sc.sigma_volatile / sc.sigma_stable, 2),
            "mu_bar_per_day": round(sc.mu_bar, 2),
            "vol_occupancy": round(sc.occ_vol, 3),
            "xi*gamma": round(sc.xi * sc.gamma, 2),
            "Vanilla mean PnL": r["vanilla"]["mean_pnl"],
            "Equil mean PnL": r["equilibrium"]["mean_pnl"],
            "dPnL_%": r["d_pnl_pct"],
            "dPnL_CI95_%": r["d_pnl_ci95_pct"],
            "Vanilla Sharpe-like": r["vanilla"]["sharpe"],
            "Equil Sharpe-like": r["equilibrium"]["sharpe"],
            "dSharpe_%": r["d_sharpe_pct"],
            "dSpread_%": r["d_spread_pct"],
            "frac_eq_wins": r["frac_paths_eq_wins"],
            "N_MC": n_mc,
            "seed": seed,
        })
        print(f"  {sc.name:18s} dPnL={r['d_pnl_pct']:+6.1f}% (+/-{r['d_pnl_ci95_pct']:.1f}) "
              f"vanSh={r['vanilla']['sharpe']:.3f} eqSh={r['equilibrium']['sharpe']:.3f} "
              f"eq-wins={r['frac_paths_eq_wins']*100:.0f}%")
    sweep_df = pd.DataFrame(rows)

    # (2) One-at-a-time axes (clean attribution): vary a single parameter, hold the rest at base.
    def axis(name, values, setter):
        out = []
        for val in values:
            r = evaluate(setter(val), n_mc, seed)
            out.append({
                name: val,
                "dPnL_%": r["d_pnl_pct"], "dPnL_CI95_%": r["d_pnl_ci95_pct"],
                "dSharpe_%": r["d_sharpe_pct"],
                "Vanilla Sharpe-like": r["vanilla"]["sharpe"],
                "Equil Sharpe-like": r["equilibrium"]["sharpe"],
            })
            print(f"  {name}={val}: dPnL={r['d_pnl_pct']:+6.1f}% (+/-{r['d_pnl_ci95_pct']:.1f}) "
                  f"dSharpe={r['d_sharpe_pct']:+6.1f}%")
        return pd.DataFrame(out)

    print("[axis] predator strength xi*gamma (sigma, rates at base)")
    xi_df = axis("xi*gamma", [0.1, 0.2, 0.4, 0.8], lambda xg: replace(base, xi=xg / base.gamma))
    print("[axis] volatility ratio sigma_v/sigma_s (rates, xi*gamma at base)")
    vol_df = axis("sigma_v/sigma_s", [1.5, 3.0, 4.5, 6.0],
                  lambda rr: replace(base, sigma_volatile=rr * base.sigma_stable))
    print("[axis] switching speed: mean rate mu_bar/day (occupancy fixed; sigma, xi*gamma at base)")
    mu_df = axis("mu_bar_per_day", [round(s * base.mu_bar, 2) for s in (0.5, 1.0, 2.0, 4.0)],
                 lambda mb: scale_rates(base, mb / base.mu_bar))

    # ---- outputs ----
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    sweep_df.to_csv(results_dir / "robustness_sweep_summary.csv", index=False)
    xi_df.to_csv(results_dir / "robustness_predator_axis.csv", index=False)
    vol_df.to_csv(results_dir / "robustness_vol_axis.csv", index=False)
    mu_df.to_csv(results_dir / "robustness_rate_axis.csv", index=False)

    # ---- figure: (a) Sharpe-like per scenario, (b) vs predator strength, (c) vs volatility ratio ----
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.0))

    x = np.arange(len(sweep_df))
    width = 0.38
    axA.bar(x - width / 2, sweep_df["Vanilla Sharpe-like"], width, label="Vanilla AS", color="#c0392b", edgecolor="black")
    axA.bar(x + width / 2, sweep_df["Equil Sharpe-like"], width, label="Equilibrium AS", color="#2c6fbb", edgecolor="black")
    axA.set_xticks(x)
    axA.set_xticklabels(sweep_df["Scenario"], rotation=12, ha="right", fontsize=9)
    axA.set_ylabel("Sharpe-like (terminal-PnL ratio)")
    axA.set_title("(a) Market-condition regimes")
    axA.grid(True, axis="y", alpha=0.3)
    axA.legend(fontsize=9)

    axB.errorbar(xi_df["xi*gamma"], xi_df["dPnL_%"], yerr=xi_df["dPnL_CI95_%"],
                 fmt="o-", color="#2c6fbb", capsize=3, label=r"$\Delta$ mean PnL")
    axB.plot(xi_df["xi*gamma"], xi_df["dSharpe_%"], "s--", color="#16a085", label=r"$\Delta$ Sharpe-like")
    axB.axhline(0, color="black", lw=0.8, alpha=0.5)
    axB.set_xlabel(r"predator strength $\xi\gamma$")
    axB.set_ylabel("equilibrium over vanilla (%)")
    axB.set_title("(b) vs adversarial intensity")
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=9)

    axC.errorbar(vol_df["sigma_v/sigma_s"], vol_df["dPnL_%"], yerr=vol_df["dPnL_CI95_%"],
                 fmt="o-", color="#8e44ad", capsize=3, label=r"$\Delta$ mean PnL")
    axC.plot(vol_df["sigma_v/sigma_s"], vol_df["dSharpe_%"], "s--", color="#16a085", label=r"$\Delta$ Sharpe-like")
    axC.axhline(0, color="black", lw=0.8, alpha=0.5)
    axC.set_xlabel(r"volatility ratio $\sigma_v/\sigma_s$")
    axC.set_ylabel("equilibrium over vanilla (%)")
    axC.set_title("(c) vs regime severity")
    axC.grid(True, alpha=0.3)
    axC.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(results_dir / "robustness_sweep.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved figure + 4 CSVs to {results_dir} (metadata: N_MC and seed columns in summary).")

    print("\n=== Scenario sweep ===")
    print(sweep_df.to_string(index=False))
    print("\n=== switching-speed axis (mean rate /day; occupancy fixed) ===")
    print(mu_df.to_string(index=False))


if __name__ == "__main__":
    main()
