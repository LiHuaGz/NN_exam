#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 numerical verification for the parameter construction in Question 1(2).

The theory constructs parameters (g_syn, mu, sigma) so that the inter-spike
interval T approximately satisfies

    E[T] = m,    Var(T) = v.

This script fixes the random seed, generates three random target settings, uses
the small-noise formulas from the derivation to construct the parameters, and
then estimates the actual mean/variance of T by Monte Carlo simulation.

Outputs are written to python/Q1_outputs.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent.parent / ".matplotlib_cache").resolve()),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


SEED = 20260627
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Q1_outputs"

N_SCENARIOS = 3
N_TRIALS = 6000
DT = 0.0015
MAX_TIME_FACTOR = 6.0
PLOT_COLORS = {
    "blue": "#6a8caf",
    "orange": "#d08c60",
    "green": "#6b9f71",
    "red": "#c44e52",
}


@dataclass(frozen=True)
class Scenario:
    label: str
    g_L: float
    tau: float
    V_th: float
    V_r: float
    tau_ref: float
    g_syn: float
    target_mean: float
    target_var: float
    alpha: float
    beta: float
    mu: float
    sigma: float
    t0: float
    C_t0: float


def configure_plot_style() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 220


def C_value(t: float, g_L: float, tau: float) -> float:
    """C(t)=Var[Z(t)] for the unit-noise perturbation used in the derivation."""
    kappa = 1.0 / tau
    if abs(g_L - kappa) < 1e-8:
        return (
            1.0
            - math.exp(-2.0 * kappa * t)
            * (1.0 + 2.0 * kappa * t + 2.0 * kappa * kappa * t * t)
        ) / (4.0 * kappa)

    return (
        kappa**2
        / (g_L - kappa) ** 2
        * (
            (1.0 - math.exp(-2.0 * kappa * t)) / (2.0 * kappa)
            - 2.0 * (1.0 - math.exp(-(g_L + kappa) * t)) / (g_L + kappa)
            + (1.0 - math.exp(-2.0 * g_L * t)) / (2.0 * g_L)
        )
    )


def construct_parameters(
    *,
    label: str,
    g_L: float,
    tau: float,
    V_th: float,
    V_r: float,
    tau_ref: float,
    g_syn: float,
    target_mean: float,
    target_var: float,
) -> Scenario:
    if target_mean <= tau_ref:
        raise ValueError("target_mean must be larger than tau_ref.")
    if target_var <= 0.0:
        raise ValueError("target_var must be positive.")

    t0 = target_mean - tau_ref
    e = math.exp(-g_L * t0)
    alpha = g_L * (V_th - V_r * e) / (1.0 - e)
    slope = alpha - g_L * V_th
    C_t0 = C_value(t0, g_L, tau)
    beta = slope * math.sqrt(target_var / C_t0)

    return Scenario(
        label=label,
        g_L=g_L,
        tau=tau,
        V_th=V_th,
        V_r=V_r,
        tau_ref=tau_ref,
        g_syn=g_syn,
        target_mean=target_mean,
        target_var=target_var,
        alpha=alpha,
        beta=beta,
        mu=alpha / g_syn,
        sigma=beta / g_syn,
        t0=t0,
        C_t0=C_t0,
    )


def make_random_scenarios(rng: np.random.Generator) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for idx in range(N_SCENARIOS):
        g_L = float(rng.uniform(0.65, 1.45))
        tau = float(rng.uniform(0.25, 0.95))
        tau_ref = float(rng.uniform(0.04, 0.12))
        t0 = float(rng.uniform(0.45, 1.10))
        g_syn = float(rng.uniform(0.7, 1.8))
        target_mean = tau_ref + t0

        # The construction is a small-noise approximation, so choose modest
        # target coefficients of variation.
        cv = float(rng.uniform(0.030, 0.055))
        target_var = (cv * target_mean) ** 2

        scenarios.append(
            construct_parameters(
                label=f"第{idx + 1}组",
                g_L=g_L,
                tau=tau,
                V_th=1.0,
                V_r=0.0,
                tau_ref=tau_ref,
                g_syn=g_syn,
                target_mean=target_mean,
                target_var=target_var,
            )
        )
    return scenarios


def transition_matrices(scenario: Scenario, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Gaussian transition for X=(V,J) over one time step."""
    a = scenario.g_L
    b = 1.0 / scenario.tau
    noise_scale = scenario.beta / scenario.tau

    e_a = math.exp(-a * dt)
    e_b = math.exp(-b * dt)
    if abs(a - b) < 1e-10:
        phi12 = dt * e_a
    else:
        phi12 = (e_b - e_a) / (a - b)

    phi = np.array([[e_a, phi12], [0.0, e_b]], dtype=float)
    x_star = np.array([scenario.alpha / scenario.g_L, scenario.alpha], dtype=float)
    mean_shift = x_star - phi @ x_star

    def integral_exp(rate: float) -> float:
        return (1.0 - math.exp(-rate * dt)) / rate

    if abs(a - b) < 1e-10:
        lam = 2.0 * a
        q11 = (
            noise_scale**2
            * 2.0
            / lam**3
            * (1.0 - math.exp(-lam * dt) * (1.0 + lam * dt + 0.5 * (lam * dt) ** 2))
        )
        q12 = (
            noise_scale**2
            / lam**2
            * (1.0 - math.exp(-lam * dt) * (1.0 + lam * dt))
        )
        q22 = noise_scale**2 * integral_exp(2.0 * a)
    else:
        denom = a - b
        i_2a = integral_exp(2.0 * a)
        i_2b = integral_exp(2.0 * b)
        i_ab = integral_exp(a + b)
        q11 = noise_scale**2 / denom**2 * (i_2b - 2.0 * i_ab + i_2a)
        q12 = noise_scale**2 / denom * (i_2b - i_ab)
        q22 = noise_scale**2 * i_2b

    q = np.array([[q11, q12], [q12, q22]], dtype=float)
    q = 0.5 * (q + q.T)
    chol = np.linalg.cholesky(q + 1e-14 * np.eye(2))
    return phi, mean_shift, chol


def simulate_intervals(
    scenario: Scenario,
    rng: np.random.Generator,
    n_trials: int = N_TRIALS,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi, mean_shift, chol = transition_matrices(scenario, dt)
    max_time = MAX_TIME_FACTOR * scenario.target_mean
    max_steps = int(math.ceil(max_time / dt))

    v = np.full(n_trials, scenario.V_r, dtype=float)
    j = np.full(n_trials, scenario.alpha, dtype=float)
    active = np.ones(n_trials, dtype=bool)
    theta = np.full(n_trials, np.nan, dtype=float)

    for step in range(1, max_steps + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        old_v = v[idx].copy()
        x = np.column_stack((v[idx], j[idx]))
        noise = rng.standard_normal((idx.size, 2)) @ chol.T
        x_next = x @ phi.T + mean_shift + noise
        v[idx] = x_next[:, 0]
        j[idx] = x_next[:, 1]

        crossed_local = v[idx] >= scenario.V_th
        if np.any(crossed_local):
            crossed_idx = idx[crossed_local]
            denom = v[crossed_idx] - old_v[crossed_local]
            frac = np.where(
                np.abs(denom) > 1e-12,
                (scenario.V_th - old_v[crossed_local]) / denom,
                1.0,
            )
            frac = np.clip(frac, 0.0, 1.0)
            theta[crossed_idx] = (step - 1 + frac) * dt
            active[crossed_idx] = False

    if np.any(active):
        raise RuntimeError(
            f"{scenario.label}: {np.sum(active)} trials did not hit the threshold. "
            "Increase MAX_TIME_FACTOR or adjust random targets."
        )

    intervals = scenario.tau_ref + theta
    return intervals, theta, j


def save_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_mean_var(summary_rows: list[dict[str, float | str]], output_dir: Path) -> None:
    labels = [str(row["label"]) for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.34
    target_mean = np.array([float(row["target_mean"]) for row in summary_rows])
    sim_mean = np.array([float(row["sim_mean"]) for row in summary_rows])
    target_var = np.array([float(row["target_var"]) for row in summary_rows])
    sim_var = np.array([float(row["sim_var"]) for row in summary_rows])
    colors = {"target": PLOT_COLORS["blue"], "simulated": PLOT_COLORS["orange"]}

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].bar(
        x - width / 2,
        target_mean,
        width,
        label="目标值",
        color=colors["target"],
        hatch="//",
        edgecolor="black",
    )
    axes[0].bar(
        x + width / 2,
        sim_mean,
        width,
        label="模拟值",
        color=colors["simulated"],
        edgecolor="black",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel(r"$\mathbb{E}[T]$")
    axes[0].grid(True, axis="y", linestyle=":", linewidth=0.7)
    axes[0].legend()

    axes[1].bar(
        x - width / 2,
        target_var,
        width,
        label="目标值",
        color=colors["target"],
        hatch="//",
        edgecolor="black",
    )
    axes[1].bar(
        x + width / 2,
        sim_var,
        width,
        label="模拟值",
        color=colors["simulated"],
        edgecolor="black",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel(r"$\operatorname{Var}(T)$")
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.7)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "target_vs_simulated.png")
    plt.close(fig)


def plot_relative_errors(summary_rows: list[dict[str, float | str]], output_dir: Path) -> None:
    labels = [str(row["label"]) for row in summary_rows]
    x = np.arange(len(labels))
    mean_err = 100.0 * np.array([float(row["mean_relative_error"]) for row in summary_rows])
    var_err = 100.0 * np.array([float(row["var_relative_error"]) for row in summary_rows])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(
        x,
        mean_err,
        marker="o",
        linestyle="-",
        color=PLOT_COLORS["blue"],
        label=r"$\mathbb{E}[T]$ 相对误差",
    )
    ax.plot(
        x,
        var_err,
        marker="s",
        linestyle="--",
        color=PLOT_COLORS["orange"],
        label=r"$\operatorname{Var}(T)$ 相对误差",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("相对误差（%）")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "relative_errors.png")
    plt.close(fig)


def plot_histograms(
    scenarios: list[Scenario],
    intervals_by_label: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8.4, 7.0))
    for ax, scenario in zip(np.atleast_1d(axes), scenarios):
        intervals = intervals_by_label[scenario.label]
        ax.hist(intervals, bins=45, density=True, color=PLOT_COLORS["blue"], alpha=0.78, edgecolor="white")
        ax.axvline(
            scenario.target_mean,
            color=PLOT_COLORS["red"],
            linestyle="--",
            linewidth=2.0,
            label="目标均值",
        )
        ax.axvline(np.mean(intervals), color="black", linestyle="-", linewidth=1.6, label="模拟均值")
        ax.set_ylabel(scenario.label)
        ax.grid(True, linestyle=":", linewidth=0.7)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("脉冲间隔 T")
    fig.tight_layout()
    fig.savefig(output_dir / "interval_histograms.png")
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    scenarios = make_random_scenarios(rng)
    summary_rows: list[dict[str, float | str]] = []
    parameter_rows: list[dict[str, float | str]] = []
    intervals_by_label: dict[str, np.ndarray] = {}

    for scenario in scenarios:
        intervals, _, _ = simulate_intervals(scenario, rng)
        intervals_by_label[scenario.label] = intervals
        sim_mean = float(np.mean(intervals))
        sim_var = float(np.var(intervals, ddof=1))

        summary_rows.append(
            {
                "label": scenario.label,
                "target_mean": scenario.target_mean,
                "sim_mean": sim_mean,
                "mean_relative_error": (sim_mean - scenario.target_mean) / scenario.target_mean,
                "target_var": scenario.target_var,
                "sim_var": sim_var,
                "var_relative_error": (sim_var - scenario.target_var) / scenario.target_var,
            }
        )
        parameter_rows.append(
            {
                "label": scenario.label,
                "g_L": scenario.g_L,
                "tau": scenario.tau,
                "V_th": scenario.V_th,
                "V_r": scenario.V_r,
                "tau_ref": scenario.tau_ref,
                "g_syn": scenario.g_syn,
                "target_mean_m": scenario.target_mean,
                "target_var_v": scenario.target_var,
                "t0": scenario.t0,
                "alpha": scenario.alpha,
                "beta": scenario.beta,
                "mu": scenario.mu,
                "sigma": scenario.sigma,
                "C_t0": scenario.C_t0,
            }
        )

    save_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    save_csv(OUTPUT_DIR / "parameters.csv", parameter_rows)
    plot_mean_var(summary_rows, OUTPUT_DIR)
    plot_relative_errors(summary_rows, OUTPUT_DIR)
    plot_histograms(scenarios, intervals_by_label, OUTPUT_DIR)

    print(f"Seed: {SEED}")
    print(f"Trials per scenario: {N_TRIALS}")
    print(f"Output directory: {OUTPUT_DIR}")
    for row in summary_rows:
        print(
            f"{row['label']}: "
            f"target mean={float(row['target_mean']):.6f}, "
            f"sim mean={float(row['sim_mean']):.6f}, "
            f"target var={float(row['target_var']):.6f}, "
            f"sim var={float(row['sim_var']):.6f}"
        )
    print("Saved: target_vs_simulated.png")
    print("Saved: relative_errors.png")
    print("Saved: interval_histograms.png")
    print("Saved: summary.csv")
    print("Saved: parameters.csv")


if __name__ == "__main__":
    main()
