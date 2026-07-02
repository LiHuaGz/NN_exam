#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4题

随机生成三组参数 (s, s_a, sigma_a, N)，模拟脉冲计数
K_a | s ~ Poisson(T f_a(s))，并用推导中的 MLE 估计 s。
统计各观测窗口 T 下的平均平方误差。

结果输出到 python/Q4_outputs。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent.parent / ".matplotlib_cache").resolve()),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


SEED = 20260627
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Q4_outputs"

TIME_WINDOWS = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
N_REPEATS = 2000
GRID_STEP_DEG = 0.05

SCENARIO_MARKERS = ["o", "s", "^"]
SCENARIO_LINESTYLES = ["-", "--", "-."]
PLOT_COLORS = {
    "blue": "#6a8caf",
    "orange": "#d08c60",
    "green": "#6b9f71",
    "red": "#c44e52",
    "band": "#d9d9d9",
    "band_edge": "#8f8f8f",
}
SCENARIO_COLORS = [PLOT_COLORS["blue"], PLOT_COLORS["orange"], PLOT_COLORS["green"]]


@dataclass(frozen=True)
class Scenario:
    label: str
    s_true: float
    s_a: np.ndarray
    sigma_a: np.ndarray

    @property
    def n_neurons(self) -> int:
        return int(self.s_a.size)


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


def tuning_rate(u: np.ndarray, s_a: np.ndarray, sigma_a: np.ndarray) -> np.ndarray:
    """计算所有 u 和神经元的 f_a(u)。"""
    return np.exp(-0.5 * ((u[:, None] - s_a[None, :]) / sigma_a[None, :]) ** 2)


def make_random_scenarios(rng: np.random.Generator, n_sets: int = 3) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for idx in range(n_sets):
        n_neurons = int(rng.integers(25, 71))
        s_true = float(rng.uniform(15.0, 165.0))
        s_a = rng.uniform(0.0, 180.0, size=n_neurons)
        sigma_a = rng.uniform(8.0, 30.0, size=n_neurons)

        order = np.argsort(s_a)
        scenarios.append(
            Scenario(
                label=f"Set {idx + 1}",
                s_true=s_true,
                s_a=s_a[order],
                sigma_a=sigma_a[order],
            )
        )
    return scenarios


def fisher_information(scenario: Scenario, time_window: float) -> float:
    f_true = np.exp(
        -0.5 * ((scenario.s_true - scenario.s_a) / scenario.sigma_a) ** 2
    )
    one_unit = np.sum(
        f_true * (scenario.s_a - scenario.s_true) ** 2 / scenario.sigma_a**4
    )
    return float(time_window * one_unit)


def mle_from_counts(
    counts: np.ndarray,
    scenario: Scenario,
    time_window: float,
    grid: np.ndarray,
    grid_rates_sum: np.ndarray,
) -> np.ndarray:
    """
    网格 MLE，并做局部抛物线插值。

    略去与 u 无关的常数：
        ell(u) = -0.5 A u^2 + B u - T sum_a f_a(u),
    其中 A=sum_a K_a/sigma_a^2，B=sum_a K_a s_a/sigma_a^2。
    """
    inv_sigma2 = 1.0 / scenario.sigma_a**2
    a_stat = counts @ inv_sigma2
    b_stat = counts @ (scenario.s_a * inv_sigma2)

    log_like = (
        -0.5 * a_stat[:, None] * grid[None, :] ** 2
        + b_stat[:, None] * grid[None, :]
        - time_window * grid_rates_sum[None, :]
    )
    max_idx = np.argmax(log_like, axis=1)
    estimates = grid[max_idx].copy()

    interior = (max_idx > 0) & (max_idx < grid.size - 1)
    rows = np.flatnonzero(interior)
    if rows.size:
        idx = max_idx[rows]
        left = log_like[rows, idx - 1]
        center = log_like[rows, idx]
        right = log_like[rows, idx + 1]
        denom = left - 2.0 * center + right
        ok = np.abs(denom) > 1e-12
        delta = np.zeros(rows.size, dtype=float)
        delta[ok] = 0.5 * (left[ok] - right[ok]) / denom[ok]
        delta = np.clip(delta, -1.0, 1.0)
        estimates[rows] = grid[idx] + delta * (grid[1] - grid[0])

    return np.clip(estimates, 0.0, 180.0)


def run_scenario(
    scenario: Scenario,
    rng: np.random.Generator,
    time_windows: np.ndarray,
    n_repeats: int,
    grid: np.ndarray,
) -> tuple[list[dict[str, float]], dict[float, np.ndarray]]:
    grid_rates = tuning_rate(grid, scenario.s_a, scenario.sigma_a)
    grid_rates_sum = np.sum(grid_rates, axis=1)
    true_rates = np.exp(
        -0.5 * ((scenario.s_true - scenario.s_a) / scenario.sigma_a) ** 2
    )

    rows: list[dict[str, float]] = []
    squared_errors_by_t: dict[float, np.ndarray] = {}

    for time_window in time_windows:
        counts = rng.poisson(time_window * true_rates, size=(n_repeats, scenario.n_neurons))
        estimates = mle_from_counts(counts, scenario, time_window, grid, grid_rates_sum)
        errors = estimates - scenario.s_true
        squared_errors = errors**2
        squared_errors_by_t[float(time_window)] = squared_errors

        info = fisher_information(scenario, float(time_window))
        rows.append(
            {
                "time_window": float(time_window),
                "mean_squared_error": float(np.mean(squared_errors)),
                "median_squared_error": float(np.median(squared_errors)),
                "bias": float(np.mean(errors)),
                "variance": float(np.var(estimates, ddof=1)),
                "crlb": float(1.0 / info) if info > 0.0 else float("inf"),
            }
        )
    return rows, squared_errors_by_t


def save_parameter_json(scenarios: list[Scenario], output_dir: Path) -> None:
    payload = []
    for scenario in scenarios:
        payload.append(
            {
                "label": scenario.label,
                "s_true": scenario.s_true,
                "N": scenario.n_neurons,
                "s_a": scenario.s_a.tolist(),
                "sigma_a": scenario.sigma_a.tolist(),
            }
        )
    with (output_dir / "scenario_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_results_csv(all_rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    fieldnames = [
        "scenario",
        "s_true",
        "N",
        "time_window",
        "mean_squared_error",
        "median_squared_error",
        "bias",
        "variance",
        "crlb",
    ]
    with (output_dir / "mse_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def plot_mse(all_rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    scenario_names = list(dict.fromkeys(str(row["scenario"]) for row in all_rows))

    for idx, name in enumerate(scenario_names):
        rows = [row for row in all_rows if row["scenario"] == name]
        t = np.array([float(row["time_window"]) for row in rows])
        mse = np.array([float(row["mean_squared_error"]) for row in rows])
        crlb = np.array([float(row["crlb"]) for row in rows])
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        marker = SCENARIO_MARKERS[idx % len(SCENARIO_MARKERS)]
        label = name.replace("Set ", "第") + "组"
        ax.loglog(
            t,
            mse,
            marker=marker,
            linestyle="-",
            color=color,
            linewidth=2.0,
            markersize=5.2,
            label=f"{label}: 经验 MSE",
        )
        ax.loglog(
            t,
            crlb,
            marker=marker,
            linestyle="--",
            color=color,
            alpha=0.85,
            linewidth=1.9,
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markevery=2,
            label=f"{label}: CRLB",
        )
    ax.set_xlim(TIME_WINDOWS[0] * 0.82, TIME_WINDOWS[-1] * 1.12)
    ax.set_xlabel(r"观测时间窗 $T$")
    ax.set_ylabel(r"均方误差 $E[(\hat{s}-s)^2]$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "mse_vs_time_window.png")
    plt.close(fig)


def plot_bias_variance(all_rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    scenario_names = list(dict.fromkeys(str(row["scenario"]) for row in all_rows))

    for idx, name in enumerate(scenario_names):
        rows = [row for row in all_rows if row["scenario"] == name]
        t = np.array([float(row["time_window"]) for row in rows])
        bias = np.array([float(row["bias"]) for row in rows])
        var = np.array([float(row["variance"]) for row in rows])
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        marker = SCENARIO_MARKERS[idx % len(SCENARIO_MARKERS)]
        linestyle = SCENARIO_LINESTYLES[idx % len(SCENARIO_LINESTYLES)]
        label = name.replace("Set ", "第") + "组"
        axes[0].semilogx(
            t,
            bias,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=2.0,
            markersize=5.2,
            label=label,
        )
        axes[1].loglog(
            t,
            var,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=2.0,
            markersize=5.2,
            label=label,
        )

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel(r"观测时间窗 $T$")
    axes[0].set_ylabel(r"数值 $E[\hat{s}-s]$")
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].set_xlabel(r"观测时间窗 $T$")
    axes[1].set_ylabel(r"$\hat{s}$ 的方差")
    axes[1].set_xlim(TIME_WINDOWS[0] * 0.82, TIME_WINDOWS[-1] * 1.42)
    axes[1].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "bias_variance_vs_time_window.png")
    plt.close(fig)


def plot_tuning_summaries(scenarios: list[Scenario], output_dir: Path) -> None:
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8.4, 7.2), sharex=True)
    u = np.linspace(0.0, 180.0, 721)
    for ax, scenario in zip(np.atleast_1d(axes), scenarios):
        rates = tuning_rate(u, scenario.s_a, scenario.sigma_a)
        mean_rate = np.mean(rates, axis=1)
        q10, q90 = np.quantile(rates, [0.10, 0.90], axis=1)
        band = ax.fill_between(u, q10, q90, color=PLOT_COLORS["band"], alpha=0.9, label="10%-90% 调谐曲线范围")
        band.set_hatch("///")
        band.set_edgecolor(PLOT_COLORS["band_edge"])
        band.set_linewidth(0.0)
        ax.plot(u, mean_rate, color=PLOT_COLORS["blue"], linewidth=2.0, linestyle="-", label="平均调谐率")
        ax.axvline(scenario.s_true, color=PLOT_COLORS["red"], linestyle="--", linewidth=2.0, label="真实 s")
        ax.scatter(
            scenario.s_a,
            np.zeros_like(scenario.s_a),
            s=12,
            color="black",
            alpha=0.55,
            label="偏好方向",
        )
        ax.set_ylabel(scenario.label.replace("Set ", "第") + "组")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, linestyle=":", linewidth=0.7)
        ax.text(
            0.01,
            0.88,
            rf"$s={scenario.s_true:.2f}^\circ$, $N={scenario.n_neurons}$",
            transform=ax.transAxes,
            fontsize=9,
        )

    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel(r"刺激方向 $u$（度）")
    fig.tight_layout()
    fig.savefig(output_dir / "random_parameter_sets.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED, help="Fixed RNG seed.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=N_REPEATS,
        help="Monte Carlo repeats per scenario and time window.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=GRID_STEP_DEG,
        help="Grid spacing in degrees for the MLE search.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for images and tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.grid_step <= 0.0:
        raise ValueError("--grid-step must be positive.")

    configure_plot_style()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    scenarios = make_random_scenarios(rng, n_sets=3)
    grid = np.arange(0.0, 180.0 + 0.5 * args.grid_step, args.grid_step)

    all_rows: list[dict[str, float | int | str]] = []
    for scenario in scenarios:
        rows, _ = run_scenario(
            scenario=scenario,
            rng=rng,
            time_windows=TIME_WINDOWS,
            n_repeats=args.repeats,
            grid=grid,
        )
        for row in rows:
            all_rows.append(
                {
                    "scenario": scenario.label,
                    "s_true": scenario.s_true,
                    "N": scenario.n_neurons,
                    **row,
                }
            )

    save_parameter_json(scenarios, output_dir)
    save_results_csv(all_rows, output_dir)
    plot_mse(all_rows, output_dir)
    plot_bias_variance(all_rows, output_dir)
    plot_tuning_summaries(scenarios, output_dir)

    print(f"Seed: {args.seed}")
    print(f"Repeats per T: {args.repeats}")
    print(f"Output directory: {output_dir}")
    for scenario in scenarios:
        print(
            f"{scenario.label}: s={scenario.s_true:.4f}, "
            f"N={scenario.n_neurons}, "
            f"mean(sigma_a)={np.mean(scenario.sigma_a):.4f}"
        )
    print("Saved: mse_vs_time_window.png")
    print("Saved: bias_variance_vs_time_window.png")
    print("Saved: random_parameter_sets.png")
    print("Saved: mse_results.csv")
    print("Saved: scenario_parameters.json")


if __name__ == "__main__":
    main()
