#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5题：BSS 混合音频盲源分离。

附件只有混合信号，无干净源信号，因此使用无参考指标：
相关性、归一化互信息、时间协方差非对角项越低越好；
绝对超额峰度通常越高越好。

比较 FastICA 和 AMUSE，结果输出到 python/Q5_outputs。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


EPS = 1e-12

METHOD_LABELS = {
    "Mixtures": "混合信号",
    "FastICA": "FastICA",
    "AMUSE": "AMUSE",
}

METRIC_LABELS = {
    "method": "方法",
    "mean_abs_corr": "平均绝对相关系数",
    "mean_pairwise_nmi": "成对归一化互信息",
    "temporal_offdiag": "时间协方差非对角项",
    "mean_abs_excess_kurtosis": "平均绝对超额峰度",
}

EXTRA_LABELS = {
    "iterations": "迭代次数",
    "convergence_delta": "收敛误差",
    "selected_lag": "选定延迟",
}

PLOT_COLORS = {
    "blue": "#6a8caf",
    "orange": "#d08c60",
    "green": "#6b9f71",
    "red": "#c44e52",
}
METHOD_COLORS = [PLOT_COLORS["blue"], PLOT_COLORS["orange"], PLOT_COLORS["green"]]
METHOD_HATCHES = ["//", "", "\\\\"]


def display_method(name: str) -> str:
    return METHOD_LABELS.get(name, name)


@dataclass
class AudioData:
    sample_rate: int
    mixtures: np.ndarray
    names: list[str]


@dataclass
class SeparationResult:
    name: str
    sources: np.ndarray
    extra: dict[str, float]


def configure_chinese_font() -> None:
    """设置中文字体。"""
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


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    """读取 PCM WAV，返回单声道浮点采样。"""
    with wave.open(str(path), "rb") as wav:
        nchannels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        nframes = wav.getnframes()
        frames = wav.readframes(nframes)

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        data_i32 = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        data_i32[data_i32 >= 2**23] -= 2**24
        data = data_i32.astype(np.float64) / float(2**23)
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float64) / float(2**31)
    else:
        raise ValueError(f"Unsupported sample width {sample_width} in {path}")

    if nchannels > 1:
        data = data.reshape(-1, nchannels).mean(axis=1)
    return sample_rate, data


def write_wav_mono(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    """写入峰值归一化的 16 位单声道 WAV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean()
    peak = np.max(np.abs(x))
    if peak > EPS:
        x = 0.98 * x / peak
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def load_bss_audio(bss_dir: Path, max_seconds: float | None = None) -> AudioData:
    wav_paths = sorted(bss_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No wav files found in {bss_dir}")

    sample_rates: list[int] = []
    signals: list[np.ndarray] = []
    for path in wav_paths:
        sample_rate, data = read_wav_mono(path)
        sample_rates.append(sample_rate)
        signals.append(data)

    if len(set(sample_rates)) != 1:
        raise ValueError(f"Sample rates differ: {sample_rates}")

    sample_rate = sample_rates[0]
    min_len = min(len(x) for x in signals)
    if max_seconds is not None:
        min_len = min(min_len, int(round(max_seconds * sample_rate)))
    if min_len <= 0:
        raise ValueError("No audio samples are available after trimming.")

    mixtures = np.vstack([x[:min_len] for x in signals])
    return AudioData(sample_rate=sample_rate, mixtures=mixtures, names=[p.name for p in wav_paths])


def center_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=1, keepdims=True)
    return x - mean, mean


def whiten_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回白化数据 z = whitening @ (x - mean)。"""
    xc, mean = center_rows(x)
    cov = (xc @ xc.T) / xc.shape[1]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], EPS)
    eigvecs = eigvecs[:, order]
    whitening = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    z = whitening @ xc
    return z, whitening, mean


def symmetric_decorrelation(w: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(w, full_matrices=False)
    return u @ vt


def fastica(
    x: np.ndarray,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> SeparationResult:
    """使用 tanh 对比函数的对称 FastICA。"""
    z, _, _ = whiten_rows(x)
    n_components, n_samples = z.shape
    rng = np.random.default_rng(seed)
    w = symmetric_decorrelation(rng.normal(size=(n_components, n_components)))

    last_delta = math.inf
    for iteration in range(1, max_iter + 1):
        y = w @ z
        g = np.tanh(y)
        g_prime_mean = (1.0 - g * g).mean(axis=1)
        w_next = (g @ z.T) / n_samples - g_prime_mean[:, None] * w
        w_next = symmetric_decorrelation(w_next)
        last_delta = float(np.max(np.abs(np.abs(np.diag(w_next @ w.T)) - 1.0)))
        w = w_next
        if last_delta < tol:
            break

    sources = fix_source_scale_and_order(w @ z)
    return SeparationResult(
        name="FastICA",
        sources=sources,
        extra={"iterations": float(iteration), "convergence_delta": last_delta},
    )


def delayed_covariance(z: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0 or lag >= z.shape[1]:
        raise ValueError(f"Invalid lag {lag} for {z.shape[1]} samples.")
    cov = (z[:, lag:] @ z[:, :-lag].T) / (z.shape[1] - lag)
    return 0.5 * (cov + cov.T)


def amuse(x: np.ndarray, sample_rate: int) -> SeparationResult:
    """
    AMUSE 通过延迟协方差对角化分离白化观测。
    用时间协方差对角性选择延迟。
    """
    z, _, _ = whiten_rows(x)
    n_samples = z.shape[1]
    candidates = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        int(round(0.005 * sample_rate)),
        int(round(0.010 * sample_rate)),
        int(round(0.020 * sample_rate)),
        int(round(0.040 * sample_rate)),
    ]
    lags = sorted({lag for lag in candidates if 0 < lag < n_samples // 2})
    if not lags:
        lags = [1]

    eval_lags = [lag for lag in [1, 2, 4, 8, 16, 32, 64, 128] if lag < n_samples // 2]
    if not eval_lags:
        eval_lags = [1]

    best_sources: np.ndarray | None = None
    best_score = math.inf
    best_lag = lags[0]
    for lag in lags:
        cov = delayed_covariance(z, lag)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(np.abs(eigvals))[::-1]
        rotation = eigvecs[:, order].T
        sources = fix_source_scale_and_order(rotation @ z)
        score = temporal_offdiag_score(sources, eval_lags)
        if score < best_score:
            best_score = score
            best_sources = sources
            best_lag = lag

    assert best_sources is not None
    return SeparationResult(name="AMUSE", sources=best_sources, extra={"selected_lag": float(best_lag)})


def standardize_rows(x: np.ndarray) -> np.ndarray:
    xc, _ = center_rows(x)
    return xc / (xc.std(axis=1, keepdims=True) + EPS)


def fix_source_scale_and_order(sources: np.ndarray) -> np.ndarray:
    """固定 BSS 输出的符号和顺序。"""
    s = standardize_rows(sources)
    for idx in range(s.shape[0]):
        if abs(float(s[idx].min())) > abs(float(s[idx].max())):
            s[idx] *= -1.0
    kurt = np.mean(s**4, axis=1) - 3.0
    order = np.argsort(np.abs(kurt))[::-1]
    return s[order]


def correlation_offdiag_score(x: np.ndarray) -> float:
    corr = np.corrcoef(standardize_rows(x))
    offdiag = corr - np.diag(np.diag(corr))
    return float(np.mean(np.abs(offdiag)))


def normalized_mutual_information(x: np.ndarray, bins: int = 64, max_samples: int = 40000) -> float:
    s = standardize_rows(x)
    if s.shape[1] > max_samples:
        idx = np.linspace(0, s.shape[1] - 1, max_samples).astype(int)
        s = s[:, idx]
    s = np.clip(s, -5.0, 5.0)

    values: list[float] = []
    for i in range(s.shape[0]):
        for j in range(i + 1, s.shape[0]):
            hist, _, _ = np.histogram2d(s[i], s[j], bins=bins)
            pxy = hist / (hist.sum() + EPS)
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            nz = pxy > 0.0
            mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :] + EPS)[nz])))
            hx = -float(np.sum(px[px > 0.0] * np.log(px[px > 0.0])))
            hy = -float(np.sum(py[py > 0.0] * np.log(py[py > 0.0])))
            values.append(mi / (math.sqrt(hx * hy) + EPS))
    return float(np.mean(values)) if values else 0.0


def temporal_offdiag_score(x: np.ndarray, lags: Iterable[int]) -> float:
    s = standardize_rows(x)
    scores: list[float] = []
    for lag in lags:
        if lag <= 0 or lag >= s.shape[1]:
            continue
        cov = (s[:, lag:] @ s[:, :-lag].T) / (s.shape[1] - lag)
        offdiag = cov - np.diag(np.diag(cov))
        scores.append(float(np.mean(np.abs(offdiag))))
    return float(np.mean(scores)) if scores else 0.0


def mean_abs_excess_kurtosis(x: np.ndarray) -> float:
    s = standardize_rows(x)
    kurt = np.mean(s**4, axis=1) - 3.0
    return float(np.mean(np.abs(kurt)))


def evaluate_signal_set(name: str, x: np.ndarray, sample_rate: int) -> dict[str, float | str]:
    n_samples = x.shape[1]
    lags = [
        lag
        for lag in [1, 2, 4, 8, 16, 32, 64, 128, int(round(0.01 * sample_rate))]
        if 0 < lag < n_samples
    ]
    return {
        "method": name,
        "mean_abs_corr": correlation_offdiag_score(x),
        "mean_pairwise_nmi": normalized_mutual_information(x),
        "temporal_offdiag": temporal_offdiag_score(x, lags),
        "mean_abs_excess_kurtosis": mean_abs_excess_kurtosis(x),
    }


def save_sources(result: SeparationResult, sample_rate: int, out_dir: Path) -> None:
    method_dir = out_dir / result.name.lower()
    for idx, source in enumerate(result.sources, start=1):
        write_wav_mono(method_dir / f"{result.name.lower()}_source{idx}.wav", sample_rate, source)


def save_metrics_csv(metrics: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "method",
        "mean_abs_corr",
        "mean_pairwise_nmi",
        "temporal_offdiag",
        "mean_abs_excess_kurtosis",
    ]
    fieldnames = [METRIC_LABELS[key] for key in keys]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    METRIC_LABELS["method"]: display_method(str(row["method"])),
                    METRIC_LABELS["mean_abs_corr"]: row["mean_abs_corr"],
                    METRIC_LABELS["mean_pairwise_nmi"]: row["mean_pairwise_nmi"],
                    METRIC_LABELS["temporal_offdiag"]: row["temporal_offdiag"],
                    METRIC_LABELS["mean_abs_excess_kurtosis"]: row["mean_abs_excess_kurtosis"],
                }
            )


def plot_waveforms(audio: AudioData, results: list[SeparationResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = audio.sample_rate
    max_points = min(audio.mixtures.shape[1], int(round(1.0 * sample_rate)))
    t = np.arange(max_points) / sample_rate

    rows = 1 + len(results)
    fig, axes = plt.subplots(rows, audio.mixtures.shape[0], figsize=(13, 2.7 * rows), sharex=True)
    axes = np.atleast_2d(axes)

    for ch in range(audio.mixtures.shape[0]):
        axes[0, ch].plot(t, audio.mixtures[ch, :max_points], color=METHOD_COLORS[0], linewidth=0.8)
        axes[0, ch].set_title(f"混合信号 {ch + 1}")
        axes[0, ch].grid(alpha=0.25)

    for row, result in enumerate(results, start=1):
        for ch in range(result.sources.shape[0]):
            axes[row, ch].plot(
                t,
                result.sources[ch, :max_points],
                color=METHOD_COLORS[row % len(METHOD_COLORS)],
                linewidth=0.8,
            )
            axes[row, ch].set_title(f"{result.name} 源信号 {ch + 1}")
            axes[row, ch].grid(alpha=0.25)

    for ax in axes[-1]:
        ax.set_xlabel("时间 (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "waveforms.png", dpi=160)
    plt.close(fig)


def plot_metric_bars(metrics: list[dict[str, float | str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [display_method(str(row["method"])) for row in metrics]
    keys = [
        "mean_abs_corr",
        "mean_pairwise_nmi",
        "temporal_offdiag",
        "mean_abs_excess_kurtosis",
    ]
    titles = [
        "平均绝对相关系数",
        "成对归一化互信息",
        "时间协方差非对角项",
        "平均绝对超额峰度",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, key, title in zip(axes.ravel(), keys, titles):
        values = [float(row[key]) for row in metrics]
        bars = ax.bar(
            methods,
            values,
            color=METHOD_COLORS[: len(methods)],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, hatch in zip(bars, METHOD_HATCHES):
            bar.set_hatch(hatch)
        ax.set_title(title, pad=10)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "quality_metrics.png", dpi=160)
    plt.close(fig)


def print_metrics(metrics: list[dict[str, float | str]], results: list[SeparationResult]) -> None:
    print("\n质量指标（附件未提供干净参考源信号，因此采用无参考指标）：")
    header = (
        f"{'方法':<10} {'平均绝对相关':>12} {'NMI':>12} "
        f"{'时间非对角项':>14} {'平均绝对峰度':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in metrics:
        print(
            f"{display_method(str(row['method'])):<10} "
            f"{float(row['mean_abs_corr']):12.6f} "
            f"{float(row['mean_pairwise_nmi']):12.6f} "
            f"{float(row['temporal_offdiag']):14.6f} "
            f"{float(row['mean_abs_excess_kurtosis']):14.6f}"
        )

    print("\n算法细节：")
    for result in results:
        detail = "，".join(f"{EXTRA_LABELS.get(k, k)}={v:.6g}" for k, v in result.extra.items())
        print(f"  {result.name}: {detail}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_bss_dir = script_dir.parent / "附件" / "附件" / "BSS"
    default_out_dir = script_dir / "Q5_outputs"

    parser = argparse.ArgumentParser(description="Blind source separation for Q5 BSS wav files.")
    parser.add_argument("--bss-dir", type=Path, default=default_bss_dir, help="Directory containing mix wav files.")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir, help="Directory for separated wavs and plots.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional duration limit for quick experiments.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for FastICA.")
    parser.add_argument("--no-plots", action="store_true", help="Skip waveform and metric plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_chinese_font()

    audio = load_bss_audio(args.bss_dir, max_seconds=args.max_seconds)
    print(f"已从 {args.bss_dir} 读取 {len(audio.names)} 路混合信号")
    print(f"文件：{', '.join(audio.names)}")
    print(f"采样率：{audio.sample_rate} Hz，每路采样点数：{audio.mixtures.shape[1]}")

    results = [
        fastica(audio.mixtures, seed=args.seed),
        amuse(audio.mixtures, sample_rate=audio.sample_rate),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        save_sources(result, audio.sample_rate, args.out_dir)

    metrics = [evaluate_signal_set("Mixtures", audio.mixtures, audio.sample_rate)]
    metrics.extend(evaluate_signal_set(result.name, result.sources, audio.sample_rate) for result in results)
    save_metrics_csv(metrics, args.out_dir / "quality_metrics.csv")

    if not args.no_plots:
        plot_waveforms(audio, results, args.out_dir)
        plot_metric_bars(metrics, args.out_dir)

    print_metrics(metrics, results)
    print(f"\n分离后的 wav 文件和报告已保存到：{args.out_dir}")


if __name__ == "__main__":
    main()
