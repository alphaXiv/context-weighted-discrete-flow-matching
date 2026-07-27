"""Regenerate the four report figures from committed terminal-log extracts."""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/qm9/data"
IMAGES = ROOT / "reports/qm9/images"
IMAGES.mkdir(parents=True, exist_ok=True)
COLORS = {"euler": "#64748b", "neighbor": "#0f766e", "entropy": "#d97706"}
NFES = [8, 16, 32, 64, 128, 256]


def read(name: str) -> list[dict]:
    with (DATA / name).open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "sce_radius" in row and row["sce_radius"] == "":
            row["sce_radius"] = None
        for key in ("seed", "sce_radius", "nfe", "valid", "novel", "count", "neighbors"):
            if key in row and row[key] not in ("", None):
                row[key] = int(float(row[key]))
        for key in ("t", "entropy", "nll"):
            if key in row and row[key] not in ("", None):
                row[key] = float(row[key])
    return rows


def mean_sd(values):
    return st.mean(values), st.stdev(values)


def finish(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sampler_figure(rows: list[dict]) -> None:
    ce = [r for r in rows if r["loss"] == "ce"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, ax in zip(("valid", "novel"), axes):
        for method in ("euler", "neighbor", "entropy"):
            means, sds = [], []
            for nfe in NFES:
                values = [r[metric] for r in ce if r["method"] == method and r["nfe"] == nfe]
                mean, sd = mean_sd(values)
                means.append(mean)
                sds.append(sd)
            ax.errorbar(
                NFES, means, yerr=sds, marker="o", capsize=3,
                color=COLORS[method], label=method.title(),
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(NFES, NFES)
        ax.set_xlabel("Function evaluations (NFE)")
        ax.set_ylabel(f"{metric.title()} molecules / 1,024")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Context weighting helps only after the very-low-NFE regime")
    finish(fig, "sampler_by_nfe.png")


def context_figure(rows: list[dict]) -> None:
    selected = [r for r in rows if r["t"] == 0.5]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for metric, color, marker in (("entropy", "#0f766e", "o"), ("nll", "#7c3aed", "s")):
        means, sds = [], []
        for neighbors in range(5):
            values = [r[metric] for r in selected if r["neighbors"] == neighbors]
            mean, sd = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        ax.errorbar(range(5), means, yerr=sds, marker=marker, capsize=3,
                    color=color, label="Entropy" if metric == "entropy" else "Negative log-likelihood")
    ax.set_xlabel("Revealed neighbors within radius 2")
    ax.set_ylabel("Token uncertainty (nats)")
    ax.set_xticks(range(5))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title("Revealed local context sharply reduces uncertainty")
    finish(fig, "context_mechanism.png")


def radius_figure(rows: list[dict]) -> None:
    selected = [r for r in rows if r["method"] == "euler" and r["nfe"] == 256]
    groups = [("CE", "ce", None), ("SCE r=1", "sce", 1), ("SCE r=2", "sce", 2), ("SCE r=3", "sce", 3)]
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for offset, metric, color in ((-width / 2, "valid", "#0f766e"), (width / 2, "novel", "#d97706")):
        means, sds = [], []
        for _, loss, radius in groups:
            values = [
                r[metric] for r in selected
                if r["loss"] == loss and r.get("sce_radius") == radius
            ]
            mean, sd = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        ax.bar(x + offset, means, width, yerr=sds, capsize=3, color=color, label=metric.title())
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylabel("Molecules / 1,024")
    ax.set_title("Scaled cross-entropy did not improve this compact model")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    finish(fig, "radius_ablation.png")


def temperature_figure(base: list[dict], robust: list[dict]) -> None:
    combined = []
    for row in base:
        if row["loss"] == "ce":
            combined.append({**row, "temperature": 1.0})
    for row in robust:
        label = row["run_label"]
        if label.startswith("ce_t08_s"):
            combined.append({**row, "temperature": 0.8})
        elif label.startswith("ce_t12_s"):
            combined.append({**row, "temperature": 1.2})
    combined = [r for r in combined if r["nfe"] == 256]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for metric, ax in zip(("valid", "novel"), axes):
        for method in ("euler", "neighbor", "entropy"):
            means, sds = [], []
            for temperature in (0.8, 1.0, 1.2):
                values = [
                    r[metric] for r in combined
                    if r["method"] == method and r["temperature"] == temperature
                ]
                mean, sd = mean_sd(values)
                means.append(mean)
                sds.append(sd)
            ax.errorbar((0.8, 1.0, 1.2), means, yerr=sds, marker="o", capsize=3,
                        color=COLORS[method], label=method.title())
        ax.set_xlabel("Sampling temperature")
        ax.set_ylabel(f"{metric.title()} / 1,024")
        ax.set_xticks((0.8, 1.0, 1.2))
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("High-NFE sampler ordering is robust to temperature")
    finish(fig, "temperature_robustness.png")


factorial = read("factorial_results.csv")
robustness = read("robustness_results.csv")
context = read("context_diagnostic.csv")
sampler_figure(factorial)
context_figure(context)
radius_figure(factorial)
temperature_figure(factorial, robustness)
