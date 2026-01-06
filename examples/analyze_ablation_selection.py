#!/usr/bin/env python3
"""Analyze and compare selection distributions across ablation runs.

This works with the current output layout produced by run_ablation.py:
  <output_root>/<CONFIG_ID>/<SCENE_ID>/seed_<SEED>/selection_history.jsonl

It summarizes *empirical* selection distributions (from actual selections):
- coverage, normalized entropy, Gini, top-k mass, effective sample size
- histogram of selection counts

Usage:
  python examples/analyze_ablation_selection.py \
    --output_root output \
    --scene 5371eff4f9 \
    --seed 0 \
    --configs B1 B2 S1 S2 S3 C1 CL

Optional:
  --out_dir output/analysis_selection/5371eff4f9_seed0

Notes:
- selection_history.jsonl only logs the selected camera each step (and its
  probability at the moment of selection). We do NOT have the full probability
  vector per iteration, so we analyze the realized sampling distribution.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_jsonl(path: str) -> pd.DataFrame:
    rows: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    df = pd.DataFrame(rows)
    required = {"iteration", "cam_uid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    return df


def _gini_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    if counts.size == 0:
        return 0.0
    if np.all(counts == 0):
        return 0.0
    counts_sorted = np.sort(counts)
    n = counts_sorted.size
    cum = np.cumsum(counts_sorted)
    return float((2.0 * np.sum((np.arange(1, n + 1)) * counts_sorted)) / (n * cum[-1]) - (n + 1) / n)


def _entropy_from_counts(counts: np.ndarray) -> Dict[str, float]:
    counts = counts.astype(np.float64)
    s = counts.sum()
    if s <= 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0}
    p = counts / s
    h = float(-(p * np.log2(p + 1e-12)).sum())
    h_max = float(np.log2(len(p))) if len(p) > 0 else 0.0
    h_norm = float(h / h_max) if h_max > 0 else 0.0
    return {"entropy": h, "normalized_entropy": h_norm}


def _effective_sample_size(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    denom = float((p * p).sum())
    return float(1.0 / denom) if denom > 0 else 0.0


@dataclass
class RunSummary:
    config: str
    scene: str
    seed: int
    n_steps: int
    n_unique_selected: int
    coverage_vs_selected_set: float
    gini: float
    entropy: float
    normalized_entropy: float
    ess: float
    top1_mass: float
    top5_mass: float
    top10_mass: float
    mean_selected_prob: Optional[float]
    std_selected_prob: Optional[float]


def summarize_run(selection_jsonl: str, config: str, scene: str, seed: int) -> RunSummary:
    df = _read_jsonl(selection_jsonl)

    counts = df["cam_uid"].value_counts().values
    n_steps = int(len(df))
    n_unique = int(df["cam_uid"].nunique())

    gini = _gini_from_counts(counts)
    ent = _entropy_from_counts(counts)
    ess = _effective_sample_size(counts)

    p = counts / counts.sum()
    p_sorted = np.sort(p)[::-1]

    def _topk_mass(k: int) -> float:
        return float(p_sorted[:k].sum()) if p_sorted.size else 0.0

    mean_prob = float(df["probability"].mean()) if "probability" in df.columns else None
    std_prob = float(df["probability"].std()) if "probability" in df.columns else None

    return RunSummary(
        config=config,
        scene=scene,
        seed=seed,
        n_steps=n_steps,
        n_unique_selected=n_unique,
        # We do not know the total number of cameras from selection_history.jsonl alone,
        # so we report coverage relative to the selected set.
        coverage_vs_selected_set=1.0,
        gini=gini,
        entropy=ent["entropy"],
        normalized_entropy=ent["normalized_entropy"],
        ess=float(ess),
        top1_mass=_topk_mass(1),
        top5_mass=_topk_mass(5),
        top10_mass=_topk_mass(10),
        mean_selected_prob=mean_prob,
        std_selected_prob=std_prob,
    )


def plot_count_histogram(df_counts: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Boxplot per config for selection counts
    configs = list(df_counts["config"].unique())
    data = [df_counts[df_counts["config"] == c]["count"].values for c in configs]
    ax.boxplot(data, labels=configs, showfliers=False)
    ax.set_title("Selection count distribution per camera (boxplot)")
    ax.set_xlabel("Config")
    ax.set_ylabel("Selections per camera")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.output_root, "analysis_selection", f"{args.scene}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    summaries: List[RunSummary] = []
    count_rows: List[dict] = []

    for config in args.configs:
        run_dir = os.path.join(args.output_root, config, args.scene, f"seed_{args.seed}")
        log_path = os.path.join(run_dir, "selection_history.jsonl")
        if not os.path.exists(log_path):
            print(f"[skip] missing {log_path}")
            continue

        s = summarize_run(log_path, config=config, scene=args.scene, seed=args.seed)
        summaries.append(s)

        df = _read_jsonl(log_path)
        vc = df["cam_uid"].value_counts()
        for uid, cnt in vc.items():
            count_rows.append({"config": config, "cam_uid": int(uid), "count": int(cnt)})

        print(
            f"{config}: steps={s.n_steps} unique={s.n_unique_selected} "
            f"gini={s.gini:.3f} Hn={s.normalized_entropy:.3f} ESS={s.ess:.1f} "
            f"top1={s.top1_mass:.3f} top10={s.top10_mass:.3f} "
            + (f"mean_p(sel)={s.mean_selected_prob:.4g}" if s.mean_selected_prob is not None else "")
        )

    if not summaries:
        raise SystemExit("No runs found. Check --output_root/--scene/--seed/--configs")

    # Save summary table
    df_sum = pd.DataFrame([s.__dict__ for s in summaries]).sort_values("config")
    df_sum.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump([s.__dict__ for s in summaries], f, indent=2)

    # Save count distribution boxplot
    df_counts = pd.DataFrame(count_rows)
    if not df_counts.empty:
        plot_count_histogram(df_counts, os.path.join(out_dir, "selection_count_boxplot.png"))

    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
