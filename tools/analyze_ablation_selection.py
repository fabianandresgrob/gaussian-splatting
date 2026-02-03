#!/usr/bin/env python3
"""Analyze and compare selection distributions across ablation runs.

This works with the current output layout produced by run_ablation.py:
  <output_root>/<CONFIG_ID>/<SCENE_ID>/seed_<SEED>/selection_history.jsonl

It summarizes *empirical* selection distributions (from actual selections):
- coverage, normalized entropy, Gini, top-k mass, effective sample size
- histogram of selection counts

Usage:
  python tools/analyze_ablation_selection.py \
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


def _load_split_summary(run_dir: str) -> Optional[dict]:
    path = os.path.join(run_dir, "split_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_num_cameras_from_log(df: pd.DataFrame) -> Optional[int]:
    # Camera uids are assigned as 0..N-1 when the scene is loaded.
    if "cam_uid" not in df.columns or df.empty:
        return None
    try:
        m = int(df["cam_uid"].max())
    except Exception:
        return None
    return (m + 1) if m >= 0 else None


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


def _load_metrics_history(run_dir: str) -> Optional[pd.DataFrame]:
    """Load metrics_history.json as a flat DataFrame.

    Expected format: list of dicts with keys: iteration, elapsed_time, train{...}, test{...}
    """
    path = os.path.join(run_dir, "metrics_history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception:
        return None
    if not isinstance(raw, list) or not raw:
        return None

    rows: List[dict] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        it = rec.get("iteration", None)
        if it is None:
            continue
        row: Dict[str, object] = {
            "iteration": int(it),
            "elapsed_time": float(rec.get("elapsed_time", np.nan)),
        }
        for split in ("train", "test"):
            d = rec.get(split, {})
            if isinstance(d, dict):
                for k, v in d.items():
                    # e.g. train_PSNR, test_PSNR
                    row[f"{split}_{k}"] = float(v)
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("iteration")
    return df


def plot_psnr_curves(df_metrics: pd.DataFrame, out_path: str, which: str) -> None:
    """Overlay PSNR curves across configs.

    which: 'test' or 'train'
    """
    if df_metrics.empty:
        return
    col = f"{which}_PSNR"
    if col not in df_metrics.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for cfg in df_metrics["config"].unique():
        df_c = df_metrics[df_metrics["config"] == cfg].sort_values("iteration")
        if df_c.empty:
            continue
        ax.plot(
            df_c["iteration"].to_numpy(),
            df_c[col].to_numpy(dtype=float),
            label=str(cfg),
            linewidth=1.8,
        )
    ax.set_title(f"{which.capitalize()} PSNR over time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


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

    # Probability diagnostics (selected probability each iteration)
    min_selected_prob: Optional[float]
    max_selected_prob: Optional[float]
    p05_selected_prob: Optional[float]
    p50_selected_prob: Optional[float]
    p95_selected_prob: Optional[float]
    uniform_prob: Optional[float]
    prob_eff_num_cameras: Optional[float]
    prob_eff_fraction_of_total: Optional[float]
    frac_selected_prob_gt_2x_uniform: Optional[float]
    frac_selected_prob_gt_5x_uniform: Optional[float]


def summarize_run(selection_jsonl: str, config: str, scene: str, seed: int) -> RunSummary:
    df = _read_jsonl(selection_jsonl)

    run_dir = os.path.dirname(selection_jsonl)
    split_summary = _load_split_summary(run_dir)
    total_cams = None
    if split_summary and isinstance(split_summary, dict):
        # split_summary is written by train_gsplat.py (if available)
        total_cams = split_summary.get("num_train", None)

    if total_cams is None:
        total_cams = _infer_num_cameras_from_log(df)
    total_cams = int(total_cams) if total_cams is not None else None
    uniform_prob = (1.0 / total_cams) if (total_cams and total_cams > 0) else None

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

    # Selected-probability distribution stats
    if "probability" in df.columns:
        probs = df["probability"].astype(float).to_numpy()
        min_p = float(np.min(probs)) if probs.size else None
        max_p = float(np.max(probs)) if probs.size else None
        p05 = float(np.quantile(probs, 0.05)) if probs.size else None
        p50 = float(np.quantile(probs, 0.50)) if probs.size else None
        p95 = float(np.quantile(probs, 0.95)) if probs.size else None

        # If probabilities sum to 1, E[p_selected] = sum_i p_i^2. Then 1/E[p_selected]
        # is the inverse-Simpson effective number of cameras (closer to N => near-uniform).
        prob_eff_n = float(1.0 / mean_prob) if (mean_prob is not None and mean_prob > 0) else None
        prob_eff_frac = float(prob_eff_n / total_cams) if (prob_eff_n is not None and total_cams) else None

        frac_2x = None
        frac_5x = None
        if uniform_prob is not None and uniform_prob > 0 and probs.size:
            frac_2x = float(np.mean(probs > 2.0 * uniform_prob))
            frac_5x = float(np.mean(probs > 5.0 * uniform_prob))
    else:
        min_p = max_p = p05 = p50 = p95 = None
        prob_eff_n = prob_eff_frac = None
        frac_2x = frac_5x = None

    return RunSummary(
        config=config,
        scene=scene,
        seed=seed,
        n_steps=n_steps,
        n_unique_selected=n_unique,
        coverage_vs_selected_set=(float(n_unique) / float(total_cams)) if (total_cams and total_cams > 0) else 1.0,
        gini=gini,
        entropy=ent["entropy"],
        normalized_entropy=ent["normalized_entropy"],
        ess=float(ess),
        top1_mass=_topk_mass(1),
        top5_mass=_topk_mass(5),
        top10_mass=_topk_mass(10),
        mean_selected_prob=mean_prob,
        std_selected_prob=std_prob,
        min_selected_prob=min_p,
        max_selected_prob=max_p,
        p05_selected_prob=p05,
        p50_selected_prob=p50,
        p95_selected_prob=p95,
        uniform_prob=uniform_prob,
        prob_eff_num_cameras=prob_eff_n,
        prob_eff_fraction_of_total=prob_eff_frac,
        frac_selected_prob_gt_2x_uniform=frac_2x,
        frac_selected_prob_gt_5x_uniform=frac_5x,
    )


def plot_selected_probability_distributions(df_probs: pd.DataFrame, out_path: str) -> None:
    """Histogram of selected probabilities per config."""
    if df_probs.empty:
        return
    configs = list(df_probs["config"].unique())
    n = len(configs)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, max(2.0, 2.0 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        p = df_probs[df_probs["config"] == cfg]["probability"].to_numpy(dtype=float)
        ax.hist(p, bins=50, alpha=0.9)
        ax.set_ylabel(cfg)
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xlabel("Selected probability (probability of chosen view at that iteration)")
    fig.suptitle("Selected-probability histograms per config", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_selected_probability_boxplot(df_probs: pd.DataFrame, out_path: str) -> None:
    """Boxplot of selected probabilities per config."""
    if df_probs.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * df_probs["config"].nunique()), 5))
    configs = list(df_probs["config"].unique())
    data = [
        df_probs[df_probs["config"] == c]["probability"].to_numpy(dtype=float)
        for c in configs
    ]
    ax.boxplot(data, tick_labels=configs, showfliers=False)
    ax.set_title("Selected-probability distribution per config")
    ax.set_xlabel("Config")
    ax.set_ylabel("Selected probability")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_selected_probability_over_time(df_probs: pd.DataFrame, out_path: str) -> None:
    """Rolling mean of selected probability vs iteration for each config."""
    if df_probs.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))

    for cfg in df_probs["config"].unique():
        df_c = df_probs[df_probs["config"] == cfg].sort_values("iteration")
        if df_c.empty:
            continue
        # Use a window that gives a readable plot across different run lengths.
        window = max(50, int(len(df_c) / 200))
        s = df_c["probability"].astype(float).rolling(window=window, min_periods=1).mean()
        ax.plot(df_c["iteration"].to_numpy(), s.to_numpy(), label=f"{cfg} (w={window})", linewidth=1.5)

    ax.set_title("Selected probability over time (rolling mean)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Selected probability")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_effective_num_cameras_over_time(df_probs: pd.DataFrame, out_path: str) -> None:
    """Plot rolling effective number of cameras over time.

    For a probability distribution p over cameras, if we sample from p then
    E[p_selected] = sum_i p_i^2.
    Thus effN = 1 / E[p_selected] is the inverse-Simpson effective number.

    We don't log the full p_i vector per iteration, but we do log p_selected.
    Using effN_hat(t) = 1 / p_selected(t) and smoothing over time is a useful
    diagnostic: uniform sampling gives ~N, peaky sampling gives smaller values.
    """
    if df_probs.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for cfg in df_probs["config"].unique():
        df_c = df_probs[df_probs["config"] == cfg].sort_values("iteration")
        if df_c.empty:
            continue
        p = df_c["probability"].astype(float)
        inv_p = (1.0 / (p + 1e-12)).clip(upper=1e9)
        window = max(50, int(len(df_c) / 200))
        s = inv_p.rolling(window=window, min_periods=1).mean()
        ax.plot(df_c["iteration"].to_numpy(), s.to_numpy(), label=f"{cfg} (w={window})", linewidth=1.5)

    ax.set_title("Effective #cameras over time (rolling mean of 1/p_selected)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Effective #cameras (proxy)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_effective_fraction_over_time(df_probs: pd.DataFrame, out_path: str, n_by_config: Dict[str, int]) -> None:
    """Plot rolling effective fraction effN/N over time.

    effN is estimated consistently with the summary metric:
      effN ~= 1 / mean(p_selected)

    To keep this interpretable as a fraction in [0, 1], we compute a rolling mean
    of p_selected first, then invert:
      effN_window = 1 / rolling_mean(p_selected)
      frac_window = effN_window / N

    IMPORTANT: Do not try to infer N from min(p_selected) for non-uniform
    selectors: low-probability tails can be far smaller than 1/N.
    """
    if df_probs.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for cfg in df_probs["config"].unique():
        df_c = df_probs[df_probs["config"] == cfg].sort_values("iteration")
        if df_c.empty:
            continue

        n = n_by_config.get(str(cfg))
        if not n or n <= 0:
            continue

        p = df_c["probability"].astype(float)
        window = max(50, int(len(df_c) / 200))
        mean_p = p.rolling(window=window, min_periods=1).mean()
        eff_n = (1.0 / (mean_p + 1e-12)).clip(upper=1e9)
        frac = eff_n / float(n)
        ax.plot(
            df_c["iteration"].to_numpy(),
            frac.to_numpy(),
            label=f"{cfg} (N={n}, w={window})",
            linewidth=1.5,
        )

    ax.set_title("Effective fraction over time (effN/N; effN=1/rolling_mean(p_selected))")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Effective fraction of cameras")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_count_histogram(df_counts: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Boxplot per config for selection counts
    configs = list(df_counts["config"].unique())
    data = [df_counts[df_counts["config"] == c]["count"].values for c in configs]
    ax.boxplot(data, tick_labels=configs, showfliers=False)
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
    prob_rows: List[dict] = []
    metric_rows: List[dict] = []
    metric_summ_rows: List[dict] = []

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

        if "probability" in df.columns:
            for it, p in zip(df["iteration"].to_numpy(), df["probability"].to_numpy()):
                prob_rows.append({"config": config, "iteration": int(it), "probability": float(p)})

        # Metrics history (PSNR curves)
        df_m = _load_metrics_history(run_dir)
        if df_m is not None and not df_m.empty:
            df_m = df_m.copy()
            df_m["config"] = config
            df_m["scene"] = args.scene
            df_m["seed"] = int(args.seed)
            metric_rows.extend(df_m.to_dict(orient="records"))

            # Per-run summary
            def _summ(col: str) -> Dict[str, float]:
                if col not in df_m.columns:
                    return {"final": np.nan, "best": np.nan, "auc": np.nan}
                y = df_m[col].to_numpy(dtype=float)
                x = df_m["iteration"].to_numpy(dtype=float)
                if y.size == 0:
                    return {"final": np.nan, "best": np.nan, "auc": np.nan}
                final = float(y[-1])
                best = float(np.nanmax(y))
                auc = float(np.trapz(y, x)) if x.size == y.size and x.size >= 2 else np.nan
                return {"final": final, "best": best, "auc": auc}

            test = _summ("test_PSNR")
            train = _summ("train_PSNR")
            metric_summ_rows.append(
                {
                    "config": config,
                    "scene": args.scene,
                    "seed": int(args.seed),
                    "n_eval_points": int(len(df_m)),
                    "test_psnr_final": test["final"],
                    "test_psnr_best": test["best"],
                    "test_psnr_auc": test["auc"],
                    "train_psnr_final": train["final"],
                    "train_psnr_best": train["best"],
                    "train_psnr_auc": train["auc"],
                }
            )

        print(
            f"{config}: steps={s.n_steps} unique={s.n_unique_selected} "
            f"gini={s.gini:.3f} Hn={s.normalized_entropy:.3f} ESS={s.ess:.1f} "
            f"top1={s.top1_mass:.3f} top10={s.top10_mass:.3f} "
            + (f"mean_p(sel)={s.mean_selected_prob:.4g}" if s.mean_selected_prob is not None else "")
            + (f" effN~{s.prob_eff_num_cameras:.1f}" if s.prob_eff_num_cameras is not None else "")
            + (f" /N={s.prob_eff_fraction_of_total:.3f}" if s.prob_eff_fraction_of_total is not None else "")
        )

    if not summaries:
        raise SystemExit("No runs found. Check --output_root/--scene/--seed/--configs")

    # Save summary table
    df_sum = pd.DataFrame([s.__dict__ for s in summaries]).sort_values("config")
    df_sum.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump([s.__dict__ for s in summaries], f, indent=2)

    # N per config (prefer split_summary-derived value via uniform_prob=1/N).
    n_by_config: Dict[str, int] = {}
    for _, row in df_sum.iterrows():
        cfg = str(row.get("config"))
        up = row.get("uniform_prob")
        try:
            up_f = float(up)
        except Exception:
            continue
        if up_f > 0:
            n_by_config[cfg] = int(round(1.0 / up_f))

    # Save count distribution boxplot
    df_counts = pd.DataFrame(count_rows)
    if not df_counts.empty:
        plot_count_histogram(df_counts, os.path.join(out_dir, "selection_count_boxplot.png"))

    # Probability analysis plots (selected probability at each iteration)
    df_probs = pd.DataFrame(prob_rows)
    if not df_probs.empty:
        plot_selected_probability_distributions(df_probs, os.path.join(out_dir, "selected_probability_hist.png"))
        plot_selected_probability_boxplot(df_probs, os.path.join(out_dir, "selected_probability_boxplot.png"))
        plot_selected_probability_over_time(df_probs, os.path.join(out_dir, "selected_probability_over_time.png"))
        plot_effective_num_cameras_over_time(df_probs, os.path.join(out_dir, "effective_num_cameras_over_time.png"))
        plot_effective_fraction_over_time(df_probs, os.path.join(out_dir, "effective_fraction_over_time.png"), n_by_config)
        df_probs.to_csv(os.path.join(out_dir, "selected_probability_timeseries.csv"), index=False)

    # Metrics history plots + summary
    df_metrics = pd.DataFrame(metric_rows)
    if not df_metrics.empty:
        df_metrics.to_csv(os.path.join(out_dir, "metrics_timeseries.csv"), index=False)
        plot_psnr_curves(df_metrics, os.path.join(out_dir, "test_psnr_over_time.png"), which="test")
        plot_psnr_curves(df_metrics, os.path.join(out_dir, "train_psnr_over_time.png"), which="train")

    df_metric_summ = pd.DataFrame(metric_summ_rows)
    if not df_metric_summ.empty:
        df_metric_summ = df_metric_summ.sort_values("config")
        df_metric_summ.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
        with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
            json.dump(df_metric_summ.to_dict(orient="records"), f, indent=2)

    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
