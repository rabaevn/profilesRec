#!/usr/bin/env python3
"""Post-hoc analysis on per_example_prof_vs_zero.parquet from slice_fusion.py.

Consumes the parquet (no GPU, no encoding) and produces:
- Aggregate R@10 / NDCG@10 with paired bootstrap 95% CIs (baseline vs oracle).
- McNemar test on Hit@10 (baseline vs oracle).
- The user-requested focus cohorts: full / cold x popular / long-hist x long-tail
  + cold x long-tail (hardest), long-hist x popular (no-regression), has_profile splits,
  gate-value tertiles.
- NDCG@10 cohort pivot alongside R@10.
- Win/loss transition table per focus cohort.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import cohort_dims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-example", type=str, required=True,
                   help="parquet from slice_fusion.py")
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-name", type=str, default=None,
                   help="If set, back-fill user_activity/target_first_ts cohort "
                        "columns from this dataset's interactions when the parquet "
                        "predates them (enables activity/age cohorts on legacy runs).")
    p.add_argument("--root", type=str, default="./raw_data")
    return p.parse_args()


def ndcg10(rank: np.ndarray) -> np.ndarray:
    r = rank.astype(np.float64)
    return np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)


def metrics(rank: np.ndarray) -> Dict[str, float]:
    hit10 = (rank <= 10).astype(np.float64)
    return {
        "n": int(len(rank)),
        "recall@10": float(hit10.mean()) if len(rank) else 0.0,
        "ndcg@10": float(ndcg10(rank).mean()) if len(rank) else 0.0,
    }


def paired_bootstrap_delta(
    a: np.ndarray, b: np.ndarray, reps: int, rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Return (mean_delta, lo95, hi95) for mean(a) - mean(b), paired by index."""
    n = len(a)
    diff = a - b
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(reps, n))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mcnemar(a_hit: np.ndarray, b_hit: np.ndarray) -> Dict[str, float]:
    """Paired McNemar on hit indicators. a vs b: b01 = b wins where a misses, b10 = a wins."""
    a = a_hit.astype(bool)
    b = b_hit.astype(bool)
    b01 = int(((~a) & b).sum())   # baseline missed, oracle hit
    b10 = int((a & (~b)).sum())   # baseline hit,    oracle missed
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "stat": 0.0, "p_approx": 1.0}
    # continuity-corrected chi^2 (1 dof); avoid scipy dep
    stat = (abs(b01 - b10) - 1) ** 2 / n
    # 1-dof chi^2 survival approx: p = exp(-stat/2)
    p = float(np.exp(-stat / 2.0))
    return {"b01": b01, "b10": b10, "stat": float(stat), "p_approx": p}


def cohort_block(
    df: pd.DataFrame, name: str, mask: np.ndarray, reps: int, rng: np.random.Generator,
) -> Dict[str, object]:
    sub = df[mask]
    n = len(sub)
    out: Dict[str, object] = {"name": name, "n": int(n)}
    if n == 0:
        return out
    rb = sub["rank_baseline"].to_numpy()
    rp = sub["rank_prof"].to_numpy()
    rz = sub["rank_zero"].to_numpy()
    out["baseline"] = metrics(rb)
    out["oracle_prof"] = metrics(rp)
    out["oracle_zero"] = metrics(rz)

    hit_b = (rb <= 10).astype(np.float64)
    hit_p = (rp <= 10).astype(np.float64)
    hit_z = (rz <= 10).astype(np.float64)
    ndcg_b = ndcg10(rb); ndcg_p = ndcg10(rp); ndcg_z = ndcg10(rz)

    d_r10_pb = paired_bootstrap_delta(hit_p, hit_b, reps, rng)
    d_n10_pb = paired_bootstrap_delta(ndcg_p, ndcg_b, reps, rng)
    d_r10_pz = paired_bootstrap_delta(hit_p, hit_z, reps, rng)
    d_n10_pz = paired_bootstrap_delta(ndcg_p, ndcg_z, reps, rng)
    out["delta_vs_baseline"] = {
        "recall@10":  {"mean": d_r10_pb[0], "ci95": [d_r10_pb[1], d_r10_pb[2]]},
        "ndcg@10":    {"mean": d_n10_pb[0], "ci95": [d_n10_pb[1], d_n10_pb[2]]},
    }
    out["delta_vs_zero"] = {
        "recall@10":  {"mean": d_r10_pz[0], "ci95": [d_r10_pz[1], d_r10_pz[2]]},
        "ndcg@10":    {"mean": d_n10_pz[0], "ci95": [d_n10_pz[1], d_n10_pz[2]]},
    }
    out["mcnemar_vs_baseline"] = mcnemar(hit_b, hit_p)
    # transition counts (baseline -> oracle+prof)
    gained = int(((hit_p > 0) & (hit_b == 0)).sum())
    lost   = int(((hit_p == 0) & (hit_b > 0)).sum())
    kept   = int(((hit_p > 0) & (hit_b > 0)).sum())
    miss   = int(((hit_p == 0) & (hit_b == 0)).sum())
    out["transitions_vs_baseline"] = {
        "gained": gained, "lost": lost, "kept": kept, "both_miss": miss,
    }
    if "gate" in sub.columns and not sub["gate"].isna().all():
        out["mean_gate"] = float(sub["gate"].mean())
    return out


def hl_bin(h: int) -> str:
    if h <= 5:  return "3-5"
    if h <= 10: return "6-10"
    if h <= 20: return "11-20"
    return "21+"


def fmt_pct(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "    -"
    return f"{x*100:6.2f}"


def fmt_delta(mean: float, lo: float, hi: float) -> str:
    sig = "*" if (lo > 0 or hi < 0) else " "
    return f"{mean*100:+6.2f} [{lo*100:+6.2f},{hi*100:+6.2f}]{sig}"


def print_cohort(c: Dict[str, object]) -> None:
    print(f"\n--- {c['name']}  (n={c['n']}) ---")
    if c["n"] == 0:
        print("  (empty)")
        return
    bl = c["baseline"]; op = c["oracle_prof"]; oz = c["oracle_zero"]
    print(f"  R@10    baseline  : {fmt_pct(bl['recall@10'])}")
    print(f"  R@10    oracle+p  : {fmt_pct(op['recall@10'])}")
    print(f"  R@10    oracle+0  : {fmt_pct(oz['recall@10'])}")
    print(f"  NDCG@10 baseline  : {fmt_pct(bl['ndcg@10'])}")
    print(f"  NDCG@10 oracle+p  : {fmt_pct(op['ndcg@10'])}")
    print(f"  NDCG@10 oracle+0  : {fmt_pct(oz['ndcg@10'])}")
    db = c["delta_vs_baseline"]
    print(f"  delta R@10    (oracle+p - baseline)  : {fmt_delta(db['recall@10']['mean'], *db['recall@10']['ci95'])}")
    print(f"  delta NDCG@10 (oracle+p - baseline)  : {fmt_delta(db['ndcg@10']['mean'],   *db['ndcg@10']['ci95'])}")
    dz = c["delta_vs_zero"]
    print(f"  delta R@10    (oracle+p - oracle+0)  : {fmt_delta(dz['recall@10']['mean'], *dz['recall@10']['ci95'])}")
    mc = c["mcnemar_vs_baseline"]
    print(f"  McNemar baseline-vs-oracle+p: b01={mc['b01']} b10={mc['b10']} "
          f"stat={mc['stat']:.2f} p~={mc['p_approx']:.3g}")
    tr = c["transitions_vs_baseline"]
    print(f"  transitions baseline -> oracle+p: "
          f"gained={tr['gained']} lost={tr['lost']} kept={tr['kept']} both_miss={tr['both_miss']}")
    if "mean_gate" in c:
        print(f"  mean gate: {c['mean_gate']:.4f}")


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.per_example)
    n_total = len(df)
    n_bl = df["rank_baseline"].notna().sum()
    print(f"Loaded {n_total:,} examples; {n_bl:,} have baseline ranks.")
    df = df.dropna(subset=["rank_baseline"]).reset_index(drop=True)

    # bins (match slice_fusion / evaluate_slices)
    df["hl"] = df["history_len"].map(hl_bin)
    pop_edges = [float(np.quantile(df["target_freq"], q)) for q in (0.2, 0.4, 0.6, 0.8)]
    def pop_bin(f: int) -> str:
        if f <= pop_edges[0]: return "Q1"
        if f <= pop_edges[1]: return "Q2"
        if f <= pop_edges[2]: return "Q3"
        if f <= pop_edges[3]: return "Q4"
        return "Q5"
    df["pop"] = df["target_freq"].map(pop_bin)
    print(f"popularity quintile edges (train freq): {pop_edges}")

    # new cohort axes: user_activity + target item age (back-fill if absent)
    have_new = {"user_activity", "target_first_ts"}.issubset(df.columns)
    if not have_new and args.dataset_name:
        print(f"Back-filling user_activity/target_first_ts from {args.dataset_name} ...")
        df = cohort_dims.backfill_from_dataset(df, args.dataset_name, root=args.root)
        have_new = True
    if have_new:
        df["activity"], act_edges = cohort_dims.activity_buckets(df["user_activity"])
        df["age"], age_edges = cohort_dims.age_buckets(df["target_first_ts"])
        print(f"activity tertile edges (total user interactions): {act_edges}")
        print(f"item-age tertile edges (target first_ts): {age_edges}")
    else:
        print("[note] user_activity/target_first_ts absent and no --dataset-name; "
              "skipping activity/age cohorts.")

    rng = np.random.default_rng(args.seed)
    R = args.bootstrap_reps

    cohorts: list[Dict[str, object]] = []
    cohorts.append(cohort_block(df, "FULL", np.ones(len(df), dtype=bool), R, rng))
    cohorts.append(cohort_block(df, "cold (hist 3-5)",         (df["hl"] == "3-5").to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "long-hist (21+)",         (df["hl"] == "21+").to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "popular (Q5)",            (df["pop"] == "Q5").to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "long-tail (Q1)",          (df["pop"] == "Q1").to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "cold x popular (3-5 x Q5)",     ((df["hl"] == "3-5") & (df["pop"] == "Q5")).to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "long-hist x long-tail (21+ x Q1)", ((df["hl"] == "21+") & (df["pop"] == "Q1")).to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "cold x long-tail (3-5 x Q1)",   ((df["hl"] == "3-5") & (df["pop"] == "Q1")).to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "long-hist x popular (21+ x Q5)", ((df["hl"] == "21+") & (df["pop"] == "Q5")).to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "has_profile=True (warm cache)",  df["has_profile"].astype(bool).to_numpy(), R, rng))
    cohorts.append(cohort_block(df, "has_profile=False (cold cache)", (~df["has_profile"].astype(bool)).to_numpy(), R, rng))

    # new axis: user activity volume (light/medium/heavy total interactions)
    if "activity" in df.columns:
        for lvl in cohort_dims.ACTIVITY_LABELS:
            cohorts.append(cohort_block(
                df, f"activity {lvl}", (df["activity"] == lvl).to_numpy(), R, rng))
    # new axis: target item age (established/mid/new by first-interaction ts)
    if "age" in df.columns:
        for lvl in cohort_dims.AGE_LABELS:
            cohorts.append(cohort_block(
                df, f"item-age {lvl}", (df["age"] == lvl).to_numpy(), R, rng))

    # gate tertiles among profile-present examples (gate is meaningful only there)
    if "gate" in df.columns and not df["gate"].isna().all():
        warm = df["has_profile"].astype(bool).to_numpy()
        g = df["gate"].to_numpy()
        gw = g[warm]
        if len(gw) >= 3:
            t1, t2 = float(np.quantile(gw, 1/3)), float(np.quantile(gw, 2/3))
            print(f"gate tertile edges (warm only): t1={t1:.4f}, t2={t2:.4f}")
            mlow  = warm & (g <= t1)
            mmid  = warm & (g >  t1) & (g <= t2)
            mhigh = warm & (g >  t2)
            cohorts.append(cohort_block(df, f"gate low  (<= {t1:.3f}, warm)",  mlow,  R, rng))
            cohorts.append(cohort_block(df, f"gate mid  ({t1:.3f}-{t2:.3f}, warm)", mmid, R, rng))
            cohorts.append(cohort_block(df, f"gate high (>  {t2:.3f}, warm)", mhigh, R, rng))

    # print blocks
    for c in cohorts:
        print_cohort(c)

    # cohort pivots (R@10, NDCG@10)
    hl_order = ["3-5", "6-10", "11-20", "21+"]
    pop_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    df["hit10_baseline_f"] = (df["rank_baseline"] <= 10).astype(float)
    df["hit10_prof_f"]     = (df["rank_prof"]     <= 10).astype(float)
    df["ndcg10_baseline"]  = ndcg10(df["rank_baseline"].to_numpy())
    df["ndcg10_prof"]      = ndcg10(df["rank_prof"].to_numpy())

    def pivot(col: str) -> pd.DataFrame:
        return (df.pivot_table(index="hl", columns="pop", values=col, aggfunc="mean")
                  .reindex(hl_order)[pop_order] * 100).round(2)

    print("\n========== Cohort pivots (history x target popularity) ==========")
    print("\nR@10 baseline (text only), %:")
    print(pivot("hit10_baseline_f").to_string())
    print("\nR@10 oracle+profile, %:")
    print(pivot("hit10_prof_f").to_string())
    print("\nR@10 DELTA (oracle+p - baseline), pp:")
    print((pivot("hit10_prof_f") - pivot("hit10_baseline_f")).to_string())
    print("\nNDCG@10 baseline, %:")
    print(pivot("ndcg10_baseline").to_string())
    print("\nNDCG@10 oracle+profile, %:")
    print(pivot("ndcg10_prof").to_string())
    print("\nNDCG@10 DELTA (oracle+p - baseline), pp:")
    print((pivot("ndcg10_prof") - pivot("ndcg10_baseline")).to_string())

    counts = (df.pivot_table(index="hl", columns="pop", values="hit10_prof_f", aggfunc="size")
                .reindex(hl_order)[pop_order])
    print("\nCounts:")
    print(counts.to_string())

    # write json
    out = {
        "n_examples": int(len(df)),
        "popularity_edges": pop_edges,
        "bootstrap_reps": R,
        "cohorts": cohorts,
    }
    if "activity" in df.columns:
        out["activity_edges"] = list(act_edges)
        out["age_edges"] = list(age_edges)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {args.output_json}")


if __name__ == "__main__":
    main()
