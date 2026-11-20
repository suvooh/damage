"""Choose a threshold on the calibration set, then report on the untouched test set.

Reads the JSONL written by real_pipeline_scores.py. An image "alerts" when its
case score is at or above the threshold. Intervals are 95% Wilson intervals.

Usage: python tools/real_pipeline_report.py SCORES.jsonl [max_clean_false_alert_rate]
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def stats(rows: list[dict], threshold: float) -> dict:
    out: dict = {}
    damage = np.array([r["case_score"] for r in rows if r["label"] == "damage"])
    clean = [r for r in rows if r["label"] == "clean"]
    out["damage_n"], out["recall"] = len(damage), float((damage >= threshold).mean())
    out["recall_ci"] = wilson(int((damage >= threshold).sum()), len(damage))
    for name, sel in (("all_clean", clean), ("wild_clean", [r for r in clean if r["source"] == "wild"]),
                      ("coco_clean", [r for r in clean if r["source"] == "coco"])):
        k = sum(r["case_score"] >= threshold for r in sel)
        out[name] = {"n": len(sel), "false_alerts": k, "rate": k / max(1, len(sel)), "ci": wilson(k, len(sel))}
    out["auroc_vs_all_clean"] = auroc(damage, np.array([r["case_score"] for r in clean]))
    out["auroc_vs_wild_clean"] = auroc(damage, np.array([r["case_score"] for r in clean if r["source"] == "wild"]))
    return out


def fmt(s: dict) -> str:
    lo, hi = s["recall_ci"]
    lines = [f"  damage recall {s['recall']:.3f}  (95% CI {lo:.3f}-{hi:.3f}, n={s['damage_n']})"]
    for k in ("all_clean", "wild_clean", "coco_clean"):
        c = s[k]
        lines.append(f"  {k:11s} false alerts {c['false_alerts']}/{c['n']} = {c['rate']:.3f}  (95% CI {c['ci'][0]:.3f}-{c['ci'][1]:.3f})")
    lines.append(f"  AUROC vs all clean {s['auroc_vs_all_clean']:.3f}, vs wild clean {s['auroc_vs_wild_clean']:.3f}")
    return "\n".join(lines)


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    cal = [r for r in rows if r["set"] == "calibration"]
    test = [r for r in rows if r["set"] == "test"]
    print(f"scored: calibration {len(cal)}, test {len(test)}")
    levels = np.round(np.linspace(0.05, 0.99, 48), 4)
    best = None
    print(f"\nCalibration sweep (budget: all-clean false-alert rate <= {budget:.0%}, wild-clean <= {budget:.0%}):")
    for t in levels:
        s = stats(cal, float(t))
        ok = s["all_clean"]["rate"] <= budget and s["wild_clean"]["rate"] <= budget
        if ok and (best is None or s["recall"] > best[1]["recall"] or (s["recall"] == best[1]["recall"] and t < best[0])):
            best = (float(t), s)
    if best is None:
        print("  no threshold meets the false-alert budget on calibration")
        chosen = float(levels[-1])
    else:
        chosen = best[0]
        print(f"  chosen threshold {chosen:.3f}")
        print(fmt(best[1]))
    for t, label in ((0.381, "current profile (0.381)"), (chosen, f"chosen ({chosen:.3f})")):
        print(f"\nTEST SET at {label}:")
        if test:
            print(fmt(stats(test, t)))
    print("\nTEST SET recall at fixed false-alert rates (threshold set from the test clean scores themselves):")
    clean = np.array([r["case_score"] for r in test if r["label"] == "clean"])
    dmg = np.array([r["case_score"] for r in test if r["label"] == "damage"])
    for fa in (0.0, 0.01, 0.03, 0.05):
        cut = np.quantile(clean, 1 - fa) if fa else clean.max() + 1e-9
        print(f"  <= {fa:.0%} clean alerts: recall {(dmg > cut if fa == 0 else dmg > cut).mean():.3f}")


if __name__ == "__main__":
    main()
