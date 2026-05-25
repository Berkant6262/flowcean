"""
metrics_helper.py
═════════════════
Zentrales Metrik-Modul für die Bachelorarbeit.
"""

import csv
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path.cwd() / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def _safe_y(value: float, ymin: float, ymax: float) -> float:
    """Klemmt einen y-Wert auf den sichtbaren Bereich, damit Text-
    Annotationen nicht außerhalb landen und savefig nicht crasht."""
    if np.isnan(value):
        return ymin
    return float(np.clip(value, ymin, ymax))


# ══════════════════════════════════════════════════════════════════════════════
#  METRIK-BERECHNUNG
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics_for_tank(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual    = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {k: float("nan") for k in ("MAE", "MSE", "RMSE", "R2", "MaxE")}

    residuals = actual - predicted
    mae  = float(np.mean(np.abs(residuals)))
    mse  = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    maxe = float(np.max(np.abs(residuals)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "MAE":  round(mae,  6),
        "MSE":  round(mse,  6),
        "RMSE": round(rmse, 6),
        "R2":   round(r2,   6),
        "MaxE": round(maxe, 6),
    }


def compute_all_metrics(
    predictions_data: dict,
    n_tanks: int,
) -> Dict[str, Dict[int, dict]]:
    if not predictions_data:
        return {}

    first_value = next(iter(predictions_data.values()))
    if isinstance(first_value, dict) and "actual" in first_value:
        predictions_data = {"Modell": predictions_data}

    all_metrics = {}
    for learner_name, preds in predictions_data.items():
        per_tank = {}
        for i in range(n_tanks):
            per_tank[i] = compute_metrics_for_tank(
                preds[i]["actual"], preds[i]["predicted"]
            )
        all_metrics[learner_name] = per_tank
    return all_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1: STANDARD-METRIKEN (MAE + MSE)
# ══════════════════════════════════════════════════════════════════════════════

def plot_basic_metrics(
    all_metrics: Dict[str, Dict[int, dict]],
    n_tanks: int,
    filename: str = "learning_results.png",
) -> None:
    plt.close("all")

    learner_names = list(all_metrics.keys())
    x      = np.arange(n_tanks)
    width  = 0.35 / max(len(learner_names), 1)
    colors_mae = ["#1f77b4", "#2ca02c", "#9467bd"]
    colors_mse = ["#d62728", "#ff7f0e", "#e377c2"]

    fig, ax = plt.subplots(figsize=(max(7, n_tanks * 2 * len(learner_names)), 5))

    for li, name in enumerate(learner_names):
        metrics = all_metrics[name]
        mae_vals = [metrics[i]["MAE"] for i in range(n_tanks)]
        mse_vals = [metrics[i]["MSE"] for i in range(n_tanks)]
        offset   = (li - len(learner_names) / 2 + 0.5) * width * 2

        bars1 = ax.bar(x + offset - width / 2, mae_vals, width,
                       label=f"MAE – {name}",
                       color=colors_mae[li % 3], alpha=0.85)
        bars2 = ax.bar(x + offset + width / 2, mse_vals, width,
                       label=f"MSE – {name}",
                       color=colors_mse[li % 3], alpha=0.85)

        for bar in (*bars1, *bars2):
            h = bar.get_height()
            label_txt = f"{h:.4f}" if not np.isnan(h) else "N/A"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h if not np.isnan(h) else 0,
                label_txt, ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Tank")
    ax.set_ylabel("Fehler")
    ax.set_title("Standard-Metriken pro Tank – MAE und MSE",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tank {i + 1}" for i in range(n_tanks)])
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    try:
        fig.tight_layout()
    except Exception:
        pass

    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150)   # OHNE bbox_inches="tight"
    plt.close("all")
    print(f"→ {filename} gespeichert")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2: ERWEITERTE METRIKEN (RMSE + R² + MaxE)
# ══════════════════════════════════════════════════════════════════════════════

def plot_extended_metrics(
    all_metrics: Dict[str, Dict[int, dict]],
    n_tanks: int,
    filename: str = "extended_metrics.png",
) -> None:
    plt.close("all")

    learner_names = list(all_metrics.keys())
    x      = np.arange(n_tanks)
    width  = 0.4 / max(len(learner_names), 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Erweiterte Metriken pro Tank – RMSE, MaxE und R²",
                 fontsize=12, fontweight="bold")

    # ── Subplot 1: RMSE und MaxE ──────────────────────────────────────────────
    ax1 = axes[0]
    for li, name in enumerate(learner_names):
        m    = all_metrics[name]
        rmse = [m[i]["RMSE"] for i in range(n_tanks)]
        maxe = [m[i]["MaxE"] for i in range(n_tanks)]
        off  = (li - len(learner_names) / 2 + 0.5) * width * 2

        bars1 = ax1.bar(x + off - width / 2, rmse, width,
                        label=f"RMSE – {name}",
                        color=colors[li % 4], alpha=0.85)
        bars2 = ax1.bar(x + off + width / 2, maxe, width,
                        label=f"MaxE – {name}",
                        color=colors[li % 4], alpha=0.45)

        for bar in (*bars1, *bars2):
            h = bar.get_height()
            label_txt = f"{h:.4f}" if not np.isnan(h) else "N/A"
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h if not np.isnan(h) else 0,
                label_txt, ha="center", va="bottom", fontsize=7,
            )

    ax1.set_xlabel("Tank")
    ax1.set_ylabel("Fehler [m]")
    ax1.set_title("RMSE  |  Max-Fehler")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Tank {i + 1}" for i in range(n_tanks)])
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Subplot 2: R² (mit Clamping gegen extreme Werte) ──────────────────────
    ax2 = axes[1]
    R2_YMIN, R2_YMAX = -0.1, 1.1
    ax2.set_ylim(R2_YMIN, R2_YMAX)

    for li, name in enumerate(learner_names):
        m   = all_metrics[name]
        r2_raw     = [m[i]["R2"] for i in range(n_tanks)]
        # Bar-Höhen klemmen, sonst sprengt extreme R² den Plot
        r2_clipped = [_safe_y(v, R2_YMIN, R2_YMAX) for v in r2_raw]
        off = (li - len(learner_names) / 2 + 0.5) * width * 2

        bars = ax2.bar(x + off, r2_clipped, width * 1.5,
                       label=name, color=colors[li % 4], alpha=0.85)

        for bar, raw in zip(bars, r2_raw):
            if np.isnan(raw):
                continue
            # Text-Position klemmen, echten Wert anzeigen
            text_y = _safe_y(raw, R2_YMIN, R2_YMAX - 0.05)
            if raw < R2_YMIN or raw > R2_YMAX:
                label_text = f"{raw:.2g} (clipped)"
            else:
                label_text = f"{raw:.4f}"
            ax2.text(bar.get_x() + bar.get_width() / 2, text_y,
                     label_text, ha="center", va="bottom", fontsize=8)

    ax2.axhline(1.0, color="green", linewidth=1.2, linestyle="--",
                label="R²=1.0 (perfekt)")
    ax2.axhline(0.0, color="gray",  linewidth=0.8, linestyle="--",
                label="R²=0.0 (Mittelwert)")
    ax2.set_xlabel("Tank")
    ax2.set_ylabel("R²")
    ax2.set_title("R² – Bestimmtheitsmaß\n(1.0 = perfekt, 0.0 = nur Mittelwert)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Tank {i + 1}" for i in range(n_tanks)])
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    try:
        fig.tight_layout()
    except Exception:
        pass

    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150)   # OHNE bbox_inches="tight"
    plt.close("all")
    print(f"→ {filename} gespeichert")


# ══════════════════════════════════════════════════════════════════════════════
#  CSV-EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_metrics_csv(
    all_metrics: Dict[str, Dict[int, dict]],
    n_tanks: int,
    filename: str = "metrics.csv",
) -> None:
    rows = []
    for name, metrics in all_metrics.items():
        for i in range(n_tanks):
            row = {"learner": name, "tank": i + 1}
            row.update(metrics[i])
            rows.append(row)

    if not rows:
        return

    out = OUTPUT_DIR / filename
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"→ {filename} gespeichert")


# ══════════════════════════════════════════════════════════════════════════════
#  KONSOLENAUSGABE
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(
    all_metrics: Dict[str, Dict[int, dict]],
    n_tanks: int,
) -> None:
    print(f"\n{'═' * 75}")
    print("  Metriken-Übersicht (alle aus Modellvorhersagen berechnet)")
    print(f"{'═' * 75}")
    print(f"  {'Lerner':<28} {'Tank':<5} "
          f"{'MAE':>8} {'MSE':>10} {'RMSE':>8} {'R²':>8} {'MaxE':>8}")
    print(f"{'─' * 75}")
    for name, metrics in all_metrics.items():
        for i in range(n_tanks):
            m = metrics[i]
            r2_str = f"{m['R2']:.4f}" if not np.isnan(m["R2"]) else "  N/A"
            print(f"  {name:<28} {i + 1:<5} "
                  f"{m['MAE']:>8.4f} {m['MSE']:>10.6f} "
                  f"{m['RMSE']:>8.4f} {r2_str:>8} {m['MaxE']:>8.4f}")
    print(f"{'═' * 75}\n")