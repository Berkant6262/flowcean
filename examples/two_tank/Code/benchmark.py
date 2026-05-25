"""
benchmark.py
════════════
Automatisierter Skalierbarkeits-Benchmark für die Bachelorarbeit.

Misst für jede Kombination aus (n_tanks × topology × mode):
  - Simulationszeit   [ms]
  - Trainingszeit     [ms]
  - Inferenzzeit      [ms]
  - MAE, MSE pro Tank (Durchschnitt über alle Tanks)

Ergebnisse werden als CSV gespeichert und als Plot visualisiert.

Verwendung:
    python benchmark.py                    # lädt benchmark_config.yaml
    python benchmark.py my_benchmark.yaml  # eigene Config
"""

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.utils.data as _tud
import yaml
from river import neural_net, optim, tree

import flowcean.cli
from flowcean.core import learn_incremental, learn_offline
from flowcean.ode import OdeEnvironment
from flowcean.polars import SlidingWindow, StreamingOfflineEnvironment, TrainTestSplit, collect
from flowcean.river import RiverLearner
from flowcean.sklearn import RegressionTree
from flowcean.torch import LightningLearner, MultilayerPerceptron
from flowcean.utils.random import initialize_random

# ── Windows-Fix ───────────────────────────────────────────────────────────────
_orig_dl_init = _tud.DataLoader.__init__
def _patched_dl_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["persistent_workers"] = False
    _orig_dl_init(self, *args, **kwargs)
_tud.DataLoader.__init__ = _patched_dl_init

# ── Importiere Klassen aus Hauptdatei ─────────────────────────────────────────
# Passe den Dateinamen ggf. an
from ntank_simulation import (
    FrameCollector,
    Valve,
    build_system,
    extract_river_model,
)
from metrics_helper import compute_metrics_for_tank

OUTPUT_DIR = Path.cwd() / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "benchmark_results.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK-KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_BENCHMARK_CONFIG = {
    # Tankanzahlen die getestet werden sollen
    "tank_sizes": [2, 3, 4, 5, 6, 8],

    # Topologien
    "topologies": ["linear", "coupled"],

    # Lernmodi
    "modes": ["offline", "incremental"],

    # Modell pro Modus
    "offline_model":      "tree",       # "tree" | "mlp" | "beide"
    "incremental_model":  "hoeffding",  # "hoeffding" | "mlp"

    # Simulationsparameter (fest für alle Läufe)
    "n_samples_offline":      250,
    "n_samples_incremental":  1000,
    "h_target_per_tank":      200.0,
    "area_per_tank":          1.0,
    "qpmax_per_tank":         1.0,
    "noise_std":              0.0,
}


def load_benchmark_config(path: str = "benchmark_config.yaml") -> dict:
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"✓ Benchmark-Config geladen: {config_path.resolve()}")
        return cfg

    print(f"  '{config_path}' nicht gefunden → verwende Standard-Konfiguration")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_BENCHMARK_CONFIG, f, allow_unicode=True, sort_keys=False)
    print(f"  → Standard benchmark_config.yaml geschrieben")
    return DEFAULT_BENCHMARK_CONFIG


# ══════════════════════════════════════════════════════════════════════════════
#  EINZELNER BENCHMARK-LAUF
# ══════════════════════════════════════════════════════════════════════════════

def run_single_benchmark(
    n_tanks: int,
    topology: str,
    mode: str,
    cfg: dict,
) -> Dict[str, Any]:
    """
    Führt einen einzelnen Benchmark-Lauf durch und gibt ein Ergebnis-Dict zurück.

    Gemessene Größen:
      sim_ms        – reine ODE-Simulationszeit
      train_ms      – Trainingszeit des ML-Modells
      inference_ms  – Zeit für Vorhersagen auf dem Testset
      mae_mean      – mittlerer MAE über alle Tanks
      mse_mean      – mittlerer MSE über alle Tanks

    HINWEIS zur Vergleichbarkeit (für die BA wichtig):
      Der inkrementelle Modus trainiert pro Tank ein separates Modell
      (n Modelle bei n Tanks). Trainings- und Inferenzzeit werden über
      alle Tanks summiert.
      Der Offline-Modus trainiert ein einziges Multi-Output-Modell für
      alle Tanks gleichzeitig.
      → Höhere Trainingszeit im incremental Modus ist KEIN Bug, sondern
        ein architektureller Unterschied der beiden Lernparadigmen.
    """
    initialize_random(seed=42)

    # ── Parameter aus Config ──────────────────────────────────────────────────
    n_samples  = cfg["n_samples_offline"] if mode == "offline" else cfg["n_samples_incremental"]
    h_target   = [cfg["h_target_per_tank"]] * n_tanks
    areas      = [cfg["area_per_tank"]]     * n_tanks
    qp_list    = [cfg["qpmax_per_tank"]]    * n_tanks
    noise_std  = cfg["noise_std"]

    n_between = (n_tanks - 1) if topology == "linear" else n_tanks * (n_tanks - 1) // 2
    valves_between = [Valve(open=True, position=1.0) for _ in range(n_between)]
    valves_out     = [Valve(open=True, position=1.0) for _ in range(n_tanks)]

    # ── Simulation messen ─────────────────────────────────────────────────────
    collector = FrameCollector(n_tanks, noise_std=noise_std)
    system = build_system(topology, n_tanks, valves_between, valves_out,
                          qp_list, h_target, areas)
    data_env = OdeEnvironment(system, dt=1.0,
                              map_to_dataframe=collector.collect_frame)

    t0 = time.perf_counter()
    df_collected = collect(data_env, n_samples)
    sim_ms = round((time.perf_counter() - t0) * 1000, 2)

    df_plot = collector.concat()

    # ── Feature Engineering ───────────────────────────────────────────────────
    data    = df_collected | SlidingWindow(window_size=3)
    inputs  = [f"h{i + 1}_{step}" for i in range(n_tanks) for step in range(2)]
    outputs = [f"h{i + 1}_2" for i in range(n_tanks)]

    train, _test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    n_windows  = len(df_plot) - 2
    test_start = int(n_windows * 0.8)

    mae_list = []
    mse_list = []
    train_ms_total = 0.0
    inference_ms_total = 0.0

    # ── Training + Evaluierung ────────────────────────────────────────────────
    if mode == "offline":
        model_choice = cfg.get("offline_model", "tree")

        # Saubere Logik: explizit jede Option behandeln.
        # "beide" macht im Benchmark keinen Sinn (würde 2× Training bedeuten)
        # → fällt auf "tree" zurück mit Warnung.
        if model_choice == "mlp":
            learner = LightningLearner(
                module=MultilayerPerceptron(
                    learning_rate=1e-3,
                    output_size=len(outputs),
                    hidden_dimensions=[32, 16],
                    activation_function=torch.nn.LeakyReLU,
                ),
                max_epochs=500,
            )
        else:
            if model_choice not in ("tree", "beide"):
                print(f"  ⚠ Unbekanntes offline_model='{model_choice}' "
                      f"→ verwende 'tree'")
            if model_choice == "beide":
                print(f"  ⚠ offline_model='beide' im Benchmark nicht sinnvoll "
                      f"→ verwende 'tree'")
            learner = RegressionTree(max_depth=5)

        t0 = time.perf_counter()
        model = learn_offline(train, learner, inputs, outputs)
        train_ms_total = round((time.perf_counter() - t0) * 1000, 2)

        # Inferenzzeit messen UND Vorhersagen für Metriken sammeln
        actuals_per_tank = [[] for _ in range(n_tanks)]
        preds_per_tank   = [[] for _ in range(n_tanks)]

        t0 = time.perf_counter()
        for j in range(test_start, n_windows):
            row = {
                f"h{ti + 1}_{step}": float(df_plot[f"h{ti + 1}"][j + step])
                for ti in range(n_tanks) for step in range(2)
            }
            pred_df = model.predict(pl.DataFrame([row]).lazy()).collect()
            for i in range(n_tanks):
                tgt = f"h{i + 1}_2"
                actuals_per_tank[i].append(float(df_plot[f"h{i + 1}"][j + 2]))
                preds_per_tank[i].append(
                    float(pred_df[tgt][0]) if tgt in pred_df.columns
                    else float("nan")
                )
        inference_ms_total = round((time.perf_counter() - t0) * 1000, 2)

        # Metriken einheitlich aus Vorhersagen berechnen
        for i in range(n_tanks):
            m = compute_metrics_for_tank(
                np.array(actuals_per_tank[i]),
                np.array(preds_per_tank[i]),
            )
            mae_list.append(m["MAE"])
            mse_list.append(m["MSE"])

    else:  # incremental
        model_choice = cfg.get("incremental_model", "hoeffding")

        for tank_idx in range(1, n_tanks + 1):
            target_name = f"h{tank_idx}_2"
            train_env = StreamingOfflineEnvironment(train, batch_size=1)

            if model_choice == "mlp":
                learner = RiverLearner(model=neural_net.MLPRegressor(
                    hidden_dims=(32, 16),
                    activations=(
                        neural_net.activations.ReLU,
                        neural_net.activations.ReLU,
                        neural_net.activations.Identity,
                    ),
                    optimizer=optim.Adam(lr=1e-3),
                    seed=42,
                ))
            else:
                learner = RiverLearner(
                    model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5)
                )

            t0 = time.perf_counter()
            trained = learn_incremental(train_env, learner, inputs, [target_name])
            train_ms_total += round((time.perf_counter() - t0) * 1000, 2)

            # Inferenzzeit messen UND Vorhersagen für Metriken sammeln
            river_model = extract_river_model(trained)
            actuals = []
            preds   = []
            t0 = time.perf_counter()
            for j in range(test_start, n_windows):
                x = {
                    f"h{ti + 1}_{step}": float(df_plot[f"h{ti + 1}"][j + step])
                    for ti in range(n_tanks) for step in range(2)
                }
                pred = river_model.predict_one(x)
                actuals.append(float(df_plot[f"h{tank_idx}"][j + 2]))
                preds.append(float(pred) if pred is not None else float("nan"))
            inference_ms_total += round((time.perf_counter() - t0) * 1000, 2)

            # Metriken einheitlich aus Vorhersagen berechnen
            m = compute_metrics_for_tank(np.array(actuals), np.array(preds))
            mae_list.append(m["MAE"])
            mse_list.append(m["MSE"])

    # ── Ergebnis zusammenstellen ──────────────────────────────────────────────
    mae_mean = float(np.nanmean(mae_list))
    mse_mean = float(np.nanmean(mse_list))

    result = {
        "n_tanks":       n_tanks,
        "topology":      topology,
        "mode":          mode,
        "n_samples":     n_samples,
        "sim_ms":        sim_ms,
        "train_ms":      train_ms_total,
        "inference_ms":  inference_ms_total,
        "mae_mean":      round(mae_mean, 6),
        "mse_mean":      round(mse_mean, 6),
    }

    print(
        f"  n={n_tanks:2d} | {topology:7s} | {mode:11s} | "
        f"sim={sim_ms:7.1f}ms | train={train_ms_total:8.1f}ms | "
        f"inf={inference_ms_total:7.1f}ms | "
        f"MAE={mae_mean:.4f} | MSE={mse_mean:.4f}"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ERGEBNISSE SPEICHERN
# ══════════════════════════════════════════════════════════════════════════════

def save_results_csv(results: List[Dict], path: Path) -> None:
    """Speichert alle Benchmark-Ergebnisse als CSV-Datei."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Ergebnisse gespeichert: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_scalability(results: List[Dict]) -> None:
    """
    Erzeugt den zentralen Skalierbarkeits-Plot der Bachelorarbeit.
    4 Subplots: MAE, MSE, Trainingszeit, Inferenzzeit – jeweils über n_tanks.
    Jede Kombination (topology × mode) bekommt eine eigene Linie.
    """
    # Alle eindeutigen Kombinationen bestimmen
    combos = sorted(set((r["topology"], r["mode"]) for r in results))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = [
        ("mae_mean",     "MAE (Ø über alle Tanks)",   axes[0]),
        ("mse_mean",     "MSE (Ø über alle Tanks)",   axes[1]),
        ("train_ms",     "Trainingszeit [ms]",         axes[2]),
        ("inference_ms", "Inferenzzeit [ms]",          axes[3]),
    ]

    for metric_key, ylabel, ax in metrics:
        for ci, (topo, mode) in enumerate(combos):
            subset = sorted(
                [r for r in results if r["topology"] == topo and r["mode"] == mode],
                key=lambda r: r["n_tanks"],
            )
            if not subset:
                continue
            xs = [r["n_tanks"]     for r in subset]
            ys = [r[metric_key]    for r in subset]
            label = f"{topo} / {mode}"
            ax.plot(xs, ys,
                    color=colors[ci % len(colors)],
                    marker=markers[ci % len(markers)],
                    linewidth=2.8, markersize=12,
                    markeredgecolor="black", markeredgewidth=0.8,
                    alpha=0.85,
                    label=label)

        ax.set_xlabel("Anzahl Tanks (n)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(sorted(set(r["n_tanks"] for r in results)))

    fig.suptitle(
        "Skalierbarkeitsanalyse – N-Tank-Hybridsystem\n"
        "(Fehlermetriken & Laufzeiten in Abhängigkeit der Tankanzahl)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "benchmark_scalability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Skalierbarkeits-Plot gespeichert: {out}")


def plot_time_breakdown(results: List[Dict]) -> None:
    """
    Gestapeltes Balkendiagramm: Sim + Train + Inferenz pro Konfiguration.

    Layout: Horizontal gruppiert. Linke Hälfte = Offline, rechte Hälfte = Incremental.
    Innerhalb jeder Hälfte: nach n_tanks und Topologie sortiert.
    Lesbarer als 24 vertikale Balken nebeneinander.
    """
    # Nach Modus sortieren - offline links, incremental rechts
    offline_results     = sorted([r for r in results if r["mode"] == "offline"],
                                  key=lambda r: (r["n_tanks"], r["topology"]))
    incremental_results = sorted([r for r in results if r["mode"] == "incremental"],
                                  key=lambda r: (r["n_tanks"], r["topology"]))
    sorted_results = offline_results + incremental_results

    labels = [
        f"n={r['n_tanks']} {r['topology'][:3]}\n[{r['mode'][:3]}]"
        for r in sorted_results
    ]
    sim_ms   = [r["sim_ms"]       for r in sorted_results]
    train_ms = [r["train_ms"]     for r in sorted_results]
    inf_ms   = [r["inference_ms"] for r in sorted_results]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))

    ax.barh(y, sim_ms,   label="Simulation [ms]",
            color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.barh(y, train_ms, left=sim_ms,
            label="Training [ms]",
            color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.barh(y, inf_ms,
            left=[s + t for s, t in zip(sim_ms, train_ms)],
            label="Inferenz [ms]",
            color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Trennlinie zwischen offline und incremental
    if offline_results and incremental_results:
        sep_y = len(offline_results) - 0.5
        ax.axhline(sep_y, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.5, sep_y - len(offline_results) / 2,
                "OFFLINE", fontsize=11, fontweight="bold", color="gray",
                ha="center", va="center", alpha=0.4)
        ax.text(ax.get_xlim()[1] * 0.5,
                sep_y + len(incremental_results) / 2,
                "INCREMENTAL", fontsize=11, fontweight="bold", color="gray",
                ha="center", va="center", alpha=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Zeit [ms]", fontsize=11)
    ax.set_title("Laufzeit-Breakdown: Simulation / Training / Inferenz",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()  # damit n=2 oben steht

    plt.tight_layout()
    out = OUTPUT_DIR / "benchmark_time_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Laufzeit-Breakdown gespeichert: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  HAUPTFUNKTION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Führt den vollautomatischen Skalierbarkeits-Benchmark durch.

    Iteriert über alle Kombinationen aus:
      tank_sizes × topologies × modes

    und speichert Ergebnisse als CSV + Plots.
    """
    config_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_config.yaml"

    flowcean.cli.initialize()
    cfg = load_benchmark_config(config_path)

    tank_sizes  = cfg["tank_sizes"]
    topologies  = cfg["topologies"]
    modes       = cfg["modes"]

    total_runs = len(tank_sizes) * len(topologies) * len(modes)
    print(f"\n{'═' * 60}")
    print(f"  Benchmark gestartet – {total_runs} Läufe geplant")
    print(f"  Tanks: {tank_sizes}")
    print(f"  Topologien: {topologies}")
    print(f"  Modi: {modes}")
    print(f"{'═' * 60}\n")

    all_results = []
    run_nr = 0

    for n_tanks in tank_sizes:
        for topology in topologies:
            for mode in modes:
                run_nr += 1
                print(f"[{run_nr:2d}/{total_runs}] ", end="", flush=True)
                try:
                    result = run_single_benchmark(n_tanks, topology, mode, cfg)
                    all_results.append(result)
                except Exception as e:
                    print(f"\n  ⚠ Fehler bei n={n_tanks}, {topology}, {mode}: {e}")
                    all_results.append({
                        "n_tanks": n_tanks, "topology": topology, "mode": mode,
                        "n_samples": 0, "sim_ms": float("nan"),
                        "train_ms": float("nan"), "inference_ms": float("nan"),
                        "mae_mean": float("nan"), "mse_mean": float("nan"),
                    })

    print(f"\n{'═' * 60}")
    print(f"  Benchmark abgeschlossen – {len(all_results)} Läufe")
    print(f"{'═' * 60}\n")

    save_results_csv(all_results, RESULTS_CSV)
    plot_scalability(all_results)
    plot_time_breakdown(all_results)

    print("\n✓ Alle Dateien gespeichert:")
    print(f"   {RESULTS_CSV}")
    print(f"   {OUTPUT_DIR / 'benchmark_scalability.png'}")
    print(f"   {OUTPUT_DIR / 'benchmark_time_breakdown.png'}")


if __name__ == "__main__":
    main()