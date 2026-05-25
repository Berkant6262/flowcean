"""
experiment_runner.py
════════════════════
Startet ein einzelnes N-Tank-Experiment vollständig aus einer YAML-Konfiguration.
Ersetzt jede interaktive Eingabe – kein input() mehr nötig.

Workflow:
    1. config.yaml laden und validieren
    2. Lernmodus ausführen (incremental oder offline)
       → Rückgabe: predictions_data
    3. Vorhersage-Plot rendern (predictions_plot.png)
    4. Alle Metriken aus den Vorhersagen berechnen (metrics_helper.py)
    5. Zwei Plots:
         learning_results.png   – MAE + MSE
         extended_metrics.png   – RMSE + R² + MaxE
    6. CSV mit allen Metriken speichern
    7. Störfall-Szenarien (fault_scenarios.py)
    8. Optional: Parametersprung (Kapitel 6.2 Ausblick)

Verwendung:
    python experiment_runner.py                  # lädt config.yaml
    python experiment_runner.py my_config.yaml   # eigene Datei
"""

import sys
from pathlib import Path

import yaml

import flowcean.cli
from flowcean.utils.random import initialize_random

# ── Eigene Module ─────────────────────────────────────────────────────────────
from ntank_simulation import (
    Valve,
    plot_predictions,
    run_incremental,
    run_offline,
)
from fault_scenarios import run_fault_scenarios, faults_from_config
from parameter_jump import run_parameter_jump
from metrics_helper import (
    compute_all_metrics,
    plot_basic_metrics,
    plot_extended_metrics,
    print_metrics,
    save_metrics_csv,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG LADEN
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str = "config.yaml") -> dict:
    """Lädt die YAML-Konfigurationsdatei."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Konfigurationsdatei '{config_path}' nicht gefunden.\n"
            f"  Bitte config.yaml erstellen oder Pfad angeben."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"✓ Konfiguration geladen: {config_path.resolve()}")
    return cfg


def validate_config(cfg: dict) -> None:
    """Prüft die wichtigsten Felder auf Konsistenz."""
    n = cfg["n_tanks"]

    assert cfg["mode"] in ("incremental", "offline"), \
        "mode muss 'incremental' oder 'offline' sein"
    assert cfg["topology"] in ("linear", "coupled"), \
        "topology muss 'linear' oder 'coupled' sein"
    assert len(cfg["pumps"])    == n, f"pumps muss {n} Einträge haben"
    assert len(cfg["h_target"]) == n, f"h_target muss {n} Einträge haben"
    assert len(cfg["areas"])    == n, f"areas muss {n} Einträge haben"

    expected_between = (n - 1) if cfg["topology"] == "linear" \
                       else n * (n - 1) // 2
    assert len(cfg["valves_between"]) == expected_between, (
        f"valves_between muss {expected_between} Einträge haben "
        f"(Topologie: {cfg['topology']}, n_tanks: {n})"
    )
    assert len(cfg["valves_out"]) == n, \
        f"valves_out muss {n} Einträge haben"

    print("✓ Konfiguration validiert")


def build_valves(cfg: dict):
    """Erzeugt Valve-Objekte aus den YAML-Einträgen."""
    valves_between = [
        Valve(open=v["open"], position=float(v["position"]))
        for v in cfg["valves_between"]
    ]
    valves_out = [
        Valve(open=v["open"], position=float(v["position"]))
        for v in cfg["valves_out"]
    ]
    return valves_between, valves_out


def print_config_summary(cfg: dict) -> None:
    """Übersichtliche Zusammenfassung der Config."""
    sep = "─" * 50
    print(f"\n{'═' * 50}")
    print("  N-Tank-Simulation – Experiment-Konfiguration")
    print(f"{'═' * 50}")
    print(f"  Modus      : {cfg['mode'].upper()}")
    print(f"  Topologie  : {cfg['topology']}")
    print(f"  Tanks      : {cfg['n_tanks']}")
    print(f"  Samples    : {cfg['n_samples']}")
    print(f"  Modell     : {cfg['model']}")
    print(sep)
    print(f"  Pumpen     : {cfg['pumps']}")
    print(f"  h_target   : {cfg['h_target']}")
    print(f"  Flächen    : {cfg['areas']}")
    print(f"  Rauschen σ : {cfg['noise_std']} m")
    print(sep)
    print(f"  Ventile (zwischen) : {len(cfg['valves_between'])} Stück")
    print(f"  Ventile (Auslass)  : {len(cfg['valves_out'])} Stück")
    print(f"{'═' * 50}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  HAUPTFUNKTION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Hauptablauf des Experiments.

    Hybride System-Architektur (Bachelorarbeit):
      Das ODE-System (NTankLinear / NTankFullyCoupled) übernimmt die
      physikalische Modellierung und generiert Trainingsdaten.
      Das ML-Modell (RegressionTree / MLP / HoeffdingTree) lernt daraus
      ohne explizite Kenntnis der Differentialgleichungen.
      Diese Kombination aus physikalischem Simulator + datengetriebenem
      Lerner definiert das 'hybride System' im Sinne der Bachelorarbeit.
    """
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    flowcean.cli.initialize()
    cfg = load_config(config_path)
    validate_config(cfg)
    print_config_summary(cfg)

    initialize_random(seed=42)

    valves_between, valves_out = build_valves(cfg)

    # Parameter aus Config
    mode      = cfg["mode"]
    topology  = cfg["topology"]
    n_tanks   = cfg["n_tanks"]
    n_samples = cfg["n_samples"]
    model     = cfg["model"]
    qp_list   = cfg["pumps"]
    h_target  = cfg["h_target"]
    areas     = cfg["areas"]
    noise_std = cfg["noise_std"]

    # ── 1. Lernmodus ausführen ───────────────────────────────────────────────
    if mode == "incremental":
        predictions_data = run_incremental(
            topology, n_tanks, n_samples,
            valves_between, valves_out,
            qp_list, h_target, areas, noise_std, model,
        )
    else:
        predictions_data = run_offline(
            topology, n_tanks, n_samples,
            valves_between, valves_out,
            qp_list, h_target, areas, noise_std, model,
        )

    # ── 2. Vorhersage-Plot ───────────────────────────────────────────────────
    plot_predictions(predictions_data, n_tanks)

    # ── 3. Metriken berechnen & plotten ──────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Metriken-Berechnung (alle aus Vorhersagen)")
    print(f"{'═' * 60}")
    all_metrics = compute_all_metrics(predictions_data, n_tanks)

    print_metrics(all_metrics, n_tanks)
    plot_basic_metrics(all_metrics, n_tanks)        # learning_results.png
    plot_extended_metrics(all_metrics, n_tanks)     # extended_metrics.png
    save_metrics_csv(all_metrics, n_tanks)          # metrics.csv

    # ── 4. Störfall-Szenarien (Kapitel 4.2) ──────────────────────────────────
    run_fault_scenarios(
        topology  = topology,
        n_tanks   = n_tanks,
        qp_list   = qp_list,
        h_target  = h_target,
        areas     = areas,
        faults    = faults_from_config(cfg),
        n_samples = cfg.get("n_samples_fault", 500),
    )

    # ── 5. Parametersprung (Kapitel 6.2 – Ausblick, optional) ────────────────
    if cfg.get("show_parameter_jump", False):
        run_parameter_jump(
            topology       = topology,
            n_tanks        = n_tanks,
            qp_list        = qp_list,
            h_target       = h_target,
            areas          = areas,
            n_samples      = cfg.get("n_samples_jump",      1000),
            jump_time      = cfg.get("jump_time",           500.0),
            jump_magnitude = cfg.get("jump_magnitude",      0.3),
            rolling_window = cfg.get("jump_rolling_window", 30),
            jump_fault_type= cfg.get("jump_fault_type",     "valve_stuck"),
        )

    print(f"\n{'═' * 60}")
    print("  ✓ Experiment abgeschlossen")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()