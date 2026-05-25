"""
parameter_jump.py
═════════════════
Parametersprung-Szenario für Kapitel 6.2 (Ausblick) der Bachelorarbeit.

Demonstriert die natürliche Stärke des inkrementellen Lerners:
  Während die Simulation läuft, wird mitten drin ein Systemparameter
  geändert (z.B. Ventil schließt sich plötzlich auf 30%). Der inkrementelle
  Lerner trainiert weiter und passt sich an die neue Dynamik an.

Was der Plot zeigt:
  ▸ Vor dem Sprung    : Vorhersagefehler ist klein (Modell hat gelernt)
  ▸ Direkt nach Sprung: Fehler springt hoch (System hat sich geändert)
  ▸ Über die Zeit     : Fehler sinkt wieder (Modell passt sich an)

Wissenschaftlicher Wert für die BA:
  Zeigt eine Eigenschaft die ein Offline-Modell prinzipiell NICHT hat –
  Anpassung an Konzeptdrift ohne komplettes Neutraining.

Erweiterungsmöglichkeit (in BA als Ausblick erwähnen):
  ADWIN-Konzeptdrift-Erkennung (in River bereits eingebaut) könnte
  automatisch erkennen wann ein Sprung passiert ist.

Verwendung:
    from parameter_jump import run_parameter_jump
    run_parameter_jump(topology, n_tanks, qp_list, h_target, areas,
                       n_samples=1000, jump_time=500, jump_magnitude=0.3)
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from river import tree
from sklearn.tree import DecisionTreeRegressor

from fault_scenarios import Fault, simulate
from ntank_simulation import Valve

OUTPUT_DIR = Path.cwd() / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETERSPRUNG-EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_parameter_jump(
    topology:        str,
    n_tanks:         int,
    qp_list:         List[float],
    h_target:        List[float],
    areas:           List[float],
    n_samples:       int   = 1000,
    jump_time:       float = 500.0,
    jump_magnitude:  float = 0.3,
    rolling_window:  int   = 30,
    jump_fault_type: str   = "valve_stuck",
) -> None:
    """
    Simuliert einen Parametersprung mid-simulation und visualisiert
    wie ein HoeffdingTree-Lerner darauf reagiert.

    Parameter:
      topology       – "linear" oder "coupled"
      n_tanks        – Anzahl Tanks
      qp_list        – Pumpenleistungen
      h_target       – Ziel-Füllstände
      areas          – Querschnittsflächen
      n_samples      – Simulationsdauer [Schritte]
      jump_time      – Zeitpunkt des Parametersprungs [s]
      jump_magnitude – neue Ventilposition (z.B. 0.3 = 30% offen)
      rolling_window – Fenster für gleitenden Mittelwert des Fehlers
      jump_fault_type – Art des Sprungs: "valve_stuck" (Default),
                       "pump_rate_change" (empfohlen: macht Adaptivität
                       sichtbar), "pump_failure" oder "leak_increase".
                       Bei "pump_rate_change" ist jump_magnitude die neue
                       Pumpenrate (z.B. 0.5 = halbierte Rate von Tank 1).
    """
    print(f"\n{'═' * 60}")
    print("  Parametersprung-Szenario (Kapitel 6.2 – Ausblick)")
    print(f"{'═' * 60}")
    print(f"  Topologie       : {topology}")
    print(f"  Tanks           : {n_tanks}")
    print(f"  Sprung bei t    : {jump_time}s")
    print(f"  Sprung-Typ      : {jump_fault_type}")
    print(f"  Sprung-Magnitude: {jump_magnitude}")
    print(f"  Rolling Window  : {rolling_window} Schritte")
    print(f"{'─' * 60}\n")

    # ── Ventile vorbereiten ───────────────────────────────────────────────────
    n_between = (n_tanks - 1) if topology == "linear" \
                else n_tanks * (n_tanks - 1) // 2
    valves_between = [Valve(open=True, position=1.0) for _ in range(n_between)]
    valves_out     = [Valve(open=True, position=1.0) for _ in range(n_tanks)]

    # ── Sprung als Fault definieren ───────────────────────────────────────────
    jump_fault = Fault(
        fault_type=jump_fault_type,
        start_time=jump_time,
        tank_idx=0,
        magnitude=jump_magnitude,
    )

    # ── Simulation MIT Sprung ─────────────────────────────────────────────────
    print("Simuliere mit Parametersprung ...")
    df = simulate(
        topology, n_tanks, valves_between, valves_out,
        qp_list, h_target, areas, n_samples,
        faults=[jump_fault],
    )

    # ── Inkrementelles Training Window-by-Window ──────────────────────────────
    # Wir trainieren einen HoeffdingTree pro Tank parallel zur Simulation
    # und sammeln den Vorhersagefehler an jedem Zeitpunkt.
    print("Trainiere HoeffdingTree inkrementell während Simulation ...")

    n_windows = len(df) - 2
    t_values  = df["t"].to_numpy()[2:]   # Zeitpunkt der Vorhersage

    # Pro Tank: Lerner + Fehlerverlauf
    learners       = [
        tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5)
        for _ in range(n_tanks)
    ]
    errors_per_tank = [[] for _ in range(n_tanks)]

    for j in range(n_windows):
        # Feature-Vektor: 2 Zeitschritte aller Tanks
        x = {
            f"h{ti + 1}_{step}": float(df[f"h{ti + 1}"][j + step])
            for ti in range(n_tanks)
            for step in range(2)
        }

        for i in range(n_tanks):
            true_value = float(df[f"h{i + 1}"][j + 2])

            # 1. Vorhersage VOR dem Lernen (das ist der ehrliche Test)
            pred = learners[i].predict_one(x)
            pred = pred if pred is not None else 0.0
            errors_per_tank[i].append(abs(true_value - pred))

            # 2. Jetzt mit dem neuen Sample lernen
            learners[i].learn_one(x, true_value)

    # ── Offline-Vergleich: einmal trainiertes, eingefrorenes Modell ───────────
    # Multi-Output-Regressionsbaum (CART, max_depth=5) analog zum Offline-Setup
    # der Arbeit. Er wird AUSSCHLIESSLICH auf der Vorsprung-Phase (t < jump_time)
    # trainiert und danach eingefroren -- kann sich also nicht an den Sprung
    # anpassen. Das macht den Kontrast zum mitlernenden Hoeffding-Baum sichtbar.
    print("Trainiere Offline-Regressionsbaum (eingefroren) zum Vergleich ...")

    X = np.array([
        [float(df[f"h{ti + 1}"][j + step])
         for ti in range(n_tanks) for step in range(2)]
        for j in range(n_windows)
    ])
    Y = np.array([
        [float(df[f"h{ti + 1}"][j + 2]) for ti in range(n_tanks)]
        for j in range(n_windows)
    ])

    pre_jump_mask = t_values < jump_time
    offline_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    offline_model.fit(X[pre_jump_mask], Y[pre_jump_mask])
    Y_pred_offline = offline_model.predict(X)
    if Y_pred_offline.ndim == 1:          # Sonderfall n_tanks == 1
        Y_pred_offline = Y_pred_offline.reshape(-1, 1)

    offline_errors_per_tank = [
        np.abs(Y[:, i] - Y_pred_offline[:, i]) for i in range(n_tanks)
    ]

    # ── Rolling Mean des Fehlers berechnen ────────────────────────────────────
    def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
        """Gleitender Mittelwert über Fenster der Breite w."""
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(n_tanks, 1, figsize=(9, 2.6 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]

    fig.suptitle(
        "Parametersprung – inkrementeller vs. eingefrorener Offline-Lerner\n"
        f"Sprung ({jump_fault_type}) bei t={jump_time}s auf {jump_magnitude} "
        f"(rolling MAE über {rolling_window} Schritte)",
        fontsize=14, fontweight="bold"
    )

    for i in range(n_tanks):
        errors        = np.array(errors_per_tank[i])
        rolling_error = rolling_mean(errors, rolling_window)
        offline_roll  = rolling_mean(offline_errors_per_tank[i], rolling_window)
        c = colors[i % len(colors)]

        axes[i].plot(t_values, errors, color=c, alpha=0.20,
                     linewidth=0.8, label="Vorhersagefehler |y − ŷ| (inkrem.)")
        axes[i].plot(t_values, rolling_error, color=c, linewidth=2.2,
                     label=f"Inkrementell (Hoeffding), Rolling MAE")
        axes[i].plot(t_values, offline_roll, color="black", linewidth=2.0,
                     linestyle="--",
                     label="Offline (eingefroren), Rolling MAE")
        axes[i].axvline(jump_time, color="red", linewidth=1.8,
                        linestyle=":", label=f"Parametersprung bei t={jump_time}s")
        axes[i].set_ylabel("Vorhersagefehler [m]", fontsize=12)
        axes[i].set_title(f"Tank {i + 1}", fontsize=13)
        axes[i].tick_params(labelsize=11)
        axes[i].legend(fontsize=10, loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Zeit t [s]", fontsize=13)

    plt.tight_layout()
    out = OUTPUT_DIR / "parameter_jump.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Zusammenfassung ausgeben ──────────────────────────────────────────────
    print(f"\n  → parameter_jump.png gespeichert\n")
    print("  Beobachtung pro Tank (mittlerer Fehler):")
    for i in range(n_tanks):
        errors = np.array(errors_per_tank[i])
        off_errors = np.array(offline_errors_per_tank[i])
        # Index für jump_time finden
        jump_idx = np.searchsorted(t_values, jump_time)

        before = errors[max(0, jump_idx - 50):jump_idx]
        after  = errors[jump_idx:jump_idx + 50]
        late   = errors[-50:]
        off_late = off_errors[-50:]

        before_mae = float(np.mean(before)) if len(before) > 0 else float("nan")
        after_mae  = float(np.mean(after))  if len(after)  > 0 else float("nan")
        late_mae   = float(np.mean(late))   if len(late)   > 0 else float("nan")
        off_late_mae = float(np.mean(off_late)) if len(off_late) > 0 else float("nan")

        print(f"    Tank {i + 1}:  inkrem. vor = {before_mae:.4f}  |  "
              f"direkt nach = {after_mae:.4f}  |  am Ende = {late_mae:.4f}  "
              f"||  Offline am Ende = {off_late_mae:.4f}")

    print(f"\n  💡 Kapitel 6.2 (Ausblick): Inkrementeller Lerner zeigt natürliche")
    print(f"     Stärke gegenüber Offline-Modellen. Mit ADWIN-Konzeptdrift-")
    print(f"     Erkennung (in River verfügbar) könnte der Sprung automatisch")
    print(f"     detektiert werden.")