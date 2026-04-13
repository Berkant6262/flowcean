import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray
from river import neural_net, optim, tree
from typing_extensions import Self, override


import flowcean.cli
from flowcean.core import evaluate_offline, learn_incremental
from flowcean.ode import OdeEnvironment, OdeState, OdeSystem
from flowcean.polars import SlidingWindow, StreamingOfflineEnvironment, TrainTestSplit
from flowcean.polars.environments.dataframe import collect
from flowcean.river import RiverLearner
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError


sys.setrecursionlimit(10000)
logger = logging.getLogger(__name__)


OUTPUT_DIR = Path.cwd()



# ─────────────────────────────────────────
#  Datenstrukturen
# ─────────────────────────────────────────



@dataclass
class Valve:
    open: bool = True
    position: float = 1.0


    def effective(self) -> float:
        return self.position if self.open else 0.0



@dataclass
class NTankState(OdeState):
    h: List[float]


    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array(self.h, dtype=np.float64)


    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state.tolist())



# ─────────────────────────────────────────
#  Frame-Collector
# ─────────────────────────────────────────



class FrameCollector:
    def __init__(self, n_tanks: int, noise_std: float = 0.0) -> None:
        self.n_tanks = n_tanks
        self.noise_std = noise_std
        self.frames: List[pl.DataFrame] = []


    def clear(self) -> None:
        self.frames.clear()


    def collect_frame(self, ts, xs) -> pl.DataFrame:
        frame = pl.DataFrame({
            "t": ts,
            **{
                f"h{i+1}": [
                    max(0.0, x.h[i] + (
                        np.random.normal(0, self.noise_std)
                        if self.noise_std > 0 else 0.0
                    ))
                    for x in xs
                ]
                for i in range(self.n_tanks)
            },
        })
        self.frames.append(frame)
        return frame


    def concat(self) -> pl.DataFrame:
        return pl.concat(self.frames)



# ─────────────────────────────────────────
#  Topologie 1: Linear (Kette)
# ─────────────────────────────────────────



class NTankLinear(OdeSystem[NTankState]):


    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_between, Cout,
                 valves_between, valves_out, h_target, initial_state, initial_t=0.0):
        super().__init__(initial_t, initial_state)
        self.n = n_tanks
        self.A = A
        self.Qpmax = Qpmax
        self.Qf = Qf
        self.C_between = C_between
        self.Cout = Cout
        self.valves_between = valves_between
        self.valves_out = valves_out
        self.h_target = h_target


    @override
    def flow(self, t, state):
        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)


        Qp = np.array([
            self.Qpmax[i] if h[i] < self.h_target[i] else 0.0
            for i in range(n)
        ])


        Q_between = np.zeros(max(n - 1, 0))
        for i in range(n - 1):
            eff = self.valves_between[i].effective()
            Q_between[i] = self.C_between[i] * eff * np.sqrt(max(h[i] - h[i + 1], 0.0))


        Q_out = np.zeros(n)
        for i in range(n):
            eff = self.valves_out[i].effective()
            Q_out[i] = self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))


        Q_leak = np.array([self.Qf * np.sqrt(max(h[i], 0.0)) for i in range(n)])


        if n == 1:
            dhdt[0] = (Qp[0] - Q_leak[0] - Q_out[0]) / self.A[0]
        else:
            dhdt[0] = (Qp[0] - Q_leak[0] - Q_between[0] - Q_out[0]) / self.A[0]
            for i in range(1, n - 1):
                dhdt[i] = (Qp[i] + Q_between[i - 1] - Q_leak[i] - Q_between[i] - Q_out[i]) / self.A[i]
            dhdt[-1] = (Qp[-1] + Q_between[-1] - Q_leak[-1] - Q_out[-1]) / self.A[-1]


        for i in range(n):
            if h[i] <= 0.0 and dhdt[i] < 0.0:
                dhdt[i] = 0.0


        return dhdt



# ─────────────────────────────────────────
#  Topologie 2: Vollvermascht
# ─────────────────────────────────────────



class NTankFullyCoupled(OdeSystem[NTankState]):


    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_all, Cout,
                 valves_between, valves_out, h_target, initial_state, initial_t=0.0):
        super().__init__(initial_t, initial_state)
        self.n = n_tanks
        self.A = A
        self.Qpmax = Qpmax
        self.Qf = Qf
        self.C_all = C_all
        self.Cout = Cout
        self.valves_between = valves_between
        self.valves_out = valves_out
        self.h_target = h_target


    def _pair_index(self, i, j):
        n = self.n
        return i * (2 * n - i - 1) // 2 + (j - i - 1)


    @override
    def flow(self, t, state):
        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)


        for i in range(n):
            if h[i] < self.h_target[i]:
                dhdt[i] += self.Qpmax[i]


        for i in range(n):
            dhdt[i] -= self.Qf * np.sqrt(max(h[i], 0.0))


        for i in range(n):
            eff = self.valves_out[i].effective()
            dhdt[i] -= self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))


        for i in range(n):
            for j in range(i + 1, n):
                idx = self._pair_index(i, j)
                eff = self.valves_between[idx].effective()
                diff = h[i] - h[j]
                if diff > 0:
                    Q = self.C_all[idx] * eff * np.sqrt(diff)
                    dhdt[i] -= Q
                    dhdt[j] += Q
                elif diff < 0:
                    Q = self.C_all[idx] * eff * np.sqrt(-diff)
                    dhdt[j] -= Q
                    dhdt[i] += Q


        for i in range(n):
            dhdt[i] /= self.A[i]


        for i in range(n):
            if h[i] <= 0.0 and dhdt[i] < 0.0:
                dhdt[i] = 0.0


        return dhdt



# ─────────────────────────────────────────
#  Ziel-Füllstand konfigurieren
# ─────────────────────────────────────────



def configure_h_target(n_tanks: int) -> List[float]:
    print("\n── Ziel-Füllstände pro Tank ──────────────────")
    print("  (Pumpe stoppt sobald h[i] >= h_target[i])")


    use_same = input(
        "  Gleichen Zielwert für alle Tanks? (j/n) [Standard: j]: "
    ).strip().lower() != "n"


    if use_same:
        while True:
            raw = input("  Ziel-Füllstand für alle Tanks (m) [Standard: 0.5]: ").strip()
            if raw == "":
                val = 0.5
                break
            try:
                val = float(raw)
                if val > 0:
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"  → Alle Tanks: h_target = {val} m")
        return [val] * n_tanks


    h_target = []
    for i in range(n_tanks):
        while True:
            raw = input(f"  Tank {i + 1} – Ziel-Füllstand (m) [Standard: 0.5]: ").strip()
            if raw == "":
                h_target.append(0.5)
                break
            try:
                val = float(raw)
                if val > 0:
                    h_target.append(val)
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"    → Tank {i + 1}: h_target = {h_target[-1]} m")


    return h_target



def print_h_target_summary(h_target: List[float]) -> None:
    print("\n── Ziel-Füllstand-Übersicht ─────────────────")
    for i, ht in enumerate(h_target):
        print(f"  Tank {i + 1}: h_target = {ht:.3f} m")
    print("─────────────────────────────────────────────\n")



# ─────────────────────────────────────────
#  Tankgröße konfigurieren
# ─────────────────────────────────────────



def configure_tank_areas(n_tanks: int) -> List[float]:
    print("\n── Tankgröße (Querschnittsfläche A) ─────────")
    print("  (Größeres A = träger, kleineres A = reaktiver)")


    use_same = input(
        "  Gleiche Tankgröße für alle? (j/n) [Standard: j]: "
    ).strip().lower() != "n"


    if use_same:
        while True:
            raw = input("  A für alle Tanks (m²) [Standard: 0.0154]: ").strip()
            if raw == "":
                val = 0.0154
                break
            try:
                val = float(raw)
                if val > 0:
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"  → Alle Tanks: A = {val} m²")
        return [val] * n_tanks


    areas = []
    for i in range(n_tanks):
        while True:
            raw = input(f"  Tank {i + 1} – A (m²) [Standard: 0.0154]: ").strip()
            if raw == "":
                areas.append(0.0154)
                break
            try:
                val = float(raw)
                if val > 0:
                    areas.append(val)
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"    → Tank {i + 1}: A = {areas[-1]} m²")
    return areas



def print_tank_areas_summary(areas: List[float]) -> None:
    print("\n── Tankgröße-Übersicht ──────────────────────")
    for i, a in enumerate(areas):
        print(f"  Tank {i + 1}: A = {a:.6f} m²")
    print("─────────────────────────────────────────────\n")



# ─────────────────────────────────────────
#  Rauschen konfigurieren
# ─────────────────────────────────────────



def configure_noise() -> float:
    print("\n── Sensor-Rauschen ───────────────────────────")
    use_noise = input(
        "  Rauschen aktivieren? (j/n) [Standard: n]: "
    ).strip().lower() == "j"


    if not use_noise:
        print("  → Kein Rauschen")
        return 0.0


    while True:
        raw = input("  Standardabweichung σ (m) [Standard: 0.01]: ").strip()
        if raw == "":
            sigma = 0.01
            break
        try:
            sigma = float(raw)
            if sigma > 0:
                break
            print("  ⚠ Wert muss größer als 0 sein.")
        except ValueError:
            print("  ⚠ Bitte eine Zahl eingeben.")


    print(f"  → Rauschen aktiv: σ = {sigma} m")
    return sigma



# ─────────────────────────────────────────
#  Pumpen-Konfiguration
# ─────────────────────────────────────────



def configure_pumps(n_tanks: int) -> List[float]:
    pump_mode = prompt_choice(
        "\nPumpen-Modus – 'alle' (jeder Tank) oder 'manuell' (auswählen): ",
        ("alle", "manuell"),
    )


    if pump_mode == "alle":
        while True:
            raw = input("  Qpmax für alle Tanks (m³/s) [Standard: 0.01]: ").strip()
            if raw == "":
                qp = 1e-2
                break
            try:
                qp = float(raw)
                if qp > 0:
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"  → Alle {n_tanks} Tanks: Pumpe aktiv (Qpmax={qp})")
        return [qp] * n_tanks


    qp_list = []
    for i in range(n_tanks):
        has_pump = input(f"  Tank {i + 1} – Pumpe? (j/n) [Standard: j]: ").strip().lower() != "n"
        if has_pump:
            while True:
                raw = input(f"    Tank {i + 1} – Qpmax (m³/s) [Standard: 0.01]: ").strip()
                if raw == "":
                    qp = 1e-2
                    break
                try:
                    qp = float(raw)
                    if qp > 0:
                        break
                    print("    ⚠ Wert muss größer als 0 sein.")
                except ValueError:
                    print("    ⚠ Bitte eine Zahl eingeben.")
            qp_list.append(qp)
            print(f"    → Tank {i + 1}: Pumpe aktiv (Qpmax={qp})")
        else:
            qp_list.append(0.0)
            print(f"    → Tank {i + 1}: keine Pumpe")
    return qp_list



def print_pump_summary(qp_list: List[float]) -> None:
    print("\n── Pumpen-Übersicht ─────────────────────────")
    for i, qp in enumerate(qp_list):
        status = f"aktiv (Qpmax={qp})" if qp > 0 else "KEINE PUMPE"
        print(f"  Tank {i + 1}: {status}")
    print("─────────────────────────────────────────────\n")



# ─────────────────────────────────────────
#  System-Instanz erstellen
# ─────────────────────────────────────────



def build_system(topology: str, n_tanks: int,
                 valves_between: List[Valve], valves_out: List[Valve],
                 qp_list: List[float], h_target: List[float],
                 areas: List[float]):
    initial_state = NTankState(h=[0.0] * n_tanks)
    if topology == "linear":
        return NTankLinear(
            n_tanks=n_tanks, A=areas,
            Qpmax=qp_list, Qf=1e-4,
            C_between=[1.5938e-4] * (n_tanks - 1),
            Cout=[1.59640e-4] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            h_target=h_target,
            initial_state=initial_state,
        )
    else:
        n_between = n_tanks * (n_tanks - 1) // 2
        return NTankFullyCoupled(
            n_tanks=n_tanks, A=areas,
            Qpmax=qp_list, Qf=1e-4,
            C_all=[1.5e-4] * n_between,
            Cout=[1.59640e-4] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            h_target=h_target,
            initial_state=initial_state,
        )



# ─────────────────────────────────────────
#  Ventil-Konfiguration
# ─────────────────────────────────────────



def configure_valve(label: str) -> Valve:
    print(f"\n  Ventil {label}")
    is_open = input("    Geöffnet? (j/n) [Standard: j]: ").strip().lower() != "n"
    if is_open:
        while True:
            raw = input("    Öffnungsgrad (0.0–1.0) [Standard: 1.0]: ").strip()
            if raw == "":
                position = 1.0
                break
            try:
                position = float(raw)
                if 0.0 <= position <= 1.0:
                    break
                print("    ⚠ Wert muss zwischen 0.0 und 1.0 liegen.")
            except ValueError:
                print("    ⚠ Ungültige Eingabe, bitte eine Zahl eingeben.")
    else:
        position = 0.0
    return Valve(open=is_open, position=position)



def configure_valves_between_linear(n_tanks: int) -> List[Valve]:
    print("\n── Zwischenventile (Tank i → Tank i+1) ──")
    return [configure_valve(f"Tank {i + 1} → Tank {i + 2}") for i in range(n_tanks - 1)]



def configure_valves_between_coupled(n_tanks: int) -> List[Valve]:
    print("\n── Zwischenventile (alle Paare) ──")
    valves = []
    for i in range(n_tanks):
        for j in range(i + 1, n_tanks):
            valves.append(configure_valve(f"Tank {i + 1} ↔ Tank {j + 1}"))
    return valves



def configure_valves_out(n_tanks: int) -> List[Valve]:
    print("\n── Auslassventile ──")
    return [configure_valve(f"Auslass Tank {i + 1}") for i in range(n_tanks)]



def print_valve_summary(valves_between: List[Valve], valves_out: List[Valve],
                        topology: str, n_tanks: int) -> None:
    print("\n── Ventil-Übersicht ─────────────────────────")
    if topology == "linear":
        for i, v in enumerate(valves_between):
            print(f"  Zwischenventil Tank {i + 1}→{i + 2}: "
                  f"{'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
    else:
        idx = 0
        for i in range(n_tanks):
            for j in range(i + 1, n_tanks):
                v = valves_between[idx]
                print(f"  Zwischenventil Tank {i + 1}↔{j + 1}: "
                      f"{'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
                idx += 1
    for i, v in enumerate(valves_out):
        print(f"  Auslassventil  Tank {i + 1}: "
              f"{'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
    print("─────────────────────────────────────────────\n")



# ─────────────────────────────────────────
#  Modell-Auswahl
# ─────────────────────────────────────────



def build_learner(model_choice: str) -> RiverLearner:
    if model_choice == "mlp":
        return RiverLearner(
            model=neural_net.MLPRegressor(
                hidden_dims=(32, 16),
                activations=(
                    neural_net.activations.ReLU,
                    neural_net.activations.ReLU,
                    neural_net.activations.Identity,
                ),
                optimizer=optim.Adam(lr=1e-3),
                seed=42,
            )
        )
    else:
        return RiverLearner(
            model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5)
        )



# ─────────────────────────────────────────
#  Metriken aus Report lesen
# ─────────────────────────────────────────



def extract_metric(report_obj, metric_name: str, target: str) -> float:
    try:
        return float(report_obj.metrics[metric_name][target])
    except (AttributeError, KeyError, TypeError):
        pass


    for line in str(report_obj).splitlines():
        if metric_name in line and target in line:
            parts = line.split()
            for part in reversed(parts):
                try:
                    return float(part)
                except ValueError:
                    continue


    logger.warning("Metrik '%s' für '%s' nicht gefunden.", metric_name, target)
    return 0.0



# ─────────────────────────────────────────
#  Plot 1: Füllstände
# ─────────────────────────────────────────



def plot_sensor_data(df: pl.DataFrame, n_tanks: int, topology: str,
                     h_target: List[float], noise_std: float) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    t = df["t"].to_numpy()
    topo_label = "Linear (Kette)" if topology == "linear" else "Fully Coupled"
    noise_label = f"  |  Rauschen σ = {noise_std} m" if noise_std > 0 else "  |  kein Rauschen"


    fig, axes = plt.subplots(n_tanks, 1, figsize=(10, 3 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]


    fig.suptitle(
        f"Füllstände – Topologie: {topo_label}{noise_label}",
        fontsize=13, fontweight="bold"
    )


    for i in range(n_tanks):
        h_vals = df[f"h{i + 1}"].to_numpy()
        c = colors[i % len(colors)]
        axes[i].plot(t, h_vals, color=c, linewidth=1.8)
        axes[i].fill_between(t, h_vals, alpha=0.12, color=c)
        axes[i].axhline(0, color="gray", linewidth=0.8, linestyle="--")
        axes[i].axhline(
            h_target[i], color="red", linewidth=1.2,
            linestyle="--", label=f"h_target = {h_target[i]:.2f} m"
        )
        axes[i].legend(fontsize=8, loc="upper right")
        axes[i].set_ylabel("Füllstand h [m]", fontsize=9)
        axes[i].set_title(f"Tank {i + 1}", fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.4)


    axes[-1].set_xlabel("Zeit t [s]", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensor_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ sensor_plot.png gespeichert in {OUTPUT_DIR}")



# ─────────────────────────────────────────
#  Plot 2: Modellfehler
# ─────────────────────────────────────────



def plot_learning_results(results: dict, n_tanks: int, model_label: str) -> None:
    x = np.arange(n_tanks)
    width = 0.35
    mae_vals = [results[i]["MAE"] for i in range(n_tanks)]
    mse_vals = [results[i]["MSE"] for i in range(n_tanks)]


    fig, ax = plt.subplots(figsize=(max(6, n_tanks * 2), 5))
    bars1 = ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + width / 2, mse_vals, width, label="MSE", color="#d62728", alpha=0.85)


    for bar in (*bars1, *bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.6f}",          # ← FIX: war :.4f
            ha="center", va="bottom", fontsize=8,
        )


    ax.set_xlabel("Tank")
    ax.set_ylabel("Fehler")
    ax.set_title(f"Modellfehler pro Tank – {model_label}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tank {i + 1}" for i in range(n_tanks)])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ learning_results.png gespeichert in {OUTPUT_DIR}")



# ─────────────────────────────────────────
#  Plot 3: Vorhersage vs. Realität
# ─────────────────────────────────────────



def plot_predictions(predictions_data: dict, n_tanks: int, model_label: str) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


    fig, axes = plt.subplots(n_tanks, 1, figsize=(12, 3 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]


    fig.suptitle(f"Vorhersage vs. Realität – {model_label}", fontsize=13, fontweight="bold")


    for i in range(n_tanks):
        actual    = predictions_data[i]["actual"]
        predicted = predictions_data[i]["predicted"]
        idx = np.arange(len(actual))
        c = colors[i % len(colors)]


        axes[i].plot(idx, actual, color=c, linewidth=1.5, label="Realität")
        axes[i].plot(idx, predicted, color="black", linewidth=1.0,
                     linestyle="--", alpha=0.75, label="Vorhersage")
        axes[i].set_ylabel("Füllstand h [m]", fontsize=9)
        axes[i].set_title(f"Tank {i + 1}", fontsize=10)
        axes[i].legend(fontsize=8, loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.4)


    axes[-1].set_xlabel("Test-Sample Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ predictions_plot.png gespeichert in {OUTPUT_DIR}")



# ─────────────────────────────────────────
#  Eingabe-Hilfsfunktionen
# ─────────────────────────────────────────



def prompt_choice(prompt: str, valid: tuple) -> str:
    while True:
        val = input(prompt).strip().lower()
        if val in valid:
            return val
        print(f"  ⚠ Bitte eine der folgenden Optionen eingeben: {valid}")



def prompt_positive_int(prompt: str) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if val > 0:
                return val
            print("  ⚠ Wert muss größer als 0 sein.")
        except ValueError:
            print("  ⚠ Bitte eine ganze Zahl eingeben.")



# ─────────────────────────────────────────
#  Hauptprogramm
# ─────────────────────────────────────────



def main() -> None:
    flowcean.cli.initialize()


    topology = prompt_choice(
        "Topologie wählen – 'linear' (Kette) oder 'coupled' (jeder mit jedem): ",
        ("linear", "coupled"),
    )
    n_tanks = prompt_positive_int("Wie viele Tanks sollen simuliert werden? ")
    n_samples = prompt_positive_int(
        "Wie viele Samples sollen gesammelt werden? [Empfehlung: 3000] "
    )
    model_choice = prompt_choice(
        "\nModell wählen – 'hoeffding' (HoeffdingTree) oder 'mlp' (Neuronales Netz): ",
        ("hoeffding", "mlp"),
    )


    model_label = (
        "HoeffdingTreeRegressor" if model_choice == "hoeffding"
        else "MLP Neuronales Netz (river)"
    )
    print(f"→ Modell: {model_label}\n")


    qp_list = configure_pumps(n_tanks)
    print_pump_summary(qp_list)


    h_target = configure_h_target(n_tanks)
    print_h_target_summary(h_target)


    areas = configure_tank_areas(n_tanks)
    print_tank_areas_summary(areas)


    noise_std = configure_noise()


    use_custom = input(
        "Ventile manuell konfigurieren? (j/n) [Standard: n, alle voll offen]: "
    ).strip().lower()


    if use_custom == "j":
        valves_between = (
            configure_valves_between_linear(n_tanks)
            if topology == "linear"
            else configure_valves_between_coupled(n_tanks)
        )
        valves_out = configure_valves_out(n_tanks)
    else:
        n_between = n_tanks - 1 if topology == "linear" else n_tanks * (n_tanks - 1) // 2
        valves_between = [Valve(open=True, position=1.0) for _ in range(n_between)]
        valves_out = [Valve(open=True, position=1.0) for _ in range(n_tanks)]


    print_valve_summary(valves_between, valves_out, topology, n_tanks)


    topo_label = "Linear" if topology == "linear" else "Fully Coupled"
    print(f"→ Topologie: {topo_label} | Tanks: {n_tanks} | "
          f"Zwischenventile: {len(valves_between)} | Auslassventile: {n_tanks}\n")


    collector_preview = FrameCollector(n_tanks, noise_std=0.0)
    preview_system = build_system(
        topology, n_tanks, valves_between, valves_out, qp_list, h_target, areas
    )
    data_preview = OdeEnvironment(
        preview_system, dt=1.0, map_to_dataframe=collector_preview.collect_frame
    )
    print("Vorschau auf die ersten 20 Schritte:")
    print(collect(data_preview, 20))


    collector = FrameCollector(n_tanks, noise_std=noise_std)
    main_system = build_system(
        topology, n_tanks, valves_between, valves_out, qp_list, h_target, areas
    )
    data_incremental = OdeEnvironment(
        main_system, dt=1.0, map_to_dataframe=collector.collect_frame
    )
    df_flowcean = collect(data_incremental, n_samples)
    df_plot = collector.concat()


    plot_sensor_data(df_plot, n_tanks, topology, h_target, noise_std)


    data = df_flowcean | SlidingWindow(window_size=3)
    inputs = [f"h{i + 1}_{step}" for i in range(n_tanks) for step in range(2)]
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)


    n_windows  = len(df_plot) - 2
    n_train    = int(n_windows * 0.8)
    test_start = n_train


    results          = {}
    predictions_data = {}


    for tank_idx in range(1, n_tanks + 1):
        target_name = f"h{tank_idx}_2"
        print(f"\n--- Learning {target_name} ({model_label}) ---")


        train_env = StreamingOfflineEnvironment(train, batch_size=1)
        learner   = build_learner(model_choice)


        t_start    = datetime.now(tz=timezone.utc)
        model      = learn_incremental(train_env, learner, inputs, [target_name])
        elapsed_ms = round((datetime.now(tz=timezone.utc) - t_start).total_seconds() * 1000, 1)
        print(f"Learning {target_name} took {elapsed_ms} ms")


        report = evaluate_offline(
            model, test, inputs, [target_name],
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print(f"Report for {target_name}:")
        print(report)


        results[tank_idx - 1] = {
            "MAE": extract_metric(report, "MeanAbsoluteError", target_name),
            "MSE": extract_metric(report, "MeanSquaredError", target_name),
        }


        river_model  = getattr(model, "model")   # ← FIX: war model.model
        actuals_list = []
        preds_list   = []


        for j in range(test_start, n_windows):
            x = {
                f"h{ti + 1}_{step}": float(df_plot[f"h{ti + 1}"][j + step])
                for ti in range(n_tanks)
                for step in range(2)
            }
            pred = river_model.predict_one(x)
            actuals_list.append(float(df_plot[f"h{tank_idx}"][j + 2]))
            preds_list.append(float(pred) if pred is not None else 0.0)


        predictions_data[tank_idx - 1] = {
            "actual":    np.array(actuals_list),
            "predicted": np.array(preds_list),
        }


    plot_learning_results(results, n_tanks, model_label)
    plot_predictions(predictions_data, n_tanks, model_label)



if __name__ == "__main__":
    main()