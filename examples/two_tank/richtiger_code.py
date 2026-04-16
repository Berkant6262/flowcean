# ═══════════════════════════════════════════════════════════════════════════════
#  N-Tank-Simulation – Incremental & Offline Machine Learning (kombiniert)
#  -----------------------------------------------------------------------
#  Beim Start wird der Lernmodus gewählt:
#    • incremental – River-Lerner (HoeffdingTree oder MLP), Sample-für-Sample
#    • offline     – sklearn RegressionTree oder PyTorch MLP, Batch-Training
#
#  Gemeinsame Basis: ODE-Simulation, Ventil-/Pumpen-/Rauschkonfiguration
#  Abhängigkeiten: flowcean, river, polars, numpy, matplotlib, torch, sklearn
# ═══════════════════════════════════════════════════════════════════════════════

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.utils.data as _tud
from numpy.typing import NDArray
from river import neural_net, optim, tree
from typing_extensions import Self, override

import flowcean.cli
from flowcean.core import evaluate_offline, learn_incremental, learn_offline
from flowcean.ode import OdeEnvironment, OdeState, OdeSystem
from flowcean.polars import (
    SlidingWindow,
    StreamingOfflineEnvironment,
    TrainTestSplit,
    collect,
)
from flowcean.river import RiverLearner
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError, RegressionTree
from flowcean.torch import LightningLearner, MultilayerPerceptron
from flowcean.utils.random import initialize_random

# ── Windows-Fix: DataLoader Worker deaktivieren ───────────────────────────────
# Verhindert Deadlock beim Spawnen von PyTorch DataLoader-Prozessen auf Windows.
_orig_dl_init = _tud.DataLoader.__init__

def _patched_dl_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["persistent_workers"] = False  # ← muss False sein wenn num_workers=0
    _orig_dl_init(self, *args, **kwargs)

_tud.DataLoader.__init__ = _patched_dl_init
# ─────────────────────────────────────────────────────────────────────────────

sys.setrecursionlimit(10000)
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path.cwd()


# ══════════════════════════════════════════════════════════════════════════════
#  GEMEINSAME DATENSTRUKTUREN & KLASSEN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Valve:
    """
    Repräsentiert ein einzelnes Ventil im Tanksystem.

    Attribute:
        open     – True = Ventil geöffnet, False = geschlossen (kein Durchfluss).
        position – Öffnungsgrad zwischen 0.0 (vollständig zu) und 1.0 (vollständig auf).
    """
    open: bool = True
    position: float = 1.0

    def effective(self) -> float:
        """Gibt den effektiven Öffnungsgrad zurück (0.0 wenn geschlossen)."""
        return self.position if self.open else 0.0


@dataclass
class NTankState(OdeState):
    """
    Zustandsvektor des N-Tank-Systems.
    h – Liste der aktuellen Füllstände [m] für jeden Tank.
    """
    h: List[float]

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array(self.h, dtype=np.float64)

    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state.tolist())


class FrameCollector:
    """
    Sammelt ODE-Ausgaben schrittweise als Polars-DataFrames und
    ermöglicht optional das Überlagern mit Gaußschem Sensorrauschen.
    """

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
    """
    ODE-System für eine lineare (kaskadierende) Tanktopologie.
    Tank 1 → Tank 2 → ... → Tank N (Schwerkraftfluss nur in eine Richtung).
    """

    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_between, Cout,
                 valves_between, valves_out, h_target,
                 initial_state, initial_t=0.0):
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
            if h[i] >= self.h_target[i] and dhdt[i] > 0.0:
                dhdt[i] = 0.0

        return dhdt


# ─────────────────────────────────────────
#  Topologie 2: Vollvermascht
# ─────────────────────────────────────────

class NTankFullyCoupled(OdeSystem[NTankState]):
    """
    ODE-System für eine vollvermaschte Tanktopologie.
    Jedes Tankpaar (i, j) ist bidirektional verbunden.
    """

    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_all, Cout,
                 valves_between, valves_out, h_target,
                 initial_state, initial_t=0.0):
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
            if h[i] >= self.h_target[i] and dhdt[i] > 0.0:
                dhdt[i] = 0.0

        return dhdt


# ══════════════════════════════════════════════════════════════════════════════
#  GEMEINSAME KONFIGURATIONSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def configure_h_target(n_tanks: int) -> List[float]:
    """Interaktive Konfiguration der Ziel-Füllstände h_target pro Tank."""
    print("\n── Ziel-Füllstände pro Tank ──────────────────")
    print("  (Pumpe stoppt & Tank kann nicht überlaufen sobald h[i] >= h_target[i])")

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


def configure_tank_areas(n_tanks: int) -> List[float]:
    """Interaktive Konfiguration der Querschnittsflächen A [m²] pro Tank."""
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


def configure_noise() -> float:
    """Interaktive Konfiguration des Gaußschen Sensorrauschens."""
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


def configure_pumps(n_tanks: int) -> List[float]:
    """Interaktive Konfiguration der maximalen Pumpenleistung Qpmax [m³/s] pro Tank."""
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


def build_system(topology: str, n_tanks: int,
                 valves_between: List[Valve], valves_out: List[Valve],
                 qp_list: List[float], h_target: List[float],
                 areas: List[float]):
    """Factory-Funktion: Erzeugt die passende ODE-System-Instanz je nach Topologie."""
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


def configure_valve(label: str) -> Valve:
    """Interaktive Konfiguration eines einzelnen Ventils."""
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


def extract_metric(report_obj, metric_name: str, target: str) -> float:
    """Extrahiert einen numerischen Metrikwert aus dem flowcean-Evaluierungsbericht."""
    try:
        return float(report_obj.metrics[metric_name][target])
    except (AttributeError, KeyError, TypeError):
        pass

    for line in str(report_obj).splitlines():
        if metric_name in line and target in line:
            candidates = []
            for part in line.split():
                try:
                    candidates.append(float(part))
                except ValueError:
                    continue
            if candidates:
                return candidates[-1]

    logger.warning(
        "Metrik '%s' für Target '%s' nicht gefunden – Report:\n%s",
        metric_name, target, str(report_obj),
    )
    return 0.0


def plot_sensor_data(df: pl.DataFrame, n_tanks: int, topology: str,
                     h_target: List[float], noise_std: float) -> None:
    """Visualisiert den zeitlichen Verlauf der Füllstände aller Tanks."""
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
            linestyle="--", label=f"h_target = {h_target[i]:.2f} m (Pumpe aus + Obergrenze)"
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


def prompt_choice(prompt: str, valid: tuple) -> str:
    """Fordert den Benutzer zur Eingabe einer von mehreren gültigen Optionen auf."""
    while True:
        val = input(prompt).strip().lower()
        if val in valid:
            return val
        print(f"  ⚠ Bitte eine der folgenden Optionen eingeben: {valid}")


def prompt_positive_int(prompt: str) -> int:
    """Fordert den Benutzer zur Eingabe einer positiven ganzen Zahl auf."""
    while True:
        try:
            val = int(input(prompt).strip())
            if val > 0:
                return val
            print("  ⚠ Wert muss größer als 0 sein.")
        except ValueError:
            print("  ⚠ Bitte eine ganze Zahl eingeben.")


# ══════════════════════════════════════════════════════════════════════════════
#  INCREMENTAL-SPEZIFISCHE FUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def build_learner(model_choice: str) -> RiverLearner:
    """
    Erzeugt den gewählten inkrementellen River-Lerner als flowcean-RiverLearner.

    Optionen:
      'hoeffding' – HoeffdingTreeRegressor: lernt schrittweise durch statistischen
                   Hoeffding-Bound-Test. grace_period=50, max_depth=5.
      'mlp'       – MLPRegressor (river): 2 Hidden Layer (32, 16), ReLU, Adam lr=1e-3.
    """
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


def extract_river_model(trained_model):
    """
    Robuster Zugriff auf das zugrunde liegende River-Modell aus dem flowcean-Wrapper.
    Probiert bekannte Attributnamen der Reihe nach: model, _model, learner, _learner.
    """
    for attr in ("model", "_model", "learner", "_learner"):
        candidate = getattr(trained_model, attr, None)
        if candidate is not None:
            return candidate

    logger.warning(
        "Kein bekanntes Wrapper-Attribut gefunden. "
        "Nutze trained_model direkt als River-Modell."
    )
    return trained_model


def plot_learning_results_incremental(results: dict, n_tanks: int,
                                      model_label: str) -> None:
    """Visualisiert MAE und MSE pro Tank als gruppiertes Balkendiagramm (Incremental)."""
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
            f"{bar.get_height():.6f}",
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


def plot_predictions_incremental(predictions_data: dict, n_tanks: int,
                                  model_label: str) -> None:
    """Vergleicht vorhergesagte Füllstände mit den tatsächlichen Testwerten (Incremental)."""
    colors_actual = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(n_tanks, 1, figsize=(10, 3 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]

    fig.suptitle(f"Vorhersage vs. Realität – {model_label}",
                 fontsize=13, fontweight="bold")

    for i in range(n_tanks):
        actual = predictions_data[i]["actual"]
        predicted = predictions_data[i]["predicted"]
        idx = np.arange(len(actual))
        ca = colors_actual[i % len(colors_actual)]

        axes[i].plot(idx, actual, color=ca, linewidth=1.5, label="Realität")
        axes[i].plot(idx, predicted, color="black", linewidth=1.0,
                     linestyle="--", alpha=0.75, label="Vorhersage")
        axes[i].set_title(f"Tank {i + 1}", fontsize=10)
        axes[i].set_ylabel("Füllstand h [m]", fontsize=9)
        axes[i].legend(fontsize=8, loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Test-Sample Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ predictions_plot.png gespeichert in {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  OFFLINE-SPEZIFISCHE FUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def build_learners(model_choice: str, n_outputs: int) -> List[Any]:
    """
    Gibt eine Liste von Offline-Learnern zurück.

    Optionen:
      'tree'  → RegressionTree (sklearn, max_depth=5)
      'mlp'   → LightningLearner mit MultilayerPerceptron (PyTorch, 32-16, LeakyReLU)
      'beide' → Beide Lerner werden nacheinander trainiert und verglichen.
    """
    learners = []
    if model_choice in ("tree", "beide"):
        learners.append(RegressionTree(max_depth=5))
    if model_choice in ("mlp", "beide"):
        learners.append(
            LightningLearner(
                module=MultilayerPerceptron(
                    learning_rate=1e-3,
                    output_size=n_outputs,
                    hidden_dimensions=[32, 16],
                    activation_function=torch.nn.LeakyReLU,
                ),
                max_epochs=1000,
            )
        )
    return learners


def learner_label(learner: Any) -> str:
    """Gibt einen lesbaren Namen für den übergebenen Lerner zurück."""
    name = type(learner).__name__
    if name == "RegressionTree":
        return "RegressionTree (sklearn)"
    if name == "LightningLearner":
        return "MLP Neuronales Netz (PyTorch)"
    return name


def plot_learning_results_offline(all_results: dict, n_tanks: int) -> None:
    """
    Visualisiert MAE und MSE pro Tank als gruppiertes Balkendiagramm (Offline).
    Unterstützt mehrere Lerner gleichzeitig.
    """
    learner_names = list(all_results.keys())
    x = np.arange(n_tanks)
    width = 0.35 / max(len(learner_names), 1)
    colors_mae = ["#1f77b4", "#2ca02c", "#9467bd"]
    colors_mse = ["#d62728", "#ff7f0e", "#e377c2"]

    fig, ax = plt.subplots(figsize=(max(6, n_tanks * 2 * len(learner_names)), 5))

    for li, name in enumerate(learner_names):
        results = all_results[name]
        mae_vals = [results[i]["MAE"] for i in range(n_tanks)]
        mse_vals = [results[i]["MSE"] for i in range(n_tanks)]
        offset = (li - len(learner_names) / 2 + 0.5) * width * 2

        bars1 = ax.bar(x + offset - width / 2, mae_vals, width,
                       label=f"MAE – {name}", color=colors_mae[li % 3], alpha=0.85)
        bars2 = ax.bar(x + offset + width / 2, mse_vals, width,
                       label=f"MSE – {name}", color=colors_mse[li % 3], alpha=0.85)

        for bar in (*bars1, *bars2):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.6f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Tank")
    ax.set_ylabel("Fehler")
    ax.set_title("Modellfehler pro Tank – Offline Learner Vergleich")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tank {i + 1}" for i in range(n_tanks)])
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ learning_results.png gespeichert in {OUTPUT_DIR}")


def plot_predictions_offline(all_predictions: dict, n_tanks: int) -> None:
    """
    Vergleicht vorhergesagte Füllstände mit den tatsächlichen Testwerten (Offline).
    Grid: n_tanks × n_learner Subplots.
    """
    colors_actual = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    learner_names = list(all_predictions.keys())

    fig, axes = plt.subplots(
        n_tanks, len(learner_names),
        figsize=(7 * len(learner_names), 3 * n_tanks),
        sharex="col", sharey="row",
        squeeze=False,
    )

    fig.suptitle("Vorhersage vs. Realität – Offline Learner",
                 fontsize=13, fontweight="bold")

    for li, name in enumerate(learner_names):
        preds = all_predictions[name]
        for i in range(n_tanks):
            actual    = preds[i]["actual"]
            predicted = preds[i]["predicted"]
            idx = np.arange(len(actual))
            ca = colors_actual[i % len(colors_actual)]

            axes[i][li].plot(idx, actual, color=ca, linewidth=1.5, label="Realität")
            axes[i][li].plot(idx, predicted, color="black", linewidth=1.0,
                             linestyle="--", alpha=0.75, label="Vorhersage")
            axes[i][li].set_title(f"Tank {i + 1} – {name}", fontsize=9)
            axes[i][li].legend(fontsize=7, loc="upper right")
            axes[i][li].grid(True, linestyle="--", alpha=0.4)
            if li == 0:
                axes[i][li].set_ylabel("Füllstand h [m]", fontsize=9)

    for li in range(len(learner_names)):
        axes[-1][li].set_xlabel("Test-Sample Index", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ predictions_plot.png gespeichert in {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  LERNMODUS: INCREMENTAL
# ══════════════════════════════════════════════════════════════════════════════

def run_incremental(
    topology: str, n_tanks: int, n_samples: int,
    valves_between: List[Valve], valves_out: List[Valve],
    qp_list: List[float], h_target: List[float], areas: List[float],
    noise_std: float, model_choice: str,
) -> None:
    """
    Führt den inkrementellen Lernmodus aus.

    Ablauf:
      1. Vorschau-Simulation (20 Schritte, kein Rauschen)
      2. Hauptsimulation mit dt=1.0 s
      3. SlidingWindow(3) Feature Engineering
      4. Train/Test-Split 80/20, shuffle=False (Zeitreihe bleibt erhalten)
      5. Pro Tank: inkrementelles Training mit learn_incremental + Evaluierung
      6. Plots: Füllstände, Fehlerbalken, Vorhersage vs. Realität
    """
    model_label = (
        "HoeffdingTreeRegressor" if model_choice == "hoeffding"
        else "MLP Neuronales Netz (river)"
    )
    print(f"→ Modell: {model_label}\n")

    # ── Vorschau-Simulation (ohne Rauschen) ───────────────────────────────────
    collector_preview = FrameCollector(n_tanks, noise_std=0.0)
    preview_system = build_system(
        topology, n_tanks, valves_between, valves_out, qp_list, h_target, areas
    )
    data_preview = OdeEnvironment(
        preview_system, dt=1.0, map_to_dataframe=collector_preview.collect_frame
    )
    print("Vorschau auf die ersten 20 Schritte:")
    print(collect(data_preview, 20))

    # ── Hauptsimulation ───────────────────────────────────────────────────────
    collector = FrameCollector(n_tanks, noise_std=noise_std)
    main_system = build_system(
        topology, n_tanks, valves_between, valves_out, qp_list, h_target, areas
    )
    data_incremental = OdeEnvironment(
        main_system, dt=1.0, map_to_dataframe=collector.collect_frame
    )
    df_flowcean = collect(data_incremental, n_samples)
    df_plot     = collector.concat()

    plot_sensor_data(df_plot, n_tanks, topology, h_target, noise_std)

    # ── Feature Engineering via SlidingWindow ─────────────────────────────────
    data   = df_flowcean | SlidingWindow(window_size=3)
    inputs = [f"h{i + 1}_{step}" for i in range(n_tanks) for step in range(2)]
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    n_windows  = len(df_plot) - 2
    n_train    = int(n_windows * 0.8)
    test_start = n_train

    results          = {}
    predictions_data = {}

    # ── Training, Evaluierung & Vorhersage pro Tank ───────────────────────────
    for tank_idx in range(1, n_tanks + 1):
        target_name = f"h{tank_idx}_2"
        print(f"\n--- Learning {target_name} ({model_label}) ---")

        train_env = StreamingOfflineEnvironment(train, batch_size=1)
        learner   = build_learner(model_choice)

        t_start       = datetime.now(tz=timezone.utc)
        trained_model = learn_incremental(train_env, learner, inputs, [target_name])
        elapsed_ms    = round((datetime.now(tz=timezone.utc) - t_start).total_seconds() * 1000, 1)
        print(f"Learning {target_name} took {elapsed_ms} ms")

        report = evaluate_offline(
            trained_model, test, inputs, [target_name],
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print(f"Report for {target_name}:")
        print(report)

        results[tank_idx - 1] = {
            "MAE": extract_metric(report, "MeanAbsoluteError", target_name),
            "MSE": extract_metric(report, "MeanSquaredError", target_name),
        }

        river_model  = extract_river_model(trained_model)
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

    # ── Ergebnisvisualisierung ────────────────────────────────────────────────
    plot_learning_results_incremental(results, n_tanks, model_label)
    plot_predictions_incremental(predictions_data, n_tanks, model_label)


# ══════════════════════════════════════════════════════════════════════════════
#  LERNMODUS: OFFLINE
# ══════════════════════════════════════════════════════════════════════════════

def run_offline(
    topology: str, n_tanks: int, n_samples: int,
    valves_between: List[Valve], valves_out: List[Valve],
    qp_list: List[float], h_target: List[float], areas: List[float],
    noise_std: float, model_choice: str,
) -> None:
    """
    Führt den Offline-Lernmodus aus.

    Ablauf:
      1. Simulation mit dt=0.1 s
      2. SlidingWindow(3) Feature Engineering
      3. Train/Test-Split 80/20, shuffle=True (zufällige Mischung erlaubt)
      4. Multi-Output-Training: ein Modell lernt ALLE Tank-Outputs gleichzeitig
      5. Evaluierung mit MAE und MSE auf allen Outputs
      6. Plots: Füllstände, Fehlerbalken-Vergleich, Vorhersage vs. Realität
    """
    # ── Simulation ────────────────────────────────────────────────────────────
    collector = FrameCollector(n_tanks, noise_std=noise_std)
    system = build_system(
        topology, n_tanks, valves_between, valves_out, qp_list, h_target, areas
    )
    data_env = OdeEnvironment(
        system, dt=0.1, map_to_dataframe=collector.collect_frame
    )

    # ── Feature Engineering ───────────────────────────────────────────────────
    data = collect(data_env, n_samples) | SlidingWindow(window_size=3)
    df_plot = collector.concat()

    plot_sensor_data(df_plot, n_tanks, topology, h_target, noise_std)

    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)

    inputs  = [f"h{i + 1}_{step}" for i in range(n_tanks) for step in range(2)]
    outputs = [f"h{i + 1}_2" for i in range(n_tanks)]

    print(f"\n→ Inputs:  {inputs}")
    print(f"→ Outputs: {outputs}\n")

    learners = build_learners(model_choice, n_outputs=len(outputs))

    all_results     = {}
    all_predictions = {}

    n_windows  = len(df_plot) - 2
    test_start = int(n_windows * 0.8)

    # ── Training, Evaluierung & Vorhersage pro Lerner ─────────────────────────
    for learner in learners:
        label = learner_label(learner)
        print(f"\n{'=' * 60}")
        print(f"  Learner: {label}")
        print(f"{'=' * 60}")

        t_start = datetime.now(tz=timezone.utc)
        model = learn_offline(train, learner, inputs, outputs)
        elapsed_ms = round(
            (datetime.now(tz=timezone.utc) - t_start).total_seconds() * 1000, 1
        )
        print(f"Learning took {elapsed_ms} ms")

        report = evaluate_offline(
            model, test, inputs, outputs,
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print(report)

        results = {}
        for i in range(n_tanks):
            target_name = f"h{i + 1}_2"
            results[i] = {
                "MAE": extract_metric(report, "MeanAbsoluteError", target_name),
                "MSE": extract_metric(report, "MeanSquaredError", target_name),
            }
        all_results[label] = results

        predictions_data = {i: {"actual": [], "predicted": []} for i in range(n_tanks)}

        for j in range(test_start, n_windows):
            x = {
                f"h{ti + 1}_{step}": float(df_plot[f"h{ti + 1}"][j + step])
                for ti in range(n_tanks)
                for step in range(2)
            }
            pred_df = model.predict(pl.DataFrame([x]).lazy()).collect()

            for i in range(n_tanks):
                target_col = f"h{i + 1}_2"
                pred_val = (
                    float(pred_df[target_col][0])
                    if target_col in pred_df.columns
                    else 0.0
                )
                predictions_data[i]["actual"].append(float(df_plot[f"h{i + 1}"][j + 2]))
                predictions_data[i]["predicted"].append(pred_val)

        for i in range(n_tanks):
            predictions_data[i]["actual"]    = np.array(predictions_data[i]["actual"])
            predictions_data[i]["predicted"] = np.array(predictions_data[i]["predicted"])

        all_predictions[label] = predictions_data

    # ── Ergebnisvisualisierung ────────────────────────────────────────────────
    plot_learning_results_offline(all_results, n_tanks)
    plot_predictions_offline(all_predictions, n_tanks)


# ══════════════════════════════════════════════════════════════════════════════
#  HAUPTPROGRAMM
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Einstiegspunkt – steuert den gesamten Ablauf:

    1. Flowcean-Infrastruktur initialisieren
    2. Lernmodus abfragen: 'incremental' oder 'offline'
    3. Gemeinsame Konfigurationsphase (Topologie, Tanks, Samples, Modell,
       Pumpen, Ziel-Füllstände, Tankflächen, Rauschen, Ventile)
    4. Delegation an run_incremental() oder run_offline()
    """
    flowcean.cli.initialize()

    # ── Modus-Auswahl (einmalige Abfrage am Anfang) ───────────────────────────
    print("\n" + "═" * 60)
    print("  N-Tank-Simulation – Lernmodus")
    print("═" * 60)
    mode = prompt_choice(
        "Lernmodus wählen – 'incremental' (River) oder 'offline' (PyTorch/sklearn): ",
        ("incremental", "offline"),
    )
    print(f"→ Modus: {mode.upper()}\n")

    if mode == "offline":
        # Reproduzierbaren Seed nur für Offline-Modus setzen (PyTorch braucht es).
        initialize_random(seed=42)

    # ── Gemeinsame Konfigurationsphase ────────────────────────────────────────
    topology = prompt_choice(
        "Topologie wählen – 'linear' (Kette) oder 'coupled' (jeder mit jedem): ",
        ("linear", "coupled"),
    )
    n_tanks = prompt_positive_int("Wie viele Tanks sollen simuliert werden? ")

    if mode == "incremental":
        n_samples = prompt_positive_int(
            "Wie viele Samples sollen gesammelt werden? [Empfehlung: 3000] "
        )
        model_choice = prompt_choice(
            "\nModell wählen – 'hoeffding' (HoeffdingTree) oder 'mlp' (River MLP): ",
            ("hoeffding", "mlp"),
        )
    else:
        n_samples = prompt_positive_int(
            "Wie viele Samples sollen gesammelt werden? [Empfehlung: 250] "
        )
        model_choice = prompt_choice(
            "\nModell wählen – 'tree' (RegressionTree), 'mlp' (PyTorch MLP) oder 'beide': ",
            ("tree", "mlp", "beide"),
        )

    qp_list  = configure_pumps(n_tanks)
    print_pump_summary(qp_list)

    h_target = configure_h_target(n_tanks)
    print_h_target_summary(h_target)

    areas    = configure_tank_areas(n_tanks)
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
        valves_out     = [Valve(open=True, position=1.0) for _ in range(n_tanks)]

    print_valve_summary(valves_between, valves_out, topology, n_tanks)

    topo_label = "Linear" if topology == "linear" else "Fully Coupled"
    print(f"→ Topologie: {topo_label} | Tanks: {n_tanks} | "
          f"Zwischenventile: {len(valves_between)} | Auslassventile: {n_tanks}\n")

    # ── Lernmodus ausführen ───────────────────────────────────────────────────
    if mode == "incremental":
        run_incremental(
            topology, n_tanks, n_samples,
            valves_between, valves_out,
            qp_list, h_target, areas, noise_std, model_choice,
        )
    else:
        run_offline(
            topology, n_tanks, n_samples,
            valves_between, valves_out,
            qp_list, h_target, areas, noise_std, model_choice,
        )


if __name__ == "__main__":
    main()