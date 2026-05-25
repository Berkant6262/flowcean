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
from flowcean.core import learn_incremental, learn_offline
from flowcean.ode import OdeEnvironment, OdeState, OdeSystem
from flowcean.polars import (
    SlidingWindow,
    StreamingOfflineEnvironment,
    TrainTestSplit,
    collect,
)
from flowcean.river import RiverLearner
from flowcean.sklearn import RegressionTree
from flowcean.torch import LightningLearner, MultilayerPerceptron
from flowcean.utils.random import initialize_random

# ── Windows-Fix: DataLoader Worker deaktivieren ───────────────────────────────
_orig_dl_init = _tud.DataLoader.__init__

def _patched_dl_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["persistent_workers"] = False
    _orig_dl_init(self, *args, **kwargs)

_tud.DataLoader.__init__ = _patched_dl_init
# ─────────────────────────────────────────────────────────────────────────────

sys.setrecursionlimit(10000)

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path.cwd() / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  GEMEINSAME DATENSTRUKTUREN & KLASSEN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Valve:
    """
    Repräsentiert ein einzelnes Ventil im Tanksystem.

    Attribute:
        open     – True = Ventil geöffnet, False = geschlossen
        position – Öffnungsgrad zwischen 0.0 (zu) und 1.0 (auf)
    """
    open: bool = True
    position: float = 1.0

    def effective(self) -> float:
        """Effektiver Öffnungsgrad (0.0 wenn geschlossen)."""
        return self.position if self.open else 0.0


@dataclass
class NTankState(OdeState):
    """Zustandsvektor – h = Liste der aktuellen Füllstände [m]."""
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
    Sammelt ODE-Ausgaben schrittweise als Polars-DataFrames und überlagert
    optional Gaußsches Sensorrauschen.
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
    ODE-System für eine lineare Tanktopologie.
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
#  SYSTEM-FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_system(topology: str, n_tanks: int,
                 valves_between: List[Valve], valves_out: List[Valve],
                 qp_list: List[float], h_target: List[float],
                 areas: List[float]):
    """
    Factory-Funktion: Erzeugt die passende ODE-System-Instanz je nach Topologie.

    Hybride System-Architektur (Bachelorarbeit):
      Das ODE-System übernimmt die physikalische Modellierung und generiert
      Trainingsdaten. Das ML-Modell lernt daraus ohne explizite Kenntnis
      der Differentialgleichungen. Diese Kombination definiert das
      'hybride System' im Sinne der Arbeit.

    Physikalische Konstanten (fest für alle Simulationen):
      Qf         = 0.01 m²/s   – Leckagekoeffizient
      C_between  = 0.01 m^0.5/s – Durchflusskoeffizient Zwischenverbindungen
      Cout       = 0.01 m^0.5/s – Durchflusskoeffizient Auslassventile
    """
    initial_state = NTankState(h=[0.0] * n_tanks)
    if topology == "linear":
        return NTankLinear(
            n_tanks=n_tanks, A=areas,
            Qpmax=qp_list, Qf=0.01,
            C_between=[0.01] * (n_tanks - 1),
            Cout=[0.01] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            h_target=h_target,
            initial_state=initial_state,
        )
    else:
        n_between = n_tanks * (n_tanks - 1) // 2
        return NTankFullyCoupled(
            n_tanks=n_tanks, A=areas,
            Qpmax=qp_list, Qf=0.01,
            C_all=[0.01] * n_between,
            Cout=[0.01] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            h_target=h_target,
            initial_state=initial_state,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT: SENSOR-VERLAUF (Füllstände über Zeit)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sensor_data(df: pl.DataFrame, n_tanks: int, topology: str,
                     h_target: List[float], noise_std: float) -> None:
    """Visualisiert den zeitlichen Verlauf der Fuellstaende aller Tanks.

    Aenderungen (Review Mai 2026):
      - suptitle entfernt (Information steht in LaTeX-caption)
      - Subplot-Titel "Tank i" durch ylabel-Annotation ersetzt
      - Topologie/Rauschen-Info entfaellt (steht in caption)
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    t = df["t"].to_numpy()

    fig, axes = plt.subplots(n_tanks, 1, figsize=(7, 2.0 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]

    for i in range(n_tanks):
        h_vals = df[f"h{i + 1}"].to_numpy()
        c = colors[i % len(colors)]
        axes[i].plot(t, h_vals, color=c, linewidth=1.8)
        axes[i].fill_between(t, h_vals, alpha=0.12, color=c)
        axes[i].axhline(0, color="gray", linewidth=0.8, linestyle="--")
        axes[i].axhline(
            h_target[i], color="red", linewidth=1.2,
            linestyle="--", label=f"$h_\\mathrm{{target}}$ = {h_target[i]:.0f} m"
        )
        axes[i].legend(fontsize=12, loc="lower right")
        # Tank-Identifikation in ylabel statt set_title
        axes[i].set_ylabel(f"Tank {i + 1}\nh [m]", fontsize=13)
        axes[i].tick_params(labelsize=12)
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Zeit t [s]", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensor_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"-> sensor_plot.png gespeichert")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT: VORHERSAGE vs. REALITÄT
# ══════════════════════════════════════════════════════════════════════════════

def plot_predictions(predictions_data: dict, n_tanks: int,
                     model_label: str = "Modell",
                     focus_tank: int = 0,
                     window_size: int = 200) -> None:
    """
    Vergleicht vorhergesagte Fuellstaende mit den tatsaechlichen Testwerten.

    Aenderungen (Review Mai 2026):
      - Zeigt nur EINEN Tank (focus_tank, default Tank 1)
      - Kurzes Zeitfenster (default 200 Samples) statt gesamtes Testset
      - Groessere Legende und dickere Linien
      - suptitle entfernt (Information in LaTeX-caption)

    Akzeptiert zwei Eingabeformate:
      a) {tank_idx: {actual, predicted}}                  (incremental)
      b) {"LernerName": {tank_idx: {actual, predicted}}}  (offline)

    Parameter:
      focus_tank   - Index des Tanks der gezeigt werden soll (0-basiert)
      window_size  - Anzahl Test-Samples die geplottet werden
    """
    # Format (a) -> wrappen
    first_value = next(iter(predictions_data.values()))
    if isinstance(first_value, dict) and "actual" in first_value:
        predictions_data = {model_label: predictions_data}

    learner_names = list(predictions_data.keys())
    n_learners = len(learner_names)

    # Side-by-side: 1 Zeile, n_learners Spalten
    fig, axes = plt.subplots(
        1, n_learners,
        figsize=(6.5 * n_learners, 4.5),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]  # nur 1 Zeile

    for li, name in enumerate(learner_names):
        preds = predictions_data[name]
        actual     = preds[focus_tank]["actual"]
        predicted  = preds[focus_tank]["predicted"]
        test_start = preds[focus_tank].get("test_start", 0)

        # Auf Fenster der ersten window_size Samples des Testsets beschraenken
        n_show = min(window_size, len(actual))
        actual_show = actual[:n_show]
        pred_show   = predicted[:n_show]
        idx         = np.arange(test_start, test_start + n_show)

        axes[li].plot(idx, actual_show, color="#1f77b4", linewidth=2.0,
                      alpha=0.85, label="Realität")
        axes[li].plot(idx, pred_show, color="#e6194b", linewidth=1.8,
                      linestyle="--", alpha=0.95, label="Vorhersage")
        axes[li].set_xlabel("Test-Sample Index", fontsize=12)
        if li == 0:
            axes[li].set_ylabel(f"Füllstand h [m] - Tank {focus_tank + 1}",
                                fontsize=12)
        # Lerner-Name als Plot-Subtitle (nicht suptitle), damit klar ist
        # welche Spalte welcher Lerner ist
        axes[li].set_title(name, fontsize=12, fontweight="bold")
        axes[li].legend(fontsize=11, loc="best", framealpha=0.9)
        axes[li].grid(True, linestyle="--", alpha=0.4)
        axes[li].tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"-> predictions_plot.png gespeichert (Tank {focus_tank + 1}, "
          f"erste {window_size} Test-Samples)")


# ══════════════════════════════════════════════════════════════════════════════
#  LERNER-FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def build_river_learner(model_choice: str) -> RiverLearner:
    """
    Erzeugt einen inkrementellen River-Lerner.

    Optionen:
      'hoeffding' – HoeffdingTreeRegressor (grace_period=50, max_depth=5)
      'mlp'       – MLPRegressor (32-16, ReLU, Adam lr=1e-3)
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
    return RiverLearner(
        model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5)
    )


def build_offline_learners(model_choice: str, n_outputs: int) -> List[Any]:
    """
    Gibt eine Liste von Offline-Learnern zurück.

    Optionen:
      'tree'  → RegressionTree (sklearn, max_depth=5)
      'mlp'   → LightningLearner mit MultilayerPerceptron (32-16, LeakyReLU)
      'beide' → Beide Lerner werden nacheinander trainiert.
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
    """Lesbarer Name für einen Lerner."""
    name = type(learner).__name__
    if name == "RegressionTree":
        return "RegressionTree (sklearn)"
    if name == "LightningLearner":
        return "MLP Neuronales Netz (PyTorch)"
    return name


def extract_river_model(trained_model):
    """Robuster Zugriff auf das zugrunde liegende River-Modell."""
    for attr in ("model", "_model", "learner", "_learner"):
        candidate = getattr(trained_model, attr, None)
        if candidate is not None:
            return candidate
    logger.warning("Kein bekanntes Wrapper-Attribut gefunden.")
    return trained_model


# ══════════════════════════════════════════════════════════════════════════════
#  LERNMODUS: INCREMENTAL
# ══════════════════════════════════════════════════════════════════════════════

def run_incremental(
    topology: str, n_tanks: int, n_samples: int,
    valves_between: List[Valve], valves_out: List[Valve],
    qp_list: List[float], h_target: List[float], areas: List[float],
    noise_std: float, model_choice: str,
) -> dict:
    """
    Inkrementeller Lernmodus.

    Ablauf:
      1. Seed setzen, Vorschau-Simulation
      2. Hauptsimulation mit dt=1.0 s
      3. SlidingWindow(3) Feature Engineering
      4. Train/Test-Split 80/20, shuffle=False
      5. Pro Tank: inkrementelles Training + Vorhersage auf Testset
      6. Plot: Sensor-Verlauf

    Rückgabe:
      predictions_data = {tank_idx: {"actual": np.array, "predicted": np.array}}
      → wird von experiment_runner.py an metrics_helper.py weitergegeben
    """
    initialize_random(seed=42)

    model_label = (
        "HoeffdingTreeRegressor" if model_choice == "hoeffding"
        else "MLP Neuronales Netz (river)"
    )
    print(f"→ Modell: {model_label}\n")

    # ── Vorschau-Simulation ───────────────────────────────────────────────────
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
    train, _test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    n_windows  = len(df_plot) - 2
    n_train    = int(n_windows * 0.8)
    test_start = n_train

    predictions_data = {}

    # ── Training & Vorhersage pro Tank ────────────────────────────────────────
    for tank_idx in range(1, n_tanks + 1):
        target_name = f"h{tank_idx}_2"
        print(f"\n--- Learning {target_name} ({model_label}) ---")

        train_env = StreamingOfflineEnvironment(train, batch_size=1)
        learner   = build_river_learner(model_choice)

        t_start       = datetime.now(tz=timezone.utc)
        trained_model = learn_incremental(train_env, learner, inputs, [target_name])
        elapsed_ms    = round((datetime.now(tz=timezone.utc) - t_start).total_seconds() * 1000, 1)
        print(f"Learning {target_name} took {elapsed_ms} ms")

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
            preds_list.append(float(pred) if pred is not None else float("nan"))

        predictions_data[tank_idx - 1] = {
            "actual":     np.array(actuals_list),
            "predicted":  np.array(preds_list),
            "test_start": test_start,
        }

    return predictions_data


# ══════════════════════════════════════════════════════════════════════════════
#  LERNMODUS: OFFLINE
# ══════════════════════════════════════════════════════════════════════════════

def run_offline(
    topology: str, n_tanks: int, n_samples: int,
    valves_between: List[Valve], valves_out: List[Valve],
    qp_list: List[float], h_target: List[float], areas: List[float],
    noise_std: float, model_choice: str,
) -> dict:
    """
    Offline-Lernmodus.

    Ablauf:
      1. Simulation mit dt=1.0 s
      2. SlidingWindow(3) Feature Engineering
      3. Train/Test-Split 80/20, shuffle=False
      4. Multi-Output-Training: ein Modell lernt ALLE Tank-Outputs gleichzeitig
      5. Vorhersage auf Testset
      6. Plot: Sensor-Verlauf

    Rückgabe:
      all_predictions = {"LernerName": {tank_idx: {"actual", "predicted"}}}
      → wird von experiment_runner.py an metrics_helper.py weitergegeben
    """
    # ── Simulation ────────────────────────────────────────────────────────────
    collector = FrameCollector(n_tanks, noise_std=noise_std)
    system = build_system(
        topology, n_tanks, valves_between, valves_out, qp_list, h_target, areas
    )
    data_env = OdeEnvironment(
        system, dt=1.0, map_to_dataframe=collector.collect_frame
    )

    # ── Feature Engineering ───────────────────────────────────────────────────
    data = collect(data_env, n_samples) | SlidingWindow(window_size=3)
    df_plot = collector.concat()

    plot_sensor_data(df_plot, n_tanks, topology, h_target, noise_std)

    train, _test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    inputs  = [f"h{i + 1}_{step}" for i in range(n_tanks) for step in range(2)]
    outputs = [f"h{i + 1}_2" for i in range(n_tanks)]

    print(f"\n→ Inputs:  {inputs}")
    print(f"→ Outputs: {outputs}\n")

    learners = build_offline_learners(model_choice, n_outputs=len(outputs))

    all_predictions = {}

    n_windows  = len(df_plot) - 2
    test_start = int(n_windows * 0.8)

    # ── Training & Vorhersage pro Lerner ──────────────────────────────────────
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

        predictions_data = {
            i: {"actual": [], "predicted": [], "test_start": test_start}
            for i in range(n_tanks)
        }

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
                    else float("nan")
                )
                predictions_data[i]["actual"].append(float(df_plot[f"h{i + 1}"][j + 2]))
                predictions_data[i]["predicted"].append(pred_val)

        for i in range(n_tanks):
            predictions_data[i]["actual"]    = np.array(predictions_data[i]["actual"])
            predictions_data[i]["predicted"] = np.array(predictions_data[i]["predicted"])

        all_predictions[label] = predictions_data

    return all_predictions