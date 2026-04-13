import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray
from river import tree
from typing_extensions import Self, override

import flowcean.cli
from flowcean.core import evaluate_offline, learn_incremental
from flowcean.ode import OdeEnvironment, OdeState, OdeSystem
from flowcean.polars import SlidingWindow, StreamingOfflineEnvironment, TrainTestSplit
from flowcean.polars.environments.dataframe import collect
from flowcean.river import RiverLearner
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError

logger = logging.getLogger(__name__)


@dataclass
class Valve:
    """Steuerbares Ventil.
    open     – True = Ventil kann Wasser durchlassen
    position – Öffnungsgrad 0.0 (zu) … 1.0 (voll offen)
    """
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
#  Topologie 1: Linear (Kette)
# ─────────────────────────────────────────

class NTankLinear(OdeSystem[NTankState]):

    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_between, Cout,
                 valves_between, valves_out, initial_state, initial_t=0.0):
        super().__init__(initial_t, initial_state)
        self.n = n_tanks
        self.A = A
        self.Qpmax = Qpmax
        self.Qf = Qf
        self.C_between = C_between
        self.Cout = Cout
        self.valves_between = valves_between
        self.valves_out = valves_out

    @override
    def flow(self, t, state):
        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)

        # Flüsse zwischen benachbarten Tanks (i → i+1)
        Q_between = np.zeros(max(n - 1, 0))
        for i in range(n - 1):
            eff = self.valves_between[i].effective()
            Q_between[i] = self.C_between[i] * eff * np.sqrt(max(h[i] - h[i+1], 0.0))

        # Auslass pro Tank
        Q_out = np.zeros(n)
        for i in range(n):
            eff = self.valves_out[i].effective()
            Q_out[i] = self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))

        # REALISTISCH: Leckage proportional zum Füllstand (je mehr Druck → mehr Leckage)
        Q_leak = np.array([self.Qf * np.sqrt(max(h[i], 0.0)) for i in range(n)])

        if n == 1:
            dhdt[0] = (self.Qpmax - Q_leak[0] - Q_out[0]) / self.A
        else:
            dhdt[0] = (self.Qpmax - Q_leak[0] - Q_between[0] - Q_out[0]) / self.A
            for i in range(1, n - 1):
                dhdt[i] = (Q_between[i-1] - Q_leak[i] - Q_between[i] - Q_out[i]) / self.A
            dhdt[-1] = (Q_between[-1] - Q_leak[-1] - Q_out[-1]) / self.A

        # Numerische Absicherung: Tank kann nicht unter 0 fallen
        for i in range(n):
            if h[i] <= 0.0 and dhdt[i] < 0.0:
                dhdt[i] = 0.0

        return dhdt


# ─────────────────────────────────────────
#  Topologie 2: Vollvermascht
# ─────────────────────────────────────────

class NTankFullyCoupled(OdeSystem[NTankState]):

    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_all, Cout,
                 valves_between, valves_out, initial_state, initial_t=0.0):
        super().__init__(initial_t, initial_state)
        self.n = n_tanks
        self.A = A
        self.Qpmax = Qpmax
        self.Qf = Qf
        self.C_all = C_all
        self.Cout = Cout
        self.valves_between = valves_between
        self.valves_out = valves_out

    def _pair_index(self, i, j):
        n = self.n
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    @override
    def flow(self, t, state):
        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)

        # Pumpe Tank 0
        dhdt[0] += self.Qpmax

        # REALISTISCH: Leckage proportional zum Füllstand
        for i in range(n):
            dhdt[i] -= self.Qf * np.sqrt(max(h[i], 0.0))

        # Auslass pro Tank
        for i in range(n):
            eff = self.valves_out[i].effective()
            dhdt[i] -= self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))

        # Kopplung: alle Paare (i, j)
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

        dhdt /= self.A

        # Numerische Absicherung: Tank kann nicht unter 0 fallen
        for i in range(n):
            if h[i] <= 0.0 and dhdt[i] < 0.0:
                dhdt[i] = 0.0

        return dhdt


# ─────────────────────────────────────────
#  Ventil-Konfiguration
# ─────────────────────────────────────────

def configure_valve(label: str) -> Valve:
    print(f"\n  Ventil {label}")
    is_open = input("    Geöffnet? (j/n) [Standard: j]: ").strip().lower() != "n"
    if is_open:
        try:
            position = max(0.0, min(1.0, float(
                input("    Öffnungsgrad (0.0–1.0) [Standard: 1.0]: ").strip()
            )))
        except ValueError:
            position = 1.0
    else:
        position = 0.0
    return Valve(open=is_open, position=position)


def configure_valves_between_linear(n_tanks):
    print("\n── Zwischenventile (Verbindung Tank i → Tank i+1) ──")
    return [configure_valve(f"Tank {i+1} → Tank {i+2}") for i in range(n_tanks - 1)]


def configure_valves_between_coupled(n_tanks):
    print("\n── Zwischenventile (alle Paare) ──")
    valves = []
    for i in range(n_tanks):
        for j in range(i + 1, n_tanks):
            valves.append(configure_valve(f"Tank {i+1} ↔ Tank {j+1}"))
    return valves


def configure_valves_out(n_tanks):
    print("\n── Auslassventile (ein Auslass pro Tank) ──")
    return [configure_valve(f"Auslass Tank {i+1}") for i in range(n_tanks)]


def print_valve_summary(valves_between, valves_out, topology, n_tanks):
    print("\n── Ventil-Übersicht ─────────────────────────")
    if topology == "linear":
        for i, v in enumerate(valves_between):
            print(f"  Zwischenventil Tank {i+1}→{i+2}: {'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
    else:
        idx = 0
        for i in range(n_tanks):
            for j in range(i + 1, n_tanks):
                v = valves_between[idx]
                print(f"  Zwischenventil Tank {i+1}↔{j+1}: {'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
                idx += 1
    for i, v in enumerate(valves_out):
        print(f"  Auslassventil  Tank {i+1}: {'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────
#  Metriken sicher aus Report-String lesen
# ─────────────────────────────────────────

def extract_metric(report_obj, metric_name: str, target: str) -> float:
    for line in str(report_obj).splitlines():
        if metric_name in line and target in line:
            try:
                return float(line.split()[-1])
            except ValueError:
                pass
    return 0.0


# ─────────────────────────────────────────
#  Plot 1: Füllstände
# ─────────────────────────────────────────

def plot_sensor_data(df: pl.DataFrame, n_tanks: int, topology: str) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    t = df["t"].to_numpy()

    fig, axes = plt.subplots(n_tanks, 1, figsize=(10, 3 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]

    topo_label = "Linear (Kette)" if topology == "linear" else "Fully Coupled"
    fig.suptitle(f"Füllstände – Topologie: {topo_label}", fontsize=13, fontweight="bold")

    for i in range(n_tanks):
        h_vals = df[f"h{i+1}"].to_numpy()
        c = colors[i % len(colors)]
        axes[i].plot(t, h_vals, color=c, linewidth=1.8)
        axes[i].fill_between(t, h_vals, alpha=0.12, color=c)
        axes[i].axhline(0, color="gray", linewidth=0.8, linestyle="--")  # Nulllinie
        axes[i].set_ylabel("Füllstand h [m]", fontsize=9)
        axes[i].set_title(f"Tank {i+1}", fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Zeit t [s]", fontsize=10)
    plt.tight_layout()
    plt.savefig("sensor_plot.png", dpi=150, bbox_inches="tight")
    print("\n→ sensor_plot.png gespeichert – Fenster schließen um fortzufahren!")
    plt.show()


# ─────────────────────────────────────────
#  Plot 2: Modellfehler
# ─────────────────────────────────────────

def plot_learning_results(results: dict, n_tanks: int) -> None:
    x = np.arange(n_tanks)
    width = 0.35
    mae_vals = [results[i]["MAE"] for i in range(n_tanks)]
    mse_vals = [results[i]["MSE"] for i in range(n_tanks)]

    fig, ax = plt.subplots(figsize=(max(6, n_tanks * 2), 5))
    bars1 = ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + width / 2, mse_vals, width, label="MSE", color="#d62728", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Tank")
    ax.set_ylabel("Fehler")
    ax.set_title("Modellfehler pro Tank (HoeffdingTreeRegressor)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tank {i+1}" for i in range(n_tanks)])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("learning_results.png", dpi=150, bbox_inches="tight")
    print("→ learning_results.png gespeichert")
    plt.show()


# ─────────────────────────────────────────
#  Hauptprogramm
# ─────────────────────────────────────────

def main() -> None:
    flowcean.cli.initialize()

    topology = ""
    while topology not in ("linear", "coupled"):
        topology = input(
            "Topologie wählen – 'linear' (Kette) oder 'coupled' (jeder mit jedem): "
        ).strip().lower()

    n_tanks = int(input("Wie viele Tanks sollen simuliert werden? "))
    n_samples = int(input("Wie viele Samples sollen gesammelt werden? "))

    use_custom = input(
        "\nVentile manuell konfigurieren? (j/n) [Standard: n, alle voll offen]: "
    ).strip().lower()

    if use_custom == "j":
        valves_between = (configure_valves_between_linear(n_tanks)
                          if topology == "linear"
                          else configure_valves_between_coupled(n_tanks))
        valves_out = configure_valves_out(n_tanks)
    else:
        n_between = n_tanks - 1 if topology == "linear" else n_tanks * (n_tanks - 1) // 2
        valves_between = [Valve(open=True, position=1.0) for _ in range(n_between)]
        valves_out = [Valve(open=True, position=1.0) for _ in range(n_tanks)]

    print_valve_summary(valves_between, valves_out, topology, n_tanks)

    # REALISTISCH: Alle Tanks starten leer
    initial_levels = [0.0] * n_tanks

    raw_frames: List[pl.DataFrame] = []

    def map_to_dataframe(ts, xs) -> pl.DataFrame:
        frame = pl.DataFrame({
            "t": ts,
            **{f"h{i+1}": [x.h[i] for x in xs] for i in range(n_tanks)},
        })
        raw_frames.append(frame)
        return frame

    if topology == "linear":
        system = NTankLinear(
            n_tanks=n_tanks, A=0.0154, Qpmax=1e-2, Qf=1e-4,
            C_between=[1.5938e-4] * (n_tanks - 1),
            Cout=[1.59640e-4] * n_tanks,
            valves_between=valves_between, valves_out=valves_out,
            initial_state=NTankState(h=initial_levels),
        )
        topo_label = "Linear"
    else:
        n_between = n_tanks * (n_tanks - 1) // 2
        system = NTankFullyCoupled(
            n_tanks=n_tanks, A=0.0154, Qpmax=1e-2, Qf=1e-4,
            C_all=[1.5e-4] * n_between,
            Cout=[1.59640e-4] * n_tanks,
            valves_between=valves_between, valves_out=valves_out,
            initial_state=NTankState(h=initial_levels),
        )
        topo_label = "Fully Coupled"

    print(f"→ Topologie: {topo_label} | Tanks: {n_tanks} | "
          f"Zwischenventile: {len(valves_between)} | Auslassventile: {n_tanks}\n")

    data_incremental = OdeEnvironment(system, dt=1.0, map_to_dataframe=map_to_dataframe)

    print("Vorschau auf die ersten 20 Schritte:")
    print(collect(data_incremental, 20))

    raw_frames.clear()
    df_flowcean = collect(data_incremental, n_samples)
    df_plot = pl.concat(raw_frames)

    # ── Plot 1: Füllstände ────────────────
    plot_sensor_data(df_plot, n_tanks, topology)

    # ── Training ──────────────────────────
    data = df_flowcean | SlidingWindow(window_size=3)

    inputs = []
    for i in range(n_tanks):
        inputs.append(f"h{i+1}_0")
        inputs.append(f"h{i+1}_1")

    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    results = {}

    for tank_idx in range(1, n_tanks + 1):
        target_name = f"h{tank_idx}_2"
        print(f"\n--- Learning {target_name} ---")

        train_env = StreamingOfflineEnvironment(train, batch_size=1)
        learner = RiverLearner(
            model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5),
        )

        t_start = datetime.now(tz=timezone.utc)
        model = learn_incremental(train_env, learner, inputs, [target_name])
        delta_t = datetime.now(tz=timezone.utc) - t_start
        print(f"Learning {target_name} took {np.round(delta_t.microseconds / 1000, 1)} ms")

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

    # ── Plot 2: Modellfehler ──────────────
    plot_learning_results(results, n_tanks)


if __name__ == "__main__":
    main()