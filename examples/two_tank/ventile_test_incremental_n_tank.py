import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import numpy as np
import polars as pl
from numpy.typing import NDArray
from river import tree
from typing_extensions import Self, override

import flowcean.cli
from flowcean.core import evaluate_offline, learn_incremental
from flowcean.ode import (
    OdeEnvironment,
    OdeState,
    OdeSystem,
)
from flowcean.polars import (
    SlidingWindow,
    StreamingOfflineEnvironment,
    TrainTestSplit,
)
from flowcean.polars.environments.dataframe import collect
from flowcean.river import RiverLearner
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  Ventil
# ─────────────────────────────────────────

@dataclass
class Valve:
    """Steuerbares Ventil.

    open     – True = Ventil kann Wasser durchlassen
    position – Öffnungsgrad 0.0 (zu) … 1.0 (voll offen)
    """
    open: bool = True
    position: float = 1.0

    def effective(self) -> float:
        """Effektiver Öffnungsgrad: 0.0 wenn geschlossen."""
        return self.position if self.open else 0.0


# ─────────────────────────────────────────
#  Gemeinsamer Zustand
# ─────────────────────────────────────────

@dataclass
class NTankState(OdeState):
    h: List[float]  # Füllstände h[0]..h[n-1]

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array(self.h, dtype=np.float64)

    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state.tolist())


# ─────────────────────────────────────────
#  Topologie 1: Linear (Kette)
#  Tank0 → Tank1 → … → TankN
#  Zwischenventile: n-1  |  Auslassventile: n
# ─────────────────────────────────────────

class NTankLinear(OdeSystem[NTankState]):

    def __init__(
        self,
        *,
        n_tanks: int,
        A: float,
        Qpmax: float,
        Qf: float,
        C_between: List[float],       # Länge n-1
        Cout: List[float],            # Länge n
        valves_between: List[Valve],  # Länge n-1
        valves_out: List[Valve],      # Länge n
        initial_state: NTankState,
        initial_t: float = 0.0,
    ) -> None:
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
    def flow(self, t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)

        # Flüsse zwischen benachbarten Tanks (i → i+1)
        Q_between = np.zeros(max(n - 1, 0))
        for i in range(n - 1):
            eff = self.valves_between[i].effective()
            Q_between[i] = (
                self.C_between[i] * eff * np.sqrt(max(h[i] - h[i + 1], 0.0))
            )

        # Auslass pro Tank
        Q_out = np.zeros(n)
        for i in range(n):
            eff = self.valves_out[i].effective()
            Q_out[i] = self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))

        if n == 1:
            dhdt[0] = (self.Qpmax - self.Qf - Q_out[0]) / self.A
        else:
            dhdt[0] = (self.Qpmax - self.Qf - Q_between[0] - Q_out[0]) / self.A
            for i in range(1, n - 1):
                dhdt[i] = (
                    Q_between[i - 1] - self.Qf - Q_between[i] - Q_out[i]
                ) / self.A
            dhdt[-1] = (Q_between[-1] - self.Qf - Q_out[-1]) / self.A

        return dhdt


# ─────────────────────────────────────────
#  Topologie 2: Vollvermascht (jeder mit jedem)
#  Zwischenventile: n*(n-1)/2  |  Auslassventile: n
# ─────────────────────────────────────────

class NTankFullyCoupled(OdeSystem[NTankState]):

    def __init__(
        self,
        *,
        n_tanks: int,
        A: float,
        Qpmax: float,
        Qf: float,
        C_all: List[float],           # Länge n*(n-1)/2
        Cout: List[float],            # Länge n
        valves_between: List[Valve],  # Länge n*(n-1)/2
        valves_out: List[Valve],      # Länge n
        initial_state: NTankState,
        initial_t: float = 0.0,
    ) -> None:
        super().__init__(initial_t, initial_state)
        self.n = n_tanks
        self.A = A
        self.Qpmax = Qpmax
        self.Qf = Qf
        self.C_all = C_all
        self.Cout = Cout
        self.valves_between = valves_between
        self.valves_out = valves_out

    def _pair_index(self, i: int, j: int) -> int:
        """Flacher Index für Paar (i, j) mit i < j."""
        n = self.n
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    @override
    def flow(self, t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)

        # Leckage alle Tanks
        dhdt -= self.Qf

        # Pumpe Tank 0
        dhdt[0] += self.Qpmax

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
        return dhdt


# ─────────────────────────────────────────
#  Ventil-Konfiguration (interaktiv)
# ─────────────────────────────────────────

def configure_valve(label: str) -> Valve:
    """Fragt den User nach Öffnungszustand und -grad eines Ventils."""
    print(f"\n  Ventil {label}")

    open_input = input("    Geöffnet? (j/n) [Standard: j]: ").strip().lower()
    is_open = open_input != "n"

    if is_open:
        pos_input = input("    Öffnungsgrad (0.0–1.0) [Standard: 1.0]: ").strip()
        try:
            position = float(pos_input)
            position = max(0.0, min(1.0, position))
        except ValueError:
            position = 1.0
    else:
        position = 0.0

    return Valve(open=is_open, position=position)


def configure_valves_between_linear(n_tanks: int) -> List[Valve]:
    print("\n── Zwischenventile (Verbindung Tank i → Tank i+1) ──")
    valves = []
    for i in range(n_tanks - 1):
        valves.append(configure_valve(f"Tank {i+1} → Tank {i+2}"))
    return valves


def configure_valves_between_coupled(n_tanks: int) -> List[Valve]:
    print("\n── Zwischenventile (alle Paare) ──")
    valves = []
    for i in range(n_tanks):
        for j in range(i + 1, n_tanks):
            valves.append(configure_valve(f"Tank {i+1} ↔ Tank {j+1}"))
    return valves


def configure_valves_out(n_tanks: int) -> List[Valve]:
    print("\n── Auslassventile (ein Auslass pro Tank) ──")
    valves = []
    for i in range(n_tanks):
        valves.append(configure_valve(f"Auslass Tank {i+1}"))
    return valves


def print_valve_summary(
    valves_between: List[Valve],
    valves_out: List[Valve],
    topology: str,
    n_tanks: int,
) -> None:
    print("\n── Ventil-Übersicht ─────────────────────────")
    if topology == "linear":
        for i, v in enumerate(valves_between):
            status = "offen" if v.open else "ZU"
            print(f"  Zwischenventil Tank {i+1}→{i+2}: {status}, Position={v.position:.2f}")
    else:
        idx = 0
        for i in range(n_tanks):
            for j in range(i + 1, n_tanks):
                v = valves_between[idx]
                status = "offen" if v.open else "ZU"
                print(f"  Zwischenventil Tank {i+1}↔{j+1}: {status}, Position={v.position:.2f}")
                idx += 1
    for i, v in enumerate(valves_out):
        status = "offen" if v.open else "ZU"
        print(f"  Auslassventil  Tank {i+1}:      {status}, Position={v.position:.2f}")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────
#  Hauptprogramm
# ─────────────────────────────────────────

def main() -> None:
    flowcean.cli.initialize()

    # ── Topologie ─────────────────────────
    topology = ""
    while topology not in ("linear", "coupled"):
        topology = input(
            "Topologie wählen – 'linear' (Kette) oder 'coupled' (jeder mit jedem): "
        ).strip().lower()

    n_tanks = int(input("Wie viele Tanks sollen simuliert werden? "))
    n_samples = int(input("Wie viele Samples sollen gesammelt werden? "))

    # ── Ventile konfigurieren ─────────────
    use_custom = input(
        "\nVentile manuell konfigurieren? (j/n) [Standard: n, alle voll offen]: "
    ).strip().lower()

    if use_custom == "j":
        if topology == "linear":
            valves_between = configure_valves_between_linear(n_tanks)
        else:
            valves_between = configure_valves_between_coupled(n_tanks)
        valves_out = configure_valves_out(n_tanks)
    else:
        n_between = (
            n_tanks - 1
            if topology == "linear"
            else n_tanks * (n_tanks - 1) // 2
        )
        valves_between = [Valve(open=True, position=1.0) for _ in range(n_between)]
        valves_out = [Valve(open=True, position=1.0) for _ in range(n_tanks)]

    print_valve_summary(valves_between, valves_out, topology, n_tanks)

    # ── System aufbauen ───────────────────
    initial_levels = [0.0] + [0.03] * (n_tanks - 1)

    if topology == "linear":
        system = NTankLinear(
            n_tanks=n_tanks,
            A=0.0154,
            Qpmax=1e-2,
            Qf=1e-4,
            C_between=[1.5938e-4] * (n_tanks - 1),
            Cout=[1.59640e-4] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            initial_state=NTankState(h=initial_levels),
        )
        topo_label = "Linear"
    else:
        n_between = n_tanks * (n_tanks - 1) // 2
        system = NTankFullyCoupled(
            n_tanks=n_tanks,
            A=0.0154,
            Qpmax=1e-2,
            Qf=1e-4,
            C_all=[1.5e-4] * n_between,
            Cout=[1.59640e-4] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            initial_state=NTankState(h=initial_levels),
        )
        topo_label = "Fully Coupled"

    print(
        f"→ Topologie: {topo_label} | Tanks: {n_tanks} | "
        f"Zwischenventile: {len(valves_between)} | Auslassventile: {n_tanks}\n"
    )

    # ── ODE-Umgebung ──────────────────────
    data_incremental = OdeEnvironment(
        system,
        dt=0.1,
        map_to_dataframe=lambda ts, xs: pl.DataFrame(
            {
                "t": ts,
                **{f"h{i + 1}": [x.h[i] for x in xs] for i in range(n_tanks)},
            }
        ),
    )

    df_preview = collect(data_incremental, 20)
    print("Vorschau auf die ersten 20 Schritte:")
    print(df_preview)

    # ── Daten & Features ──────────────────
    data = collect(data_incremental, n_samples) | SlidingWindow(window_size=3)

    inputs = []
    for i in range(n_tanks):
        inputs.append(f"h{i + 1}_0")
        inputs.append(f"h{i + 1}_1")

    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    # ── Lernen & Evaluation ───────────────
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
        print(
            f"Learning {target_name} took "
            f"{np.round(delta_t.microseconds / 1000, 1)} ms"
        )

        report = evaluate_offline(
            model,
            test,
            inputs,
            [target_name],
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print(f"Report for {target_name}:")
        print(report)


if __name__ == "__main__":
    main()