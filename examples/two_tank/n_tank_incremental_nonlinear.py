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


# ---------- Zustand: n Tanks ----------

@dataclass
class NTankState(OdeState):
    h: List[float]  # Füllstände der Tanks, h[0]..h[n-1]

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array(self.h, dtype=np.float64)

    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state.tolist())


# ---------- ODE-System: n Tanks, alle miteinander gekoppelt ----------

class FullyCoupledNTank(OdeSystem[NTankState]):

    def __init__(
        self,
        *,
        n_tanks: int,
        A: float,
        Qpmax: float,
        Qf: float,
        C_all: float,
        Cout: float,
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

    @override
    def flow(self, t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        h = state.astype(float)  # h[0]..h[n-1]
        n = self.n

        dhdt = np.zeros_like(h)

        # Pumpe in Tank 0
        Qp = self.Qpmax

        # Abfluss nur aus letztem Tank
        Q_out = self.Cout * np.sqrt(max(h[-1], 0.0))

        # Start: Leckage in allen Tanks
        for i in range(n):
            dhdt[i] = -self.Qf

        # Zufluss zu Tank 0
        dhdt[0] += Qp

        # Abfluss aus letztem Tank
        dhdt[-1] -= Q_out

        # Kopplung: alle Tanks miteinander verbunden
        for i in range(n):
            for j in range(i + 1, n):
                if h[i] > h[j]:
                    Q = self.C_all * np.sqrt(h[i] - h[j])
                    dhdt[i] -= Q
                    dhdt[j] += Q
                elif h[j] > h[i]:
                    Q = self.C_all * np.sqrt(h[j] - h[i])
                    dhdt[j] -= Q
                    dhdt[i] += Q
                # Bei exakt gleichen Höhen kein Fluss

        # Auf Tankfläche normieren
        dhdt = dhdt / self.A

        return dhdt


def main() -> None:
    flowcean.cli.initialize()

    # --------- Benutzerabfragen ---------
    n_tanks = int(input("Wie viele Tanks sollen simuliert werden? "))
    n_samples = int(input("Wie viele Samples sollen gesammelt werden? "))

    # Anfangsfüllstände: erster 0.0, Rest 0.03
    initial_levels = [0.0] + [0.03] * (n_tanks - 1)

    system = FullyCoupledNTank(
        n_tanks=n_tanks,
        A=0.0154,
        Qpmax=1e-2,
        Qf=1e-4,
        C_all=1.5e-4,
        Cout=1.6e-4,
        initial_state=NTankState(h=initial_levels),
    )

    data_incremental = OdeEnvironment(
        system,
        dt=0.1,
        map_to_dataframe=lambda ts, xs: pl.DataFrame(
            {
                "t": ts,
                **{f"h{i+1}": [x.h[i] for x in xs] for i in range(n_tanks)},
            },
        ),
    )

    # Optional: Rohdaten anschauen
    df_preview = collect(data_incremental, 20)
    print("Vorschau auf die ersten 20 Schritte:")
    print(df_preview)

    # Daten für das Lernen
    data = collect(data_incremental, n_samples) | SlidingWindow(window_size=3)

    # Inputs: zwei Zeitschritte Historie von allen Tanks
    inputs = []
    for i in range(n_tanks):
        idx = i + 1
        inputs.append(f"h{idx}_0")
        inputs.append(f"h{idx}_1")

    # Train/Test-Split (zeitlich, kein Shuffle)
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    # Für jeden Tank ein eigenes Modell: h1_2, h2_2, ..., hN_2
    for tank_idx in range(1, n_tanks + 1):
        target_name = f"h{tank_idx}_2"
        print(f"\n--- Learning {target_name} ---")

        train_env = StreamingOfflineEnvironment(train, batch_size=1)

        learner = RiverLearner(
            model=tree.HoeffdingTreeRegressor(
                grace_period=100,
                max_depth=3,
            ),
        )

        t_start = datetime.now(tz=timezone.utc)
        model = learn_incremental(
            train_env,
            learner,
            inputs,
            [target_name],
        )
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