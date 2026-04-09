#!/usr/bin/env python

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from typing_extensions import Self, override

from flowcean.cli import initialize
from flowcean.core import evaluate_offline, learn_offline
from flowcean.ode import OdeEnvironment, OdeState, OdeSystem
from flowcean.polars import SlidingWindow, TrainTestSplit, collect
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.torch import LightningLearner, MultilayerPerceptron
from flowcean.utils.random import initialize_random

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


# ---------- ODE-System: n Tanks, alle miteinander gekoppelt (nichtlinear) ----------

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

        # Kopplung: alle Tanks miteinander verbunden (voll vernetzt)
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
                # bei exakt gleichen Höhen kein Fluss

        # Auf Tankfläche normieren
        dhdt = dhdt / self.A

        return dhdt


def main() -> None:
    initialize()
    initialize_random(seed=42)

    # --------- Benutzerabfragen ---------
    n_tanks = int(input("Wie viele Tanks sollen simuliert werden? "))
    n_samples = int(input("Wie viele Samples sollen gesammelt werden? "))

    # Anfangsfüllstände: erster 0.0, Rest 0.03 (kannst du anpassen)
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

    # ---------- Datenerzeugung über ODE ----------

    data_env = OdeEnvironment(
        system,
        dt=0.1,
        map_to_dataframe=lambda ts, xs: pl.DataFrame(
            {
                "t": ts,
                **{f"h{i+1}": [x.h[i] for x in xs] for i in range(n_tanks)},
            },
        ),
    )

    # Kurzer Preview zum Checken
    df_preview = collect(data_env, 20)
    print("Vorschau auf die ersten 20 Schritte:")
    print(df_preview)

    # Für das Lernen:
    data = collect(data_env, n_samples) | SlidingWindow(window_size=3)

    # Train/Test-Split (offline: Shuffle erlaubt)
    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)

    # Inputs: 2 Zeitschritte Historie von allen Tanks
    inputs = []
    for i in range(n_tanks):
        idx = i + 1
        inputs.append(f"h{idx}_0")
        inputs.append(f"h{idx}_1")

    # Outputs: alle Tanks gleichzeitig, nächster Zeitschritt
    outputs = [f"h{i}_2" for i in range(1, n_tanks + 1)]

    # ---------- Offline-Lernen: RegressionTree + MLP ----------

    learners = [
        RegressionTree(max_depth=5),
        LightningLearner(
            module=MultilayerPerceptron(
                learning_rate=1e-3,
                output_size=len(outputs),
                hidden_dimensions=[10, 10],
                activation_function=torch.nn.LeakyReLU,
            ),
            max_epochs=1000,
        ),
    ]

    for learner in learners:
        print("\n=== Training", learner.__class__.__name__, "===")
        t_start = datetime.now(tz=timezone.utc)

        model = learn_offline(
            train,
            learner,
            inputs,
            outputs,
        )

        delta_t = datetime.now(tz=timezone.utc) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        report = evaluate_offline(
            model,
            test,
            inputs,
            outputs,
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print("Report:")
        print(report)


if __name__ == "__main__":
    main()