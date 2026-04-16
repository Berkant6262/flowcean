#!/usr/bin/env python

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

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


@dataclass
class TwoTankState(OdeState):
    h1: float
    h2: float

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.h1, self.h2])

    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0], state[1])


class TwoTank(OdeSystem[TwoTankState]):
    def __init__(
        self,
        *,
        A1: float,
        A2: float,
        Qpmax: float,
        Qf1: float,
        Qf2: float,
        Cvb: float,
        Cvo: float,
        initial_t: float = 0.0,
        initial_state: TwoTankState,
    ) -> None:
        super().__init__(initial_t, initial_state)
        self.A1 = A1
        self.A2 = A2
        self.Qpmax = Qpmax
        self.Qf1 = Qf1
        self.Qf2 = Qf2
        self.Cvb = Cvb
        self.Cvo = Cvo

    @override
    def flow(self, t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        h1 = float(state[0])
        h2 = float(state[1])

        Qp = self.Qpmax
        Q12 = self.Cvb * np.sqrt(max(h1 - h2, 0.0))
        Q2out = self.Cvo * np.sqrt(max(h2, 0.0))

        dh1dt = (Qp - self.Qf1 - Q12) / self.A1
        dh2dt = (Q12 - self.Qf2 - Q2out) / self.A2

        return np.array([dh1dt, dh2dt], dtype=np.float64)


def main() -> None:
    initialize()
    initialize_random(seed=42)

    system = TwoTank(
        A1=0.0154,
        A2=0.0154,
        Qpmax=1e-2,
        Qf1=1e-4,
        Qf2=1e-4,
        Cvb=1.5938e-4,
        Cvo=1.59640e-4,
        initial_state=TwoTankState(h1=0.0, h2=0.03),
    )

    data_env = OdeEnvironment(
        system,
        dt=0.1,
        map_to_dataframe=lambda ts, xs: pl.DataFrame(
            {
                "t": ts,
                "h1": [x.h1 for x in xs],
                "h2": [x.h2 for x in xs],
            },
        ),
    )

    data = collect(data_env, 250) | SlidingWindow(window_size=3)

    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)

    # Eingaben: 2 Zeitschritte Historie von h1 und h2
    inputs = ["h1_0", "h1_1", "h2_0", "h2_1"]
    # Ausgaben: beide nächsten Werte gleichzeitig
    outputs = ["h1_2", "h2_2"]

    for learner in [
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
    ]:
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
        print(report)


if __name__ == "__main__":
    main()