import logging
from dataclasses import dataclass
from datetime import datetime, timezone

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


@dataclass
class TwoTankState(OdeState):
    h1: float  # Füllstand Tank 1
    h2: float  # Füllstand Tank 2

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.h1, self.h2])

    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0], state[1])


@dataclass
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
        super().__init__(
            initial_t,
            initial_state,
        )
        self.A1 = A1
        self.A2 = A2
        self.Qpmax = Qpmax
        self.Qf1 = Qf1
        self.Qf2 = Qf2
        self.Cvb = Cvb
        self.Cvo = Cvo

    @override
    def flow(self, t: float, state: NDArray[np.float64]) -> NDArray[np.float64]: #
        h1 = float(state[0])
        h2 = float(state[1])

        Qp = self.Qpmax
        Q12 = self.Cvb * np.sqrt(max(h1 - h2, 0.0))
        Q2out = self.Cvo * np.sqrt(max(h2, 0.0))

        dh1dt = (Qp - self.Qf1 - Q12) / self.A1
        dh2dt = (Q12 - self.Qf2 - Q2out) / self.A2

        return np.array([dh1dt, dh2dt], dtype=np.float64)


def main() -> None:
    flowcean.cli.initialize()

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

    data_incremental = OdeEnvironment(
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

    # Optional: Rohdaten anschauen
    df = collect(data_incremental, 20)
    print(df)

    # Daten für das Lernen
    data = collect(data_incremental, 250) | SlidingWindow(window_size=3)

    inputs = ["h1_0", "h1_1", "h2_0", "h2_1"]

    # Gemeinsame Train/Test-Splits
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)
    train_env = StreamingOfflineEnvironment(train, batch_size=1)

    # ---------- Modell für h2 ----------
    learner_h2 = RiverLearner(
        model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5),
    )

    t_start = datetime.now(tz=timezone.utc)
    model_h2 = learn_incremental(
        train_env,
        learner_h2,
        inputs,
        ["h2_2"],
    )
    delta_t = datetime.now(tz=timezone.utc) - t_start
    print(f"Learning h2 took {np.round(delta_t.microseconds / 1000, 1)} ms")

    report_h2 = evaluate_offline(
        model_h2,
        test,
        inputs,
        ["h2_2"],
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print("Report for h2_2:")
    print(report_h2)

    # ---------- Modell für h1 ----------
    # Train-Environment neu aufsetzen, weil der erste Durchlauf es verbraucht hat
    train_env_h1 = StreamingOfflineEnvironment(train, batch_size=1)

    learner_h1 = RiverLearner(
        model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5),
    )

    t_start = datetime.now(tz=timezone.utc)
    model_h1 = learn_incremental(
        train_env_h1,
        learner_h1,
        inputs,
        ["h1_2"],
    )
    delta_t = datetime.now(tz=timezone.utc) - t_start
    print(f"Learning h1 took {np.round(delta_t.microseconds / 1000, 1)} ms")

    report_h1 = evaluate_offline(
        model_h1,
        test,
        inputs,
        ["h1_2"],
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print("Report for h1_2:")
    print(report_h1)


if __name__ == "__main__":
    main() 