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
    h1: float #Füllstand Tank 1
    h2: float #Füllstand Tank 2

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.h1,self.h2])
    
    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0],state[1])
        

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
        # initial_t=0.0  # optional, weil Default
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

    df = collect(data_incremental, 20)      # aus Flowcean-DataFrame ein echtes polars.DataFrame machen
    print(df)

if __name__ == "__main__":
    main()