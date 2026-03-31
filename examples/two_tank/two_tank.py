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
        
