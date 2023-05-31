from abc import ABCMeta, abstractmethod

import numpy as np
from absl import logging
from ml_collections import ConfigDict


class VMesh(object, metaclass=ABCMeta):
    """Base abstract velocity mesh class."""

    def __init__(self, config: ConfigDict, *args, **kwargs):
        # Get dimension
        self._ndim = config.collision_model.dim
        logging.info(f"{self._ndim} dimensional collision model.")
        # Get the config
        self.config = config.velocity_mesh
        self._center = None
        self._centers = None
        self._weights = None
        # Construct the velocity mesh
        self._build_velocity_mesh()

    @property
    def centers(self) -> list[np.ndarray]:
        if self._centers is None:
            index = np.indices(self.num_nodes)
            self._centers = [self.center[index[i, ...]] for i in range(self.num_dim)]
        return self._centers

    @property
    def num_dim(self) -> int:
        return self._ndim

    @property
    def delta(self) -> float:
        return 2 * self.L / self.nv

    @property
    def vsquare(self) -> np.ndarray:
        vsq = 0.0
        for v in self.centers:
            vsq += v**2
        return vsq

    def get_F(self, f):
        vaxis = tuple(-(i + 1) for i in range(self.num_dim))
        rho = np.sum(f * self.weights, axis=vaxis)
        m = [np.sum(f * v * self.weights, axis=vaxis) for v in self.centers]
        E = 0.5 * np.sum(f * self.vsquare * self.weights, axis=vaxis)

        return [rho, m, E]

    def get_p(self, f):
        rho, m, E = self.get_F(f)
        u = [m_i / rho for m_i in m]
        usq = 0.0
        for ui in u:
            usq += ui**2
        T = (2 * E / rho - usq) / self.num_dim

        return [rho, u, T]

    @abstractmethod
    def _build_velocity_mesh(self):
        pass
