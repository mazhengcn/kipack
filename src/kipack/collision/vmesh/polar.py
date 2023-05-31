import numpy as np
from absl import logging

from .base import VMesh


class PolarMesh(VMesh):
    def _build_velocity_mesh(self):
        self._nv = self.config.nv
        logging.info(f"Number of velocity cells: {self._nv}.")

        # Define the phsical domain
        self._radius = self.config.radius
        self._lower = self.config.lower
        self._upper = self.config.upper

        if self._ndim == 2:
            logging.info(f"Velocity domain: disk with raidus {self._radius}.")
            self._build_polar_mesh()
        else:
            raise ValueError("Only polar coordinate systme is implemented currently.")

    def _build_polar_mesh(self):
        self._wphi = (self._upper - self._lower) / self._nv
        self._phi = np.arange(self._lower, self._upper, self._wphi)

    @property
    def centers(self):
        if self._centers is None:
            self._centers = np.meshgrid(
                self._radius * np.cos(self.center),
                self._radius * np.sin(self.center),
            )
        return self._centers

    @property
    def center(self):
        return self._phi

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def L(self):
        return (self.upper - self.lower) / 2

    @property
    def nv(self):
        return self._nv

    @property
    def num_nodes(self):
        return [self.nv] * self.num_dim

    @property
    def delta(self):
        return self._wphi

    @property
    def weights(self):
        return (self.delta / (2 * self.L)) * np.eye(*self.num_nodes)
