from abc import ABCMeta, abstractmethod

import numpy as np


class Collision(object, metaclass=ABCMeta):
    def __init__(self, config, velocity_mesh, heat_bath=None, device="cpu"):

        # Load configuration
        self.config = config
        self.vm = velocity_mesh
        # Load dimension
        self.num_dim = self.config.collision_model.dim
        assert (
            self.num_dim == self.vm.num_dim
        ), "'spectral_mesh' has different \
            velocity dimensions from collision model! This may be caused by \
            using different config files. Please check the consistency."

        self.heat_bath = heat_bath
        # Device: cpu or gpu
        self.device = device
        # Read model parameters
        self.load_parameters()
        # Perform precomputation
        self.perform_precomputation()

        if self.device == "cpu":
            self.build = self._build_cpu
        elif self.device == "gpu":
            self.build = self._build_gpu
        else:
            raise ValueError("Device must be 'cpu' or 'gpu'.")

        self._built = False
        self._input_shape = None

    def __call__(self, input_f, heat_bath=0.0):
        """Compute one step collision."""
        if not self._built:
            self.build(input_f.shape)

        output = self.collide(input_f)

        if heat_bath:
            output += heat_bath * self.laplacian(input_f)

        return output

    # get primitive macroscopic quantities [rho, u, T]
    def get_p(self, input_f: np.ndarray):
        return self.vm.get_p(input_f)

    # get conserved macroscopic quantities [rho, m, E]
    def get_F(self, input_f: np.ndarray):
        return self.vm.get_F(input_f)

    @abstractmethod
    def load_parameters(self):
        pass
