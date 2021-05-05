from abc import ABCMeta, abstractmethod

import cupy as cp


class BaseCollision(object, metaclass=ABCMeta):
    def __init__(self, config, velocity_mesh, **kwargs):

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

        if "sigma" in kwargs.keys():
            self.sigma = kwargs["sigma"]

        # Read model parameters
        self.load_parameters()

        # Perform precomputation
        self.perform_precomputation()

        self._built_cpu = False
        self._built_gpu = False
        self._input_shape = None

    def __call__(self, input_f, heat_bath=None, device="cpu"):
        output = None
        if device == "cpu":
            if not self._built_cpu or input_f.shape != self._input_shape:
                # Broadcast kernels and build
                self._build_cpu(input_f.shape)
            # Select cpu
            self._set_to_cpu()
            # Collide
            output = self.collide(input_f)
            if heat_bath:
                output += heat_bath * self.laplacian(input_f)
        elif device == "gpu":
            if not self._built_gpu or input_f.shape != self._input_shape:
                # Broadcast kernels and build
                self._build_gpu(input_f.shape)
            # Select cpu
            self._set_to_gpu()
            # Copy input to gpu
            gpu_f = cp.asarray(input_f)
            # Collide
            output = self.collide(gpu_f)
            if heat_bath:
                output += heat_bath * self.laplacian(gpu_f)
            # Copy back to cpu
            output = output.get()

        return output

    # get primitive macroscopic quantities [rho, u, T]
    def get_p(self, input_f):
        return self.vm.get_p(input_f)

    # get conserved macroscopic quantities [rho, m, E]
    def get_F(self, input_f):
        return self.vm.get_F(input_f)

    @abstractmethod
    def load_parameters(self):
        pass

    @abstractmethod
    def collide(self, input_f):
        pass
