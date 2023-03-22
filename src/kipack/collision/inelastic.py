import math

import jax
import jax.numpy as jnp
import numpy as np
import pyfftw
from absl import logging
from scipy import special

from .base import Collision


class FSInelasticVHSCollision(Collision):
    """Fast spectral method to compute (in)elastic collsions."""

    collision_model = "vhs"

    @property
    def e(self):
        return self._e

    def load_parameters(self):
        """Load computation parameters."""

        # Load collision model (e and gamma)
        collision_model = self.config.collision_model
        self._e = collision_model.e
        logging.info(f"e: {self._e}")
        self._gamma = collision_model.gamma

        # Compute prefactor and exponential factor
        L = self.vm.L
        self._exp_fac = 0.25 * math.pi * (1 + self._e) / L

        # Special functions
        if self.num_dim == 2:
            self._sp_func = lambda x: special.jv(0, x)
        elif self.num_dim == 3:
            self._sp_func = lambda x: np.sinc(x / np.pi)
        else:
            raise ValueError("Dimension must be 2 or 3.")

        # Sphere area constant
        self._sphr_fac = (
            2
            * math.pi ** (0.5 * self.num_dim)
            / math.gamma(0.5 * self.num_dim)
        )

    def _pre_fac(self, r):
        return self._sphr_fac * r ** (self._gamma + self.num_dim - 1)

    def perform_precomputation(self):
        """Perform precomputation."""

        # Spectral index
        # Compute index for spectral method
        n = self.vm.nv
        # Spectral index and norm
        idx = np.fft.fftshift(np.arange(-int(n / 2), int(n / 2)))
        # idx_norm has shape (nv, nv, nv) or (nv, nv)
        idx_norm, idx_square = 0, idx**2
        for i in range(self.num_dim):
            idx_norm = idx_norm + idx_square[(...,) + (None,) * i]
        idx_norm = np.sqrt(idx_norm)
        # Laplacian
        lapl = -math.pi**2 / self.vm.L**2 * idx_norm**2

        # Quadrature points and weights
        r, wr = self.vm.rquad()
        sigma, wsigma = self.vm.circ_or_sphr_quad()
        quad_slice = (...,) + (None,) * self.num_dim

        # Gain term
        # Dot with index
        # index_mesh has shape (nr, nsigma, nv, nv) (2D)
        # or (nv, nsigma, nv, nv, nv)
        idx_mesh = 0
        for i in range(self.num_dim):
            idx_mesh = (
                np.expand_dims(idx_mesh, axis=-1)
                + idx * sigma[(slice(None), slice(i, i + 1)) + (None,) * i]
            )
        # Compute gain kernel
        r_gain = r[quad_slice + (None,)]
        wr_gain = wr[quad_slice + (None,)]
        idx_mesh = idx_mesh * r_gain
        gain_kern = (
            self._pre_fac(r_gain)
            * np.exp(1j * self._exp_fac * idx_mesh)
            * self._sp_func(self._exp_fac * r_gain * idx_norm)
        )
        gain_kern *= wr_gain
        gain_kern *= wsigma
        # Exponential kernel
        exp = np.exp(-1j * math.pi * idx_mesh / self.vm.L)

        # Loss term
        # Compute loss kernel
        r_loss = r[quad_slice]
        wr_loss = wr[quad_slice]
        loss_kern = self._pre_fac(r_loss)
        loss_kern *= wr_loss
        # Special function
        sp = self._sp_func(math.pi * r_loss * idx_norm / self.vm.L)
        # CPU Kernels as a dict
        self._cpu_kernels = {
            "gain": gain_kern,
            "exp": exp,
            "loss": loss_kern,
            "sp": sp,
            "lapl": lapl,
        }

        # Copy arrays to GPU
        gain_kern_gpu = jnp.array(gain_kern)
        exp_gpu = jnp.array(exp)
        loss_kern_gpu = jnp.array(loss_kern)
        sp_gpu = jnp.array(sp)
        lapl_gpu = jnp.array(lapl)
        # GPU kernels as a dict
        self._gpu_kernels = {
            "gain": gain_kern_gpu,
            "exp": exp_gpu,
            "loss": loss_kern_gpu,
            "sp": sp_gpu,
            "lapl": lapl_gpu,
        }

        logging.info("Collision model precomputation finished!")

    def _build_cpu(self, input_shape: list[int] | tuple[int]):
        # Pyfftw routines
        # Pyfftw config
        pyfftw.config.NUM_THREADS = 8
        pyfftw.config.PLANNER_EFFORT = "FFTW_ESTIMATE"
        # Compute axis
        axis = tuple(-(i + 1) for i in range(self.num_dim))
        # fft0
        arr0 = pyfftw.empty_aligned(input_shape)
        fftw0 = pyfftw.builders.fftn(arr0, axes=axis)
        ifftw0 = pyfftw.builders.ifftn(arr0, axes=axis)
        # fft1
        arr1 = pyfftw.empty_aligned((self.vm.nr,) + input_shape)
        fftw1 = pyfftw.builders.fftn(arr1, axes=axis)
        ifftw1 = pyfftw.builders.ifftn(arr1, axes=axis)
        # fft2
        arr2 = pyfftw.empty_aligned(
            (self.vm.nr, self.vm.ncirc_or_nsphr) + input_shape
        )
        fftw2 = pyfftw.builders.fftn(arr2, axes=axis)
        ifftw2 = pyfftw.builders.ifftn(arr2, axes=axis)
        # ffts dict (cpu)
        self.ffts_cpu = [fftw0, fftw1, fftw2]
        self.iffts_cpu = [ifftw0, ifftw1, ifftw2]

        # Broadcast kernels
        num_extr_dim = len(input_shape) - self.num_dim
        self._cpu_kernels = _broadcast_kernels(
            self._cpu_kernels, self.num_dim, num_extr_dim
        )
        # Save shape
        self._input_shape = input_shape
        # Set built as true
        self._built_cpu = True

    def _build_gpu(self, input_shape: list[int] | tuple[int]):
        # Compute axis
        axis = tuple(-(i + 1) for i in range(self.num_dim))

        # cufft
        def cufft(x):
            return jnp.fft.fftn(x, axes=axis)

        def cuifft(x):
            return jnp.fft.ifftn(x, axes=axis)

        # cufft = lambda x: cp.fft.fftn(x, axes=axis)
        # cuifft = lambda x: cp.fft.ifftn(x, axes=axis)
        self.ffts_gpu = [cufft] * 3
        self.iffts_gpu = [cuifft] * 3

        # Broadcast kernels
        num_extr_dim = len(input_shape) - self.num_dim
        self._gpu_kernels = _broadcast_kernels(
            self._gpu_kernels, self.num_dim, num_extr_dim
        )
        # Save shape
        self._input_shape = input_shape
        # Set built as true
        self._built_gpu = True

    def _set_to_cpu(self):
        self.kernels = self._cpu_kernels
        self.ffts = self.ffts_cpu
        self.iffts = self.iffts_cpu

    def _set_to_gpu(self):
        self.kernels = self._gpu_kernels
        self.ffts = self.ffts_gpu
        self.iffts = self.iffts_gpu

    def collide(
        self, input_f: np.ndarray | jax.Array
    ) -> np.ndarray | jax.Array:
        """Compute the collision for given density function f

        Arguments:
            input_f: density function has shape [..., vmesh]

        Returns:
            collision results
        """
        # fft of input
        f_hat = self.ffts[0](input_f)
        # Gain
        gain_hat = self.ffts[2](
            self.iffts[2](self.kernels["exp"] * f_hat) * input_f
        )
        # Multiplied by the gain kernel
        gain_hat *= self.kernels["gain"]
        gain_hat = gain_hat.sum(axis=(0, 1))
        gain = self.iffts[0](gain_hat)
        # Loss
        loss = self.iffts[1](self.kernels["sp"] * f_hat)
        loss *= self.kernels["loss"] * input_f
        loss = loss.sum(axis=0)
        # Output
        return (gain / (self._sphr_fac) - loss).real

    def laplacian(self, input_f: np.ndarray) -> np.ndarray:
        return self.iffts[0](self.kernels["lapl"] * self.ffts[0](input_f)).real


def _broadcast_kernels(
    kernels: dict[str, np.ndarray], dim: int, num_extr_dim: int
) -> dict[str, np.ndarray]:
    """Compute broadcasted shapes."""

    # Expand gain kernels
    expand_loss_slice = (slice(None),) + (None,) * num_extr_dim
    kernels["loss"] = kernels["loss"].squeeze()[
        expand_loss_slice + (None,) * dim
    ]
    kernels["sp"] = kernels["sp"].squeeze()[expand_loss_slice]
    # Expand loss kernels
    expand_gain_slice = (slice(None),) + expand_loss_slice
    kernels["gain"] = kernels["gain"].squeeze()[expand_gain_slice]
    kernels["exp"] = kernels["exp"].squeeze()[expand_gain_slice]

    return kernels
