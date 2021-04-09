# encoding: utf-8
r"""
Euler 2D Quadrants example
==========================

Simple example solving the Euler equations of compressible fluid dynamics:

.. math::
    \rho_t + (\rho u)_x + (\rho v)_y & = 0 \\
    (\rho u)_t + (\rho u^2 + p)_x + (\rho uv)_y & = 0 \\
    (\rho v)_t + (\rho uv)_x + (\rho v^2 + p)_y & = 0 \\
    E_t + (u (E + p) )_x + (v (E + p))_y & = 0.

Here :math:`\rho` is the density, (u,v) is the velocity,
and E is the total energy. The initial condition is one of the 2D Riemann
problems from the paper of Liska and Wendroff.

"""
from __future__ import absolute_import

import numpy as np
from clawpack import pyclaw, riemann
from clawpack.riemann.euler_4wave_2D_constants import (
    density,
    energy,
    num_eqn,
    x_momentum,
    y_momentum,
)


def q_src(solver, state, dt):
    q = state.q
    q[3] += 0.5 * dt * (0.5 * (q[1] ** 2 + q[2] ** 2) - q[0] * q[3])


class Euler2D(object):
    def __init__(self, domain_config):
        solver = pyclaw.ClawSolver2D(riemann.euler_4wave_2D)
        solver.all_bcs = pyclaw.BC.periodic

        xmin, xmax = domain_config.x_range
        ymin, ymax = domain_config.y_range
        nx, ny = domain_config.num_cells

        domain = pyclaw.Domain([xmin, ymin], [xmax, ymax], [nx, ny])
        solution = pyclaw.Solution(num_eqn, domain)
        solution.problem_data["gamma"] = 2.0
        solver.dimensional_split = False
        solver.transverse_waves = 2
        solver.step_source = q_src

        claw = pyclaw.Controller()
        claw.solution = solution
        claw.solver = solver

        claw.output_format = "ascii"
        claw.outdir = "./_output"

        self._solver = solver
        self._domain = domain
        self._solution = solution
        self._claw = claw

    def set_initial(
        self, rho_mean, rho_var, ux=0.0, uy=0.0, T=1.0, dist="uniform"
    ):
        self._solution.t = 0.0
        # Set initial data
        xx, _ = self._domain.grid.p_centers
        solution = self._solution
        solution.q[density, ...] = rho_mean + rho_var * np.random.uniform(
            0.0, 1.0, size=xx.shape
        )
        u, v = ux, uy
        solution.q[x_momentum, ...] = solution.q[density, ...] * u
        solution.q[y_momentum, ...] = solution.q[density, ...] * v
        T = T
        solution.q[energy, ...] = (
            0.5 * solution.q[density, ...] * (u ** 2 + v ** 2)
            + solution.q[density, ...] * T
        )

    def solve(self, tfinal=5.0):
        self._claw.tfinal = tfinal
        # remove old frames
        # self._claw.frames = []
        self._claw.run()

    def plot(self):
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(figsize=(8, 6))
        cs = axes.contourf(
            self._claw.solution.q[energy, ...]
            / self._claw.solution.q[density, ...]
            - 0.5
            * (
                self._claw.solution.q[x_momentum, ...] ** 2
                + self._claw.solution.q[y_momentum, ...] ** 2
            )
        )
        plt.colorbar(cs)
        plt.savefig("out_put.pdf")
        plt.show()
