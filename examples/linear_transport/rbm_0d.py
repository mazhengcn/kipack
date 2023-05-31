import copy
import math

import jax
import numpy as np

from kipack import collision, pykinetic
from kipack.utils import Progbar

from .configs import linear

rkcoeff = {
    "RK3": {
        "a": np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0], [-1.0, 2.0, 0.0]]),
        "b": np.array([1 / 6, 2 / 3, 1 / 6]),
        "c": np.array([0, 0.5, 1.0]),
    },
    "RK4": {
        "a": np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        "b": np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        "c": np.array([0.0, 0.5, 0.5, 1.0]),
    },
}


def bkw_fn(vmesh, t):
    vsq = vmesh.vsquare

    K = 1 - 0.5 * np.exp(-t / 8)
    return (
        1
        / (2 * math.pi * K**2)
        * np.exp(-0.5 * vsq / K)
        * (2 * K - 1 + 0.5 * vsq * (1 - K) / K)
    )


def ext_Q(vmesh, t):
    vsq = vmesh.vsquare

    K = 1 - np.exp(-t / 8) / 2
    dK = np.exp(-t / 8) / 16
    df = (-2 / K + vsq / (2 * K**2)) * bkw_fn(vmesh, t) + 1 / (
        2 * math.pi * K**2
    ) * np.exp(-vsq / (2 * K)) * (2 - vsq / (2 * K**2))
    return df * dK


def maxwellian(vmesh, K):
    vsq = vmesh.vsquare
    return 1 / math.sqrt(2 * math.pi * K) * np.exp(-0.5 * vsq / K)


def qinit(state, vmesh):
    state.q[:] = maxwellian(vmesh, 1.0)


def compute_rho(state, vmesh):
    vaxis = tuple(-(i + 1) for i in range(vmesh.num_dim))
    return np.sum(state.q[0, ...] * vmesh.weights, axis=vaxis)


def l2_err(sol, vmesh):
    return np.sqrt(np.sum((sol.q - bkw_fn(vmesh, sol.t)) ** 2)) * vmesh.delta


class RandomBoltzmannSolver0D(pykinetic.BoltzmannSolver0D):
    def __init__(self, riemann_solver, collision_operator, seed: int = 42):
        self.rng = jax.random.PRNGKey(seed)
        super().__init__(
            riemann_solver=riemann_solver,
            collision_operator=collision_operator,
        )

    def dq_collision(self, state):
        collisions = np.zeros(state.q.shape)
        for i in range(state.num_eqn):
            collisions[i, :], self.rng = self.coll[i](state.q[i, :], self.rng)
        return collisions * self.dt / self.kn


def run(kn=1.0, dt=0.01, nt=1000, coll="linear", scheme="Euler"):
    cfg = linear.get_config("1d")

    vmesh = collision.CartesianMesh(cfg)
    if coll == "linear":
        coll_op = collision.LinearCollision(cfg, vmesh)
    elif coll == "rbm":
        coll_op = collision.RandomBatchLinearCollision(cfg, vmesh)
    else:
        raise NotImplementedError(
            "Collision method {} is not implemented.".format(coll)
        )

    solver = pykinetic.BoltzmannSolver0D(collision_operator=coll_op, kn=kn)
    if "RK" in scheme:
        solver.time_integrator = "RK"
        solver.a = rkcoeff[scheme]["a"]
        solver.b = rkcoeff[scheme]["b"]
        solver.c = rkcoeff[scheme]["c"]
    else:
        solver.time_integrator = scheme
    solver.dt = dt

    domain = pykinetic.Domain([])
    state = pykinetic.State(domain, vmesh, 1)

    qinit(state, vmesh)

    sol = pykinetic.Solution(state, domain)

    output_dict = {}
    sol_frames, macro_frames, ts, l2_errs = (
        [copy.deepcopy(sol)],
        [compute_rho(sol.state, vmesh)],
        [0.0],
        [l2_err(sol, vmesh)],
    )
    pbar = Progbar(nt)
    for t in range(nt):
        solver.evolve_to_time(sol)
        l2_errs.append(l2_err(sol, vmesh))
        sol_frames.append([copy.deepcopy(sol), l2_err])
        macro_frames.append(compute_rho(sol.state, vmesh))
        ts.append(0.0 + (t + 1) * dt)
        pbar.update(t + 1, finalize=False)
    pbar.update(nt, finalize=True)

    output_dict.update(
        {
            "macro_frames": macro_frames,
            "sol_frames": sol_frames,
            "t": ts,
            "errors": l2_errs,
        }
    )

    return output_dict


def flat(vmesh, T0):
    vx, vy = vmesh.centers
    w = np.sqrt(3 * T0)
    return 1 / 4 / w**2 * (vx <= w) * (vx >= -w) * (vy <= w) * (vy >= -w)


def anisotropic_f(v):
    return (
        0.8
        * math.pi ** (-1.5)
        * (
            np.exp(
                -(16 ** (1 / 3))
                * ((v - 2)[:, None, None] ** 2 + (v - 2)[:, None] ** 2 + (v - 2) ** 2)
            )
            + np.exp(
                -(v + 0.5)[:, None, None] ** 2
                - (v + 0.5)[:, None] ** 2
                - (v + 0.5) ** 2
            )
        )
    )
