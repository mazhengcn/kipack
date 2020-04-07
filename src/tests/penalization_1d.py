import copy
import json
import math

import numpy as np
from tqdm.notebook import tnrange

import collision
import pykinetic
from .euler_1d import Euler1D
from pykinetic import riemann


class PenalizationSolver1D(pykinetic.BoltzmannSolver0D):
    def __init__(
        self, kn, penalty, riemann_solver=None, collision_operator=None
    ):
        self.kn = kn
        self.p = penalty
        super(PenalizationSolver1D).__init__(
            riemann_solver, collision_operator
        )

    def dqdt(self, state):
        r"""
        Evaluate dq/dt*(delta t).  This routine is used for implicit time
        stepping.
        """
        kndt = self.dt / (self.kn + self.p * self.dt)
        dqdt = kndt * self.dq_collision(state)
        return dqdt


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


def run(dt=0.001, nt=100, scheme="Euler"):
    with open("./configs/vhs_2d.json") as f:
        config = json.load(f)

    vmesh = collision.VMesh(config)
    rp = riemann.advection_1D
    coll_op = collision.FSInelasticVHSCollision(config, vmesh)

    # tau = None

    def coll(x):
        return coll_op(x, device="gpu")

    if scheme == "BEuler":
        solver = pykinetic.PenalizationSolver1D(1e-3, 10, rp, coll)
        solver.time_integrator = "BEuler"
    else:
        solver = pykinetic.BoltzmannSolver1D(1e-4, rp, coll)
        solver.dt = dt
        if "RK" in scheme:
            solver.time_integrator = "RK"
            solver.a = rkcoeff[scheme]["a"]
            solver.b = rkcoeff[scheme]["b"]
            solver.c = rkcoeff[scheme]["c"]
        else:
            solver.time_integrator = scheme

    print("dt is {}".format(solver.dt))
    # solver.order = 1
    solver.lim_type = 2
    solver.bc_lower[0] = pykinetic.BC.periodic
    solver.bc_upper[0] = pykinetic.BC.periodic

    x = pykinetic.Dimension(0.0, 1.0, 100, name="x")
    domain = pykinetic.Domain([x])
    state = pykinetic.State(domain, vdof=vmesh.nv_s)
    state.problem_data["vx"] = vmesh.v_centers[0]
    qinit(state, vmesh)
    sol = pykinetic.Solution(state, domain)

    with open("./configs/config_1d_2d.json") as f:
        domain_config = json.load(f)
    euler = Euler1D(domain_config)
    euler.set_initial(vmesh.get_p(sol.q)[0], 0, 1.0)

    sol_frames = []
    macro_frames = []
    for _ in tnrange(nt):
        solver.evolve_to_time(sol)
        # l2_err = (
        #     np.sqrt(np.sum((sol.q - bkw_fn(vmesh, sol.t)) ** 2))
        # * vmesh.delta
        # )
        # sol_frames.append([copy.deepcopy(sol), l2_err])
        macro_frames.append(vmesh.get_p(sol.q))
        sol_frames.append(copy.deepcopy(sol))

    euler.solve(dt * nt)

    return macro_frames, euler.macros(2)


def qinit(state, vmesh):
    x = state.grid.x.centers
    rho = 1.0 + 0.5 * np.cos(4 * math.pi * x)
    state.q[:] = maxwellian_vec(vmesh, 0.0, 1.0, rho)


def maxwellian_vec(vmesh, u, T, rho):
    v = vmesh.v_center
    return (
        rho[:, None, None]
        / (2 * math.pi * T)
        * np.exp(-((v - u)[:, None] ** 2 + v ** 2) / (2 * T))
    )


def bkw_fn(vmesh, t):
    vsq = vmesh.vsquare

    K = 1 - 0.5 * np.exp(-t / 8)
    return (
        1
        / (2 * math.pi * K ** 2)
        * np.exp(-0.5 * vsq / K)
        * (2 * K - 1 + 0.5 * vsq * (1 - K) / K)
    )


def ext_Q(vmesh, t):
    vsq = vmesh.vsquare

    K = 1 - np.exp(-t / 8) / 2
    dK = np.exp(-t / 8) / 16
    df = (-2 / K + vsq / (2 * K ** 2)) * bkw_fn(vmesh, t) + 1 / (
        2 * math.pi * K ** 2
    ) * np.exp(-vsq / (2 * K)) * (2 - vsq / (2 * K ** 2))
    return df * dK


def ext_T(rho0, T0, e, tau, t):
    # exact evolution of temperature
    # assume u0 = 0
    return (T0 - 8 * tau / (1 - e ** 2)) * np.exp(
        -(1 - e ** 2) * t * rho0 / 4
    ) + 8 * tau / (1 - e ** 2)


def flat(vmesh, T0):
    vx, vy = vmesh.v_centers
    w = np.sqrt(3 * T0)
    return 1 / 4 / w ** 2 * (vx <= w) * (vx >= -w) * (vy <= w) * (vy >= -w)


def maxwellian(vmesh, K):
    return 1 / (2 * math.pi * K) * np.exp(-0.5 * vmesh.vsq / K)


def anisotropic_f(v):
    return (
        0.8
        * math.pi ** (-1.5)
        * (
            np.exp(
                -(16 ** (1 / 3))
                * (
                    (v - 2)[:, None, None] ** 2
                    + (v - 2)[:, None] ** 2
                    + (v - 2) ** 2
                )
            )
            + np.exp(
                -(v + 0.5)[:, None, None] ** 2
                - (v + 0.5)[:, None] ** 2
                - (v + 0.5) ** 2
            )
        )
    )
