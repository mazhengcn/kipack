import copy
import math

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from kipack import collision, pykinetic
from kipack.utils import Progbar

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file(
    name="config",
    help_string="Configuration file.",
)


class PenalizationSolver0D(pykinetic.BoltzmannSolver0D):
    def __init__(self, collision_operator, **kwargs):
        self.p = kwargs.get("penalty", 0.0)

        super().__init__(collision_operator=collision_operator, **kwargs)

    def dqdt(self, state):
        r"""
        Evaluate dq/dt*(delta t).  This routine is used for implicit time
        stepping.
        """
        pdt_kn = 1.0 - self.dt * self.p / self.kn
        dqdt = self.dq_collision(state) / pdt_kn
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


def main(kn=1.0, tau=0.1, p=1.0, dt=0.01, nt=1000, scheme="Euler"):
    config = collision.utils.CollisionConfig.from_json("./configs/penalty.json")
    vmesh = collision.SpectralMesh(config)
    coll_op = collision.FSInelasticVHSCollision(config, vmesh)

    tau = tau * kn

    print("tau is {}.".format(tau))

    if scheme == "BEuler":
        solver = PenalizationSolver0D(
            collision_operator=coll_op, penalty=p, kn=kn, heat_bath=tau
        )
        solver.time_integrator = "BEuler"
    else:
        solver = pykinetic.BoltzmannSolver0D(
            collision_operator=coll_op, kn=kn, heat_bath=tau
        )
        if "RK" in scheme:
            print("OK")
            solver.time_integrator = "RK"
            solver.a = rkcoeff[scheme]["a"]
            solver.b = rkcoeff[scheme]["b"]
            solver.c = rkcoeff[scheme]["c"]
        else:
            solver.time_integrator = scheme
    solver.dt = dt

    domain = pykinetic.Domain([])
    state = pykinetic.State(domain, vdof=vmesh.nvs)

    qinit(state, vmesh)

    sol = pykinetic.Solution(state, domain)
    sol_frames = []
    T_frames = []
    pbar = Progbar(nt)
    for t in range(nt):
        solver.evolve_to_time(sol)
        T_frames.append(vmesh.get_p(sol.q)[-1])
        sol_frames.append(copy.deepcopy(sol))
        pbar.update(t + 1, finalize=False)
    pbar.update(nt, finalize=True)

    return T_frames, sol_frames, vmesh, coll_op, solver


def qinit(state, vmesh):
    state.q[:] = bkw_fn(vmesh, 8.0)


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


def ext_T(t, e, kn, tau, rho0, T0):
    # exact evolution of temperature
    # assume u0 = 0
    t = t / kn
    tau = tau * kn
    return (T0 - 8 * tau / (1 - e**2)) * np.exp(
        -(1 - e**2) * t * rho0 / 4
    ) + 8 * tau / (1 - e**2)


def flat(vmesh, T0):
    vx, vy = vmesh.centers
    w = np.sqrt(3 * T0)
    return 1 / 4 / w**2 * (vx <= w) * (vx >= -w) * (vy <= w) * (vy >= -w)


def maxwellian(vmesh, K):
    return 1 / (2 * math.pi * K) * np.exp(-0.5 * vmesh.vsq / K)


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


if __name__ == "__main__":
    app.run(main)
