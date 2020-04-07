import json
import math
import copy
import numpy as np

from tqdm.notebook import tnrange
import collision
import pykinetic


class PenalizationSolver0D(pykinetic.BoltzmannSolver0D):
    def __init__(self, kn, penalty, collision_operator):
        self.kn = kn
        self.p = penalty
        super(PenalizationSolver0D, self).__init__(
            kn, penalty, collision_operator
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


def run(kn=1, tau=0.1, p=5, dt=0.01, nt=1000, scheme="Euler"):
    with open("./configs/vhs_2d.json") as f:
        config = json.load(f)

    vmesh = collision.VMesh(config)
    coll_op = collision.FSInelasticVHSCollision(config, vmesh)

    tau = tau * kn

    print(tau)

    def coll(x):
        return coll_op(x, heat_bath=tau, device="gpu")

    if scheme == "BEuler":
        solver = PenalizationSolver0D(
            kn=kn, penalty=p, collision_operator=coll
        )
        solver.time_integrator = "BEuler"
    else:
        solver = pykinetic.BoltzmannSolver0D(kn=kn, collision_operator=coll)
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
    state = pykinetic.State(domain, vdof=vmesh.nv_s)

    qinit(state, vmesh)

    sol = pykinetic.Solution(state, domain)
    sol_frames = []
    T_frames = []
    for _ in tnrange(nt):
        solver.evolve_to_time(sol)
        # l2_err = (
        #     np.sqrt(np.sum((sol.q - bkw_fn(vmesh, sol.t)) ** 2))
        # * vmesh.delta
        # )
        # sol_frames.append([copy.deepcopy(sol), l2_err])
        T_frames.append(vmesh.get_p(sol.q)[-1])
        sol_frames.append(copy.deepcopy(sol))

    return T_frames, sol_frames, vmesh, coll, solver


def qinit(state, vmesh):

    state.q[:] = flat(vmesh, 8.0)


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


def ext_T(t, e, kn, tau, rho0, T0):
    # exact evolution of temperature
    # assume u0 = 0
    t = t / kn
    tau = tau * kn
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
