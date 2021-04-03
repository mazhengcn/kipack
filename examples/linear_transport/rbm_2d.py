import copy
import math

import numpy as np
from examples.utils import Progbar
from kipack import collision, pykinetic
from kipack.pykinetic.boltzmann.solver import BoltzmannSolver2D


def maxwellian_vec_init(vmesh, u, T, rho):
    v = vmesh.center
    return (
        rho[:, None]
        / np.sqrt(2 * math.pi * T)
        * np.exp(-((v - u) ** 2) / (2 * T))
    )


def qinit(state, vmesh, init_func):
    x = state.grid.x.centers
    rho = 1.0 + 0.5 * np.cos(4 * math.pi * x)
    vx, vy = vmesh.centers
    state.q[0, ...] = init_func(vx, vy, 0.0, 0.0, 1.0, rho)


def compute_rho(state, vmesh):
    vaxis = tuple(-(i + 1) for i in range(vmesh.num_dim))
    return np.sum(state.q[0, ...] * vmesh.weights, axis=vaxis)


class DiffusiveRegimeSolver2D(BoltzmannSolver2D):
    def __init__(
        self,
        riemann_solver,
        collision_operator,
        kn,
        sigma_s,
        sigma_a,
        Q,
        **kwargs
    ):
        self.sigma_s, self.sigma_a, self.Q = map(
            self._convert_params, [sigma_s, sigma_a, Q]
        )

        super().__init__(
            riemann_solver=riemann_solver,
            collision_operator=collision_operator,
            kn=kn,
            **kwargs,
        )

    def dq(self, state):
        deltaq = self.dq_hyperbolic(state) / self.kn
        deltaq += self.dq_collision(state) / self.kn * self.sigma_s
        deltaq -= self.sigma_a * state.q * self.dt
        deltaq += self.Q * self.dt
        return deltaq


def run(
    kn=1.0,
    sigma_s=lambda x, y: 1.0,
    sigma_a=lambda x, y: 0.0,
    Q=lambda x, y: 0.0,
    xmin=[0.0, 0.0],
    xmax=[1.0, 1.0],
    nx=40,
    dt=0.01,
    nt=1000,
    coll="linear",
    scheme="Euler",
    BC="periodic",
    f_l=lambda vx, vy: 0.0,
    f_b=lambda vx, vy: 0.0,
    f_r=lambda vx, vy: 0.0,
    f_t=lambda vx, vy: 0.0,
    init_func=lambda vmesh, u, T, rho: 0.0,
):
    # Load config
    config = collision.utils.CollisionConfig.from_json(
        "./examples/linear_transport/configs/" + "linear_2d" + ".json"
    )

    # Collision
    vmesh = collision.PolarMesh(config)
    if coll == "linear":
        coll_op = collision.LinearCollision(config, vmesh)
    elif coll == "rbm":
        coll_op = collision.RandomBatchLinearCollision(config, vmesh)
    elif coll == "rbm_symm":
        coll_op = collision.SymmetricRBMLinearCollision(config, vmesh)
    else:
        raise NotImplementedError(
            "Collision method {} is not implemented.".format(coll)
        )

    # x domian
    x = pykinetic.Dimension(xmin[0], xmax[0], nx, name="x")
    y = pykinetic.Dimension(xmin[1], xmax[1], nx, name="y")
    domain = pykinetic.Domain([x, y])

    # Riemann solver
    rp = pykinetic.riemann.advection_2D
    solver = DiffusiveRegimeSolver2D(
        rp,
        coll_op,
        kn=kn(x.centers, y.centers),
        sigma_s=sigma_s(x.centers, y.centers),
        sigma_a=sigma_a(x.centers, y.centers),
        Q=Q(x.centers, y.centers),
    )
    solver.order = 1
    # solver.lim_type = 2
    # Time integrator
    solver.time_integrator = scheme
    solver.dt = dt
    print("dt is {}".format(solver.dt))

    # Boundary condition
    def dirichlet_lower_BC(state, dim, t, qbc, auxbc, num_ghost):
        vx, vy = state.problem_data["v"]
        if dim.name == "x":
            for i in range(num_ghost):
                qbc[0, i, :, (vx > 0)[0], :] = f_l(vx[(vx > 0)[0]], vy)
        elif dim.name == "y":
            for i in range(num_ghost):
                qbc[0, :, i, :, (vy > 0)[:, 0]] = f_b(vx, vy[(vy > 0)[:, 0]])
        else:
            raise ValueError("Dim could be only x or y.")

    def dirichlet_upper_BC(state, dim, t, qbc, auxbc, num_ghost):
        vx, vy = state.problem_data["v"]
        if dim.name == "x":
            for i in range(num_ghost):
                qbc[0, -i - 1, :, (vx < 0)[0], :] = f_r(vx[(vx < 0)[0]], vy)
        elif dim.name == "y":
            for i in range(num_ghost):
                qbc[0, :, -i - 1, :, (vy < 0)[:, 0]] = f_t(
                    vx, vy[(vy < 0)[:, 0]]
                )
        else:
            raise ValueError("Dim could be only 'x' or 'y'.")

    if BC == "periodic":
        solver.all_bcs = pykinetic.BC.periodic
    elif BC == "dirichlet":
        solver.all_bcs = pykinetic.BC.custom
        solver.user_bc_lower = dirichlet_lower_BC
        solver.user_bc_upper = dirichlet_upper_BC
    else:
        raise ValueError("Given BC type is not avaliable!")

    state = pykinetic.State(domain, vmesh, 1)
    state.problem_data["v"] = vmesh.centers
    qinit(state, vmesh, init_func)
    sol = pykinetic.Solution(state, domain)

    output_dict = {}
    sol_frames, macro_frames, ts = (
        [copy.deepcopy(sol)],
        [compute_rho(sol.state, vmesh)],
        [0.0],
    )
    pbar = Progbar(nt)
    for t in range(nt):
        solver.evolve_to_time(sol)
        sol_frames.append(copy.deepcopy(sol))
        macro_frames.append(compute_rho(sol.state, vmesh))
        ts.append(0.0 + (t + 1) * dt)
        pbar.update(t + 1, finalize=False)
    pbar.update(nt, finalize=True)

    output_dict["macro_frames"] = macro_frames
    output_dict["mesh"] = state.c_centers
    output_dict["t"] = ts

    return output_dict
