import numpy as np


def advection_1D(q_l, q_r, aux_l, aux_r, problem_data):
    r"""Basic 1d advection riemann solver
    *problem_data* should contain -
     - *vx* - (float) Determines advection speed
    """

    # wave shape: q.shape = (nx, nv, nv) (2D)
    wave = q_r - q_l
    # s shape: v shape = (nv, nv) (2D)
    s = problem_data["vx"]
    apdq = np.maximum(s, 0.0) * wave
    amdq = np.minimum(s, 0.0) * wave

    return wave, s, amdq, apdq
