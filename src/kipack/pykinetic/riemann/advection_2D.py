import numpy as np

num_eqn = 1
num_waves = 1


def advection_2D(q_l, q_r, aux_l, aux_r, problem_data, idim):
    # Number of Riemann problems we are solving
    num_rp = q_l.shape[1:3]
    num_vdof = q_l.shape[3:]

    # Return values
    wave = np.empty((num_eqn, num_waves) + num_rp + num_vdof)
    s = np.empty((num_waves, 1, 1) + num_vdof)
    amdq = np.zeros((num_eqn,) + num_rp + num_vdof)
    apdq = np.zeros((num_eqn,) + num_rp + num_vdof)

    wave[0, 0, :] = q_r[0, :] - q_l[0, :]
    s[0, 0, 0, :] = problem_data["v"][idim - 1]

    apdq[0, :] = np.maximum(s[0, 0, 0, :], 0.0) * wave[0, 0, :]
    amdq[0, :] = np.minimum(s[0, 0, 0, :], 0.0) * wave[0, 0, :]

    return wave, s, amdq, apdq
