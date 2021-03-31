import numpy as np

num_eqn = 4
num_waves = 2


def parity_2D(q_l, q_r, aux_l, aux_r, problem_data, idim):
    # Convenience
    num_rp = q_l.shape[1:3]
    num_vdof = q_l.shape[3:]

    # Return values
    wave = np.empty((num_eqn, num_waves) + num_rp + num_vdof)
    s = np.empty((num_waves, 1, 1) + num_vdof)
    amdq = np.zeros((num_eqn,) + num_rp + num_vdof)
    apdq = np.zeros((num_eqn,) + num_rp + num_vdof)

    # Local values
    delta = np.empty(np.shape(q_l))
    delta = q_r - q_l

    # Compute the waves
    # 1-Wave
    a1 = 0.5 * (delta[::2, :] - delta[1::2, :] / problem_data["sqrt_phi"])
    wave[::2, 0, :] = a1
    wave[1::2, 0, :] = -a1 * problem_data["sqrt_phi"]
    s[0, 0] = -problem_data["sqrt_phi"] * problem_data["v"][idim - 1]

    # 2-Wave
    a2 = 0.5 * (delta[::2, :] + delta[1::2, :] / problem_data["sqrt_phi"])
    wave[::2, 1, :] = a2
    wave[1::2, 1, :] = a2 * problem_data["sqrt_phi"]
    s[1, 0] = problem_data["sqrt_phi"] * problem_data["v"][idim - 1]

    # Compute the left going and right going fluctuations
    for m in range(num_eqn):
        amdq[m, :] = s[0, :] * wave[m, 0, :]
        apdq[m, :] = s[1, :] * wave[m, 1, :]

    if idim == 2:
        amdq[:2, :] = -amdq[:2, :]
        apdq[:2, :] = -apdq[:2, :]

    return wave, s, amdq, apdq
