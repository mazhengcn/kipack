import numpy as np


def limit(num_eqn, wave, s, limiter, dtdx):
    r"""
    Apply a limiter to the waves

    Function that limits the given waves using the methods contained
    in limiter.  This is the vectorized version of the function acting on a
    row of waves at a time.

    :Input:
     - *wave* - (ndarray(num_eqn,num_waves,:)) The waves at each interface
     - *s* - (ndarray(:,num_waves)) Speeds for each wave
     - *limiter* - (``int`` list) Array of type ``int`` determining which
         limiter to use
     - *dtdx* - (ndarray(:)) :math:`\Delta t / \Delta x` ratio, used for CFL
        dependent limiters

    :Output:
     - (ndarray(:,num_eqn,num_waves)) - Returns the limited waves
    """

    # wave_norm2 is the sum of the squares along the num_eqn axis,
    # so the norm of the cell i for wave number j is addressed
    # as wave_norm2[i,j]
    wave_norm2 = np.sum(wave**2, axis=0)
    wave_zero_mask = np.array((wave_norm2 == 0), dtype=float)
    wave_nonzero_mask = 1.0 - wave_zero_mask

    # dotls contains the products of adjacent cell values summed
    # along the num_eqn axis.  For reference, dotls[0,:,:] is the dot
    # product of the 0 cell and the 1 cell.
    dotls = np.sum(
        wave[:, :, 1:] * wave[:, :, :-1], axis=0
    )  # (num_waves, rp, num_vnodes)
    spos = np.array(s > 0.0, dtype=float)  # (num_waves, 1, num_vnodes)

    # Here we construct a masked array, then fill the empty values with 0,
    # this is done in case wave_norm2 is 0 or close to it
    # Take upwind dot product
    r = np.ma.array((spos * dotls[:, :-1] + (1.0 - spos) * dotls[:, 1:]))
    # Divide it by the norm**2
    r /= np.ma.array(wave_norm2[:, 1:-1])
    # Fill the rest of the array
    r.fill_value = 0
    r = r.filled()  # (num_waves, rp, num_vnodes)

    for mw in range(wave.shape[1]):
        limit_func = limiter_functions.get(limiter)
        if limit_func is not None:
            for m in range(num_eqn):
                cfl = np.abs(
                    s[mw, :] * dtdx[1:-2] * spos[mw, :] + (1 - spos[mw, :]) * dtdx[2:-1]
                )
                wlimitr = limit_func(r[mw, :], cfl)
                wave[m, mw, 1:-1] = (
                    wave[m, mw, 1:-1] * wave_zero_mask[mw, 1:-1]
                    + wlimitr * wave[m, mw, 1:-1] * wave_nonzero_mask[mw, 1:-1]
                )

    return wave


def minmod_limiter(r, cfl):
    r"""
    Minmod vectorized limiter
    """
    a = np.ones((2,) + r.shape)
    b = np.zeros((2,) + r.shape)

    a[1, :] = r
    b[1, :] = np.minimum(a[0], a[1])

    return np.maximum(b[0], b[1])


def superbee_limiter(r, cfl):
    r"""
    Superbee vectorized limiter
    """
    a = np.ones((2,) + r.shape)
    b = np.zeros((2,) + r.shape)
    c = np.zeros((3,) + r.shape)

    a[1, :] = 2.0 * r

    b[1, :] = r

    c[1, :] = np.minimum(a[0], a[1])
    c[2, :] = np.minimum(b[0], b[1])

    return np.max(c, axis=0)


def van_leer_klein_sharpening_limiter(r, cfl):
    r"""
    van Leer with Klein sharpening, k=2
    """
    a = np.ones((2,) + r.shape) * 1.0e-5
    a[0, :] = r

    rcorr = np.maximum(a[0], a[1])
    a[1, :] = 1 / rcorr
    sharg = np.minimum(a[0], a[1])
    sharp = 1.0 + sharg * (1.0 - sharg) * (1.0 - sharg**2)

    return (r + np.abs(r)) / (1.0 + np.abs(r)) * sharp


limiter_functions = {
    1: minmod_limiter,
    2: superbee_limiter,
    3: van_leer_klein_sharpening_limiter,
}
