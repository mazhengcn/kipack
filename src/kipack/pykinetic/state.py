#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing all Pyclaw solution objects
"""


from __future__ import absolute_import


class State(object):
    r"""
    A State object contains the current state on a particular patch,
    including the unkowns q, the time t, and the auxiliary coefficients aux.

    The variables num_eqn and num_aux determine the length of the first
    dimension of the q and aux arrays.

    :State Data:

        The arrays :attr:`q`, and :attr:`aux` have variable
        extents based on the patch dimensions and the values of
        :attr:`num_eqn` and :attr:`num_aux`.

    A State object is automatically created upon instantiation of a Solution
    object from a Domain object:

        >>> x = pykinetic.Dimension('x',0.0,1.0,100)
        >>> domain = pykinetic.Domain(x)
        >>> vdof = [64, 64]
        >>> solution = pykinetic.Solution(domain, vdof)
        >>> print solution.state
        PyKinetic State object
        Patch dimensions: [100]
        Time  t=0.0
        Number of conserved quantities: 1
        <BLANKLINE>

    A State lives on a Patch, and can be instantiated directly
    by first creating a Patch:

        >>> x = pykinetic.Dimension('x',0.,1.,100)
        >>> patch = pykinetic.Patch((x))

    The arguments to the constructor are the patch, the number of equations,
    and the number of auxiliary fields:

        >>> state = pykinetic.State(patch,3,2)
        >>> state.q.shape
        (3, 100)
        >>> state.aux.shape
        (2, 100)
        >>> state.t
        0.0

    Note that state.q and state.aux are initialized as empty arrays
    (not zeroed).
    Additional parameters, such as scalar values that are used in the Riemann
    solver,
    can be set using the dictionary state.problem_data.
    """

    def __getattr__(self, key):
        if key in (
            "num_dim",
            "p_centers",
            "p_edges",
            "c_centers",
            "c_edges",
            "num_cells",
            "lower",
            "upper",
            "delta",
            "centers",
            "edges",
            "gauges",
        ):
            return self._get_grid_attribute(key)
        else:
            raise AttributeError(
                "'State' object has no attribute '" + key + "'"
            )

    def _get_grid_attribute(self, name):
        r"""
        Return grid attribute

        :Output:
         - (id) - Value of attribute from ``grid``
        """
        return getattr(self.grid, name)

    # ========== Property Definitions ========================================
    @property
    def vdof(self):
        r"""(int) - Number of unknowns (components of q)"""
        if self.q is None:
            raise Exception("state.vdof has not been set.")
        else:
            return self.q.shape[self.num_dim :]

    @property
    def num_aux(self):
        r"""(int) - Number of unknowns (components of q)"""
        if self.aux is None:
            return None
        else:
            return self.aux.shape[self.num_dim :]

    # @property
    # def num_aux(self):
    #     r"""(int) - Number of auxiliary fields"""
    #     if self.aux is not None:
    #         return self.aux.shape[-1]
    #     else:
    #         return 0

    @property
    def grid(self):
        return self.patch.grid

    @property
    def mp(self):
        r"""(int) - Number of derived quantities"""
        if self.p is not None:
            return self.p.shape[-1]
        else:
            return 0

    @mp.setter
    def mp(self, mp):
        if self.p is not None:
            raise Exception("Cannot change state.mp after aux is initialized.")
        else:
            self.p = self.new_array(mp)

    @property
    def mF(self):
        r"""(int) - Number of output functionals"""
        if self.F is not None:
            return self.F.shape[-1]
        else:
            return 0

    @mF.setter
    def mF(self, mF):
        if self.F is not None:
            raise Exception("Cannot change state.mF after aux is initialized.")
        else:
            self.F = self.new_array(mF)

    # ========== Class Methods ===============================================
    def __init__(self, geom, vdof, num_aux=None):
        from kipack.pykinetic import geometry

        if isinstance(geom, geometry.Patch):
            self.patch = geom
        elif isinstance(geom, geometry.Domain):
            self.patch = geom.patches[0]
        else:
            raise Exception(
                """A State object must be initialized with
                             a Patch object."""
            )

        # ========== Attribute Definitions ===================================
        r"""pykinetic.Patch.patch - The patch this state lives on"""
        self.p = None
        r"""(ndarray(mp,...)) - Cell averages of derived quantities."""
        self.F = None
        r"""(ndarray(mF,...)) - Cell averages of output functional densities.
        """
        self.problem_data = {}
        r"""(dict) - Dictionary of global values for this patch,
            ``default = {}``"""
        self.t = 0.0
        r"""(float) - Current time represented on this patch,
            ``default = 0.0``"""
        self.index_capa = -1
        self.keep_gauges = False
        r"""(bool) - Keep gauge values in memory for every time step,
        ``default = False``"""
        self.gauge_data = []
        r"""(list) - List of numpy.ndarray objects. Each element of the list
        stores the values of the corresponding gauge if ``keep_gauges`` is set
        to ``True``"""

        self.q = self.new_array(vdof)
        self.aux = self.new_array(num_aux)

    def __str__(self):
        output = "PyKinetic State object\n"
        output += "Patch dimensions: %s\n" % str(self.patch.num_cells_global)
        output += "Time  t=%s\n" % (self.t)
        output += "Number of conserved quantities: %s\n" % str(self.q.shape[0])
        if self.aux is not None:
            output += "Number of auxiliary fields: %s\n" % str(
                self.aux.shape[0]
            )
        if self.problem_data != {}:
            output += "problem_data: " + self.problem_data.__str__()
        return output

    def is_valid(self):
        r"""
        Checks to see if this state is valid

        The state is declared valid based on the following criteria:
            - :attr:`q` is Fortran contiguous
            - :attr:`aux` is Fortran contiguous

        A debug logger message will be sent documenting exactly what was not
        valid.

        :Output:
         - (bool) - True if valid, false otherwise.

        """
        # import logging

        valid = True
        # logger = logging.getLogger("pyclaw.solution")
        # if not self.q.flags["F_CONTIGUOUS"]:
        #     logger.debug("q array is not Fortran contiguous.")
        #     valid = False
        # if self.aux is not None:
        #     if not self.aux.flags["F_CONTIGUOUS"]:
        #         logger.debug("aux array is not Fortran contiguous.")
        #         valid = False
        return valid

    def set_num_ghost(self, num_ghost):
        """
        Virtual routine (does nothing).  Overridden in the petclaw.state class.
        """
        pass

    def set_q_from_qbc(self, num_ghost, qbc):
        """
        Set the value of q using the array qbc. Typically this is called
        after qbc has been updated by the solver.
        """

        num_dim = self.patch.num_dim
        if num_dim == 0:
            self.q = qbc[:]
        elif num_dim == 1:
            self.q = qbc[num_ghost:-num_ghost]
        elif num_dim == 2:
            self.q = qbc[num_ghost:-num_ghost, num_ghost:-num_ghost]
        elif num_dim == 3:
            self.q = qbc[
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
            ]
        else:
            raise Exception("Assumption (0 <= num_dim <= 3) violated.")

    def set_aux_from_auxbc(self, num_ghost, auxbc):
        """
        Set the value of aux using the array auxbc.
        """

        patch = self.patch
        if patch.num_dim == 0:
            self.aux = auxbc[:]
        elif patch.num_dim == 1:
            self.aux = auxbc[num_ghost:-num_ghost]
        elif patch.num_dim == 2:
            self.aux = auxbc[num_ghost:-num_ghost, num_ghost:-num_ghost]
        elif patch.num_dim == 3:
            self.aux = auxbc[
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
            ]
        else:
            raise Exception("Assumption (0 <= num_dim <= 3) violated.")

    def get_qbc_from_q(self, num_ghost, qbc):
        """
        Fills in the interior of qbc by copying q to it.
        """
        num_dim = self.patch.num_dim
        if num_dim == 0:
            qbc[:] = self.q
        elif num_dim == 1:
            qbc[num_ghost:-num_ghost] = self.q
        elif num_dim == 2:
            qbc[num_ghost:-num_ghost, num_ghost:-num_ghost] = self.q
        elif num_dim == 3:
            qbc[
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
            ] = self.q
        else:
            raise Exception("Assumption (0 <= num_dim <= 3) violated.")

        return qbc

    def get_auxbc_from_aux(self, num_ghost, auxbc):
        """
        Fills in the interior of auxbc by copying aux to it.
        """
        num_dim = self.patch.num_dim
        if num_dim == 0:
            auxbc[:] = self.aux
        elif num_dim == 1:
            auxbc[num_ghost:-num_ghost] = self.aux
        elif num_dim == 2:
            auxbc[num_ghost:-num_ghost, num_ghost:-num_ghost] = self.aux
        elif num_dim == 3:
            auxbc[
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
            ] = self.aux
        else:
            raise Exception("Assumption (0 <= num_dim <= 3) violated.")

        return auxbc

    # ========== Copy functionality ==========================================
    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo={}):
        import copy

        result = self.__class__(
            copy.deepcopy(self.patch), self.vdof, self.num_aux
        )
        result.__init__(copy.deepcopy(self.patch), self.vdof, self.num_aux)

        for attr in "t":
            setattr(result, attr, copy.deepcopy(getattr(self, attr)))

        if self.q is not None:
            result.q = copy.deepcopy(self.q)
        if self.aux is not None:
            result.aux = copy.deepcopy(self.aux)
        result.problem_data = copy.deepcopy(self.problem_data)

        return result

    def sum_F(self, i):
        import numpy as np

        return np.sum(np.abs(self.F[i, ...]))

    def new_array(self, dof):
        if dof:
            import numpy as np

            if not isinstance(dof, (list, tuple)):
                dof = [dof]
            shape = self.grid.num_cells
            shape.extend(dof)
            return np.empty(shape)
        else:
            return None

    def get_q_global(self):
        r"""
        Returns a copy of state.q.
        """
        return self.q.copy()

    def get_aux_global(self):
        r"""
        Returns a copy of state.aux.
        """
        return self.aux.copy()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
