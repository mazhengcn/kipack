from __future__ import absolute_import

import numpy as np
from kipack.pykinetic import geometry


class State(object):
    def __init__(self, geom, vmesh, num_eqn, num_aux=0):

        if isinstance(geom, geometry.Patch):
            self.patch = geom
        elif isinstance(geom, geometry.Domain):
            self.patch = geom.patches[0]
        else:
            raise Exception(
                "A State object must be initialized with a Patch object."
            )
        # Velocity descritization
        self.vmesh = vmesh
        # ========== Attribute Definitions ===================================
        # pykinetic.Patch.patch - The patch this state lives on.
        self.p = None
        # (ndarray(mp,...)) - Cell averages of derived quantities.
        self.F = None
        # (dict) - Dictionary of global values for this patch, default = {}
        self.problem_data = {}
        # (float) - Current time represented on this patch, default = 0.0
        self.t = 0.0
        # (bool) - Keep gauge values in memory for every time step,
        # default = False
        self.keep_gauges = False
        # (list) - List of numpy.ndarray objects. Each element of the list
        # stores the values of the corresponding gauge if "keep_gauges" is
        # set to "True"
        self.gauge_data = []

        self.q = self.new_array(num_eqn)
        self.aux = self.new_array(num_aux)

    # ========== Class Methods ===============================================
    def set_q_from_qbc(self, num_ghost, qbc):
        """Set the value of q using the array qbc. Typically this is called
        after qbc has been updated by the solver.
        """
        num_dim = self.patch.num_dim
        if num_dim == 0:
            self.q = qbc[:]
        elif num_dim == 1:
            self.q = qbc[:, num_ghost:-num_ghost]
        elif num_dim == 2:
            self.q = qbc[:, num_ghost:-num_ghost, num_ghost:-num_ghost]
        elif num_dim == 3:
            self.q = qbc[
                :num_ghost:-num_ghost,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
            ]
        else:
            raise Exception("Assumption (0 <= num_dim <= 3) violated.")

    def set_aux_from_auxbc(self, num_ghost, auxbc):
        """Set the value of aux using the array auxbc."""
        patch = self.patch
        if patch.num_dim == 0:
            self.aux = auxbc[:]
        elif patch.num_dim == 1:
            self.aux = auxbc[:, num_ghost:-num_ghost]
        elif patch.num_dim == 2:
            self.aux = auxbc[:, num_ghost:-num_ghost, num_ghost:-num_ghost]
        elif patch.num_dim == 3:
            self.aux = auxbc[
                :,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
            ]
        else:
            raise Exception("Assumption (0 <= num_dim <= 3) violated.")

    def get_qbc_from_q(self, num_ghost, qbc):
        """Fills in the interior of qbc by copying q to it."""
        num_dim = self.patch.num_dim
        if num_dim == 0:
            qbc[:] = self.q
        elif num_dim == 1:
            qbc[:, num_ghost:-num_ghost] = self.q
        elif num_dim == 2:
            qbc[:, num_ghost:-num_ghost, num_ghost:-num_ghost] = self.q
        elif num_dim == 3:
            qbc[
                :,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
            ] = self.q
        else:
            raise Exception("Assumption (0 <= num_dim <= 3) violated.")

        return qbc

    def get_auxbc_from_aux(self, num_ghost, auxbc):
        """Fills in the interior of auxbc by copying aux to it."""
        num_dim = self.patch.num_dim
        if num_dim == 0:
            auxbc[:] = self.aux
        elif num_dim == 1:
            auxbc[:, num_ghost:-num_ghost] = self.aux
        elif num_dim == 2:
            auxbc[:, num_ghost:-num_ghost, num_ghost:-num_ghost] = self.aux
        elif num_dim == 3:
            auxbc[
                :,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
                num_ghost:-num_ghost,
            ] = self.aux
        else:
            raise Exception("Assumption (0 <= num_dim <= 3) violated.")

        return auxbc

    def sum_F(self, i):
        return np.sum(np.abs(self.F[i, ...]))

    def new_array(self, dof):
        if dof == 0:
            return None
        shape = [dof]
        shape.extend(self.grid.num_cells)
        shape.extend(self.vmesh.num_nodes)
        return np.empty(shape, order="F")

    def get_q_global(self):
        """Returns a copy of state.q."""
        return self.q.copy()

    def get_aux_global(self):
        """Returns a copy of state.aux."""
        return self.aux.copy()

    def is_valid(self):
        """Checks to see if this state is valid

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

    # ========== Property Definitions ========================================
    @property
    def num_eqn(self):
        r"""(int) - Number of unknowns (components of q)"""
        if self.q is None:
            raise Exception("state.num_eqn has not been set.")
        else:
            return self.q.shape[0]

    @property
    def num_aux(self):
        r"""(int) - Number of auxiliary fields"""
        if self.aux is not None:
            return self.aux.shape[0]
        else:
            return 0

    @property
    def num_vnodes(self):
        return self._get_vmesh_attribute("num_nodes")

    @property
    def grid(self):
        return self.patch.grid

    @property
    def mp(self):
        """(int) - Number of derived quantities"""
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
        """(int) - Number of output functionals"""
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

    # ========== Copy functionality ==========================================
    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo={}):
        import copy

        result = self.__class__(
            copy.deepcopy(self.patch),
            copy.deepcopy(self.vmesh),
            self.num_eqn,
            self.num_aux,
        )
        result.__init__(
            copy.deepcopy(self.patch),
            copy.deepcopy(self.vmesh),
            self.num_eqn,
            self.num_aux,
        )

        for attr in "t":
            setattr(result, attr, copy.deepcopy(getattr(self, attr)))

        if self.q is not None:
            result.q = copy.deepcopy(self.q)
        if self.aux is not None:
            result.aux = copy.deepcopy(self.aux)
        result.problem_data = copy.deepcopy(self.problem_data)

        return result

    def _get_grid_attribute(self, name):
        """Return grid attribute

        :Output:
         - (id) - Value of attribute from ``grid``
        """
        return getattr(self.grid, name)

    def _get_vmesh_attribute(self, name):
        return getattr(self.vmesh, name)

    def __getattr__(self, key):
        if key in (
            "num_dim",
            "p_centers",
            "c_centers",
            "num_cells",
            "lower",
            "upper",
            "delta",
            "centers",
            "gauges",
        ):
            return self._get_grid_attribute(key)
        else:
            raise AttributeError(
                "'State' object has no attribute '" + key + "'"
            )

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
