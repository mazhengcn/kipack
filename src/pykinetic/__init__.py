"""Main pykinetic package"""

from __future__ import absolute_import

import logging
import logging.config  # noqa
import os  # noqa

# Sub-packages
from . import limiters  # noqa
from .boltzmann.solver import BoltzmannSolver0D, BoltzmannSolver1D  # noqa
from .cfl import CFL  # noqa

# from .controller import Controller
from .geometry import Dimension, Domain, Patch  # noqa
from .limiters import *  # noqa

# from .sharpclaw.solver import (
#     SharpClawSolver1D,
#     SharpClawSolver2D,
#     SharpClawSolver3D,
# )
from .solution import Solution  # noqa
from .solver import BC  # noqa
from .state import State  # noqa

# from .tests.test_io import IOTest

# To get pyclaw.examples
# _path = os.path.dirname(os.path.dirname(__path__[0]))
# if os.path.isdir(_path):
#     __path__.append(_path)
# del _path

# Default logging configuration file
# _DEFAULT_LOG_CONFIG_PATH = os.path.join(
#     os.path.dirname(__file__), "log.config"
# )
# del os

# Setup loggers
# logging.config.fileConfig(_DEFAULT_LOG_CONFIG_PATH)

__all__ = []

# Module imports
__all__.extend(
    [
        "Controller",
        "Dimension",
        "Patch",
        "Domain",
        "Solution",
        "State",
        "CFL",
        "plot",
    ]
)

# from clawpack.pyclaw import plot

__all__.extend(
    [
        "ClawSolver1D",
        "ClawSolver2D",
        "ClawSolver3D",
        "SharpClawSolver1D",
        "SharpClawSolver2D",
        "SharpClawSolver3D",
    ]
)


# __all__.extend(limiters.__all__)

__all__.append("BC")

__all__.extend("IOTest")
