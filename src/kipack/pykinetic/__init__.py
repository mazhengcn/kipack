"""Main pykinetic package"""
import logging
import logging.config  # noqa
import os  # noqa

# Sub-packages
from kipack.pykinetic import limiters  # noqa
from kipack.pykinetic import riemann  # noqa
from kipack.pykinetic.boltzmann.solver import BoltzmannSolver0D  # noqa
from kipack.pykinetic.boltzmann.solver import BoltzmannSolver1D  # noqa
from kipack.pykinetic.cfl import CFL  # noqa
from kipack.pykinetic.geometry import Dimension, Domain, Patch  # noqa
from kipack.pykinetic.solution import Solution  # noqa
from kipack.pykinetic.solver import BC  # noqa
from kipack.pykinetic.state import State  # noqa

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

__all__.extend(["BoltzmannSolver0D", "BoltzmannSolver1D"])  # noqa


# __all__.extend(limiters.__all__)

__all__.append("BC")

__all__.extend("IOTest")
