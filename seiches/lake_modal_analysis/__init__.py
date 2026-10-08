"""
Lake Modal Analysis Package
============================

A Python package for modal analysis of internal waves in stratified lakes.

This package performs eigenvalue analysis on a two-layer stratified system
to compute normal oscillation modes, including Coriolis effects and 
variable bathymetry.

Original MATLAB code: Copyright (C) 2010 Carlos Rozas Rojas
Python conversion: 2025

License: GNU General Public License v3.0

Main modules:
- mesh: Mesh generation and grid setup
- bathymetry: Bathymetry interpolation and assignment
- coriolis: Coriolis parameter calculation
- matrices: System matrix assembly (K and M matrices)
- eigensolve: Eigenvalue problem solution
- velocities: Velocity field computation
- visualization: Plotting and animation functions
"""

__version__ = "1.1.1"
__author__ = "Converted from MATLAB by Carlos Rozas Rojas"
__license__ = "GPL-3.0"

from .mesh import MeshGenerator
from .bathymetry import BathymetryProcessor
from .coriolis import CoriolisCalculator
from .matrices import MatrixGenerator
from .eigensolve import EigenSolver
from .velocities import VelocityProcessor
from .visualization import Visualizer
from .modal_analysis import ModalAnalysis
from .two_layers_analysis import compute_two_layers_analysis
from .modal_decomposition import *

__all__ = [
    'MeshGenerator',
    'BathymetryProcessor',
    'CoriolisCalculator',
    'MatrixGenerator',
    'EigenSolver',
    'VelocityProcessor',
    'Visualizer',
    'ModalAnalysis',
]
