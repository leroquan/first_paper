"""
Coriolis Parameter Module
==========================

Calculates Coriolis parameter (f) at each mesh node, with optional
beta-plane approximation for latitude-dependent variations.

Original MATLAB: fvariable.m
"""

import numpy as np


class CoriolisCalculator:
    """
    Calculate Coriolis parameter at mesh nodes.
    
    Supports both constant and variable (beta-plane) Coriolis parameter.
    """
    
    def __init__(self, lat0, y0, beta0=None):
        """
        Initialize Coriolis calculator.
        
        Parameters:
        -----------
        lat0 : float
            Central latitude (degrees)
        y0 : float
            Location of central latitude in y-coordinate (m)
        beta0 : float, optional
            Beta parameter for variable Coriolis (1/m/s)
            If None, uses constant Coriolis parameter
        """
        self.lat0 = lat0
        self.y0 = y0
        self.beta0 = beta0 if beta0 is not None else 0.0
        
        # Earth's rotation rate
        omega = 7.27e-5  # rad/s
        
        # Coriolis parameter at central latitude
        self.f0 = 2 * omega * np.sin(np.deg2rad(lat0))
    
    def calculate_coriolis(self, mesh_data):
        """
        Calculate Coriolis parameter at all velocity nodes.
        
        Parameters:
        -----------
        mesh_data : dict
            Dictionary containing mesh information
            
        Returns:
        --------
        dict : Dictionary with Coriolis arrays
        """
        xeta = mesh_data['xeta']
        yu = mesh_data['yu']
        yv = mesh_data['yv']
        nodos_u1 = mesh_data['nodos_u1']
        nodos_v1 = mesh_data['nodos_v1']
        nodos_u2 = mesh_data['nodos_u2']
        nodos_v2 = mesh_data['nodos_v2']
        CBu1 = mesh_data['CBu1']
        CBv1 = mesh_data['CBv1']
        CBu2 = mesh_data['CBu2']
        CBv2 = mesh_data['CBv2']
        if "rotation" in mesh_data:
            grid_rotation_in_rad = np.deg2rad(mesh_data["rotation"])
        else:
            grid_rotation_in_rad = 0
        
        nu, mu = nodos_u1.shape
        nv, mv = nodos_v1.shape
        
        # Generate fu1 array
        n_u1 = np.sum(nodos_u1 > 0)
        fu1 = np.zeros(n_u1)
        for i in range(nu):
            for j in range(mu):
                if nodos_u1[i, j] > 0:
                    k = nodos_u1[i, j] - 1
                    ycorr = yu[i, j] * np.cos(grid_rotation_in_rad) + xeta[i, j] * np.sin(grid_rotation_in_rad)
                    fu1[k] = self.f0 + self.beta0 * (ycorr - self.y0)
        
        # Generate fv1 array
        n_v1 = np.sum(nodos_v1 > 0)
        fv1 = np.zeros(n_v1)
        for i in range(nv):
            for j in range(mv):
                if nodos_v1[i, j] > 0:
                    k = nodos_v1[i, j] - 1
                    ycorr = yv[i, j] * np.cos(grid_rotation_in_rad) + xeta[i, j] * np.sin(grid_rotation_in_rad)
                    fv1[k] = self.f0 + self.beta0 * (ycorr - self.y0)
        
        # Generate fu2 array
        n_u2 = np.sum(nodos_u2 > 0)
        fu2 = np.zeros(n_u2)
        for i in range(nu):
            for j in range(mu):
                if nodos_u2[i, j] > 0:
                    k = nodos_u2[i, j] - 1
                    ycorr = yu[i, j] * np.cos(grid_rotation_in_rad) + xeta[i, j] * np.sin(grid_rotation_in_rad)
                    fu2[k] = self.f0 + self.beta0 * (ycorr - self.y0)
        
        # Generate fv2 array
        n_v2 = np.sum(nodos_v2 > 0)
        fv2 = np.zeros(n_v2)
        for i in range(nv):
            for j in range(mv):
                if nodos_v2[i, j] > 0:
                    k = nodos_v2[i, j] - 1
                    ycorr = yv[i, j] * np.cos(grid_rotation_in_rad) + xeta[i, j] * np.sin(grid_rotation_in_rad)
                    fv2[k] = self.f0 + self.beta0 * (ycorr - self.y0)
        
        # Extract non-boundary Coriolis values
        nnfu1 = fu1[CBu1 != 0]
        nnfv1 = fv1[CBv1 != 0]
        nnfu2 = fu2[CBu2 != 0]
        nnfv2 = fv2[CBv2 != 0]
        
        coriolis_data = {
            'fu1': fu1, 'fv1': fv1,
            'fu2': fu2, 'fv2': fv2,
            'nnfu1': nnfu1, 'nnfv1': nnfv1,
            'nnfu2': nnfu2, 'nnfv2': nnfv2,
            'f0': self.f0
        }
        
        return coriolis_data
