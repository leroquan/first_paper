"""
Bathymetry Processing Module
=============================

Assigns bathymetry values to each mesh node for the two-layer system.

Original MATLAB: generaBatimetria.m
"""

import numpy as np
import matplotlib.pyplot as plt


class BathymetryProcessor:
    """
    Process and assign bathymetry to mesh nodes.
    
    Handles:
    - Layer thickness assignment
    - Depth interpolation at velocity nodes
    - Bathymetry visualization
    """
    
    def __init__(self, h1, dx, dy):
        """
        Initialize bathymetry processor.
        
        Parameters:
        -----------
        h1 : float
            Thickness of upper layer (meters)
        dx : float
            Grid spacing in x-direction (meters)
        dy : float
            Grid spacing in y-direction (meters)
        """
        self.h1 = h1
        self.dx = dx
        self.dy = dy
    
    def process_bathymetry(self, mesh_data):
        """
        Assign bathymetry to all mesh nodes.
        
        Parameters:
        -----------
        mesh_data : dict
            Dictionary containing mesh information
            
        Returns:
        --------
        dict : Dictionary with bathymetry arrays
        """
        Heta = mesh_data['Heta']
        nodos_eta1 = mesh_data['nodos_eta1']
        nodos_eta2 = mesh_data['nodos_eta2']
        nodos_u1 = mesh_data['nodos_u1']
        nodos_v1 = mesh_data['nodos_v1']
        nodos_u2 = mesh_data['nodos_u2']
        nodos_v2 = mesh_data['nodos_v2']
        CBu1 = mesh_data['CBu1']
        CBv1 = mesh_data['CBv1']
        CBu2 = mesh_data['CBu2']
        CBv2 = mesh_data['CBv2']
        
        neta, meta = Heta.shape
        nu, mu = nodos_u1.shape
        nv, mv = nodos_v1.shape
        
        # Generate H1 matrix (upper layer thickness)
        H1 = np.zeros((neta, meta))
        for i in range(neta):
            for j in range(meta):
                if nodos_eta1[i, j] > 0:
                    if Heta[i, j] < self.h1:
                        H1[i, j] = Heta[i, j]
                    else:
                        H1[i, j] = self.h1
        
        # Generate H2 matrix (lower layer thickness)
        H2 = np.zeros((neta, meta))
        for i in range(neta):
            for j in range(meta):
                if nodos_eta2[i, j] > 0:
                    H2[i, j] = Heta[i, j] - self.h1
        
        # Generate heta1 array (depths at eta1 nodes)
        n_eta1 = np.sum(nodos_eta1 > 0)
        heta1 = np.zeros((n_eta1, 2))
        for i in range(neta):
            for j in range(meta):
                if nodos_eta1[i, j] > 0:
                    k = nodos_eta1[i, j] - 1
                    heta1[k, 0] = H1[i, j]
                    heta1[k, 1] = H2[i, j]
        
        # Generate heta2 array (depths at eta2 nodes)
        n_eta2 = np.sum(nodos_eta2 > 0)
        heta2 = np.zeros(n_eta2)
        for i in range(neta):
            for j in range(meta):
                if nodos_eta2[i, j] > 0:
                    k = nodos_eta2[i, j] - 1
                    heta2[k] = H2[i, j]
        
        # Generate hu1 array (depths at u1 nodes)
        n_u1 = np.sum(nodos_u1 > 0)
        hu1 = np.zeros(n_u1)
        for i in range(nu):
            for j in range(mu):
                if nodos_u1[i, j] > 0:
                    k = nodos_u1[i, j] - 1
                    if j == 0:
                        hu1[k] = H1[i, j]
                    elif H1[i, j-1] == 0:
                        hu1[k] = H1[i, j]
                    elif H1[i, j] == 0:
                        hu1[k] = H1[i, j-1]
                    else:
                        hu1[k] = min(H1[i, j-1], H1[i, j])
        
        # Generate hv1 array (depths at v1 nodes)
        n_v1 = np.sum(nodos_v1 > 0)
        hv1 = np.zeros(n_v1)
        for i in range(nv):
            for j in range(mv):
                if nodos_v1[i, j] > 0:
                    k = nodos_v1[i, j] - 1
                    if i == 0:
                        hv1[k] = H1[i, j]
                    elif H1[i-1, j] == 0:
                        hv1[k] = H1[i, j]
                    elif H1[i, j] == 0:
                        hv1[k] = H1[i-1, j]
                    else:
                        hv1[k] = min(H1[i-1, j], H1[i, j])
        
        # Generate hu2 array (depths at u2 nodes)
        n_u2 = np.sum(nodos_u2 > 0)
        hu2 = np.zeros(n_u2)
        for i in range(nu):
            for j in range(mu):
                if nodos_u2[i, j] > 0:
                    k = nodos_u2[i, j] - 1
                    if j == 0:
                        hu2[k] = H2[i, j]
                    elif H2[i, j-1] == 0:
                        hu2[k] = H2[i, j]
                    elif H2[i, j] == 0:
                        hu2[k] = H2[i, j-1]
                    else:
                        hu2[k] = min(H2[i, j-1], H2[i, j])
        
        # Generate hv2 array (depths at v2 nodes)
        n_v2 = np.sum(nodos_v2 > 0)
        hv2 = np.zeros(n_v2)
        for i in range(nv):
            for j in range(mv):
                if nodos_v2[i, j] > 0:
                    k = nodos_v2[i, j] - 1
                    if i == 0:
                        hv2[k] = H2[i, j]
                    elif H2[i-1, j] == 0:
                        hv2[k] = H2[i, j]
                    elif H2[i, j] == 0:
                        hv2[k] = H2[i-1, j]
                    else:
                        hv2[k] = min(H2[i-1, j], H2[i, j])
        
        # Extract non-boundary depths
        nnhu1 = hu1[CBu1 != 0]
        nnhv1 = hv1[CBv1 != 0]
        nnhu2 = hu2[CBu2 != 0]
        nnhv2 = hv2[CBv2 != 0]
        
        bathy_data = {
            'H1': H1, 'H2': H2,
            'heta1': heta1, 'heta2': heta2,
            'hu1': hu1, 'hv1': hv1,
            'hu2': hu2, 'hv2': hv2,
            'nnhu1': nnhu1, 'nnhv1': nnhv1,
            'nnhu2': nnhu2, 'nnhv2': nnhv2
        }
        
        return bathy_data
    
    def plot_bathymetry(self, bathy_data):
        """Plot bathymetry for both layers."""
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        X1, Y1 = np.meshgrid(range(bathy_data['H1'].shape[1]), 
                             range(bathy_data['H1'].shape[0]))
        ax1.plot_surface(X1, Y1, bathy_data['H1'], cmap='viridis')
        ax1.invert_zaxis()
        ax1.set_title('Layer 1 Thickness')
        ax1.set_xlabel('X index')
        ax1.set_ylabel('Y index')
        ax1.set_zlabel('Depth (m)')
        
        ax2 = fig.add_subplot(122, projection='3d')
        X2, Y2 = np.meshgrid(range(bathy_data['H2'].shape[1]), 
                             range(bathy_data['H2'].shape[0]))
        ax2.plot_surface(X2, Y2, bathy_data['H2'], cmap='viridis')
        ax2.invert_zaxis()
        ax2.set_title('Layer 2 Thickness')
        ax2.set_xlabel('X index')
        ax2.set_ylabel('Y index')
        ax2.set_zlabel('Depth (m)')
        
        plt.tight_layout()
        return fig
