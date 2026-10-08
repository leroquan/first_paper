"""
Velocity Processing Module
===========================

Averages velocities at eta nodes for visualization.

Original MATLAB: Velocidades.m
"""

import numpy as np


class VelocityProcessor:
    """
    Process velocity fields for visualization.
    
    Averages velocities from staggered grid to cell centers for plotting.
    """
    
    def __init__(self):
        """Initialize velocity processor."""
        pass
    
    def process_velocities(self, eigen_results, mesh_data):
        """
        Average velocities to eta nodes and prepare for plotting.
        
        Parameters:
        -----------
        eigen_results : dict
            Results from eigenvalue solver
        mesh_data : dict
            Mesh information
            
        Returns:
        --------
        dict : Dictionary with processed velocity fields
        """
        neta = mesh_data['neta']
        meta = mesh_data['meta']
        nodos_eta1 = mesh_data['nodos_eta1']
        u1tilde = eigen_results['u1tilde']
        v1tilde = eigen_results['v1tilde']
        u2tilde = eigen_results['u2tilde']
        v2tilde = eigen_results['v2tilde']
        xeta = mesh_data['xeta']
        yeta = mesh_data['yeta']

        n_modes = u1tilde.shape[2]

        # Initialize velocity fields at eta nodes
        u1_plot = np.full((neta, meta, n_modes), np.nan, dtype=complex)
        v1_plot = np.full((neta, meta, n_modes), np.nan, dtype=complex)
        u2_plot = np.full((neta, meta, n_modes), np.nan, dtype=complex)
        v2_plot = np.full((neta, meta, n_modes), np.nan, dtype=complex)

        # Average velocities to eta nodes
        for r in range(n_modes):
            for i in range(neta):
                for j in range(meta):
                    if nodos_eta1[i, j] != 0:
                        # Average u velocities (east and west faces)
                        if j < meta - 1:
                            u1_plot[i, j, r] = 0.5 * (
                                u1tilde[i, j, r] + u1tilde[i, j + 1, r]
                            )
                            u2_plot[i, j, r] = 0.5 * (
                                u2tilde[i, j, r] + u2tilde[i, j + 1, r]
                            )

                        # Average v velocities (north and south faces)
                        if i < neta - 1:
                            v1_plot[i, j, r] = 0.5 * (
                                v1tilde[i, j, r] + v1tilde[i + 1, j, r]
                            )
                            v2_plot[i, j, r] = 0.5 * (
                                v2tilde[i, j, r] + v2tilde[i + 1, j, r]
                            )

        # Subsample for cleaner visualization
        spacing = 3  # Plot every 3rd arrow
        ind_i = slice(1, neta, spacing)
        ind_j = slice(1, meta, spacing)

        xs = xeta[ind_i, ind_j]
        ys = yeta[ind_i, ind_j]

        u1s = np.zeros((xs.shape[0], xs.shape[1], n_modes), dtype=complex)
        v1s = np.zeros((xs.shape[0], xs.shape[1], n_modes), dtype=complex)
        u2s = np.zeros((xs.shape[0], xs.shape[1], n_modes), dtype=complex)
        v2s = np.zeros((xs.shape[0], xs.shape[1], n_modes), dtype=complex)

        for r in range(n_modes):
            u1s[:, :, r] = u1_plot[ind_i, ind_j, r]
            v1s[:, :, r] = v1_plot[ind_i, ind_j, r]
            u2s[:, :, r] = u2_plot[ind_i, ind_j, r]
            v2s[:, :, r] = v2_plot[ind_i, ind_j, r]

        velocity_data = {
            'u1_plot': u1_plot,
            'v1_plot': v1_plot,
            'u2_plot': u2_plot,
            'v2_plot': v2_plot,
            'xs': xs,
            'ys': ys,
            'u1s': u1s,
            'v1s': v1s,
            'u2s': u2s,
            'v2s': v2s
        }

        return velocity_data
