"""
Visualization Module
====================

Plotting and animation functions for modal analysis results.

Original MATLAB: Grafico_modos.m, animacion_modos.m, plot_results.m
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from numpy.f2py.auxfuncs import throw_error


class Visualizer:
    """
    Visualization tools for modal analysis results.
    
    Provides:
    - Mode shape plotting
    - Velocity field visualization
    - Animation of modal evolution
    """
    
    def __init__(self, output_dir):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save figures for the selected lake
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_save_path(self, filename):
        """Get full path for saving a file."""
        if filename:
            return str(self.output_dir / filename)
        return None
    
    def plot_modes(self, eigen_results, velocity_data, mesh_data, 
                   n_modes=9, save_path=None, layer=1):
        """
        Plot mode shapes with velocity vectors.
        
        Parameters:
        -----------
        eigen_results : dict
            Eigenvalue solution results
        velocity_data : dict
            Processed velocity fields
        mesh_data : dict
            Mesh information
        n_modes : int
            Number of modes to plot (default: 9)
        save_path : str, optional
            Filename to save figure (will be saved in the configured output directory)
            If None, defaults to 'modes.png'
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        eta2tilde = eigen_results['eta2tilde']
        wshorasID = eigen_results['wshorasID']

        if layer==1:
            us = velocity_data['u1s']
            vs = velocity_data['v1s']
        elif layer==2:
            us = velocity_data['u2s']
            vs = velocity_data['v2s']
        else:
            throw_error("layer must be 1 or 2")

        xs = velocity_data['xs']
        ys = velocity_data['ys']
        xeta = mesh_data['xeta']
        yeta = mesh_data['yeta']
        xb = mesh_data['xb']
        yb = mesh_data['yb']
        
        # Normalize modes
        eta2norm = np.zeros_like(eta2tilde[:, :, :n_modes])
        for j in range(min(n_modes, eta2tilde.shape[2])):
            max_val = np.nanmax(np.abs(eta2tilde[:, :, j] + 
                                       np.conj(eta2tilde[:, :, j])))
            if max_val > 0:
                eta2norm[:, :, j] = eta2tilde[:, :, j] / max_val
        
        # Calculate real parts for plotting
        eta2 = np.zeros_like(eta2norm)
        u = np.zeros((xs.shape[0], xs.shape[1], n_modes))
        v = np.zeros((xs.shape[0], xs.shape[1], n_modes))
        
        for r in range(min(n_modes, eta2tilde.shape[2])):
            eta2[:, :, r] = np.real(eta2norm[:, :, r] + 
                                    np.conj(eta2norm[:, :, r]))
            u[:, :, r] = np.real(us[:, :, r] + np.conj(us[:, :, r]))
            v[:, :, r] = np.real(vs[:, :, r] + np.conj(vs[:, :, r]))

        # Add velocity scale
        if u.size > 0:
            u[-1, -1, :] = 0.01
            v[-1, -1, :] = 0
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))

        nrows = 3
        ncol = 3
        for j in range(nrows):
            for s in range(ncol):
                pos = ncol * j + s
                if pos >= n_modes:
                    break
                    
                ax = plt.subplot(nrows, ncol, pos + 1)
                
                # Plot elevation
                im = ax.contourf(xeta, yeta, eta2[:, :, pos], 
                               levels=20, cmap='RdBu_r', 
                               vmin=-1, vmax=1)
                
                # Plot shoreline
                ax.plot(xb, yb, 'k', linewidth=1)
                
                # Plot velocity vectors with better visibility
                ax.quiver(xs, ys, u[:, :, pos], v[:, :, pos],
                         scale=1e-1, width=0.003, headwidth=3, headlength=4,
                         color='k', alpha=0.7)
                
                # Add title with period
                period = np.round(wshorasID[pos, 0] * 100) / 100
                ax.text(0.01, 0.9, f'{pos}) {period:.2f} [hr]',
                       transform=ax.transAxes, fontsize=12)
                
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(True)
        
        # Add colorbar
        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.05, 0.1, 0.89, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                           ticks=np.arange(-1, 1.2, 0.2))
        cbar.set_label(r'$\eta_2/\eta_{2max}$', fontsize=14)
        
        plt.tight_layout(rect=[0, 0.13, 1, 1])
        
        # Save to the configured output directory
        if save_path is None:
            save_path = 'modes.png'
        full_path = self._get_save_path(save_path)

        Path(full_path).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {full_path}")
        
        return fig
    
    def animate_modes(self, eigen_results, velocity_data, mesh_data,
                      n_modes=9, n_frames=128, dt=600, save_path=None, scale=0.5):
        """
        Create animation of modal evolution.
        
        Parameters:
        -----------
        eigen_results : dict
            Eigenvalue solution results
        velocity_data : dict
            Processed velocity fields
        mesh_data : dict
            Mesh information
        n_modes : int
            Number of modes to animate
        n_frames : int
            Number of animation frames
        dt : float
            Time step (seconds)
        save_path : str, optional
            Filename to save animation (will be saved in the configured output directory)
            If None, defaults to 'modes_animation.gif'
            
        Returns:
        --------
        matplotlib.animation.FuncAnimation
        """
        ws = eigen_results['ws']
        eta2tilde = eigen_results['eta2tilde']
        wshorasID = eigen_results['wshorasID']
        etilde = eigen_results['etilde']
        u1s = velocity_data['u1s']
        v1s = velocity_data['v1s']
        xs = velocity_data['xs']
        ys = velocity_data['ys']
        xeta = mesh_data['xeta']
        yeta = mesh_data['yeta']
        xb = mesh_data['xb']
        yb = mesh_data['yb']
        
        # Solve for amplitudes using Crank-Nicolson
        amp = np.ones((n_frames + 1, n_modes), dtype=complex)
        fr = np.zeros((n_frames + 1, n_modes))
        
        for t in range(n_frames):
            for j in range(n_modes):
                # Crank-Nicolson time stepping
                amp[t+1, j] = (amp[t, j]/dt + 0.5*(1j*ws[j]*amp[t, j] + 
                              fr[t+1, j]/etilde[j] + fr[t, j]/etilde[j])) / \
                              (1/dt - 0.5*1j*ws[j])
        
        # Normalize modes
        eta2norm = np.zeros_like(eta2tilde[:, :, :n_modes])
        for j in range(n_modes):
            max_val = np.nanmax(np.abs(eta2tilde[:, :, j] + 
                                       np.conj(eta2tilde[:, :, j])))
            if max_val > 0:
                eta2norm[:, :, j] = eta2tilde[:, :, j] / max_val
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        panels = ['A) ', 'B) ', 'C) ', 'D) ', 'E) ', 'F) ', 
                  'G) ', 'H) ', 'I) ']
        
        axes = []
        ims = []
        quivers = []
        
        for j in range(3):
            for s in range(3):
                pos = 3 * j + s
                if pos >= n_modes:
                    break
                    
                ax = plt.subplot(3, 3, pos + 1)
                axes.append(ax)
                
                # Initial plot
                eta2_init = np.real(amp[0, pos] * eta2norm[:, :, pos] + 
                                   np.conj(amp[0, pos]) * 
                                   np.conj(eta2norm[:, :, pos]))
                
                im = ax.contourf(xeta, yeta, eta2_init, 
                               levels=20, cmap='RdBu_r', 
                               vmin=-1, vmax=1)
                ims.append(im)
                
                ax.plot(xb, yb, 'k', linewidth=1)
                
                u1_init = np.real(amp[0, pos] * u1s[:, :, pos] + 
                                 np.conj(amp[0, pos]) * np.conj(u1s[:, :, pos]))
                v1_init = np.real(amp[0, pos] * v1s[:, :, pos] + 
                                 np.conj(amp[0, pos]) * np.conj(v1s[:, :, pos]))
                
                q = ax.quiver(xs, ys, u1_init, v1_init,
                            scale=scale, width=0.003, headwidth=3, headlength=4,
                            color='k', alpha=0.7)
                quivers.append(q)
                
                period = np.round(wshorasID[pos, 0] * 100) / 100
                ax.text(0.01, 0.9, f'{panels[pos]}{period:.2f} [hr]',
                       transform=ax.transAxes, fontsize=12)
                
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
        
        def animate_frame(frame):
            """Animation update function."""
            for pos in range(min(n_modes, 9)):
                # Calculate fields at this time
                eta2 = np.real(amp[frame, pos] * eta2norm[:, :, pos] + 
                              np.conj(amp[frame, pos]) * 
                              np.conj(eta2norm[:, :, pos]))
                u1 = np.real(amp[frame, pos] * u1s[:, :, pos] + 
                            np.conj(amp[frame, pos]) * np.conj(u1s[:, :, pos]))
                v1 = np.real(amp[frame, pos] * v1s[:, :, pos] + 
                            np.conj(amp[frame, pos]) * np.conj(v1s[:, :, pos]))
                
                # Update contour (requires clearing and redrawing)
                axes[pos].clear()
                axes[pos].contourf(xeta, yeta, eta2, 
                                  levels=20, cmap='RdBu_r', 
                                  vmin=-1, vmax=1)
                axes[pos].plot(xb, yb, 'k', linewidth=1)
                axes[pos].quiver(xs, ys, u1, v1,
                               scale=scale, width=0.003, headwidth=3, headlength=4,
                               color='k', alpha=0.7)
                
                period = np.round(wshorasID[pos, 0] * 100) / 100
                axes[pos].text(0.01, 0.9, f'{panels[pos]}{period:.2f} [hr]',
                              transform=axes[pos].transAxes, fontsize=12)
                axes[pos].set_aspect('equal')
                axes[pos].set_xticks([])
                axes[pos].set_yticks([])
            
            return axes
        
        anim = animation.FuncAnimation(fig, animate_frame, frames=n_frames,
                                      interval=100, blit=False)
        
        # Save to the configured output directory
        if save_path is None:
            save_path = 'modes_animation.gif'
        full_path = self._get_save_path(save_path)
        anim.save(full_path, writer='pillow', fps=10)
        print(f"✓ Saved: {full_path}")
        
        return anim
    
    def plot_single_mode(self, eigen_results, velocity_data, mesh_data,
                        mode_idx=0, save_path=None, layer=1):
        """
        Plot a single mode in detail.
        
        Parameters:
        -----------
        eigen_results : dict
            Eigenvalue solution results
        velocity_data : dict
            Processed velocity fields
        mesh_data : dict
            Mesh information
        mode_idx : int
            Index of mode to plot
        save_path : str, optional
            Filename to save figure (will be saved in the configured output directory)
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        eta2tilde = eigen_results['eta2tilde']
        wshorasID = eigen_results['wshorasID']
        if layer==1:
            us = velocity_data['u1s']
            vs = velocity_data['v1s']
        elif layer==2:
            us = velocity_data['u2s']
            vs = velocity_data['v2s']
        else:
            throw_error("layer must be 1 or 2")

        xs = velocity_data['xs']
        ys = velocity_data['ys']
        xeta = mesh_data['xeta']
        yeta = mesh_data['yeta']
        xb = mesh_data['xb']
        yb = mesh_data['yb']
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Normalize and compute real part
        max_val = np.nanmax(np.abs(eta2tilde[:, :, mode_idx] + 
                                   np.conj(eta2tilde[:, :, mode_idx])))
        eta2_norm = eta2tilde[:, :, mode_idx] / max_val
        eta2 = np.real(eta2_norm + np.conj(eta2_norm))
        
        u1 = np.real(us[:, :, mode_idx] + np.conj(us[:, :, mode_idx]))
        v1 = np.real(vs[:, :, mode_idx] + np.conj(vs[:, :, mode_idx]))
        
        # Plot
        im = ax.contourf(xeta, yeta, eta2, levels=20, cmap='RdBu_r')
        ax.plot(xb, yb, 'k', linewidth=2)
        ax.quiver(xs, ys, u1, v1, scale=0.05, color='k', alpha=0.7)
        
        period = np.round(wshorasID[mode_idx, 0] * 100) / 100
        ax.set_title(f'Mode {mode_idx + 1}: Period = {period:.2f} hours',
                    fontsize=14)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\eta_2/\eta_{2max}$', fontsize=12)
        
        plt.tight_layout()
        
        # Save to the configured output directory
        if save_path:
            full_path = self._get_save_path(save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {full_path}")
        
        return fig


def plot_projected_mode(eta2tilde, us, vs, eigen_results, velocity_data, mesh_data,
                    mode_idx=0, scale=0.05):
    """
    Plot a single mode in detail.
    
    Parameters:
    -----------
    eigen_results : dict
        Eigenvalue solution results
    velocity_data : dict
        Processed velocity fields
    mesh_data : dict
        Mesh information
    mode_idx : int
        Index of mode to plot
    save_path : str, optional
        Filename to save figure (will be saved in the configured output directory)
        
    Returns:
    --------
    matplotlib.figure.Figure
    """

    wshorasID = eigen_results['wshorasID']

    xs = velocity_data['xs']
    ys = velocity_data['ys']
    xeta = mesh_data['xeta']
    yeta = mesh_data['yeta']
    xb = mesh_data['xb']
    yb = mesh_data['yb']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Normalize and compute real part
    max_val = np.nanmax(np.abs(eta2tilde[:, :, mode_idx] + 
                                np.conj(eta2tilde[:, :, mode_idx])))
    eta2_norm = eta2tilde[:, :, mode_idx] / max_val
    eta2 = np.real(eta2_norm + np.conj(eta2_norm))
    
    u1 = np.real(us[:, :, mode_idx] + np.conj(us[:, :, mode_idx]))
    v1 = np.real(vs[:, :, mode_idx] + np.conj(vs[:, :, mode_idx]))
    
    # Plot
    im = ax.contourf(xeta, yeta, eta2, levels=20, cmap='RdBu_r')
    ax.plot(xb, yb, 'k', linewidth=2)
    ax.quiver(xs, ys, u1, v1, scale=scale, color='k', alpha=0.7)
    
    period = np.round(wshorasID[mode_idx, 0] * 100) / 100
    ax.set_title(f'Mode {mode_idx + 1}: Period = {period:.2f} hours',
                fontsize=14)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\eta_2/\eta_{2max}$', fontsize=12)
    
    plt.tight_layout()
        
    return fig