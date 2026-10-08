"""
Modal Analysis Module
=====================

Main class that orchestrates the complete modal analysis workflow.

Original MATLAB: ModalAnalysis_Geneva.m
"""
from pathlib import Path
import json
import time
import numpy as np
import os
from .mesh import MeshGenerator, MeshGeneratorMitgcmGrid
from .bathymetry import BathymetryProcessor
from .coriolis import CoriolisCalculator
from .matrices import MatrixGenerator
from .eigensolve import EigenSolver
from .velocities import VelocityProcessor
from .visualization import Visualizer


class ModalAnalysis:
    """
    Complete modal analysis workflow for internal waves in stratified lakes.
    
    This class manages the entire analysis pipeline:
    1. Mesh generation
    2. Bathymetry processing
    3. Coriolis parameter calculation
    4. System matrix assembly
    5. Eigenvalue problem solution
    6. Velocity field processing
    7. Visualization
    """
    
    def __init__(self, config):
        """
        Initialize modal analysis.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary with keys:
            - g: Gravitational acceleration (m/s^2)
            - h1: Upper layer thickness (m)
            - dx: Grid spacing in x (m)
            - dy: Grid spacing in y (m)
            - rho1: Upper layer density (kg/m^3)
            - rho2: Lower layer density (kg/m^3)
            - lat0: Central latitude (degrees)
            - y0: Central latitude y-coordinate (km)
            - Rt: Earth radius (km)
            - beta0: Beta parameter (optional, for variable Coriolis)
            - n_modes: Number of modes to compute
            - period_target: Target period (hours)
            - x_range: (x_min, x_max) in km
            - y_range: (y_min, y_max) in km
        """
        self.config = config

        # Extract parameters
        self.g = config['g']
        self.h1 = config.get('h1', None)
        self.rho1 = config.get('rho1', None)
        self.rho2 = config.get('rho2', None)
        self.lat0 = config.get('lat0', None)
        self.y0 = config.get('y0', 0)
        self.n_modes = config['n_modes']
        self.period_target = config['period_target']
        self.mitgcm_grid = config.get('mitgcm_grid', False)

        # Calcualte Target frequency (rad/s)
        self.freq_target = 2*np.pi/(self.period_target*3600)

        # Initialize components
        if self.mitgcm_grid == True:
            print('Fetching MITgcm grid...')

            if self.lat0 is None or self.lat0 == 'None':
                lat_grid = np.load(os.path.join(config['mitgcm_grid_folder'], 'lat_grid.npy'))
                self.lat0 = lat_grid[0, 0]

            with open(os.path.join(config['mitgcm_grid_folder'], 'parameters.json'), "r") as f:
                grid_parameters = json.load(f)

            self.dx = grid_parameters['resolution']
            self.dy = grid_parameters['resolution']
            self.mesh_gen = MeshGeneratorMitgcmGrid(config['mitgcm_grid_folder'])
        else:
            self.dx = config['dx']
            self.dy = config['dy']
            self.mesh_gen = MeshGenerator(self.dx, self.dy,
                                     config['x_range'],
                                     config['y_range'])

        # Calculate beta parameter if not provided
        if 'beta0' not in config or config['beta0'] is None or config['beta0'] == 'None':
            print('Computing beta0 from Earth parameters...')
            Rt = config.get('Rt', 6371)  # Earth radius in km
            omega = 7.27e-5  # rad/s
            self.beta0 = 2 * omega * np.cos(np.deg2rad(self.lat0)) / Rt
            print('beta0 = ' + str(self.beta0))
        else:
            self.beta0 = config['beta0']


        self.bathy_proc = BathymetryProcessor(self.h1, self.dx, self.dy)
        self.coriolis_calc = CoriolisCalculator(self.lat0, self.y0, 
                                                self.beta0)
        self.matrix_gen = MatrixGenerator(self.g, self.rho1, self.rho2, 
                                         self.dx, self.dy)
        self.eigen_solver = EigenSolver(self.rho2, self.dx, self.dy)
        self.vel_proc = VelocityProcessor()
        self.viz = Visualizer(Path(config["modes_file"]).parent / "results" / "modes")
        
        # Storage for results
        self.mesh_data = None
        self.bathy_data = None
        self.coriolis_data = None
        self.K = None
        self.M = None
        self.indices = None
        self.eigen_results = None
        self.velocity_data = None
    
    def run_analysis(self, shoreline_file, bathymetry_file, verbose=True):
        """
        Run complete modal analysis.
        
        Parameters:
        -----------
        shoreline_file : str
            Path to shoreline coordinates file
        bathymetry_file : str
            Path to bathymetry data file
        verbose : bool
            Print progress messages
            
        Returns:
        --------
        dict : Complete results dictionary
        """
        start_time = time.time()
        
        # Step 1: Generate mesh
        if verbose:
            print("Step 1/7: Generating mesh...")
        t0 = time.time()
        self.mesh_data = self.mesh_gen.generate_grid(self.h1, shoreline_file, bathymetry_file)

        if verbose:
            print(f"  Completed in {time.time() - t0:.2f}s")
            print(f"  Grid size: {self.mesh_data['neta']}x{self.mesh_data['meta']}")
            print(f"  Active eta1 nodes: {np.sum(self.mesh_data['nodos_eta1'] > 0)}")
            print(f"  Active eta2 nodes: {np.sum(self.mesh_data['nodos_eta2'] > 0)}")
        
        # Step 2: Process bathymetry
        if verbose:
            print("\nStep 2/7: Processing bathymetry...")
        t0 = time.time()
        self.bathy_data = self.bathy_proc.process_bathymetry(self.mesh_data)
        if verbose:
            print(f"  Completed in {time.time() - t0:.2f}s")
        
        # Step 3: Calculate Coriolis parameters
        if verbose:
            print("\nStep 3/7: Calculating Coriolis parameters...")
        t0 = time.time()
        self.coriolis_data = self.coriolis_calc.calculate_coriolis(
            self.mesh_data)
        if verbose:
            print(f"  Completed in {time.time() - t0:.2f}s")
            print(f"  f0 = {self.coriolis_data['f0']:.6e} rad/s")
        
        # Step 4: Generate system matrices
        if verbose:
            print("\nStep 4/7: Generating system matrices...")
        t0 = time.time()
        self.K, self.M, self.indices = self.matrix_gen.generate_matrices(
            self.mesh_data, self.bathy_data, self.coriolis_data)
        if verbose:
            print(f"  Completed in {time.time() - t0:.2f}s")
            print(f"  Matrix size: {self.K.shape[0]}x{self.K.shape[1]}")
            print(f"  K matrix sparsity: {self.K.nnz / (self.K.shape[0]**2) * 100:.2f}%")

        print("K NaNs:", np.sum(np.isnan(self.K.data)))
        print("K infs:", np.sum(np.isinf(self.K.data)))

        print("M NaNs:", np.sum(np.isnan(self.M.data)))
        print("M infs:", np.sum(np.isinf(self.M.data)))
        nan_idx = np.where(np.isnan(self.K.data))[0]

        print("Number of NaN entries:", len(nan_idx))
        print("First NaN values:", self.K.data[nan_idx[:20]])
        K_coo = self.K.tocoo()

        nan = np.isnan(K_coo.data)

        print("NaN rows:", K_coo.row[nan][:50])
        print("NaN cols:", K_coo.col[nan][:50])
        
        # Step 5: Solve eigenvalue problem
        if verbose:
            print(f"\nStep 5/7: Solving eigenvalue problem ({self.n_modes} modes)...")
        t0 = time.time()
        self.eigen_results = self.eigen_solver.solve_eigenvalue_problem(
            self.K, self.M, self.n_modes, self.freq_target,
            self.mesh_data, self.indices)
        if verbose:
            print(f"  Completed in {time.time() - t0:.2f}s")
            print(f"  Mode periods (hours):")
            for i in range(min(5, self.n_modes)):
                period = self.eigen_results['wshorasID'][i, 0]
                print(f"    Mode {i+1}: {period:.2f} hours")
        
        # Step 6: Process velocities
        if verbose:
            print("\nStep 6/7: Processing velocity fields...")
        t0 = time.time()
        self.velocity_data = self.vel_proc.process_velocities(
            self.eigen_results, self.mesh_data)
        if verbose:
            print(f"  Completed in {time.time() - t0:.2f}s")
        
        # Step 7: Complete
        total_time = time.time() - start_time
        if verbose:
            print(f"\nAnalysis complete! Total time: {total_time:.2f}s")
        
        # Compile results
        results = {
            'mesh': self.mesh_data,
            'bathymetry': self.bathy_data,
            'coriolis': self.coriolis_data,
            'matrices': {'K': self.K, 'M': self.M, 'indices': self.indices},
            'eigenvalues': self.eigen_results,
            'velocities': self.velocity_data,
            'config': self.config
        }
        
        return results
    
    def plot_mesh(self):
        """Plot the generated mesh."""
        if self.mesh_data is None:
            raise ValueError("No mesh data available. Run analysis first.")
        return self.mesh_gen.plot_mesh(self.mesh_data)
    
    def plot_bathymetry(self):
        """Plot bathymetry."""
        if self.bathy_data is None:
            raise ValueError("No bathymetry data available. Run analysis first.")
        return self.bathy_proc.plot_bathymetry(self.bathy_data)
    
    def plot_modes(self, n_modes=9, save_path=None, layer=1):
        """
        Plot mode shapes.
        
        Parameters:
        -----------
        n_modes : int
            Number of modes to plot
        save_path : str, optional
            Path to save figure
        """
        if self.eigen_results is None or self.velocity_data is None:
            raise ValueError("No results available. Run analysis first.")
        
        return self.viz.plot_modes(self.eigen_results, self.velocity_data,
                                  self.mesh_data, n_modes, save_path, layer)
    
    def plot_single_mode(self, mode_idx=0, layer=1):
        """
        Plot a single mode in detail.
        
        Parameters:
        -----------
        mode_idx : int
            Index of mode to plot (0-based)
        """
        if self.eigen_results is None or self.velocity_data is None:
            raise ValueError("No results available. Run analysis first.")
        
        return self.viz.plot_single_mode(self.eigen_results, 
                                        self.velocity_data,
                                        self.mesh_data, mode_idx, layer=layer)
    
    def animate_modes(self, n_modes=9, n_frames=128, dt=600, 
                     save_path=None, scale=0.5):
        """
        Create animation of modal evolution.
        
        Parameters:
        -----------
        n_modes : int
            Number of modes to animate
        n_frames : int
            Number of animation frames
        dt : float
            Time step (seconds)
        save_path : str, optional
            Path to save animation
        """
        if self.eigen_results is None or self.velocity_data is None:
            raise ValueError("No results available. Run analysis first.")
        
        return self.viz.animate_modes(self.eigen_results, self.velocity_data,
                                     self.mesh_data, n_modes, n_frames, 
                                     dt, save_path, scale=scale)
    
    def save_results(self, filename):
        """
        Save results to file.
        
        Parameters:
        -----------
        filename : str
            Output filename (.npz format)
        """
        if self.eigen_results is None:
            raise ValueError("No results available. Run analysis first.")
        
        # Prepare data for saving
        save_dict = {
            'xeta': self.mesh_data['xeta'],
            'yeta': self.mesh_data['yeta'],
            'xb': self.mesh_data['xb'],
            'yb': self.mesh_data['yb'],
            'eta2tilde': self.eigen_results['eta2tilde'],
            'u1s': self.velocity_data['u1s'],
            'v1s': self.velocity_data['v1s'],
            'xs': self.velocity_data['xs'],
            'ys': self.velocity_data['ys'],
            'wshorasID': self.eigen_results['wshorasID'],
            'ws': self.eigen_results['ws'],
            'config': str(self.config)
        }
        
        np.savez_compressed(filename, **save_dict)
        print(f"Results saved to {filename}")
    
    @staticmethod
    def load_results(filename):
        """
        Load results from file.
        
        Parameters:
        -----------
        filename : str
            Input filename (.npz format)
            
        Returns:
        --------
        dict : Loaded results
        """
        data = np.load(filename, allow_pickle=True)
        return {key: data[key] for key in data.files}
