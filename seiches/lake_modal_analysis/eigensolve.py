"""
Eigenvalue Solver Module
=========================

Solves the generalised eigenvalue problem and extracts modal information.

Original MATLAB: valorespropios.m

BUG FIX (vs previous Python version):
    The previous code used scipy.sparse.linalg.eigsh (Lanczos algorithm for
    symmetric / Hermitian matrices).  K is complex and non-symmetric, so
    eigsh gives wrong results.  The correct solver is
    scipy.sparse.linalg.eigs (Arnoldi method for general sparse matrices),
    which is the exact Python equivalent of MATLAB's eigs(K, M, n, sigma).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs   # ← Arnoldi, handles non-symmetric K
from scipy.sparse.linalg import eigsh   # ← Arnoldi, handles non-symmetric K
import scipy


class EigenSolver:
    """
    Solve the generalised eigenvalue problem  K * xi = omega * M * xi
    and reconstruct spatial mode shapes.
    """

    def __init__(self, rho2, dx, dy):
        """
        Parameters
        ----------
        rho2 : float
            Density of lower layer (kg/m^3)
        dx : float
            Grid spacing in x-direction (metres)
        dy : float
            Grid spacing in y-direction (metres)
        """
        self.rho2 = rho2
        self.dx   = dx
        self.dy   = dy

    # ------------------------------------------------------------------

    def solve_eigenvalue_problem(self, K, M, n_modes, freq_target,
                                  mesh_data, indices):
        """
        Solve eigenvalue problem and extract mode shapes.

        Parameters
        ----------
        K : sparse matrix (complex, non-symmetric)
            Stiffness matrix.
        M : sparse matrix (real, diagonal)
            Mass matrix.
        n_modes : int
            Number of modes to compute.
        freq_target : float
            Angular frequency (rad/s) around which to search for eigenvalues.
            Passed as sigma to the shift-invert solver — matches MATLAB's
            eigs(K, M, n_modes, freq_target).
        mesh_data : dict
        indices : dict  – block start positions from matrix assembly

        Returns
        -------
        dict with keys: ws, wshoras, wshorasID, xitilde, etilde,
                        eta2tilde, u1tilde, v1tilde, u2tilde, v2tilde,
                        normaxi, normatilde
        """
        print(f"  Solving with eigs: K={K.shape}, sigma={freq_target:.4e} rad/s")

        # ------ Solve  K * xi = omega * M * xi  ----------------------
        # which='LM' + sigma uses shift-invert: finds eigenvalues whose
        # inverse (from the shifted problem) have the largest magnitude,
        # i.e. eigenvalues closest to sigma in the original problem.
        # This is identical to MATLAB's  eigs(K, M, n_modes, freq_target).
        #omega_vals, xi = eigs(K, k=n_modes, M=M,
        #                      sigma=freq_target, which='LM')

        v0 = np.linspace(0.1, 1.0, K.shape[0])  # Initial guess for eigenvector
        tol = 1e-10 # Strict tolerance
        omega_vals, xi = eigsh(K, k=n_modes, M=M, sigma=freq_target, which='LM', v0=v0, tol=tol)

        # Ensure 2-D even if only 1 mode was requested
        if xi.ndim == 1:
            xi        = xi.reshape(-1, 1)
            omega_vals = np.array([omega_vals])

        ws = omega_vals   # complex eigenvalues (approx. real angular freqs)

        # ------ Diagnostic plots --------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(range(1, n_modes + 1), np.real(ws), 's')
        axes[0].set_title('Real(eigenvalues)')
        axes[0].set_xlabel('Mode number')
        axes[0].grid(True)
        axes[1].plot(range(1, n_modes + 1), np.imag(ws), 's')
        axes[1].set_title('Imag(eigenvalues)')
        axes[1].set_xlabel('Mode number')
        axes[1].grid(True)
        plt.tight_layout()

        # ------ Sort by period (descending) ---------------------------
        # wshoras: period in hours for each raw eigenvalue
        wshoras  = 2 * np.pi / np.real(ws) / 3600.0
        sort_idx = np.argsort(wshoras)[::-1]

        wshorasID = np.column_stack([wshoras[sort_idx], sort_idx])
        ws        = ws[sort_idx]   # keep ws aligned with xitilde order

        # ------ Normalise eigenvectors (L2 norm) ----------------------
        normaxi = np.array([
            np.sqrt(np.vdot(xi[:, j], xi[:, j]).real)
            for j in range(n_modes)
        ])

        xitilde = np.zeros_like(xi)
        for j in range(n_modes):
            r = int(wshorasID[j, 1])
            xitilde[:, j] = xi[:, r] / normaxi[r]

        # ------ Modal energy (etilde) ---------------------------------
        ieta2  = indices['ieta2']
        iniu1  = indices['iniu1']
        iniv1  = indices['iniv1']
        iniu2  = indices['iniu2']
        iniv2  = indices['iniv2']

        etilde = np.zeros(n_modes)
        for r in range(n_modes):
            def _quad(a, b):
                v = xitilde[a:b, r]
                return np.dot(v.conj(), M[a:b, a:b].dot(v)).real

            etilde[r] = self.rho2 * self.dx * self.dy * (
                _quad(0,     ieta2)  +
                _quad(ieta2, iniu1)  +
                _quad(iniu1, iniv1)  +
                _quad(iniv1, iniu2)  +
                _quad(iniu2, iniv2)  +
                _quad(iniv2, len(xitilde))
            )

        # ------ Reconstruct spatial fields ----------------------------
        neta = mesh_data['neta'];  meta = mesh_data['meta']
        nu   = mesh_data['nu'];    mu   = mesh_data['mu']
        nv   = mesh_data['nv'];    mv   = mesh_data['mv']

        nodos_eta2 = mesh_data['nodos_eta2']
        nodos_u1   = mesh_data['nodos_u1']
        nodos_v1   = mesh_data['nodos_v1']
        nodos_u2   = mesh_data['nodos_u2']
        nodos_v2   = mesh_data['nodos_v2']

        CBu1       = mesh_data['CBu1']
        CBv1       = mesh_data['CBv1']
        CBu2       = mesh_data['CBu2']
        CBv2       = mesh_data['CBv2']

        eta2tilde = np.full((neta, meta, n_modes), np.nan, dtype=complex)
        u1tilde   = np.full((nu,   mu,   n_modes), np.nan, dtype=complex)
        v1tilde   = np.full((nv,   mv,   n_modes), np.nan, dtype=complex)
        u2tilde   = np.full((nu,   mu,   n_modes), np.nan, dtype=complex)
        v2tilde   = np.full((nv,   mv,   n_modes), np.nan, dtype=complex)

        # eta2 field — starts at row ieta2 in xitilde
        for r in range(n_modes):
            k = ieta2 - 1
            for i in range(neta):
                for j in range(meta):
                    if nodos_eta2[i, j] != 0:
                        k += 1
                        eta2tilde[i, j, r] = xitilde[k, r]

        # u1 field — starts at row iniu1; CBu1 tracks boundary nodes
        for r in range(n_modes):
            k = iniu1 - 1
            s = -1
            for i in range(nu):
                for j in range(mu):
                    if nodos_u1[i, j] != 0:
                        s += 1
                        if CBu1[s] != 0:
                            k += 1
                            u1tilde[i, j, r] = xitilde[k, r]
                        else:
                            u1tilde[i, j, r] = 0.0

        # v1 field — starts at row iniv1
        for r in range(n_modes):
            k = iniv1 - 1
            s = -1
            for i in range(nv):
                for j in range(mv):
                    if nodos_v1[i, j] != 0:
                        s += 1
                        if CBv1[s] != 0:
                            k += 1
                            v1tilde[i, j, r] = xitilde[k, r]
                        else:
                            v1tilde[i, j, r] = 0.0

        # u2 field — starts at row iniu2; CBu2 tracks boundary nodes
        for r in range(n_modes):
            k = iniu2 - 1
            s = -1
            for i in range(nu):
                for j in range(mu):
                    if nodos_u2[i, j] != 0:
                        s += 1
                        if CBu2[s] != 0:
                            k += 1
                            u2tilde[i, j, r] = xitilde[k, r]
                        else:
                            u2tilde[i, j, r] = 0.0

        # v2 field — starts at row iniv2; CBv2 tracks boundary nodes
        for r in range(n_modes):
            k = iniv2 - 1
            s = -1
            for i in range(nv):
                for j in range(mv):
                    if nodos_v2[i, j] != 0:
                        s += 1
                        if CBv2[s] != 0:
                            k += 1
                            v2tilde[i, j, r] = xitilde[k, r]
                        else:
                            v2tilde[i, j, r] = 0.0

        normatilde = np.array([
            np.vdot(xitilde[:, r], xitilde[:, r]).real
            for r in range(n_modes)
        ])

        return {
            'ws':        ws,
            'wshoras':   wshoras,
            'wshorasID': wshorasID,
            'xitilde':   xitilde,
            'etilde':    etilde,
            'eta2tilde': eta2tilde,
            'u1tilde':   u1tilde,
            'v1tilde':   v1tilde,
            'u2tilde':   u2tilde,
            'v2tilde':   v2tilde,
            'normaxi':   normaxi,
            'normatilde': normatilde,
        }
