"""
System Matrices Module
======================

Generates the stiffness (K) and mass (M) matrices for the eigenvalue problem.
Implements the finite-difference discretisation of the shallow-water equations
for a two-layer stratified system with rotation.

Original MATLAB: generaMatrices.m

BUG FIX (vs previous Python version):
    When filling the rotation (Coriolis) terms in the u1, u2 and v2 equation
    rows, the previous code used post-BC-row-removal row offsets (iniv1,
    iniv2, iniu2) as COLUMN offsets.  That is wrong: at the point where those
    rows are filled, BC columns have NOT yet been removed, so the original
    pre-removal column offsets (iv1, iv2, iu2) must be used.
    Concretely the errors were:
        u1 rotation → v1 columns: iniv1  should be  iv1
        u2 rotation → v2 columns: iniv2  should be  iv2
        v2 rotation → u2 columns: iniu2  should be  iu2
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


class MatrixGenerator:
    """
    Generate system matrices K and M for eigenvalue analysis.

    Eigenvalue problem: K * xi = omega * M * xi
    """

    def __init__(self, g, rho1, rho2, dx, dy):
        """
        Parameters
        ----------
        g : float
            Gravitational acceleration (m/s^2)
        rho1 : float
            Density of upper layer (kg/m^3)
        rho2 : float
            Density of lower layer (kg/m^3)
        dx : float
            Grid spacing in x-direction (metres)
        dy : float
            Grid spacing in y-direction (metres)
        """
        self.g    = g
        self.rho1 = rho1
        self.rho2 = rho2
        self.dx   = dx
        self.dy   = dy

        # Density ratios (MATLAB variables e1, e2)
        self.e1 = rho1 / rho2
        self.e2 = (rho2 - rho1) / rho2

    # ------------------------------------------------------------------

    def generate_matrices(self, mesh_data, bathy_data, coriolis_data):
        """
        Generate K (stiffness) and M (mass) matrices.

        Returns
        -------
        K : csr_matrix (complex)
        M : csr_matrix (float)
        indices : dict  – row/column start positions in the reduced matrices
        """
        # ---- Unpack inputs -------------------------------------------
        IDeta1 = mesh_data['IDeta1'];  IDeta2 = mesh_data['IDeta2']
        IDu1   = mesh_data['IDu1'];    IDv1   = mesh_data['IDv1']
        IDu2   = mesh_data['IDu2'];    IDv2   = mesh_data['IDv2']
        CBu1   = mesh_data['CBu1'];    CBv1   = mesh_data['CBv1']
        CBu2   = mesh_data['CBu2'];    CBv2   = mesh_data['CBv2']

        heta1  = bathy_data['heta1']
        hu1    = bathy_data['hu1'];    hv1    = bathy_data['hv1']
        hu2    = bathy_data['hu2'];    hv2    = bathy_data['hv2']
        nnhu1  = bathy_data['nnhu1'];  nnhv1  = bathy_data['nnhv1']
        nnhu2  = bathy_data['nnhu2'];  nnhv2  = bathy_data['nnhv2']

        fu1    = coriolis_data['fu1']; fv1    = coriolis_data['fv1']
        fu2    = coriolis_data['fu2']; fv2    = coriolis_data['fv2']
        nnfu1  = coriolis_data['nnfu1']; nnfv1 = coriolis_data['nnfv1']
        nnfu2  = coriolis_data['nnfu2']; nnfv2 = coriolis_data['nnfv2']

        # ---- Dimensions ----------------------------------------------
        nIDeta1 = len(IDeta1)
        nIDeta2 = len(IDeta2)
        nIDu1   = len(IDu1)
        nIDv1   = len(IDv1)
        nIDu2   = len(IDu2)
        nIDv2   = len(IDv2)

        # ---- Block start positions in the FULL (pre-BC-removal) matrix
        # (mirrors MATLAB variables ieta2, iu1, iv1, iu2, iv2)
        ieta2 = nIDeta1
        iu1   = nIDeta1 + nIDeta2
        iv1   = iu1 + len(CBu1)
        iu2   = iv1 + len(CBv1)
        iv2   = iu2 + len(CBu2)

        total = nIDeta1 + nIDeta2 + len(CBu1) + len(CBv1) + len(CBu2) + len(CBv2)
        print(f"K matrix size before BC removal: {total} x {total}")

        # ---- Initialise sparse K ------------------------------------
        K = lil_matrix((total, total), dtype=complex)

        # Row indices to delete (boundary velocity nodes)
        u1del = iu1 + np.where(CBu1 == 0)[0]
        v1del = iv1 + np.where(CBv1 == 0)[0]
        u2del = iu2 + np.where(CBu2 == 0)[0]
        v2del = iv2 + np.where(CBv2 == 0)[0]

        # ---- Equations for eta1 (rows 0..nIDeta1-1) -----------------
        for n in range(nIDeta1):
            # Layer-1 contributions
            K[n, iu1 + IDeta1[n, 0]] =  1j * self.e1 * self.g * hu1[IDeta1[n, 0]] / self.dx
            K[n, iv1 + IDeta1[n, 1]] =  1j * self.e1 * self.g * hv1[IDeta1[n, 1]] / self.dy
            K[n, iu1 + IDeta1[n, 2]] = -1j * self.e1 * self.g * hu1[IDeta1[n, 2]] / self.dx
            K[n, iv1 + IDeta1[n, 3]] = -1j * self.e1 * self.g * hv1[IDeta1[n, 3]] / self.dy

            if heta1[n, 1] != 0:
                # Add layer-2 contributions when both layers are active
                K[n, iu2 + IDeta1[n, 4]] =  1j * self.e1 * self.g * hu2[IDeta1[n, 4]] / self.dx
                K[n, iv2 + IDeta1[n, 5]] =  1j * self.e1 * self.g * hv2[IDeta1[n, 5]] / self.dy
                K[n, iu2 + IDeta1[n, 6]] = -1j * self.e1 * self.g * hu2[IDeta1[n, 6]] / self.dx
                K[n, iv2 + IDeta1[n, 7]] = -1j * self.e1 * self.g * hv2[IDeta1[n, 7]] / self.dy

        # ---- Equations for eta2 (rows ieta2..iu1-1) -----------------
        for n in range(nIDeta2):
            K[ieta2 + n, iu2 + IDeta2[n, 0]] =  1j * self.e2 * self.g * hu2[IDeta2[n, 0]] / self.dx
            K[ieta2 + n, iv2 + IDeta2[n, 1]] =  1j * self.e2 * self.g * hv2[IDeta2[n, 1]] / self.dy
            K[ieta2 + n, iu2 + IDeta2[n, 2]] = -1j * self.e2 * self.g * hu2[IDeta2[n, 2]] / self.dx
            K[ieta2 + n, iv2 + IDeta2[n, 3]] = -1j * self.e2 * self.g * hv2[IDeta2[n, 3]] / self.dy

        # ---- Remove BC rows -----------------------------------------
        mask = np.ones(total, dtype=bool)
        mask[u1del] = False
        mask[v1del] = False
        mask[u2del] = False
        mask[v2del] = False
        K = K[mask, :]

        # ---- Post-removal ROW positions of velocity blocks ----------
        # (No rows deleted before iu1, so iniu1 == iu1.)
        iniu1 = iu1                         # u1 interior rows start here
        iniv1 = iu1 + nIDu1                 # v1 interior rows start here
        iniu2 = iu1 + nIDu1 + nIDv1        # u2 interior rows start here
        iniv2 = iu1 + nIDu1 + nIDv1 + nIDu2  # v2 interior rows start here

        # NOTE: columns have NOT been removed yet. When filling the rotation
        # (Coriolis) terms below, we MUST use the original pre-removal column
        # block offsets (iv1, iu2, iv2), NOT the row offsets above.

        # ---- Equations for u1 (interior, rows iniu1..) ---------------
        for n in range(nIDu1):
            row = iniu1 + n
            # Pressure gradient
            K[row, IDu1[n, 0]] =  1j * self.e1 * self.g * nnhu1[n] / self.dx
            K[row, IDu1[n, 1]] = -1j * self.e1 * self.g * nnhu1[n] / self.dx
            # Rotation (Coriolis) — columns into the FULL v1 block (iv1)
            for j in range(2, 6):
                idx = IDu1[n, j]
                if idx >= 0:
                    K[row, iv1 + idx] = (
                        -1j * nnhu1[n] * 0.125 * hv1[idx]
                        * (nnfu1[n] / nnhu1[n] + fv1[idx] / hv1[idx])
                    )

        # ---- Equations for v1 (interior, rows iniv1..) ---------------
        for n in range(nIDv1):
            row = iniv1 + n
            # Pressure gradient
            K[row, IDv1[n, 0]] =  1j * self.e1 * self.g * nnhv1[n] / self.dy
            K[row, IDv1[n, 1]] = -1j * self.e1 * self.g * nnhv1[n] / self.dy
            # Rotation — columns into the FULL u1 block (iu1)
            # (iniu1 == iu1, so this is correct either way, kept as iu1
            #  for clarity and symmetry with the other blocks below)
            for j in range(2, 6):
                idx = IDv1[n, j]
                if idx >= 0:
                    K[row, iu1 + idx] = (
                        1j * nnhv1[n] * 0.125 * hu1[idx]
                        * (nnfv1[n] / nnhv1[n] + fu1[idx] / hu1[idx])
                    )

        # ---- Equations for u2 (interior, rows iniu2..) ---------------
        for n in range(nIDu2):
            row = iniu2 + n
            # Pressure gradients (eta1 and eta2)
            K[row, IDu2[n, 0]]          =  1j * self.e1 * self.g * nnhu2[n] / self.dx
            K[row, IDu2[n, 1]]          = -1j * self.e1 * self.g * nnhu2[n] / self.dx
            K[row, ieta2 + IDu2[n, 2]]  =  1j * self.e2 * self.g * nnhu2[n] / self.dx
            K[row, ieta2 + IDu2[n, 3]]  = -1j * self.e2 * self.g * nnhu2[n] / self.dx
            # Rotation — columns into the FULL v2 block (iv2)  ← FIX
            for j in range(4, 8):
                idx = IDu2[n, j]
                if idx >= 0:
                    K[row, iv2 + idx] = (
                        -1j * nnhu2[n] * 0.125 * hv2[idx]
                        * (nnfu2[n] / nnhu2[n] + fv2[idx] / hv2[idx])
                    )

        # ---- Equations for v2 (interior, rows iniv2..) ---------------
        for n in range(nIDv2):
            row = iniv2 + n
            # Pressure gradients (eta1 and eta2)
            K[row, IDv2[n, 0]]          =  1j * self.e1 * self.g * nnhv2[n] / self.dy
            K[row, IDv2[n, 1]]          = -1j * self.e1 * self.g * nnhv2[n] / self.dy
            K[row, ieta2 + IDv2[n, 2]]  =  1j * self.e2 * self.g * nnhv2[n] / self.dy
            K[row, ieta2 + IDv2[n, 3]]  = -1j * self.e2 * self.g * nnhv2[n] / self.dy
            # Rotation — columns into the FULL u2 block (iu2)  ← FIX
            for j in range(4, 8):
                idx = IDv2[n, j]
                if idx >= 0:
                    K[row, iu2 + idx] = (
                        1j * nnhv2[n] * 0.125 * hu2[idx]
                        * (nnfv2[n] / nnhv2[n] + fu2[idx] / hu2[idx])
                    )

        # ---- Remove BC columns (same mask as rows) ------------------
        K = K[:, mask]
        K = csr_matrix(K)

        N = K.shape[0]
        print(f"K matrix size after BC removal : {N} x {N}")

        # ---- Mass matrix M (diagonal) --------------------------------
        M = lil_matrix((N, N), dtype=float)

        for n in range(nIDeta1):
            M[n, n] = self.e1 * self.g

        k = nIDeta1
        for n in range(nIDeta2):
            M[k + n, k + n] = self.e2 * self.g

        k = iniu1
        for n in range(nIDu1):
            M[k + n, k + n] = nnhu1[n]

        k = iniv1
        for n in range(nIDv1):
            M[k + n, k + n] = nnhv1[n]

        k = iniu2
        for n in range(nIDu2):
            M[k + n, k + n] = nnhu2[n]

        k = iniv2
        for n in range(nIDv2):
            M[k + n, k + n] = nnhv2[n]

        M = csr_matrix(M)

        indices = {
            'ieta2': ieta2,
            'iniu1': iniu1,
            'iniv1': iniv1,
            'iniu2': iniu2,
            'iniv2': iniv2,
        }

        return K, M, indices
