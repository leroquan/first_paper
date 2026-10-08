"""
Mesh Generation Module
======================

Creates the computational mesh for modal analysis using a staggered grid
(Arakawa C-grid) for a two-layer stratified system.

Original MATLAB: generaMalla.m

BUG FIX (vs previous Python version):
    np.arange excludes the stop value; MATLAB's colon operator includes it
    when the step lands exactly on it.  Without the fix the v-grid (yi) can
    be one row short, leading to wrong node counts and matrix sizes.
    Fix: add a small epsilon (1e-10 km) to the stop of every arange call.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import griddata


def _extract_contour_paths(cs):
    """
    Extract vertex arrays from a matplotlib QuadContourSet in a way that
    works across all matplotlib versions.

    Strategy (tried in order):
      1. cs.get_paths()     — works on matplotlib >= 3.8
      2. cs.allsegs         — works on matplotlib < 3.8
      3. cs.collections     — older fallback (deprecated in 3.7, removed 3.8)

    Returns
    -------
    list of np.ndarray, each shaped (N, 2)

    Raises
    ------
    ValueError
        If no contour segment is found.
    """
    all_vertices = []

    if hasattr(cs, 'get_paths'):
        try:
            for path in cs.get_paths():
                if len(path.vertices) > 0:
                    all_vertices.append(path.vertices)
            if all_vertices:
                return all_vertices
        except Exception:
            pass

    if hasattr(cs, 'allsegs'):
        try:
            for level_segs in cs.allsegs:
                for seg in level_segs:
                    if len(seg) > 0:
                        all_vertices.append(np.asarray(seg))
            if all_vertices:
                return all_vertices
        except Exception:
            pass

    if hasattr(cs, 'collections'):
        try:
            for collection in cs.collections:
                for path in collection.get_paths():
                    if len(path.vertices) > 0:
                        all_vertices.append(path.vertices)
            if all_vertices:
                return all_vertices
        except Exception:
            pass

    return all_vertices


class MeshGenerator:
    """
    Generate computational mesh for two-layer modal analysis.

    Uses staggered grid with:
    - eta nodes : water surface elevation (cell centres)
    - u   nodes : velocity in x-direction (cell faces perpendicular to x)
    - v   nodes : velocity in y-direction (cell faces perpendicular to y)
    """

    def __init__(self, dx, dy, x_range, y_range):
        """
        Parameters
        ----------
        dx : float
            Grid spacing in x-direction (metres).
        dy : float
            Grid spacing in y-direction (metres).
        x_range : tuple
            (x_min, x_max) in kilometres.
        y_range : tuple
            (y_min, y_max) in kilometres.
        """
        self.dx = dx
        self.dy = dy
        self.x_range = x_range
        self.y_range = y_range

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_grid(self, shoreline_file, bathymetry_file, h1):
        """
        Generate the complete staggered grid mesh.

        Parameters
        ----------
        shoreline_file : str
            Path to shoreline coordinates file (columns: x_m, y_m).
        bathymetry_file : str
            Path to bathymetry data file (columns: x_m, y_m, depth_m).
        h1 : float
            Thickness of the upper layer in metres.

        Returns
        -------
        dict
            All mesh arrays required by the matrix-assembly step.
        """
        # ---- Load shoreline ------------------------------------------
        shoreline_data = np.loadtxt(shoreline_file)
        xb = shoreline_data[:, 0] / 1000.0   # m → km
        yb = shoreline_data[:, 1] / 1000.0
        xb = np.append(xb, xb[0])            # close polygon
        yb = np.append(yb, yb[0])

        # ---- Build staggered grids -----------------------------------
        dx_km = self.dx / 1000.0
        dy_km = self.dy / 1000.0

        # Small epsilon so that arange includes the endpoint when the step
        # lands exactly on it — matches MATLAB colon operator behaviour.
        # Using half a grid step is safe: it will never add an extra point
        # beyond the intended boundary.
        eps_x = dx_km * 0.5
        eps_y = dy_km * 0.5

        # eta grid: cell centres (offset by half a step from the boundary)
        xp = np.arange(self.x_range[0] + dx_km / 2,
                        self.x_range[1] + eps_x, dx_km)
        yp = np.arange(self.y_range[0] + dy_km / 2,
                        self.y_range[1] + eps_y, dy_km)

        # u / v grids: cell faces (start at boundary)
        xi = np.arange(self.x_range[0], self.x_range[1] + eps_x, dx_km)
        yi = np.arange(self.y_range[0], self.y_range[1] + eps_y, dy_km)

        xeta, yeta = np.meshgrid(xp, yp)
        xu,   yu   = np.meshgrid(xi, yp)
        xv,   yv   = np.meshgrid(xp, yi)

        # ---- Load and interpolate bathymetry -------------------------
        bathy_data = np.loadtxt(bathymetry_file)
        xc    = bathy_data[:, 0] / 1000.0
        yc    = bathy_data[:, 1] / 1000.0
        beril = -bathy_data[:, 2]   # depth stored as negative elevation

        Heta = griddata((xc, yc), beril, (xeta, yeta),
                        method='linear', fill_value=0.0)
        Heta[np.isnan(Heta)] = 0.0

        # ---- Extract the h1-depth contour (layer-2 boundary) ---------
        fig, ax = plt.subplots()
        cs = ax.contour(xeta, yeta, Heta, levels=[h1])
        plt.close(fig)

        all_segments = _extract_contour_paths(cs)

        if not all_segments:
            raise ValueError(
                f"No contour found at depth h1={h1} m.  "
                f"Check that the bathymetry data contains points deeper "
                f"than {h1} m inside the lake domain.  "
                f"Depth range in Heta: [{Heta.min():.1f}, {Heta.max():.1f}] m."
            )

        main_segment = max(all_segments, key=len)
        xb20 = main_segment[:, 0]
        yb20 = main_segment[:, 1]

        # ---- Identify nodes inside polygons --------------------------
        INeta1 = self._points_in_polygon(xeta, yeta, xb,   yb)
        INu1   = self._points_in_polygon(xu,   yu,   xb,   yb)
        INv1   = self._points_in_polygon(xv,   yv,   xb,   yb)

        INeta2 = self._points_in_polygon(xeta, yeta, xb20, yb20)
        INu2   = self._points_in_polygon(xu,   yu,   xb20, yb20)
        INv2   = self._points_in_polygon(xv,   yv,   xb20, yb20)

        neta, meta = xeta.shape
        nu,   mu   = xu.shape
        nv,   mv   = xv.shape

        # ---- Prune nodes (mirrors MATLAB logic exactly) --------------
        INeta1 = self._remove_boundary_eta_nodes(INeta1, INu1, INv1, neta, meta)
        INeta1 = self._remove_isolated_nodes(INeta1)

        INeta2 = self._remove_boundary_eta_nodes(INeta2, INu2, INv2, neta, meta)
        INeta2 = self._remove_isolated_nodes(INeta2)
        INeta2 = INeta2 & INeta1   # layer-2 requires layer-1 above

        INu1 = self._remove_boundary_u_nodes(INu1, INeta1, nu, mu)
        INv1 = self._remove_boundary_v_nodes(INv1, INeta1, nv, mv)
        INu2 = self._remove_boundary_u_nodes(INu2, INeta2, nu, mu)
        INv2 = self._remove_boundary_v_nodes(INv2, INeta2, nv, mv)

        # ---- Number nodes (1-based, row-major) -----------------------
        nodos_eta1 = self._number_nodes(INeta1)
        nodos_eta2 = self._number_nodes(INeta2)
        nodos_u1   = self._number_nodes(INu1)
        nodos_v1   = self._number_nodes(INv1)
        nodos_u2   = self._number_nodes(INu2)
        nodos_v2   = self._number_nodes(INv2)

        # ---- Connectivity matrices (0-based indices) ----------------
        IDeta1 = self._generate_IDeta1(nodos_eta1, nodos_u1, nodos_v1,
                                        nodos_u2,   nodos_v2, neta, meta)
        IDeta2 = self._generate_IDeta2(nodos_eta2, nodos_u2, nodos_v2,
                                        neta, meta)
        IDu1   = self._generate_IDu1(nodos_u1, nodos_eta1, nodos_v1, nu, mu)
        IDv1   = self._generate_IDv1(nodos_v1, nodos_eta1, nodos_u1, nv, mv)
        IDu2   = self._generate_IDu2(nodos_u2, nodos_eta1, nodos_eta2,
                                      nodos_v2, nu, mu)
        IDv2   = self._generate_IDv2(nodos_v2, nodos_eta1, nodos_eta2,
                                      nodos_u2, nv, mv)

        # ---- Boundary-condition masks --------------------------------
        CBu1 = self._generate_CB(nodos_u1, nodos_eta1, nu, mu, 'u')
        CBv1 = self._generate_CB(nodos_v1, nodos_eta1, nv, mv, 'v')
        CBu2 = self._generate_CB(nodos_u2, nodos_eta2, nu, mu, 'u')
        CBv2 = self._generate_CB(nodos_v2, nodos_eta2, nv, mv, 'v')

        # Remove boundary rows from connectivity (matches MATLAB)
        IDu1 = IDu1[CBu1 != 0]
        IDv1 = IDv1[CBv1 != 0]
        IDu2 = IDu2[CBu2 != 0]
        IDv2 = IDv2[CBv2 != 0]

        # ---- Diagnostics --------------------------------------------
        print(f"Grid size       : neta={neta}, meta={meta}")
        print(f"                  nu={nu},   mu={mu}")
        print(f"                  nv={nv},   mv={mv}")
        print(f"nIDeta1={np.sum(nodos_eta1>0)},  nIDeta2={np.sum(nodos_eta2>0)}")
        print(f"nIDu1  ={np.sum(nodos_u1>0)},    nIDv1  ={np.sum(nodos_v1>0)}")
        print(f"nIDu2  ={np.sum(nodos_u2>0)},    nIDv2  ={np.sum(nodos_v2>0)}")
        print(f"Interior u1={np.sum(CBu1!=0)},   Interior v1={np.sum(CBv1!=0)}")
        print(f"Interior u2={np.sum(CBu2!=0)},   Interior v2={np.sum(CBv2!=0)}")
        total = (np.sum(nodos_eta1 > 0) + np.sum(nodos_eta2 > 0)
                 + np.sum(CBu1 != 0) + np.sum(CBv1 != 0)
                 + np.sum(CBu2 != 0) + np.sum(CBv2 != 0))
        print(f"Expected K matrix size: {total} x {total}")

        return {
            'xeta': xeta, 'yeta': yeta,
            'xu':   xu,   'yu':   yu,
            'xv':   xv,   'yv':   yv,
            'xb':   xb,   'yb':   yb,
            'xb20': xb20, 'yb20': yb20,
            'Heta': Heta,
            'INeta1': INeta1, 'INu1': INu1, 'INv1': INv1,
            'INeta2': INeta2, 'INu2': INu2, 'INv2': INv2,
            'nodos_eta1': nodos_eta1, 'nodos_eta2': nodos_eta2,
            'nodos_u1':   nodos_u1,   'nodos_v1':   nodos_v1,
            'nodos_u2':   nodos_u2,   'nodos_v2':   nodos_v2,
            'IDeta1': IDeta1, 'IDeta2': IDeta2,
            'IDu1':   IDu1,   'IDv1':   IDv1,
            'IDu2':   IDu2,   'IDv2':   IDv2,
            'CBu1':   CBu1,   'CBv1':   CBv1,
            'CBu2':   CBu2,   'CBv2':   CBv2,
            'neta': neta, 'meta': meta,
            'nu':   nu,   'mu':   mu,
            'nv':   nv,   'mv':   mv,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _points_in_polygon(self, x, y, xpoly, ypoly):
        """Return boolean mask — True where (x,y) lies inside the polygon."""
        points  = np.column_stack([x.ravel(), y.ravel()])
        polygon = Path(np.column_stack([xpoly, ypoly]))
        return polygon.contains_points(points).reshape(x.shape)

    def _remove_boundary_eta_nodes(self, INeta, INu, INv, n, m):
        """
        Deactivate eta nodes that do not have all four surrounding
        u / v nodes active (mirrors MATLAB nested loop).
        """
        result = INeta.copy()
        _, mu = INu.shape
        nv_rows, _ = INv.shape

        for i in range(n):
            for j in range(m):
                if result[i, j]:
                    if j >= mu - 1 or i >= nv_rows - 1:
                        result[i, j] = False
                    elif (not INu[i, j] or not INu[i, j + 1]
                          or not INv[i, j] or not INv[i + 1, j]):
                        result[i, j] = False
        return result

    def _remove_isolated_nodes(self, IN):
        """Deactivate nodes that have no active neighbours."""
        result = IN.copy()
        n, m = IN.shape
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                if result[i, j]:
                    if (not IN[i - 1, j] and not IN[i + 1, j]
                            and not IN[i, j - 1] and not IN[i, j + 1]):
                        result[i, j] = False
        return result

    def _remove_boundary_u_nodes(self, INu, INeta, n, m):
        """Deactivate u nodes with no active eta neighbour on either side."""
        result = INu.copy()
        neta_n, neta_m = INeta.shape

        for i in range(n):
            for j in range(m):
                if result[i, j]:
                    if j == 0:
                        if i >= neta_n or not INeta[i, j]:
                            result[i, j] = False
                    elif j == m - 1:
                        if i >= neta_n or not INeta[i, j - 1]:
                            result[i, j] = False
                    else:
                        if (i >= neta_n or j - 1 >= neta_m or j >= neta_m
                                or (not INeta[i, j - 1] and not INeta[i, j])):
                            result[i, j] = False
        return result

    def _remove_boundary_v_nodes(self, INv, INeta, n, m):
        """Deactivate v nodes with no active eta neighbour above or below."""
        result = INv.copy()
        neta_n, neta_m = INeta.shape

        for i in range(n):
            for j in range(m):
                if result[i, j]:
                    if i == 0:
                        if j >= neta_m or not INeta[i, j]:
                            result[i, j] = False
                    elif i == n - 1:
                        if i - 1 >= neta_n or j >= neta_m or not INeta[i - 1, j]:
                            result[i, j] = False
                    else:
                        if (i - 1 >= neta_n or i >= neta_n or j >= neta_m
                                or (not INeta[i - 1, j] and not INeta[i, j])):
                            result[i, j] = False
        return result

    def _number_nodes(self, IN):
        """Assign 1-based sequential numbers to active nodes (row-major)."""
        nodos = np.zeros_like(IN, dtype=int)
        k = 0
        for i in range(IN.shape[0]):
            for j in range(IN.shape[1]):
                if IN[i, j]:
                    k += 1
                    nodos[i, j] = k
        return nodos

    # ---- Connectivity matrices (0-based indices, -1 = no node) ----------

    def _generate_IDeta1(self, nodos_eta1, nodos_u1, nodos_v1,
                          nodos_u2, nodos_v2, n, m):
        count  = int(np.sum(nodos_eta1 > 0))
        IDeta1 = np.zeros((count, 8), dtype=int)
        for i in range(n):
            for j in range(m):
                if nodos_eta1[i, j] > 0:
                    k = nodos_eta1[i, j] - 1
                    IDeta1[k, 0] = nodos_u1[i,     j + 1] - 1
                    IDeta1[k, 1] = nodos_v1[i + 1, j    ] - 1
                    IDeta1[k, 2] = nodos_u1[i,     j    ] - 1
                    IDeta1[k, 3] = nodos_v1[i,     j    ] - 1
                    IDeta1[k, 4] = nodos_u2[i,     j + 1] - 1
                    IDeta1[k, 5] = nodos_v2[i + 1, j    ] - 1
                    IDeta1[k, 6] = nodos_u2[i,     j    ] - 1
                    IDeta1[k, 7] = nodos_v2[i,     j    ] - 1
        return IDeta1

    def _generate_IDeta2(self, nodos_eta2, nodos_u2, nodos_v2, n, m):
        count  = int(np.sum(nodos_eta2 > 0))
        IDeta2 = np.zeros((count, 4), dtype=int)
        for i in range(n):
            for j in range(m):
                if nodos_eta2[i, j] > 0:
                    k = nodos_eta2[i, j] - 1
                    IDeta2[k, 0] = nodos_u2[i,     j + 1] - 1
                    IDeta2[k, 1] = nodos_v2[i + 1, j    ] - 1
                    IDeta2[k, 2] = nodos_u2[i,     j    ] - 1
                    IDeta2[k, 3] = nodos_v2[i,     j    ] - 1
        return IDeta2

    def _generate_IDu1(self, nodos_u1, nodos_eta1, nodos_v1, n, m):
        count = int(np.sum(nodos_u1 > 0))
        IDu1  = np.zeros((count, 6), dtype=int)
        for i in range(n):
            for j in range(m):
                if nodos_u1[i, j] > 0:
                    k = nodos_u1[i, j] - 1
                    IDu1[k, 0] = nodos_eta1[i, j    ] - 1
                    IDu1[k, 1] = nodos_eta1[i, j - 1] - 1 if j > 0 else -1
                    IDu1[k, 2] = nodos_v1[i + 1, j - 1] - 1 if j > 0 else -1
                    IDu1[k, 3] = nodos_v1[i + 1, j    ] - 1
                    IDu1[k, 4] = nodos_v1[i,     j - 1] - 1 if j > 0 else -1
                    IDu1[k, 5] = nodos_v1[i,     j    ] - 1
        return IDu1

    def _generate_IDv1(self, nodos_v1, nodos_eta1, nodos_u1, n, m):
        count = int(np.sum(nodos_v1 > 0))
        IDv1  = np.zeros((count, 6), dtype=int)
        for i in range(n):
            for j in range(m):
                if nodos_v1[i, j] > 0:
                    k = nodos_v1[i, j] - 1
                    IDv1[k, 0] = nodos_eta1[i,     j] - 1
                    IDv1[k, 1] = nodos_eta1[i - 1, j] - 1 if i > 0 else -1
                    IDv1[k, 2] = nodos_u1[i,     j    ] - 1
                    IDv1[k, 3] = nodos_u1[i,     j + 1] - 1
                    IDv1[k, 4] = nodos_u1[i - 1, j    ] - 1 if i > 0 else -1
                    IDv1[k, 5] = nodos_u1[i - 1, j + 1] - 1 if i > 0 else -1
        return IDv1

    def _generate_IDu2(self, nodos_u2, nodos_eta1, nodos_eta2,
                        nodos_v2, n, m):
        count = int(np.sum(nodos_u2 > 0))
        IDu2  = np.zeros((count, 8), dtype=int)
        for i in range(n):
            for j in range(m):
                if nodos_u2[i, j] > 0:
                    k = nodos_u2[i, j] - 1
                    IDu2[k, 0] = nodos_eta1[i, j    ] - 1
                    IDu2[k, 1] = nodos_eta1[i, j - 1] - 1 if j > 0 else -1
                    IDu2[k, 2] = nodos_eta2[i, j    ] - 1
                    IDu2[k, 3] = nodos_eta2[i, j - 1] - 1 if j > 0 else -1
                    IDu2[k, 4] = nodos_v2[i + 1, j - 1] - 1 if j > 0 else -1
                    IDu2[k, 5] = nodos_v2[i + 1, j    ] - 1
                    IDu2[k, 6] = nodos_v2[i,     j - 1] - 1 if j > 0 else -1
                    IDu2[k, 7] = nodos_v2[i,     j    ] - 1
        return IDu2

    def _generate_IDv2(self, nodos_v2, nodos_eta1, nodos_eta2,
                        nodos_u2, n, m):
        count = int(np.sum(nodos_v2 > 0))
        IDv2  = np.zeros((count, 8), dtype=int)
        for i in range(n):
            for j in range(m):
                if nodos_v2[i, j] > 0:
                    k = nodos_v2[i, j] - 1
                    IDv2[k, 0] = nodos_eta1[i,     j] - 1
                    IDv2[k, 1] = nodos_eta1[i - 1, j] - 1 if i > 0 else -1
                    IDv2[k, 2] = nodos_eta2[i,     j] - 1
                    IDv2[k, 3] = nodos_eta2[i - 1, j] - 1 if i > 0 else -1
                    IDv2[k, 4] = nodos_u2[i,     j    ] - 1
                    IDv2[k, 5] = nodos_u2[i,     j + 1] - 1
                    IDv2[k, 6] = nodos_u2[i - 1, j    ] - 1 if i > 0 else -1
                    IDv2[k, 7] = nodos_u2[i - 1, j + 1] - 1 if i > 0 else -1
        return IDv2

    def _generate_CB(self, nodos_vel, nodos_eta, n, m, vel_type):
        """
        Build the boundary-condition mask for a set of velocity nodes.

        CB[k] == 0  →  node k is on the boundary (zero-normal-flow BC).
        CB[k] >  0  →  node k is interior; value is its interior index.
        """
        count = int(np.sum(nodos_vel > 0))
        CB = np.zeros(count, dtype=int)
        s  = 0

        for i in range(n):
            for j in range(m):
                if nodos_vel[i, j] > 0:
                    k = nodos_vel[i, j] - 1

                    if vel_type == 'u':
                        if j == 0:
                            on_boundary = (nodos_eta[i, j] == 0)
                        elif j == m - 1:
                            on_boundary = (nodos_eta[i, j - 1] == 0)
                        else:
                            on_boundary = (nodos_eta[i, j - 1] == 0
                                           or nodos_eta[i, j] == 0)
                    else:   # 'v'
                        if i == 0:
                            on_boundary = (nodos_eta[i, j] == 0)
                        elif i == n - 1:
                            on_boundary = (nodos_eta[i - 1, j] == 0)
                        else:
                            on_boundary = (nodos_eta[i - 1, j] == 0
                                           or nodos_eta[i, j] == 0)

                    if not on_boundary:
                        s += 1
                        CB[k] = s

        return CB

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_mesh(self, mesh_data):
        """Plot layer-1 and layer-2 meshes side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, layer, eta_key, u_key, v_key, in_eta, in_u, in_v in [
            (axes[0], 'Layer 1',
             'xeta', 'xu', 'xv', 'INeta1', 'INu1', 'INv1'),
            (axes[1], 'Layer 2',
             'xeta', 'xu', 'xv', 'INeta2', 'INu2', 'INv2'),
        ]:
            ax.plot(mesh_data['xb'],   mesh_data['yb'],   'k-',  label='Shoreline')
            ax.plot(mesh_data['xb20'], mesh_data['yb20'], 'r--', label='Interface')
            ax.plot(mesh_data[eta_key][mesh_data[in_eta]],
                    mesh_data['yeta'][mesh_data[in_eta]],
                    'go', markersize=2, label='eta')
            ax.plot(mesh_data[u_key][mesh_data[in_u]],
                    mesh_data['yu'][mesh_data[in_u]],
                    'r>', markersize=2, label='u')
            ax.plot(mesh_data[v_key][mesh_data[in_v]],
                    mesh_data['yv'][mesh_data[in_v]],
                    'b^', markersize=2, label='v')
            ax.set_title(f'{layer} Mesh')
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.legend(markerscale=3)
            ax.set_aspect('equal')

        plt.tight_layout()
        return fig
