"""
Mesh Generation Module
======================

Creates the computational mesh for modal analysis using a staggered grid
(Arakawa C-grid) for a two-layer stratified system.

Original MATLAB: generaMalla.m
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import griddata


def _path_to_polygons(path):
    vertices = path.vertices
    codes = path.codes

    polys = []
    current = []

    for v, c in zip(vertices, codes):
        if c == Path.MOVETO:
            if len(current) > 2:
                polys.append(np.array(current))
            current = [v]

        elif c == Path.LINETO:
            current.append(v)

        elif c == Path.CLOSEPOLY:
            if len(current) > 2:
                polys.append(np.array(current))
            current = []

    if len(current) > 2:
        polys.append(np.array(current))

    return polys


def _extract_contour_paths(cs):
    """
    Extract vertex arrays from a matplotlib QuadContourSet in a way that
    works across all matplotlib versions.

    Strategy (tried in order):
      1. cs.get_paths()          — works on matplotlib >= 3.8 where
                                   .allsegs and .collections were removed.
      2. cs.allsegs              — works on matplotlib < 3.8.
      3. cs.collections          — older fallback (deprecated in 3.7,
                                   removed in 3.8).

    Returns
    -------
    list of np.ndarray, each shaped (N, 2)
        Every non-empty contour segment found.  May be empty if the
        contour level does not intersect the data.

    Raises
    ------
    RuntimeError
        If none of the three strategies succeeds (indicates an unexpected
        matplotlib API change).
    """
    all_vertices = []

    # --- Strategy 1: get_paths() on the ContourSet itself (matplotlib >= 3.8) ---
    if hasattr(cs, 'get_paths'):
        try:
            for path in cs.get_paths():
                if len(path.vertices) > 0:
                    all_vertices.extend(_path_to_polygons(path))
            if all_vertices:
                return all_vertices
        except Exception:
            pass  # fall through to next strategy

    # --- Strategy 2: allsegs attribute (matplotlib < 3.8) ---
    if hasattr(cs, 'allsegs'):
        try:
            for level_segs in cs.allsegs:
                for seg in level_segs:
                    if len(seg) > 0:
                        all_vertices.extend(_path_to_polygons(np.asarray(seg)))
            if all_vertices:
                return all_vertices
        except Exception:
            pass

    # --- Strategy 3: collections attribute (matplotlib < 3.7) ---
    if hasattr(cs, 'collections'):
        try:
            for collection in cs.collections:
                for path in collection.get_paths():
                    if len(path.vertices) > 0:
                        all_vertices.extend(_path_to_polygons(path))
            if all_vertices:
                return all_vertices
        except Exception:
            pass

    # None of the strategies found anything — let caller decide
    return all_vertices


class MeshGenerator:
    """
    Generate computational mesh for two-layer modal analysis.

    Uses staggered grid with:
    - eta nodes: water surface elevation (cell centres)
    - u   nodes: velocity in x-direction (cell faces ⊥ x)
    - v   nodes: velocity in y-direction (cell faces ⊥ y)
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

    def generate_grid(self, h1, shoreline_file, bathymetry_file):
        """
        Generate the complete staggered grid mesh.

        Parameters
        ----------
        shoreline_file : str
            Path to shoreline coordinates file (columns: x_m, y_m).
        bathymetry_file : str
            Path to bathymetry data file (columns: x_m, y_m, depth_m).
        h1 : float
            Thickness of the upper layer in metres.  The layer-2 domain
            boundary is the contour where total depth == h1.

        Returns
        -------
        dict
            All mesh arrays required by the matrix-assembly step.
        """
        # ---- Load shoreline ------------------------------------------
        shoreline_data = np.loadtxt(shoreline_file)
        xb = shoreline_data[:, 0] / 1000.0   # convert m → km
        yb = shoreline_data[:, 1] / 1000.0
        xb = np.append(xb, xb[0])            # close the polygon
        yb = np.append(yb, yb[0])

        # ---- Build staggered grids -----------------------------------
        dx_km = self.dx / 1000.0
        dy_km = self.dy / 1000.0

        # eta grid: cell centres
        xp = np.arange(self.x_range[0] + dx_km / 2, self.x_range[1], dx_km)
        yp = np.arange(self.y_range[0] + dy_km / 2, self.y_range[1], dy_km)

        # u / v grids: cell faces
        # Must have mu = meta+1 columns (xu) and nv = neta+1 rows (xv).
        # np.arange can miss the final face line due to floating-point rounding,
        # so we extend the stop by one step then trim to the exact required length.
        xi = np.arange(self.x_range[0], self.x_range[1] + dx_km, dx_km)[:len(xp) + 1]
        yi = np.arange(self.y_range[0], self.y_range[1] + dy_km, dy_km)[:len(yp) + 1]

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
        #
        # This is the critical step that was previously broken: the old
        # code would silently replace xb20/yb20 with the shoreline whenever
        # contour extraction failed, making INeta2 == INeta1 and producing
        # a matrix that is far too large.
        #
        # We use a non-interactive Agg backend figure so there is no
        # display dependency, then call _extract_contour_paths() which
        # tries every known matplotlib API in order.
        import matplotlib.figure
        _fig = matplotlib.figure.Figure()
        _ax = _fig.add_subplot(111)
        cs = _ax.contour(xeta, yeta, Heta, levels=[h1])

        all_segments = _extract_contour_paths(cs)

        if not all_segments:
            raise ValueError(
                f"No contour found at depth h1={h1} m.  "
                f"Check that the bathymetry data contains points deeper "
                f"than {h1} m inside the lake domain.  "
                f"Depth range in Heta: [{Heta.min():.1f}, {Heta.max():.1f}] m."
            )

        # Pick the longest segment — that is the main interface boundary.
        # (Isolated tiny contour fragments can appear near the shoreline.)
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
        INeta2 = INeta2 & INeta1   # layer-2 cannot exist without layer-1

        INu1 = self._remove_boundary_u_nodes(INu1, INeta1, nu, mu)
        INv1 = self._remove_boundary_v_nodes(INv1, INeta1, nv, mv)
        INu2 = self._remove_boundary_u_nodes(INu2, INeta2, nu, mu)
        INv2 = self._remove_boundary_v_nodes(INv2, INeta2, nv, mv)

        # ---- Number nodes --------------------------------------------
        nodos_eta1 = self._number_nodes(INeta1)
        nodos_eta2 = self._number_nodes(INeta2)
        nodos_u1   = self._number_nodes(INu1)
        nodos_v1   = self._number_nodes(INv1)
        nodos_u2   = self._number_nodes(INu2)
        nodos_v2   = self._number_nodes(INv2)

        # ---- Connectivity matrices -----------------------------------
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
        """Return boolean mask — True where (x,y) lies inside the polygon.

        radius=1e-10 ensures that points exactly on the polygon boundary are
        included, matching MATLAB's inpolygon() behaviour where boundary
        points return 1 (inside).
        """
        points  = np.column_stack([x.ravel(), y.ravel()])
        polygon = Path(np.column_stack([xpoly, ypoly]))
        return polygon.contains_points(points, radius=1e-10).reshape(x.shape)

    def _remove_boundary_eta_nodes(self, INeta, INu, INv, n, m):
        """
        Deactivate eta nodes that do not have all four surrounding
        u / v nodes active (mirrors MATLAB's nested loop).
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
        """
        Deactivate u nodes that have no active eta neighbour on either
        side (i.e. they sit on the domain boundary).
        """
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
        """
        Deactivate v nodes that have no active eta neighbour above or
        below (i.e. they sit on the domain boundary).
        """
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

    # ---- Connectivity matrices (0-based indices stored, -1 = no node) ----

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
        CB[k] >  0  →  node k is interior; the value is its interior index.
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


import json
class MeshGeneratorMitgcmGrid:
    """
    Generate computational mesh for two-layer modal analysis.

    Uses staggered grid with:
    - eta nodes: water surface elevation (cell centres)
    - u   nodes: velocity in x-direction (cell faces ⊥ x)
    - v   nodes: velocity in y-direction (cell faces ⊥ y)
    """

    def __init__(self, mitgcm_grid_folder):
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

        with open(os.path.join(mitgcm_grid_folder, 'parameters.json'), "r") as f:
            grid_parameters = json.load(f)

        self.mitgcm_grid_folder = mitgcm_grid_folder
        self.resolution_in_km = grid_parameters['resolution'] / 1000.0
        self.rotation_in_degrees = grid_parameters['rotation']
    
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_grid(self, h1, shoreline_file=None, bathymetry_file=None):
        """
        Generate the complete staggered grid mesh.

        Parameters
        ----------
        h1 : float
            Thickness of the upper layer in metres.  The layer-2 domain
            boundary is the contour where total depth == h1.

        Returns
        -------
        dict
            All mesh arrays required by the matrix-assembly step.
        """

        x_mitgcm = np.load(os.path.join(self.mitgcm_grid_folder, "x.npy")) / 1000 # Grid is in km
        y_mitgcm = np.load(os.path.join(self.mitgcm_grid_folder, "y.npy")) / 1000 # Grid is in km

        # eta grid: cell centres
        xp = x_mitgcm + self.resolution_in_km / 2
        yp = y_mitgcm + self.resolution_in_km / 2

        # u / v grids: cell faces
        # Must have mu = meta+1 columns (xu) and nv = neta+1 rows (xv).
        # np.arange can miss the final face line due to floating-point rounding,
        # so we extend the stop by one step then trim to the exact required length.
        xi = np.append(x_mitgcm, x_mitgcm[-1] + self.resolution_in_km)
        yi = np.append(y_mitgcm, y_mitgcm[-1] + self.resolution_in_km)

        xeta, yeta = np.meshgrid(xp, yp)
        xu,   yu   = np.meshgrid(xi, yp)
        xv,   yv   = np.meshgrid(xp, yi)

        # ---- Load and interpolate bathymetry -------------------------
        with open(os.path.join(self.mitgcm_grid_folder, "bathy.bin"), 'rb') as fid:
            bathy_bin_data = -1 * np.fromfile(fid, dtype='>f8')

        Heta = np.reshape(bathy_bin_data, (len(y_mitgcm), len(x_mitgcm)))
        Heta[np.isnan(Heta)] = 0.0
        Heta[Heta<0] = 0.0

        # ---- Extract the h1-depth contour (layer-2 boundary) ---------
        #
        # This is the critical step that was previously broken: the old
        # code would silently replace xb20/yb20 with the shoreline whenever
        # contour extraction failed, making INeta2 == INeta1 and producing
        # a matrix that is far too large.
        #
        # We use a non-interactive Agg backend figure so there is no
        # display dependency, then call _extract_contour_paths() which
        # tries every known matplotlib API in order.
        import matplotlib.figure
        _fig = matplotlib.figure.Figure()
        _ax = _fig.add_subplot(111)
        cs2 = _ax.contour(xeta, yeta, Heta, levels=[h1])

        all_segments2 = _extract_contour_paths(cs2)

        if not all_segments2:
            raise ValueError(
                f"No contour found at depth h1={h1} m.  "
                f"Check that the bathymetry data contains points deeper "
                f"than {h1} m inside the lake domain.  "
                f"Depth range in Heta: [{Heta.min():.1f}, {Heta.max():.1f}] m."
            )

        # Pick the longest segment — that is the main interface boundary.
        # (Isolated tiny contour fragments can appear near the shoreline.)
        main_segment2 = max(all_segments2, key=len)
        xb20 = main_segment2[:, 0]
        yb20 = main_segment2[:, 1]

        _fig = matplotlib.figure.Figure()
        _ax = _fig.add_subplot(111)
        cs1 = _ax.contour(xeta, yeta, Heta, levels=[0.01])

        all_segments1 = _extract_contour_paths(cs1)

        if not all_segments1:
            raise ValueError(
                f"No contour found at depth = 0 m.  "
                f"Check that the bathymetry data contains points deeper "
                f"than 0 m inside the lake domain.  "
                f"Depth range in Heta: [{Heta.min():.1f}, {Heta.max():.1f}] m."
            )

        # Pick the longest segment — that is the main interface boundary.
        # (Isolated tiny contour fragments can appear near the shoreline.)
        main_segment1 = max(all_segments1, key=len)
        xb = main_segment1[:, 0]
        yb = main_segment1[:, 1]

        # ---- Identify nodes inside polygons --------------------------
        INeta1 = self._points_in_polygon(xeta, yeta, main_segment1)
        INu1   = self._points_in_polygon(xu,   yu,   main_segment1)
        INv1   = self._points_in_polygon(xv,   yv,   main_segment1)

        INeta2 = self._points_in_polygon(xeta, yeta, main_segment2)
        INu2   = self._points_in_polygon(xu,   yu,   main_segment2)
        INv2   = self._points_in_polygon(xv,   yv,   main_segment2)

        neta, meta = xeta.shape
        nu,   mu   = xu.shape
        nv,   mv   = xv.shape

        # ---- Prune nodes (mirrors MATLAB logic exactly) --------------
        INeta1 = self._remove_boundary_eta_nodes(INeta1, INu1, INv1, neta, meta)
        INeta1 = self._remove_isolated_nodes(INeta1)

        INeta2 = self._remove_boundary_eta_nodes(INeta2, INu2, INv2, neta, meta)
        INeta2 = self._remove_isolated_nodes(INeta2)
        INeta2 = INeta2 & INeta1   # layer-2 cannot exist without layer-1

        INu1 = self._remove_boundary_u_nodes(INu1, INeta1, nu, mu)
        INv1 = self._remove_boundary_v_nodes(INv1, INeta1, nv, mv)
        INu2 = self._remove_boundary_u_nodes(INu2, INeta2, nu, mu)
        INv2 = self._remove_boundary_v_nodes(INv2, INeta2, nv, mv)

        # ---- Number nodes --------------------------------------------
        nodos_eta1 = self._number_nodes(INeta1)
        nodos_eta2 = self._number_nodes(INeta2)
        nodos_u1   = self._number_nodes(INu1)
        nodos_v1   = self._number_nodes(INv1)
        nodos_u2   = self._number_nodes(INu2)
        nodos_v2   = self._number_nodes(INv2)

        # ---- Connectivity matrices -----------------------------------
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
            'rotation': self.rotation_in_degrees
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _points_in_polygon(self, x, y, poly):
        """Return boolean mask — True where (x,y) lies inside the polygon.

        radius=1e-10 ensures that points exactly on the polygon boundary are
        included, matching MATLAB's inpolygon() behaviour where boundary
        points return 1 (inside).
        """
        points  = np.column_stack([x.ravel(), y.ravel()])
        inside = np.zeros(len(points), dtype=bool)
        path = Path(poly, closed=True)
        inside |= path.contains_points(points)
        return inside.reshape(x.shape)
    

    def _remove_boundary_eta_nodes(self, INeta, INu, INv, n, m):
        """
        Deactivate eta nodes that do not have all four surrounding
        u / v nodes active (mirrors MATLAB's nested loop).
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
        """
        Deactivate u nodes that have no active eta neighbour on either
        side (i.e. they sit on the domain boundary).
        """
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
        """
        Deactivate v nodes that have no active eta neighbour above or
        below (i.e. they sit on the domain boundary).
        """
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

    # ---- Connectivity matrices (0-based indices stored, -1 = no node) ----

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
        CB[k] >  0  →  node k is interior; the value is its interior index.
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