from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import torch
import torch.nn.functional as F

@dataclass
class ElementBlock:
    """One element block (homogeneous element topology)."""
    block_id: int
    name: str
    elem_type: str                  # e.g. "HEX8", "TET4", "QUAD4"
    connectivity: np.ndarray        # shape (num_elements, nodes_per_elem), 0-indexed
    attributes: Optional[np.ndarray] = None  # shape (num_elements, num_attr)


@dataclass
class NodeSet:
    """A named group of nodes (e.g. Dirichlet BC locations)."""
    set_id: int
    name: str
    nodes: np.ndarray               # shape (N,), 0-indexed
    dist_factors: Optional[np.ndarray] = None  # shape (N,)


@dataclass
class SideSet:
    """A named group of element faces/edges."""
    set_id: int
    name: str
    elements: np.ndarray            # shape (N,), 0-indexed element indices
    sides: np.ndarray               # shape (N,), 1-indexed local face number
    dist_factors: Optional[np.ndarray] = None  # shape (N,)


@dataclass
class ExodusMesh:
    """
    Complete parsed representation of an Exodus II file.

    Arrays are numpy by default; call `.to_torch()` to convert everything
    to torch tensors.
    """
    # ── geometry ──────────────────────────────────────────────────────────
    coords: np.ndarray              # shape (num_nodes, num_dim)
    coord_names: list[str]          # e.g. ["x", "y", "z"]

    # ── topology ──────────────────────────────────────────────────────────
    element_blocks: list[ElementBlock] = field(default_factory=list)
    node_sets:      list[NodeSet]      = field(default_factory=list)
    side_sets:      list[SideSet]      = field(default_factory=list)

    # ── time / fields ─────────────────────────────────────────────────────
    times: Optional[np.ndarray] = None          # shape (T,)

    # Dict key = variable name, value shape = (T, num_nodes)
    nodal_vars: dict[str, np.ndarray] = field(default_factory=dict)

    # Dict key = variable name, value = dict{block_id: array shape (T, num_elem)}
    element_vars: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)

    # Dict key = variable name, value shape = (T,)
    global_vars: dict[str, np.ndarray] = field(default_factory=dict)

    # ── metadata ──────────────────────────────────────────────────────────
    title: str = ""
    qa_records: list[tuple] = field(default_factory=list)
    info_records: list[str] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def num_dim(self) -> int:
        return self.coords.shape[1]

    @property
    def num_time_steps(self) -> int:
        return 0 if self.times is None else len(self.times)

    def all_connectivity(self) -> np.ndarray:
        """
        Concatenate connectivity from all blocks into one array.
        Useful when all blocks share the same element type.
        Returns shape (total_elements, nodes_per_elem).
        """
        return np.concatenate([b.connectivity for b in self.element_blocks], axis=0)

    def to_torch(self, dtype_float=None, dtype_int=None, device=None):
        """
        Convert all numpy arrays to torch tensors in-place.

        Args:
            dtype_float: torch float dtype (default: torch.float32)
            dtype_int:   torch int   dtype (default: torch.int64)
            device:      torch device string or object (default: "cpu")
        """
        ft = dtype_float or torch.float32
        it = dtype_int   or torch.int64
        dev = device or "cpu"

        def _f(a):
            return torch.as_tensor(np.asarray(a), dtype=ft, device=dev)

        def _i(a):
            return torch.as_tensor(np.asarray(a), dtype=it, device=dev)

        self.coords = _f(self.coords)

        for blk in self.element_blocks:
            blk.connectivity = _i(blk.connectivity)
            if blk.attributes is not None:
                blk.attributes = _f(blk.attributes)

        for ns in self.node_sets:
            ns.nodes = _i(ns.nodes)
            if ns.dist_factors is not None:
                ns.dist_factors = _f(ns.dist_factors)

        for ss in self.side_sets:
            ss.elements = _i(ss.elements)
            ss.sides    = _i(ss.sides)
            if ss.dist_factors is not None:
                ss.dist_factors = _f(ss.dist_factors)

        if self.times is not None:
            self.times = _f(self.times)

        # Convert in-place one entry at a time to avoid briefly holding
        # both the old numpy arrays and new torch tensors simultaneously.
        for k in list(self.nodal_vars):
            self.nodal_vars[k] = _f(self.nodal_vars[k])
        for k in list(self.global_vars):
            self.global_vars[k] = _f(self.global_vars[k])
        for vname in list(self.element_vars):
            for bid in list(self.element_vars[vname]):
                self.element_vars[vname][bid] = _f(self.element_vars[vname][bid])

        return self  # allow chaining

    def __repr__(self) -> str:
        lines = [
            f"ExodusMesh('{self.title}')",
            f"  nodes      : {self.num_nodes}",
            f"  dims       : {self.num_dim}",
            f"  el. blocks : {len(self.element_blocks)}"
            + (f"  [{', '.join(b.elem_type for b in self.element_blocks)}]"
               if self.element_blocks else ""),
            f"  node sets  : {len(self.node_sets)}",
            f"  side sets  : {len(self.side_sets)}",
            f"  time steps : {self.num_time_steps}",
            f"  nodal vars : {list(self.nodal_vars.keys())}",
            f"  elem  vars : {list(self.element_vars.keys())}",
            f"  global vars: {list(self.global_vars.keys())}",
        ]
        return "\n".join(lines)

    def _infer_regular_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Infer the regular nx × ny × nz node grid from the coordinate arrays.

        Returns unique sorted coordinate values along each axis. Result is
        cached since self.coords is fixed after parsing.
        """
        if hasattr(self, '_regular_grid_cache'):
            return self._regular_grid_cache

        tol_fraction = 1e-6
        result = []
        for axis in range(3):
            vals = self.coords[:, axis]
            span = vals.max() - vals.min()
            tol  = tol_fraction * span if span > 0 else 1e-12
            sorted_vals = np.sort(np.unique(np.round(vals / tol).astype(int))) * tol
            result.append(sorted_vals)

        xs, ys, zs = result
        expected_nodes = len(xs) * len(ys) * len(zs)
        if expected_nodes != self.num_nodes:
            raise ValueError(
                f"Mesh does not appear to be a regular rectangular grid: "
                f"inferred grid {len(xs)}×{len(ys)}×{len(zs)} = {expected_nodes} nodes "
                f"but found {self.num_nodes} nodes."
            )
        self._regular_grid_cache = (xs, ys, zs)
        return xs, ys, zs

    def _build_node_index(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the (ix, iy, iz) grid indices for every node.
        Cached on self._node_index_cache keyed by grid shape.
        """
        key = (len(xs), len(ys), len(zs))
        if hasattr(self, '_node_index_cache') and self._node_index_cache[0] == key:
            return self._node_index_cache[1]

        nx, ny, nz = key
        ix = np.clip(np.searchsorted(xs, self.coords[:, 0]), 0, nx - 1)
        iy = np.clip(np.searchsorted(ys, self.coords[:, 1]), 0, ny - 1)
        iz = np.clip(np.searchsorted(zs, self.coords[:, 2]), 0, nz - 1)

        self._node_index_cache = (key, (ix, iy, iz))
        return ix, iy, iz

    def _nodal_field_to_volume(
        self,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        ts: slice = slice(None),
    ) -> np.ndarray:
        """
        Reshape a flat nodal field [T, N] → [T', nx, ny, nz].

        ``ts`` is applied before the reshape so only the selected timesteps
        are ever materialized in memory.
        """
        nx, ny, nz = len(xs), len(ys), len(zs)
        ix, iy, iz = self._build_node_index(xs, ys, zs)

        sliced = field[ts]          # [T', N] — only the needed timesteps
        T_out  = sliced.shape[0]
        volume = np.empty((T_out, nx, ny, nz), dtype=sliced.dtype)
        volume[:, ix, iy, iz] = sliced
        return volume

    def _element_field_to_volume(
        self,
        field_by_block: dict[int, np.ndarray],
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        ts: slice = slice(None),
    ) -> np.ndarray:
        """
        Reshape element-level fields into a volume [T', nx-1, ny-1, nz-1].
    
        ``ts`` is applied before the reshape so only the selected timesteps
        are materialised in memory. Assumes HEX8: centroid = mean of 8 nodes.
        """
        nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
        T_out = next(iter(field_by_block.values()))[ts].shape[0]
    
        volume = np.zeros((T_out, nx, ny, nz), dtype=next(iter(field_by_block.values())).dtype)
    
        if not hasattr(self, '_elem_index_cache'):
            self._elem_index_cache = {}
    
        for block in self.element_blocks:
            if block.block_id not in field_by_block:
                continue
            sliced = field_by_block[block.block_id][ts]  # [T', E]
            conn   = block.connectivity                   # [E, nodes_per_elem]
    
            cache_key = f'_elem_idx_{block.block_id}_{len(xs)}'
            if cache_key not in self._elem_index_cache:
                node_coords = self.coords[conn]
                centroids   = node_coords.mean(axis=1)
                ex = np.clip(np.searchsorted(xs[:-1], centroids[:, 0], side='right') - 1, 0, nx-1)
                ey = np.clip(np.searchsorted(ys[:-1], centroids[:, 1], side='right') - 1, 0, ny-1)
                ez = np.clip(np.searchsorted(zs[:-1], centroids[:, 2], side='right') - 1, 0, nz-1)
                self._elem_index_cache[cache_key] = (ex, ey, ez)
            ex, ey, ez = self._elem_index_cache[cache_key]
    
            volume[:, ex, ey, ez] = sliced
    
        return volume

    def resample_to_crystal_grid(
        self,
        crystal,
        supercell_size: tuple[int, int, int],
        time_steps: Optional[slice] = None,
        disp_names:  tuple[str, str, str] = ('disp_x', 'disp_y', 'disp_z'),
        polar_names: tuple[str, str, str] = ('polar_x', 'polar_y', 'polar_z'),
        strain_names: tuple[str, ...] = ('e00', 'e01', 'e02', 'e11', 'e12', 'e22'),
        coord_scale: float = 1e-9,
        anchor: Optional[tuple[float, float, float]] = None,
        device: str = 'cpu',
        dtype: 'torch.dtype' = None,
    ) -> 'CrystalGrid':
        """
        Resample exodus nodal/element fields onto the diffraction supercell grid
        defined by a Crystal object and supercell_size.

        Assumes the CIF lattice vectors are aligned with the exodus simulation box
        axes, so no rotation is needed before resampling.

        Parameters
        ----------
        crystal : Crystal
            The Crystal instance defining lattice_vectors and crystal_size.
        supercell_size : tuple of int
            (d1, d2, d3) unit cells per supercell, as passed to the diffraction methods.
        time_steps : slice, optional
            Subset of time steps to resample, e.g. slice(0, 100) or slice(-1, None).
            Defaults to all time steps.
        disp_names : tuple of str
            Names of the x, y, z continuum displacement nodal variables.
            Falls back to ('u_x', 'u_y', 'u_z') for any missing name.
        polar_names : tuple of str
            Names of the x, y, z polarization nodal variables.
        strain_names : tuple of str
            Names of the 6 independent strain components (Voigt order:
            e00, e01, e02, e11, e12, e22) in element_vars.
        coord_scale : float
            Multiplicative factor converting exodus coordinate units to meters.
            Default 1e-9 (nanometers → meters), the most common MOOSE convention.
            Use 1e-10 for Angstroms, or 1.0 if already in meters.
            Applied to bounding-box coordinates and to continuum_displacement values.
            Strain (dimensionless) and polarization (C/m²) are not rescaled.
        anchor : tuple of float, optional
            Position in exodus coordinates (same units as coord_scale, i.e. nanometers
            by default) of the crystal grid's origin — the point in the simulation box
            that corresponds to fractional position (0, 0, 0) of the crystal.
            If None, the crystal grid is centered within the simulation box.
            Only used when the crystal grid is smaller than the simulation box.
            Ignored (with a printed notice) when the crystal is cropped to fit the box.
        device : str
            Torch device for output tensors.
        dtype : torch.dtype, optional
            Output float dtype. Defaults to crystal.dtype.

        Returns
        -------
        CrystalGrid
            Dataclass with resampled tensors ready for the diffraction methods.

        Notes
        -----
        Internal coordinate normalisation uses float64 regardless of ``dtype``
        to avoid precision loss when mapping supercell positions to [-1, 1].
        Field values are cast to ``dtype`` only after interpolation.
        grid_sample requires float32 input, so field tensors are temporarily
        cast to float32 for interpolation then cast to the requested dtype.
        """
        dtype = dtype or crystal.dtype

        # float64 is used for all coordinate arithmetic to avoid precision loss
        # when normalising to [-1, 1]. grid_sample itself requires float32, so
        # field tensors are cast to float32 just for interpolation.
        _f64 = torch.float64

        # ── 0. time slice ─────────────────────────────────────────────────
        ts = time_steps if time_steps is not None else slice(None)

        # ── 1. infer regular grid ─────────────────────────────────────────
        xs, ys, zs = self._infer_regular_grid()
        nx, ny, nz = len(xs), len(ys), len(zs)

        # ── 2. resolve supercell grid dimensions ──────────────────────────
        d1, d2, d3 = supercell_size
        n1, n2, n3 = crystal.crystal_size

        # ── 3. compute lattice and bounding box ───────────────────────────
        L = crystal.original_lattice_vectors.to(dtype=_f64, device='cpu')

        box_min  = torch.tensor([xs.min(), ys.min(), zs.min()], dtype=_f64) * coord_scale
        box_max  = torch.tensor([xs.max(), ys.max(), zs.max()], dtype=_f64) * coord_scale
        box_span = box_max - box_min

        # ── 4. crop if crystal exceeds box, then resolve anchor ───────────
        lat_params   = torch.linalg.norm(L, dim=1)                      # [3]
        max_cells    = (box_span / lat_params).floor().to(torch.int64)   # [3]
        n1_orig, n2_orig, n3_orig = n1, n2, n3

        if torch.any(torch.tensor([n1, n2, n3], dtype=torch.int64) > max_cells):
            n1_new = int((min(n1, int(max_cells[0])) // d1) * d1)
            n2_new = int((min(n2, int(max_cells[1])) // d2) * d2)
            n3_new = int((min(n3, int(max_cells[2])) // d3) * d3)
            print(
                f"Crystal size ({n1}, {n2}, {n3}) exceeds simulation box along one or more "
                f"axes. Cropping to ({n1_new}, {n2_new}, {n3_new}) unit cells to fit."
            )
            n1, n2, n3 = n1_new, n2_new, n3_new
            crystal.crystal_size = (n1, n2, n3)
        elif torch.all(torch.tensor([n1, n2, n3], dtype=torch.int64) < max_cells):
            print(
                f"Crystal size ({n1}, {n2}, {n3}) is smaller than the simulation box. "
                f"Aligning top corner of crystal with top corner of simulation box. "
                f"To override, pass an explicit anchor in exodus coordinates "
                f"({coord_scale:.0e} m units)."
            )

        # crystal_span always computed after n1/n2/n3 are finalised
        crystal_extent = torch.tensor([n1, n2, n3], dtype=_f64).unsqueeze(0) * L
        crystal_span   = crystal_extent.sum(dim=0)  # [3] Cartesian extent

        if anchor is None:
            anchor_m = box_max - crystal_span
        else:
            anchor_m = torch.tensor(anchor, dtype=_f64) * coord_scale
            # On axes that were cropped, override the user anchor with top-corner alignment
            cropped_axes = torch.tensor(
                [n1 < n1_orig, n2 < n2_orig, n3 < n3_orig]
            )
            if torch.any(cropped_axes):
                top_anchor = box_max - crystal_span
                for ax in range(3):
                    if cropped_axes[ax]:
                        anchor_m[ax] = top_anchor[ax]
                        print(f"  (anchor overridden on axis {ax} — crystal was cropped on that axis)")
            anchor_max = anchor_m + crystal_span
            if torch.any(anchor_m < box_min) or torch.any(anchor_max > box_max):
                print(
                    f"Warning: anchor places the crystal grid partially outside the "
                    f"simulation box. Out-of-bounds supercells will sample the nearest "
                    f"in-box value (padding_mode='border')."
                )

        n_sc1, n_sc2, n_sc3 = n1 // d1, n2 // d2, n3 // d3

        i_frac = (torch.arange(n_sc1, dtype=_f64) + 0.5) * d1
        j_frac = (torch.arange(n_sc2, dtype=_f64) + 0.5) * d2
        k_frac = (torch.arange(n_sc3, dtype=_f64) + 0.5) * d3

        ii, jj, kk = torch.meshgrid(i_frac, j_frac, k_frac, indexing='ij')
        sc_indices  = torch.stack([ii, jj, kk], dim=-1)  # [n_sc1, n_sc2, n_sc3, 3]

        # Cartesian positions relative to crystal origin, then shifted by anchor
        sc_positions = torch.matmul(sc_indices, L) + anchor_m  # [n_sc1, n_sc2, n_sc3, 3]

        # grid_sample expects coords in (z, y, x) order normalised to [-1, 1]
        sc_norm = 2.0 * (sc_positions - box_min) / box_span - 1.0
        sc_grid = sc_norm[..., [2, 1, 0]]
        grid_5d = sc_grid.unsqueeze(0)

        # Pre-cast the sampling grid to float32 once (shared across all fields)
        grid_5d_f32 = grid_5d.to(torch.float32)

        def _interp_volume(vol_np: np.ndarray) -> 'torch.Tensor':
            """
            Trilinear interpolation: [T', nx, ny, nz] -> [T', n_sc1, n_sc2, n_sc3].

            vol_np must already be sliced to the desired timesteps.
            Creates the field tensor directly as float32 (required by grid_sample)
            to avoid holding a second copy at a different dtype during interpolation.
            The output is cast to ``dtype`` only after interpolation, when the
            spatial dimensions have been reduced from (nx, ny, nz) to
            (n_sc1, n_sc2, n_sc3).
            """
            T_out = vol_np.shape[0]
            # Create directly as float32 — avoids a double-copy when dtype != float32
            inp  = torch.as_tensor(vol_np, dtype=torch.float32).unsqueeze(1)
            grid = grid_5d_f32.expand(T_out, -1, -1, -1, -1)
            out  = F.grid_sample(
                inp, grid,
                mode='bilinear', padding_mode='border', align_corners=True,
            )
            return out.squeeze(1).to(dtype=dtype, device=device)

        # ── 5. resample continuum displacement ────────────────────────────
        continuum_displacement = None
        disp_components = []
        for name, fallback in zip(disp_names, ('u_x', 'u_y', 'u_z')):
            key = name if name in self.nodal_vars else fallback
            if key in self.nodal_vars:
                vol = self._nodal_field_to_volume(self.nodal_vars[key], xs, ys, zs, ts)
                disp_components.append(_interp_volume(vol) * coord_scale)

        if len(disp_components) == 3:
            continuum_displacement = torch.stack(disp_components, dim=-1)

        # ── 6. resample polarization ──────────────────────────────────────
        # Polarization (C/m²) is not affected by coord_scale
        polarization     = None
        polar_components = []
        for name in polar_names:
            if name in self.nodal_vars:
                vol = self._nodal_field_to_volume(self.nodal_vars[name], xs, ys, zs, ts)
                polar_components.append(_interp_volume(vol))

        if len(polar_components) == 3:
            polarization = torch.stack(polar_components, dim=-1)

        # ── 7. resample strain tensor → lattice perturbation ─────────────
        # Strain is dimensionless — coord_scale does not apply to the values
        lattice_strain = None
        voigt_to_ij = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
        strain_vols: dict = {}

        for name, (i, j) in zip(strain_names, voigt_to_ij):
            if name in self.element_vars:
                vol = self._element_field_to_volume(self.element_vars[name], xs, ys, zs, ts)
                # Element variables live on an (nx-1)×(ny-1)×(nz-1) centroid grid;
                # build a separate normalized coordinate system for that grid.
                xs_e = 0.5 * (xs[:-1] + xs[1:])
                ys_e = 0.5 * (ys[:-1] + ys[1:])
                zs_e = 0.5 * (zs[:-1] + zs[1:])
                box_min_e  = torch.tensor(
                    [xs_e.min(), ys_e.min(), zs_e.min()], dtype=_f64) * coord_scale
                box_span_e = torch.tensor(
                    [xs_e.max()-xs_e.min(), ys_e.max()-ys_e.min(), zs_e.max()-zs_e.min()],
                    dtype=_f64) * coord_scale
                sc_norm_e  = 2.0 * (sc_positions - box_min_e) / box_span_e - 1.0
                sc_grid_e  = sc_norm_e[..., [2, 1, 0]].unsqueeze(0).to(torch.float32)

                T_out = vol.shape[0]
                inp   = torch.as_tensor(vol, dtype=torch.float32).unsqueeze(1)
                grid_e = sc_grid_e.expand(T_out, -1, -1, -1, -1)
                out    = F.grid_sample(
                    inp, grid_e,
                    mode='bilinear', padding_mode='border', align_corners=True,
                ).squeeze(1).to(dtype=dtype, device=device)
                strain_vols[(i, j)] = out

            elif name in self.nodal_vars:
                vol = self._nodal_field_to_volume(self.nodal_vars[name], xs, ys, zs, ts)
                strain_vols[(i, j)] = _interp_volume(vol)

        if strain_vols:
            T_out = next(iter(strain_vols.values())).shape[0]
            eps   = torch.zeros(T_out, n_sc1, n_sc2, n_sc3, 3, 3, dtype=dtype, device=device)
            for (i, j), val in strain_vols.items():
                eps[..., i, j] = val
                eps[..., j, i] = val  # symmetrize off-diagonal

            # δL = ε @ L_0  (strain dimensionless, L_0 in meters)
            L0 = crystal.original_lattice_vectors.to(dtype=dtype, device=device)
            lattice_strain = torch.matmul(
                eps, L0.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )

        # ── 8. rotate fields from original structure frame to current lab frame ──
        # continuum_displacement and lattice_strain are expressed in the exodus
        # simulation frame, which is aligned with the original (unrotated) crystal.
        # crystal.rotation_matrix R satisfies L_curr = L_orig @ R.T, so vectors
        # and lattice matrices transform by the same right-multiplication by R.T.
        R = crystal.rotation_matrix.to(dtype=dtype, device=device)  # [3, 3]

        if continuum_displacement is not None:
            # u has shape [T, n_sc1, n_sc2, n_sc3, 3] — rotate each 3-vector
            continuum_displacement = continuum_displacement @ R.T

        if lattice_strain is not None:
            # lattice_strain has shape [T, n_sc1, n_sc2, n_sc3, 3, 3]
            # each δL matrix transforms as δL_rotated = δL @ R.T
            lattice_strain = lattice_strain @ R.T
            
        return CrystalGrid(
            continuum_displacement=continuum_displacement,
            polarization=polarization,
            lattice_strain=lattice_strain,
            supercell_size=supercell_size,
            crystal_size=crystal.crystal_size,
            rotation_applied=R,
            times=torch.tensor(
                self.times[ts] if self.times is not None else [],
                dtype=dtype, device=device,
            ),
        )


@dataclass
class CrystalGrid:
    """
    Exodus fields resampled onto the diffraction supercell grid.

    All tensors follow the convention [T, n_sc1, n_sc2, n_sc3, ...] where T is
    the number of selected time steps.

    Attributes
    ----------
    continuum_displacement : Tensor | None
        Shape [T, n_sc1, n_sc2, n_sc3, 3].
        Rigid per-supercell shift in the lab frame (Cartesian meters).
        From disp_x/y/z. Pass as ``continuum_displacement`` to diffraction methods.

    polarization : Tensor | None
        Shape [T, n_sc1, n_sc2, n_sc3, 3].
        Polarization per supercell (C/m²). Convert via Born effective charges
        to get ``sublattice_displacements`` for diffraction methods.

    lattice_strain : Tensor | None
        Shape [T, n_sc1, n_sc2, n_sc3, 3, 3].
        Additive perturbation to L_0: L_local = L_0 + lattice_strain.
        Derived from strain tensor via delta_L = eps @ L_0.

    supercell_size   : tuple of int
    crystal_size     : tuple of int
    rotation_applied : Tensor — R that was applied during resampling
    times            : Tensor — time values for selected steps.
    """
    continuum_displacement: Optional['torch.Tensor']
    polarization:           Optional['torch.Tensor']
    lattice_strain:         Optional['torch.Tensor']
    supercell_size:         tuple
    crystal_size:           tuple
    rotation_applied:       'torch.Tensor'
    times:                  'torch.Tensor'

    @property
    def n_time_steps(self) -> int:
        return len(self.times)

    @property
    def grid_shape(self) -> tuple:
        n1, n2, n3 = self.crystal_size
        d1, d2, d3 = self.supercell_size
        return (n1 // d1, n2 // d2, n3 // d3)

    def reorient(self, crystal):
        """
        Re-apply rotation for a newly rotated crystal without reloading the mesh.
        Undoes the previously applied rotation and applies the current one.
        """
        R_old = self.rotation_applied                             # [3, 3]
        R_new = crystal.rotation_matrix.to(
            dtype=R_old.dtype, device=R_old.device)               # [3, 3]
        # Combined transform: undo old R, apply new R
        # For a vector: v_new = v_orig @ R_new.T = v_rotated_old @ R_old @ R_new.T
        R_update = R_old @ R_new.T                                # [3, 3]
    
        if self.continuum_displacement is not None:
            self.continuum_displacement = self.continuum_displacement @ R_update
    
        if self.lattice_strain is not None:
            self.lattice_strain = self.lattice_strain @ R_update
    
        self.rotation_applied = R_new
    
    def __repr__(self) -> str:
        n_sc1, n_sc2, n_sc3 = self.grid_shape
        lines = [
            "CrystalGrid(",
            f"  supercell grid         : {n_sc1} x {n_sc2} x {n_sc3}",
            f"  time steps             : {self.n_time_steps}",
            f"  continuum_displacement : {'yes' if self.continuum_displacement is not None else 'missing'}",
            f"  polarization           : {'yes' if self.polarization is not None else 'missing'}",
            f"  lattice_strain         : {'yes' if self.lattice_strain is not None else 'missing'}",
            ")",
        ]
        return "\n".join(lines)


class ExodusParser:
    """
    Parse an Exodus II (.e / .exo) file.

    Example
    -------
    >>> parser = ExodusParser("results.e")
    >>> mesh = parser.parse()
    >>> print(mesh)
    >>> mesh.to_torch(device="cuda")
    >>> coords = mesh.coords          # torch.Tensor on GPU
    >>> conn   = mesh.element_blocks[0].connectivity
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def parse(self) -> ExodusMesh:
        try:
            from netCDF4 import Dataset
        except ImportError:
            raise ImportError(
                "netCDF4 is required: pip install netCDF4"
            )

        with Dataset(self.filepath, "r") as nc:
            mesh = ExodusMesh(
                coords=self._coords(nc),
                coord_names=self._coord_names(nc),
                title=getattr(nc, "title", "").strip(),
                qa_records=self._qa_records(nc),
                info_records=self._info_records(nc),
            )
            mesh.element_blocks = self._element_blocks(nc)
            mesh.node_sets      = self._node_sets(nc)
            mesh.side_sets      = self._side_sets(nc)
            mesh.times          = self._times(nc)
            mesh.global_vars    = self._global_vars(nc)
            mesh.nodal_vars     = self._nodal_vars(nc)
            mesh.element_vars   = self._element_vars(nc, mesh.element_blocks)

        return mesh

    @staticmethod
    def _str(char_array) -> str:
        """Convert a NetCDF char array to a Python string."""
        raw = char_array[:]
        if hasattr(raw, "tobytes"):
            return raw.tobytes().decode("utf-8", errors="replace").rstrip("\x00").strip()
        return "".join(c.decode("utf-8", errors="replace") for c in raw).rstrip("\x00").strip()

    @staticmethod
    def _names(nc, var_name: str, count: int) -> list[str]:
        """Read a 2-D char variable of names, fall back to empty strings."""
        if var_name in nc.variables:
            raw = nc.variables[var_name][:]
            # If it's a masked array, fill masked values with empty bytes so
            # they don't raise errors when we try to encode them.
            if hasattr(raw, "filled"):
                raw = raw.filled(b"")
            result = []
            for i in range(min(count, raw.shape[0])):
                row = raw[i]
                chars = []
                for c in row:
                    if isinstance(c, (bytes, bytearray)):
                        chars.append(c)
                    elif isinstance(c, str):
                        chars.append(c.encode("utf-8"))
                    elif isinstance(c, np.bytes_):
                        chars.append(bytes(c))
                    else:
                        # masked fill value or anything else — stop here
                        break
                s = b"".join(chars).decode("utf-8", errors="replace")
                result.append(s.rstrip("\x00").strip())
            # pad if needed
            while len(result) < count:
                result.append("")
            return result
        return [""] * count

    def _coords(self, nc) -> np.ndarray:
        n_dim = nc.dimensions["num_dim"].size
        n_nodes = nc.dimensions["num_nodes"].size

        # coords may be stored as one 2-D array or separate coordx/y/z
        if "coord" in nc.variables:
            c = nc.variables["coord"][:]            # (num_dim, num_nodes)
            return np.asarray(c, dtype=np.float64).T  # → (num_nodes, num_dim)

        axes = ["coordx", "coordy", "coordz"][:n_dim]
        cols = [np.asarray(nc.variables[ax][:], dtype=np.float64) for ax in axes if ax in nc.variables]
        if not cols:
            return np.zeros((n_nodes, n_dim), dtype=np.float64)
        return np.stack(cols, axis=1)               # (num_nodes, num_dim)

    def _coord_names(self, nc) -> list[str]:
        n_dim = nc.dimensions["num_dim"].size
        defaults = ["x", "y", "z"][:n_dim]
        if "coor_names" in nc.variables:
            names = self._names(nc, "coor_names", n_dim)
            return [n or defaults[i] for i, n in enumerate(names)]
        return defaults

    def _element_blocks(self, nc) -> list[ElementBlock]:
        if "num_el_blk" not in nc.dimensions:
            return []
        n_blk = nc.dimensions["num_el_blk"].size
        ids   = list(nc.variables.get("eb_prop1", range(1, n_blk + 1))[:])
        names = self._names(nc, "eb_names", n_blk)
        blocks = []
        for i in range(n_blk):
            idx = i + 1
            conn_key = f"connect{idx}"
            if conn_key not in nc.variables:
                continue
            conn_var  = nc.variables[conn_key]
            elem_type = getattr(conn_var, "elem_type", "UNKNOWN").strip()
            conn      = np.asarray(conn_var[:], dtype=np.int64) - 1  # → 0-indexed

            attr_key  = f"attrib{idx}"
            attr = np.asarray(nc.variables[attr_key][:], dtype=np.float64) \
                   if attr_key in nc.variables else None

            blocks.append(ElementBlock(
                block_id=int(ids[i]),
                name=names[i] or f"block_{ids[i]}",
                elem_type=elem_type,
                connectivity=conn,
                attributes=attr,
            ))
        return blocks

    def _node_sets(self, nc) -> list[NodeSet]:
        if "num_node_sets" not in nc.dimensions:
            return []
        n = nc.dimensions["num_node_sets"].size
        ids   = list(nc.variables.get("ns_prop1", range(1, n + 1))[:])
        names = self._names(nc, "ns_names", n)
        sets  = []
        for i in range(n):
            idx      = i + 1
            node_key = f"node_ns{idx}"
            if node_key not in nc.variables:
                continue
            nodes = np.asarray(nc.variables[node_key][:], dtype=np.int64) - 1

            df_key = f"dist_fact_ns{idx}"
            df = np.asarray(nc.variables[df_key][:], dtype=np.float64) \
                 if df_key in nc.variables else None

            sets.append(NodeSet(
                set_id=int(ids[i]),
                name=names[i] or f"nodeset_{ids[i]}",
                nodes=nodes,
                dist_factors=df,
            ))
        return sets

    def _side_sets(self, nc) -> list[SideSet]:
        if "num_side_sets" not in nc.dimensions:
            return []
        n = nc.dimensions["num_side_sets"].size
        ids   = list(nc.variables.get("ss_prop1", range(1, n + 1))[:])
        names = self._names(nc, "ss_names", n)
        sets  = []
        for i in range(n):
            idx     = i + 1
            el_key  = f"elem_ss{idx}"
            si_key  = f"side_ss{idx}"
            if el_key not in nc.variables or si_key not in nc.variables:
                continue
            elements = np.asarray(nc.variables[el_key][:], dtype=np.int64) - 1
            sides    = np.asarray(nc.variables[si_key][:], dtype=np.int64)

            df_key = f"dist_fact_ss{idx}"
            df = np.asarray(nc.variables[df_key][:], dtype=np.float64) \
                 if df_key in nc.variables else None

            sets.append(SideSet(
                set_id=int(ids[i]),
                name=names[i] or f"sideset_{ids[i]}",
                elements=elements,
                sides=sides,
                dist_factors=df,
            ))
        return sets

    def _times(self, nc) -> Optional[np.ndarray]:
        if "time_whole" in nc.variables:
            t = np.asarray(nc.variables["time_whole"][:], dtype=np.float64)
            return t if t.ndim > 0 and len(t) > 0 else None
        return None

    def _global_vars(self, nc) -> dict[str, np.ndarray]:
        if "vals_glo_var" not in nc.variables:
            return {}
        raw   = np.asarray(nc.variables["vals_glo_var"][:], dtype=np.float64)
        # raw shape: (T, num_glo_var)
        n_var = raw.shape[1] if raw.ndim == 2 else 1
        names = self._names(nc, "name_glo_var", n_var)
        if raw.ndim == 1:
            raw = raw[:, None]
        return {names[i] or f"glo_{i}": raw[:, i] for i in range(n_var)}

    def _nodal_vars(self, nc) -> dict[str, np.ndarray]:
        if "name_nod_var" not in nc.variables:
            return {}
        n_var = nc.dimensions.get("num_nod_var", None)
        if n_var is None:
            return {}
        n_var = n_var.size
        names = self._names(nc, "name_nod_var", n_var)
        result = {}
        for i in range(n_var):
            key = f"vals_nod_var{i + 1}"
            if key in nc.variables:
                arr = np.asarray(nc.variables[key][:], dtype=np.float64)
                result[names[i] or f"nod_{i}"] = arr  # shape (T, num_nodes)
        return result

    def _element_vars(
        self, nc, blocks: list[ElementBlock]
    ) -> dict[str, dict[int, np.ndarray]]:
        if "name_elem_var" not in nc.variables:
            return {}
        n_var = nc.dimensions.get("num_elem_var", None)
        if n_var is None:
            return {}
        n_var = n_var.size
        names = self._names(nc, "name_elem_var", n_var)
        result: dict[str, dict[int, np.ndarray]] = {
            names[i] or f"elem_{i}": {} for i in range(n_var)
        }
        var_names = list(result.keys())
        for i, vname in enumerate(var_names):
            for j, blk in enumerate(blocks):
                # eb index is always positional (j+1), never block_id
                key = f"vals_elem_var{i + 1}eb{j + 1}"
                if key in nc.variables:
                    result[vname][blk.block_id] = np.asarray(
                        nc.variables[key][:], dtype=np.float64
                    )
        return result

    def _qa_records(self, nc) -> list[tuple]:
        if "qa_records" not in nc.variables:
            return []
        raw = nc.variables["qa_records"][:]
        records = []
        for i in range(raw.shape[0]):
            rec = tuple(
                b"".join(c if isinstance(c, bytes) else c.encode()
                         for c in raw[i, j]).decode("utf-8", errors="replace")
                         .rstrip("\x00").strip()
                for j in range(raw.shape[1])
            )
            records.append(rec)
        return records

    def _info_records(self, nc) -> list[str]:
        if "info_records" not in nc.variables:
            return []
        raw = nc.variables["info_records"][:]
        lines = []
        for i in range(raw.shape[0]):
            try:
                s = b"".join(c if isinstance(c, bytes) else c.encode()
                             for c in raw[i]).decode("utf-8", errors="replace")
                lines.append(s.rstrip("\x00").strip())
            except Exception:
                lines.append("")
        return lines


def load_exodus(filepath: str | Path, to_torch: bool = False, **torch_kwargs) -> ExodusMesh:
    """
    Load an Exodus II file and return an ExodusMesh.

    Args:
        filepath:    Path to the .e / .exo file.
        to_torch:    If True, convert all arrays to torch tensors.
        **torch_kwargs: Passed to ExodusMesh.to_torch()
                        (e.g. dtype_float=torch.float64, device="cuda")
    """
    mesh = ExodusParser(filepath).parse()
    if to_torch:
        mesh.to_torch(**torch_kwargs)
    return mesh