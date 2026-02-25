from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

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

    # ── convenience properties ────────────────────────────────────────────
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
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.")

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

        self.nodal_vars   = {k: _f(v) for k, v in self.nodal_vars.items()}
        self.global_vars  = {k: _f(v) for k, v in self.global_vars.items()}
        self.element_vars = {
            vname: {bid: _f(arr) for bid, arr in blk_dict.items()}
            for vname, blk_dict in self.element_vars.items()
        }

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

    # ── public ────────────────────────────────────────────────────────────

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
            for blk in blocks:
                key = f"vals_elem_var{i + 1}eb{blk.block_id}"
                # also try sequential block index
                if key not in nc.variables:
                    # find positional index of this block
                    pos = next(
                        (j + 1 for j, b in enumerate(blocks) if b.block_id == blk.block_id),
                        None,
                    )
                    if pos is not None:
                        key = f"vals_elem_var{i + 1}eb{pos}"
                if key in nc.variables:
                    result[vname][blk.block_id] = np.asarray(
                        nc.variables[key][:], dtype=np.float64
                    )  # shape (T, num_elem_in_block)
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
