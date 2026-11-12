"""Aji-4 dam dataset loader for KAN-free HydroGraphNet.

Derived from the White River (Zenodo 15491288) loader in this repository; only the
dataset-specific parts changed (see ``docs/CHANGES.md`` for the full diff):

  1. **Feature set.** No infiltration (``M80_IP`` is all zeros in Aji-4) and no
     precipitation (``M80_Pr`` is all zeros and no rain falls on the mesh). Node input
     is ``9 + 2 + 2*K`` = **15 at K=2** (2 XY + 7 statics + 2 forcings + 2*K history).
  2. **Two forcings, not one.** ``release`` (``M80_US_InF``: dam -> flood plain) and
     ``inflow`` (``reservoir_inflow_hourly.csv``: catchment -> reservoir). In White River
     the single ``US_InF`` channel played the role of both.
  3. **Window.** White River kept ``[0 : argmax(Q)+25]``. Aji-4's global release maximum
     is the start-up burst (row 1) in 42 % of floods, so that rule keeps the wrong part
     of the series. Here the window starts at the flood's own burst onset
     (``skip="auto"``) or a fixed row, and runs to the end of the record.
  4. **Memory.** 272 floods x 2 fields x 181 x 15,990 does not fit in RAM. Arrays are
     cached as ``float32`` and memory-mapped; the ASCII ``.npz`` cache of the original is
     dropped because it would need 6.3 GB of decompression per epoch.
  5. **Splits.** Train and test only (no validation), and ``train`` here can be pointed at
     any id list, so train+val can be pooled into one training set.
"""

from __future__ import annotations

import gc
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import KDTree
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .graph import GraphData

STATIC_NORM_STATS_FILE = "static_norm_stats.json"
DYNAMIC_NORM_STATS_FILE = "dynamic_norm_stats.json"
RESERVOIR_INFLOW_FILE = "reservoir_inflow_hourly.csv"
CACHE_VERSION = 1

# Aji-4 has no infiltration (IP is all zeros) and no rain on the mesh (Pr is all zeros).
STATIC_FEATURES = ["CA", "CE", "CS", "A", "CU", "N", "FA"]
STATIC_KEYS = ["xy_coords", "area", "elevation", "slope", "aspect",
               "curvature", "manning", "flow_accum"]
FORCING_KEYS = ["release", "inflow"]
DYNAMIC_KEYS = ["water_depth", "volume"] + FORCING_KEYS


def _fast_load_txt(path: Path) -> np.ndarray:
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return np.empty((0,), dtype=np.float32)
    n_rows = len(lines)
    n_cols = len(lines[0].split())
    arr = np.fromstring(text, dtype=np.float32, sep=" ")
    if n_cols > 1 and arr.size == n_rows * n_cols:
        return arr.reshape(n_rows, n_cols)
    return arr


class _MmapCache:
    """Small LRU of open ``np.load(mmap_mode="r")`` handles for the ``float32`` cache.

    Aji-4 has 272 floods x 2 fields x 181 x 15,990 float32 = 6.3 GB. Opening an mmap costs
    ~6 ms, so re-opening per window fetch would dominate the data path. Keeping the last
    ``size`` floods open costs O(size) descriptors and page cache instead of O(dataset).
    """

    def __init__(self, size: int = 24) -> None:
        self.size = max(4, int(size))
        self._handles: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def get(self, hid: str, paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
        if hid in self._handles:
            self._handles.move_to_end(hid)
            return self._handles[hid]
        loaded = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
        self._handles[hid] = loaded
        if len(self._handles) > self.size:
            self._handles.popitem(last=False)
        return loaded


class HydroGraphDataset(Dataset):
    """Dataset for 2D flood forecasting on the Aji-4 dam mesh.

    Node feature layout (``9 + 2 + 2 * n_time_steps`` = **15** columns at ``K=2``):
      - ``[0:2]``               : normalized ``(x, y)`` cell-centre coordinates (UTM 42N)
      - ``[2:9]``               : normalized area, elevation, slope, aspect, curvature,
                                  Manning's n, flow accumulation
                                  (``M80_CA, CE, CS, A, CU, N, FA``)
      - ``[9:11]``              : normalized ``[release[t], inflow[t]]``
                                  ``release`` = dam -> flood plain (``M80_US_InF``)
                                  ``inflow``  = catchment -> reservoir (``reservoir_inflow_hourly.csv``)
      - ``[11:11+K]``           : normalized water-depth history ``[t-K .. t-1]``
      - ``[11+K:11+2*K]``       : normalized volume history ``[t-K .. t-1]``

    Target ``y`` layout (``2`` columns):
      - ``[wd[t] - wd[t-1], vol[t] - vol[t-1]]`` (normalized increments)

    Dropped relative to the White River loader: ``infiltration`` (``M80_IP`` is identically
    zero) and ``precipitation`` (``M80_Pr`` is identically zero; no rain on the mesh).
    Added: reservoir inflow, absent from White River. Elapsed hour is intentionally excluded.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        prefix: str = "M80",
        num_samples: Optional[int] = 500,
        n_time_steps: int = 2,
        k: int = 4,
        noise_std: float = 0.01,
        noise_type: str = "random_walk",
        hydrograph_ids_file: Optional[str] = None,
        hydrograph_ids_extra: Optional[str] = None,
        split: str = "train",
        rollout_length: Optional[int] = None,
        skip: Union[int, str] = "auto",
        interval: int = 1,
        use_existing_stats: bool = True,
        use_cache: bool = True,
        mmap_cache_size: int = 6,
        stats_dir: Optional[Union[str, Path]] = None,
        delta_t: float = 1800.0,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")
        if noise_type not in {"none", "only_last", "random_walk"}:
            raise ValueError(
                f"Invalid noise_type '{noise_type}'. Choose from: 'random_walk', 'only_last', 'none'."
            )

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory '{self.data_dir}' does not exist. "
                f"Extract the aji4_flood_data_G1O archive there, then rerun."
            )

        self.stats_dir = Path(stats_dir) if stats_dir is not None else self.data_dir
        self.prefix = prefix
        self.num_samples = num_samples
        self.n_time_steps = n_time_steps
        self.k = k
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.hydrograph_ids_file = hydrograph_ids_file
        self.split = split
        self.rollout_length = rollout_length if rollout_length is not None else 0
        self.random_ids_extra = hydrograph_ids_extra
        self.skip_mode = skip                      # int row offset, or "auto"
        self.skip = int(skip) if not isinstance(skip, str) else 0
        self.interval = interval
        self.delta_t = float(delta_t)
        self.use_existing_stats = use_existing_stats
        self.use_cache = use_cache
        self._mmaps = _MmapCache(mmap_cache_size)
        self._onset: Dict[str, int] = {}
        self._lengths: Dict[str, int] = {}
        self._reservoir_inflow: Dict[str, np.ndarray] = {}

        self.static_stats: Dict[str, dict] = {}
        self.dynamic_stats: Dict[str, dict] = {}
        self.static_data: Dict[str, np.ndarray] = {}
        self.dynamic_data: List[Dict[str, object]] = []
        self.sample_index: List[Tuple[int, int]] = []
        self.hydrograph_ids: List[str] = []

        self._process()

    @staticmethod
    def normalize(
        data: np.ndarray,
        mean: Union[float, list, np.ndarray],
        std: Union[float, list, np.ndarray],
    ) -> np.ndarray:
        return (data - np.asarray(mean)) / (np.asarray(std) + 1e-8)

    @staticmethod
    def denormalize(
        data: np.ndarray,
        mean: Union[float, list, np.ndarray],
        std: Union[float, list, np.ndarray],
    ) -> np.ndarray:
        return data * np.asarray(std) + np.asarray(mean)

    def _find_stats_file(self, filename: str) -> Optional[Path]:
        for candidate in [
            self.stats_dir / filename,
            self.data_dir / filename,
            self.data_dir.parent / filename,
        ]:
            if candidate.is_file():
                return candidate
        return None

    STATIC_WIDTH = 9          # 2 XY + 7 terrain/land-cover
    NUM_FORCINGS = 2          # release, reservoir inflow

    @property
    def node_input_dim(self) -> int:
        return self.STATIC_WIDTH + self.NUM_FORCINGS + 2 * self.n_time_steps

    def _load_static_data(self) -> None:
        static_stats_path = self._find_stats_file(STATIC_NORM_STATS_FILE)
        if (self.split == "test" or self.use_existing_stats) and static_stats_path is not None:
            self.static_stats = json.loads(static_stats_path.read_text())
        elif self.split == "test":
            raise FileNotFoundError(f"Could not find {STATIC_NORM_STATS_FILE} in {self.data_dir}")

        p = self.prefix
        xy_raw = _fast_load_txt(self.data_dir / f"{p}_XY.txt")
        if xy_raw.ndim == 1:
            xy_raw = xy_raw.reshape(-1, 2)
        n_nodes = xy_raw.shape[0]

        def _load_col(suffix: str) -> np.ndarray:
            arr = _fast_load_txt(self.data_dir / f"{p}_{suffix}.txt").reshape(-1)[:n_nodes]
            return arr.reshape(-1, 1)

        # Aji-4 statics. ``M80_IP`` (infiltration) is dropped: it is 0 in all 15,990 cells.
        raw_map = {
            "xy_coords": xy_raw,
            "area": _load_col("CA"),
            "elevation": _load_col("CE"),
            "slope": _load_col("CS"),
            "aspect": _load_col("A"),
            "curvature": _load_col("CU"),
            "manning": _load_col("N"),
            "flow_accum": _load_col("FA"),
        }

        normed: Dict[str, np.ndarray] = {}
        computed_new = False
        for key, arr in raw_map.items():
            if key not in self.static_stats:
                self.static_stats[key] = {
                    "mean": np.mean(arr, axis=0).tolist(),
                    "std": np.std(arr, axis=0).tolist(),
                }
                computed_new = True
            std = np.asarray(self.static_stats[key]["std"], dtype=np.float64)
            std = np.where(std < 1e-12, 1.0, std)      # guard constant channels -> no NaN
            normed[key] = self.normalize(
                arr, self.static_stats[key]["mean"], std
            ).astype(np.float32)

        if computed_new and self.split == "train":
            (self.data_dir / STATIC_NORM_STATS_FILE).write_text(json.dumps(self.static_stats, indent=2))

        xy_std = normed["xy_coords"]
        _, neighbors = KDTree(xy_std).query(xy_std, k=self.k + 1)
        src_list, dst_list = [], []
        for i, nbrs in enumerate(neighbors):
            for nbr in np.atleast_1d(nbrs):
                if int(nbr) != i:
                    src_list.append(i)
                    dst_list.append(int(nbr))
        edge_index = np.array([src_list, dst_list], dtype=np.int64)

        rel = xy_std[edge_index[0]] - xy_std[edge_index[1]]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        rel_norm = (rel - rel.mean(axis=0)) / (rel.std(axis=0) + 1e-8)
        dist_norm = (dist - dist.mean()) / (dist.std() + 1e-8)
        edge_features = np.hstack([rel_norm, dist_norm]).astype(np.float32)

        static_matrix = np.hstack(
            [
                normed["xy_coords"],
                normed["area"],
                normed["elevation"],
                normed["slope"],
                normed["aspect"],
                normed["curvature"],
                normed["manning"],
                normed["flow_accum"],
            ]
        ).astype(np.float32)

        self.static_data = {
            "static_matrix": static_matrix,
            "xy_coords_raw": xy_raw.astype(np.float32),
            "xy_coords": xy_std,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }
        self.num_nodes = n_nodes
        self._edge_index_t = torch.from_numpy(edge_index)
        self._edge_attr_t = torch.from_numpy(edge_features)

    # ---------------------------------------------------------------- ids
    def _discover_hydrograph_ids(self) -> List[str]:
        cache_dir = self.data_dir / ".cache_npy"
        ids: List[str] = []
        for fname in (self.hydrograph_ids_file, self.random_ids_extra):
            if not fname:
                continue
            id_path = self.data_dir / fname
            if not id_path.is_file():
                raise FileNotFoundError(f"id list '{fname}' not found in {self.data_dir}")
            ids += [ln.strip() for ln in id_path.read_text().splitlines() if ln.strip()]

        if ids:                                   # explicit lists (train / test)
            seen, out = set(), []
            for hid in ids:
                if hid in seen:
                    continue
                if (self.data_dir / f"{self.prefix}_WD_{hid}.txt").is_file():
                    seen.add(hid)
                    out.append(hid)
            if not out:
                raise RuntimeError(f"none of the ids in {ids[:3]}... have data files")
            return out

        # no list given: take every flood present on disk
        tag = f"{self.prefix}_WD_"
        found = {
            e[len(tag):-4]
            for e in os.listdir(self.data_dir)
            if e.startswith(tag) and e.endswith(".txt")
        }
        return sorted(found)

    # ------------------------------------------------------ burst onset / window
    def onset_for(self, hid: str) -> int:
        """Row at which this flood's start-up burst has drained away."""
        if hid in self._onset:
            return self._onset[hid]
        if isinstance(self.skip_mode, str):          # "auto" not yet resolved for this id
            self._onset[hid] = self.detect_burst_end(hid)
            return self._onset[hid]
        return int(self.skip_mode)

    def detect_burst_end(
        self, hid: str, min_row: int = 8, max_row: int = 40, tol: float = 0.10
    ) -> int:
        """Row where the Aji-4 start-up burst ends, measured per flood.

        Every run opens with the reservoir at 20.40 m feeding a dry channel, so the dam
        releases ~4,800-5,400 m3/s (G1O) for the first hours. That burst is *larger than
        the real flood peak* in 42 % of the 272 floods, which is why an argmax-anchored
        window (the White River rule) keeps the wrong part of the series.

        Method: find the post-burst trough (argmin of the release over
        ``[min_row, max_row]``, valid only below half the peak), then return the first row
        within ``tol`` of that trough - the moment the channel has finished draining.
        """
        q = np.loadtxt(self.data_dir / f"{self.prefix}_US_InF_{hid}.txt")
        q = np.asarray(q, dtype=np.float64)
        q = q[:, 1] if q.ndim == 2 else q
        hi = float(q.max())
        if hi <= 0 or not np.isfinite(hi):
            return int(min_row)
        up = min(max_row + 1, len(q))
        win = q[min_row:up]
        trough = float(win.min())
        if trough > 0.5 * hi:                     # no clear trough: fixed fallback
            return int(min_row)
        near = np.where(win <= trough * (1.0 + tol))[0]
        row = min_row + (int(near[0]) if near.size else int(np.argmin(win)))
        return int(min(max(row, min_row), max_row))

    # --------------------------------------------------------------- reservoir
    def _load_reservoir_inflow(self) -> Dict[str, np.ndarray]:
        """Catchment -> reservoir inflow, m3/s, hourly, closed with the final value.

        ``reservoir_inflow_hourly.csv`` is long format ``flood,hour,inflow_m3s`` covering
        hours 0-100. It is interpolated onto the model's 30-minute row grid and extended
        past its last sample with the final value (the recession tail is flat).
        """
        if self._reservoir_inflow:
            return self._reservoir_inflow
        path = self.data_dir / RESERVOIR_INFLOW_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing - Aji-4 needs both forcings (release and reservoir inflow)"
            )
        import csv as _csv

        per: Dict[str, List[Tuple[float, float]]] = {}
        with path.open() as fh:
            for row in _csv.DictReader(fh):
                if row["flood"] not in self.hydrograph_ids:
                    continue
                per.setdefault(row["flood"], []).append(
                    (float(row["hour"]), float(row["inflow_m3s"]))
                )
        self._reservoir_inflow = {
            hid: np.asarray(sorted(pts), dtype=np.float64) for hid, pts in per.items()
        }
        return self._reservoir_inflow

    def reservoir_inflow_series(self, hid: str, n_rows: int) -> np.ndarray:
        """Reservoir inflow sampled on the model's row grid (hours 0, 0.5 .. )."""
        table = self._load_reservoir_inflow().get(hid)
        hours = np.arange(n_rows, dtype=np.float64) * self.delta_t / 3600.0
        if table is None or table.shape[0] == 0:
            return np.zeros(n_rows, dtype=np.float32)
        return np.interp(
            hours, table[:, 0], table[:, 1], left=table[0, 1], right=table[-1, 1]
        ).astype(np.float32)

    # ------------------------------------------------------------------ cache
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / ".cache_npy"

    def _cache_key(self) -> str:
        sk = self.skip_mode if isinstance(self.skip_mode, str) else f"i{self.skip_mode}"
        return f"v{CACHE_VERSION}_s{sk}_i{self.interval}_dt{self.delta_t:g}"

    def _cache_paths(self, hid: str) -> Dict[str, Path]:
        base = self.cache_dir / self._cache_key()
        return {
            "water_depth": base / f"{self.prefix}_{hid}_wd.npy",
            "volume": base / f"{self.prefix}_{hid}_v.npy",
            "release": base / f"{self.prefix}_{hid}_q.npy",
            "inflow": base / f"{self.prefix}_{hid}_qi.npy",
        }

    def _build_cache(self, hid: str) -> None:
        """Convert one flood's ASCII tables to ``float32 .npy`` (once per flood)."""
        paths = self._cache_paths(hid)
        if all(p.is_file() for p in paths.values()):
            return
        s0 = self.onset_for(hid)
        wd = _fast_load_txt(self.data_dir / f"{self.prefix}_WD_{hid}.txt")[s0:: self.interval, : self.num_nodes]
        vol = _fast_load_txt(self.data_dir / f"{self.prefix}_V_{hid}.txt")[s0:: self.interval, : self.num_nodes]
        q_raw = _fast_load_txt(self.data_dir / f"{self.prefix}_US_InF_{hid}.txt")
        rel = q_raw[s0:: self.interval, 1] if q_raw.ndim == 2 else q_raw[s0:: self.interval]
        n = int(min(wd.shape[0], vol.shape[0], rel.shape[0]))
        inflow = self.reservoir_inflow_series(hid, n_rows=181)[s0:: self.interval][:n]
        if inflow.shape[0] < n:                      # extend with the final value
            inflow = np.concatenate([inflow, np.full(n - inflow.shape[0], inflow[-1], np.float32)])
        paths["water_depth"].parent.mkdir(parents=True, exist_ok=True)
        np.save(paths["water_depth"], np.ascontiguousarray(wd[:n], dtype=np.float32))
        np.save(paths["volume"], np.ascontiguousarray(vol[:n], dtype=np.float32))
        np.save(paths["release"], np.ascontiguousarray(rel[:n], dtype=np.float32))
        np.save(paths["inflow"], np.ascontiguousarray(inflow[:n], dtype=np.float32))

    def _load_single_hydrograph(self, hid: str) -> Dict[str, np.ndarray]:
        """Return the kept window of one flood: depth, volume, release, inflow."""
        if self.use_cache:
            self._build_cache(hid)
            cached = self._mmaps.get(hid, self._cache_paths(hid))
            wd = np.asarray(cached["water_depth"])
            vol = np.asarray(cached["volume"])
            rel = np.asarray(cached["release"])
        else:
            s0 = self.onset_for(hid)
            wd = _fast_load_txt(self.data_dir / f"{self.prefix}_WD_{hid}.txt")[s0:: self.interval, : self.num_nodes]
            vol = _fast_load_txt(self.data_dir / f"{self.prefix}_V_{hid}.txt")[s0:: self.interval, : self.num_nodes]
            q_raw = _fast_load_txt(self.data_dir / f"{self.prefix}_US_InF_{hid}.txt")
            rel = q_raw[s0:: self.interval, 1] if q_raw.ndim == 2 else q_raw[s0:: self.interval]
            wd = np.asarray(wd, dtype=np.float32)
            vol = np.asarray(vol, dtype=np.float32)
            rel = np.asarray(rel, dtype=np.float32)

        n = int(min(wd.shape[0], vol.shape[0], rel.shape[0]))
        # reservoir inflow on the *original* row grid, then the same slice/stride
        inflow = self.reservoir_inflow_series(hid, n_rows=181)[self.onset_for(hid) :: self.interval][:n]
        if inflow.shape[0] < n:                                   # pad with last value
            inflow = np.concatenate(
                [inflow, np.full(n - inflow.shape[0], inflow[-1], np.float32)]
            )
        return {
            "water_depth": wd[:n],
            "volume": vol[:n],
            "release": rel[:n].astype(np.float32),
            "inflow": inflow[:n].astype(np.float32),
        }

    def _process(self) -> None:
        self._load_static_data()

        ids = self._discover_hydrograph_ids()
        if not ids:
            raise RuntimeError(
                f"No hydrograph files found in {self.data_dir} with prefix '{self.prefix}'."
            )
        if self.num_samples is not None and len(ids) > self.num_samples:
            ids = ids[: self.num_samples]
        self.hydrograph_ids = ids

        # per-flood window start: measured burst end, or a fixed row offset
        if isinstance(self.skip_mode, str):
            if self.skip_mode != "auto":
                raise ValueError("skip must be an integer row or 'auto'")
            for hid in tqdm(self.hydrograph_ids, desc="Detecting burst onset", leave=False):
                self._onset[hid] = self.detect_burst_end(hid)
        else:
            self._onset = {hid: int(self.skip_mode) for hid in self.hydrograph_ids}

        # ---- normalisation statistics (streamed: 6.3 GB cannot be concatenated)
        dyn_path = self._find_stats_file(DYNAMIC_NORM_STATS_FILE)
        reuse = (self.split == "test" or self.use_existing_stats) and dyn_path is not None
        shipped = json.loads(dyn_path.read_text()) if reuse else {}
        missing = [key for key in DYNAMIC_KEYS if key not in shipped]
        # The original Aji-4 JSON lacks release/inflow, so the first training run
        # computes those fields. Later train/test runs reuse the complete TRAIN mapping.
        computed = self._compute_dynamic_stats(self.hydrograph_ids) if missing or not reuse else {}
        self.dynamic_stats = {k: shipped[k] for k in DYNAMIC_KEYS if k in shipped}
        for key in missing:
            self.dynamic_stats[key] = computed[key]
        # Persist the complete TRAIN normalization mapping, including the renamed
        # release/inflow channels, so test evaluation never derives its own statistics.
        if self.split == "train":
            (self.data_dir / DYNAMIC_NORM_STATS_FILE).write_text(
                json.dumps(self.dynamic_stats, indent=2)
            )

        # Release the mmaps used by the statistics pass: on a 15,990-node mesh each flood
        # is 4 x ~11 MB, so the pages touched while scanning 234 floods would otherwise
        # stay resident on top of the training tensors.
        self._mmaps = _MmapCache(self._mmaps.size)
        gc.collect()

        # a constant channel cannot be standardised; keep it finite
        for key in DYNAMIC_KEYS:
            if float(self.dynamic_stats[key]["std"]) < 1e-12:
                self.dynamic_stats[key] = {"mean": 0.0, "std": 1.0}

        # ---- store runs LAZILY: the depth/volume arrays stay memory-mapped and are
        # normalised per sample in __getitem__. Materialising all of them would need
        # 272 x 2 x 181 x 15,990 x 4 B = 6.3 GB of RAM, which is what OOM-kills the
        # original design on anything smaller than a large node.
        self.dynamic_data = []
        for hid in self.hydrograph_ids:
            self._build_cache(hid)
            n_rows = int(np.load(self._cache_paths(hid)["water_depth"], mmap_mode="r").shape[0])
            self.dynamic_data.append({
                "hydro_id": hid,
                "n_rows": n_rows,
                })
        self._mmaps = _MmapCache(self._mmaps.size)
        gc.collect()

        K = self.n_time_steps
        if self.split == "train":
            for h_idx, dyn in enumerate(self.dynamic_data):
                T = int(dyn["n_rows"])                                  # type: ignore[arg-type]
                for t_idx in range(max(0, T - K)):
                    self.sample_index.append((h_idx, t_idx))
        else:
            for dyn in self.dynamic_data:
                T = int(dyn["n_rows"])                                  # type: ignore[arg-type]
                need = K + self.rollout_length if self.rollout_length > 0 else K + 1
                if T < need:
                    raise ValueError(
                        f"Test hydrograph {dyn['hydro_id']} has length {T}, "
                        f"shorter than the required {need} "
                        f"(n_time_steps + rollout_length)."
                    )

    def _compute_dynamic_stats(self, hydro_ids: List[str]) -> Dict[str, dict]:
        """Mean/std of every dynamic field, streamed from the memmap in chunks.

        Two things matter here on the full package. First, a flood is never read as a
        whole: each field is 11.6 MB float32 and a 4-field flood is 46 MB, which is fine
        once but not 234 times in a loop. Second, the accumulator is float64 (sum,
        sum-of-squares, count) so a single pass gives an exact mean and variance without
        holding anything but one chunk.
        """
        acc = {k: np.zeros(3, dtype=np.float64) for k in DYNAMIC_KEYS}

        def push(key: str, arr: np.ndarray) -> None:
            flat = np.asarray(arr, dtype=np.float64).reshape(-1)
            acc[key][0] += flat.sum()
            acc[key][1] += np.square(flat).sum()
            acc[key][2] += flat.size

        for hid in tqdm(hydro_ids, desc="Statistics", leave=False):
            self._build_cache(hid)
            arrays = self._mmaps.get(hid, self._cache_paths(hid))
            for key in DYNAMIC_KEYS:
                mm = arrays[key]
                for i in range(0, mm.shape[0], 8):              # 8 rows x 15,990 cells
                    push(key, np.asarray(mm[i : i + 8]))
            del arrays
            gc.collect()

        stats: Dict[str, dict] = {}
        for key in DYNAMIC_KEYS:
            total, sumsq, count = acc[key]
            mean = total / max(count, 1.0)
            stats[key] = {
                "mean": float(mean),
                "std": float(np.sqrt(max(sumsq / max(count, 1.0) - mean * mean, 0.0))),
            }
        return stats

    def __len__(self) -> int:
        return len(self.sample_index) if self.split == "train" else len(self.dynamic_data)

    def _apply_window_noise(self, win: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random-walk or last-step Gaussian noise to history window ``(K, N)``.

        Returns ``(noisy_window, last_step_noise)`` so the target increment can be
        corrected to ``u_{t+1} - (u_t + eps_last)`` following Pfaff et al. (2020),
        teaching the model to actively damp accumulated rollout drift.
        """
        K, N = win.shape
        if self.noise_type == "none" or self.noise_std <= 0.0:
            return win, np.zeros((N,), dtype=np.float32)
        out = win.copy()
        if self.noise_type == "only_last":
            eps = np.random.normal(0.0, self.noise_std, size=(N,)).astype(np.float32)
            out[-1] += eps
            return out, eps
        step_std = self.noise_std / np.sqrt(float(K))
        walk = np.cumsum(
            np.random.normal(0.0, step_std, size=(K, N)).astype(np.float32),
            axis=0,
        )
        out += walk
        return out, walk[-1]

    def _norm(self, key: str, arr):
        """Standardise ``arr`` (array or scalar) with the stored statistics."""
        st = self.dynamic_stats[key]
        out = (np.asarray(arr, dtype=np.float32) - float(st["mean"])) / (
            float(st["std"]) + 1e-8)
        return out.astype(np.float32) if isinstance(out, np.ndarray) else np.float32(out)

    def _forcing(
        self, arrays: Dict[str, np.ndarray], t: int, N: int
    ) -> np.ndarray:
        """``(N, 2)`` normalised forcing block: release and reservoir inflow."""
        out = np.empty((N, 2), dtype=np.float32)
        out[:, 0] = self._norm("release", float(arrays["release"][t]))
        out[:, 1] = self._norm("inflow", float(arrays["inflow"][t]))
        return out

    def _arrays(self, hydro_idx: int) -> Dict[str, np.ndarray]:
        """Depth/volume/release/inflow for one flood, via the LRU mmap cache."""
        hid = str(self.dynamic_data[hydro_idx]["hydro_id"])
        if not self.use_cache:                                   # tiny runs only
            return self._load_single_hydrograph(hid)
        return self._mmaps.get(hid, self._cache_paths(hid))

    def __getitem__(self, idx: int):
        static_mat = self.static_data["static_matrix"]      # (N, 9)
        N = self.num_nodes
        K = self.n_time_steps

        if self.split == "train":
            hydro_idx, t_idx = self.sample_index[idx]
            dyn = self.dynamic_data[hydro_idx]
            arrays = self._arrays(hydro_idx)

            prev_time = t_idx + K - 1
            target_time = t_idx + K

            wd_all = self._norm("water_depth", np.asarray(arrays["water_depth"][t_idx : t_idx + K + 1]))
            vol_all = self._norm("volume", np.asarray(arrays["volume"][t_idx : t_idx + K + 1]))

            wd_win, wd_eps = self._apply_window_noise(wd_all[:K])
            vol_win, vol_eps = self._apply_window_noise(vol_all[:K])

            forcing = self._forcing(arrays, target_time, N)
            x_np = np.hstack([static_mat, forcing, wd_win.T, vol_win.T])
            target_np = np.stack(
                [
                    wd_all[K] - (wd_all[K - 1] + wd_eps),
                    vol_all[K] - (vol_all[K - 1] + vol_eps),
                ],
                axis=1,
            )
            return GraphData(
                x=torch.from_numpy(np.ascontiguousarray(x_np)),
                edge_index=self._edge_index_t,
                edge_attr=self._edge_attr_t,
                y=torch.from_numpy(np.ascontiguousarray(target_np)),
            )

        # ---- test rollout: start at row K-1, predict K, K+1, ...
        dyn = self.dynamic_data[idx]
        arrays = self._arrays(idx)
        T = int(dyn["n_rows"])
        R = self.rollout_length if self.rollout_length > 0 else (T - K)

        wd_all = self._norm("water_depth", np.asarray(arrays["water_depth"][: K + R]))
        vol_all = self._norm("volume", np.asarray(arrays["volume"][: K + R]))

        forcing_init = self._forcing(arrays, K, N)
        x_np = np.hstack([static_mat, forcing_init, wd_all[:K].T, vol_all[:K].T])
        graph = GraphData(
            x=torch.from_numpy(np.ascontiguousarray(x_np)),
            edge_index=self._edge_index_t,
            edge_attr=self._edge_attr_t,
        )
        rollout_data = {
            "hydro_id": dyn["hydro_id"],
            "release": torch.from_numpy(np.ascontiguousarray(
                self._norm("release", np.asarray(arrays["release"][K : K + R])))),
            "inflow": torch.from_numpy(np.ascontiguousarray(
                self._norm("inflow", np.asarray(arrays["inflow"][K : K + R])))),
            "water_depth_gt": torch.from_numpy(np.ascontiguousarray(wd_all[K : K + R])),
            "volume_gt": torch.from_numpy(np.ascontiguousarray(vol_all[K : K + R])),
        }
        return graph, rollout_data
