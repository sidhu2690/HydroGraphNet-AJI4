# Aji-4 dataset guide

## 1. What the dataset represents

The Aji-4 packages contain HEC-RAS 2D flood simulations downstream of **Aji-4 dam, Gujarat, India**. Each gate-case folder contains the same **272 reservoir-flood events** under a different fixed dam-gate configuration.

| Folder suffix | Gate configuration |
|---|---|
| `G1O` | One gate group open |
| `G12O` | Two gate groups open |
| `G123O` | All three gate groups open |

The mesh covers approximately **103.6 km²**, primarily below the dam and toward the downstream/coastal boundary. The code supports all three cases without source changes; only `data_dir` changes.

## 2. Spatial and temporal dimensions

| Quantity | Value |
|---|---:|
| Mesh cells / graph nodes | **15,990** |
| Nearest neighbors per node | 4 |
| Directed graph edges | **63,960** = 15,990 × 4 |
| Floods per gate-case folder | **272** |
| Original rows per flood | **181** |
| Original saved times | 0, 0.5, 1.0, …, 90 hours |
| Time between rows | **30 minutes = 1,800 seconds** |
| Original duration | **90 hours** |
| Water-depth array per flood | **181 × 15,990**, physical unit m |
| Cell-volume array per flood | **181 × 15,990**, physical unit m³ |
| Dam-release array per flood | **181 × 2** |

Rows are time steps; columns are mesh cells. A single model graph represents one time window across all 15,990 cells.

## 3. File naming convention

The files retain the main repository’s `M80_` prefix for loader compatibility. **In this package, `M80` does not mean the White River dataset.**

### Static files — one value/row per mesh cell

| File | Meaning | Shape/unit |
|---|---|---|
| `M80_XY.txt` | Cell-center coordinates | `(15990, 2)`, UTM 42N meters |
| `M80_CA.txt` | Cell area | `(15990,)` or `(15990, 1)`, m² |
| `M80_CE.txt` | Cell ground elevation | `(15990,)`, m |
| `M80_CS.txt` | Cell slope | `(15990,)`, degrees |
| `M80_A.txt` | Aspect/direction of slope | `(15990,)`, degrees |
| `M80_CU.txt` | Terrain curvature | `(15990,)`, 1/m |
| `M80_N.txt` | Manning roughness coefficient | `(15990,)` |
| `M80_FA.txt` | Flow accumulation | `(15990,)` |
| `M80_IP.txt` | Infiltration | `(15990,)`; all zero and **not loaded** |

### Dynamic files — one set per flood ID

Replace `<ID>` with an ID such as `R001`, `S109`, or `V028`.

| File pattern | Meaning | Shape/unit |
|---|---|---|
| `M80_WD_<ID>.txt` | Water depth in every cell over time | `(181, 15990)`, m |
| `M80_V_<ID>.txt` | Water volume in every cell over time | `(181, 15990)`, m³ |
| `M80_US_InF_<ID>.txt` | Time and **dam release downstream** | `(181, 2)`, hour and m³/s |
| `M80_Pr_<ID>.txt` | Precipitation | All zero and **not loaded** |

`US_InF` is potentially confusing: here it means water **leaving the dam/reservoir and entering the downstream flooded valley**. It is the direct hydraulic forcing of the HEC-RAS 2D mesh.

### Reservoir inflow

`reservoir_inflow_hourly.csv` is a separate long-format CSV:

```text
flood,hour,inflow_m3s
R001,0,75.23
R001,1,80.10
...
```

It represents water **entering the Aji-4 reservoir from its catchment**. It is not the same as dam release.

- Source time spacing: hourly.
- Model time spacing: 30 minutes.
- The loader linearly interpolates hourly values onto the 30-minute grid.
- Before the first/after the last supplied value, it uses the nearest endpoint value.
- Both reservoir inflow and dam release are scalar per time row and are broadcast to all 15,990 nodes.

### Other metadata

| File | Purpose |
|---|---|
| `train.txt` | 195 training flood IDs |
| `val.txt` | 39 validation flood IDs |
| `test.txt` | 38 held-out test flood IDs |
| `floods.csv` | Flood group, split, shape, and peak metadata |
| `static_norm_stats.json` | Static-feature normalization |
| `dynamic_norm_stats.json` | Dynamic-feature normalization |
| `mesh_cells_utm42n.csv` | Cell centers and polygon geometry |
| `MD5SUMS.txt` | Source-file checksums |

The default training recipe pools `train.txt + val.txt` into **234 optimization floods**. `test.txt` remains a separate **38-flood** set.

## 4. Flood ID groups

| Prefix | Events | General origin |
|---|---:|---|
| `R###` | 150 | New design/synthetic reservoir-inflow shapes and variants |
| `S###` | 72 | HBV rainfall-runoff events and scaled variants |
| `V###` | 50 | Probable maximum flood variants |

The test set includes entire inflow-shape families absent from train/validation, plus selected HBV and PMF events. It is therefore an event-generalization test, not random window holdout.

## 5. Features used by this implementation

### 5.1 Static features: 9 per node

| Node columns | Count | Values |
|---|---:|---|
| `0:2` | 2 | Normalized cell-center x, y |
| `2` | 1 | Normalized cell area |
| `3` | 1 | Normalized elevation |
| `4` | 1 | Normalized slope |
| `5` | 1 | Normalized aspect |
| `6` | 1 | Normalized curvature |
| `7` | 1 | Normalized Manning’s n |
| `8` | 1 | Normalized flow accumulation |

Infiltration is omitted because `M80_IP.txt` is zero throughout the Aji-4 model.

### 5.2 Dynamic forcing features: 2 per node

| Node column | Value | Physical direction |
|---|---|---|
| `9` | Normalized `US_InF` / dam release | Dam/reservoir → downstream valley |
| `10` | Normalized reservoir inflow | Catchment → Aji-4 reservoir |

Precipitation is omitted because `M80_Pr_<ID>.txt` is zero. **There is no elapsed-hour feature.**

### 5.3 History features: 4 per node when `K=2`

| Node columns | Count | Values |
|---|---:|---|
| `11:13` | 2 | Normalized water depth at the two preceding rows |
| `13:15` | 2 | Normalized cell volume at the same two preceding rows |

History is ordered oldest to newest.

### 5.4 Total input dimension

```text
9 static + 2 forcing + 2 depth history + 2 volume history = 15
```

For general history length `K`:

```text
input_dim_nodes = 11 + 2K
```

Thus `K=2` gives 15. One training sample has:

- node input `x`: **`(15990, 15)`**;
- edge index: **`(2, 63960)`**;
- edge attributes: **`(63960, 3)`**;
- target `y`: **`(15990, 2)`**.

## 6. Edge features

The graph is a directed four-nearest-neighbor graph over normalized XY coordinates.

Each edge has three normalized features:

```text
[Δx, Δy, Euclidean distance]
```

The relative displacement is source coordinates minus destination coordinates. Messages are aggregated at destination nodes by sum, exactly as in the main repository.

## 7. Targets and time alignment

For history length `K=2`, a training window starting at retained row `s` uses:

```text
history:       rows s and s+1
current state: row s+1
forcing:       row s+2
target state:  row s+2
```

Targets are normalized increments:

```text
y[:, 0] = normalized Δwater_depth
y[:, 1] = normalized Δcell_volume
```

During training with random-walk noise, the increment is measured from the noisy latest state so the model learns to correct injected drift.

During rollout, only depth and volume are fed back recursively. Future dam release and reservoir inflow are supplied from the known forcing series.

## 8. Startup burst and retained windows

Every HEC-RAS run begins with a large artificial/startup dam release as the initially full reservoir drains into a dry valley. This startup peak can exceed the later reservoir-flood peak.

The loader therefore defaults to `skip_initial_steps: auto`:

1. examines release rows 8–40;
2. detects the post-startup trough;
3. chooses the first row within 10% of that trough;
4. retains data from that row to the record end.

Detected onsets vary by flood. Consequently, retained row counts and training-window counts vary. With `T` retained rows and `K=2`, each flood contributes `T−2` one-step windows. The complete 234-flood G1O training pool contains **34,938** such windows.

Set `skip_initial_steps: 72` to use the dataset README’s fixed hours-36-to-90 recipe instead.

## 9. Cache dimensions and purpose

The loader creates versioned float32 NPY files under:

```text
<gate-case>/.cache_npy/
```

Per flood it stores retained-window arrays for:

- water depth `(T, 15990)`;
- cell volume `(T, 15990)`;
- dam release `(T,)`;
- reservoir inflow `(T,)`.

Large spatial arrays are memory-mapped and normalized only when a window is requested. This avoids loading all floods into RAM simultaneously. Cache keys include onset mode, row interval, and `delta_t`.

## 10. Normalization

Static and dynamic inputs are z-score normalized. The first training run completes the dynamic-statistics JSON with the Aji-4 names:

- `water_depth`;
- `volume`;
- `release`;
- `inflow`.

Test evaluation reuses the complete training mapping. WD/volume predictions are denormalized to meters and cubic meters before metric calculation. Rollout states are clamped so physical depth and volume cannot become negative.
