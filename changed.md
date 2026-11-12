# Changes from the main HydroGraphNet repository

Reference: [`sidhu2690/HydroGraphNet`](https://github.com/sidhu2690/HydroGraphNet), main White River implementation.

## Design rule

This is a **minimal-change Aji-4 adaptation**, not a new model. The network, message passing, loss, metrics, and graph batching are retained. Changes are concentrated in `dataset.py`, the Aji-4 wiring in `train.py`/`evaluate.py`, and the full-data configuration.

## File-by-file summary

| Main-repository file | Status here | What changed |
|---|---|---|
| `hydrographnet/__init__.py` | **Unchanged, byte-for-byte** | Same package exports |
| `hydrographnet/graph.py` | **Unchanged, byte-for-byte** | Same graph containers, disjoint batching, sum/mean aggregation |
| `hydrographnet/models.py` | **Unchanged, byte-for-byte** | Same MLP encoder–processor–decoder network |
| `hydrographnet/losses.py` | **Unchanged, byte-for-byte** | Same normalized depth/volume MSE |
| `hydrographnet/metrics.py` | **Unchanged, byte-for-byte** | Same RMSE, MAE, NSE, CSI, and volume metrics |
| `hydrographnet/dataset.py` | **Changed for Aji-4** | New features, forcings, windows, lazy cache, 15-column layout |
| `train.py` | **Changed where required** | 15-feature data wiring, full Aji-4 splits, path overrides, memory-safe full training |
| `evaluate.py` | **Changed where required** | Two Aji-4 forcings, shifted history slices, path overrides, baselines |
| `requirements.txt` | **Changed slightly** | White River downloader dependencies removed |
| `configs/config.yaml` | Replaced | `configs/aji4_15.yaml` is the complete 150-epoch recipe |
| `configs/sample_cpu.yaml` | **Dropped** | No sampling/subset recipe included |
| `scripts/download_data.py` | **Dropped** | It downloads White River, not Aji-4 |
| `White_River_M80.ipynb` | **Dropped** | White River-specific and not used by the command-line pipeline |
| `HydroGraphNet_Final_00.pdf` | **Dropped** | Background document, not runtime code |
| `assets/`, White River outputs/checkpoints | **Dropped** | Not applicable to Aji-4 training |
| `README.md` | Rewritten for this package | Installation, full G1O/G12O/G123O training and evaluation |
| `DATASET.md` | **Added** | Aji-4 files, dimensions, features, timing, and conventions |
| `changed.md` | **Added** | This comparison |

## 1. Files intentionally left identical

### `models.py`

No changes to:

- `MeshGraphMLP` Linear/activation/LayerNorm structure;
- residual edge updates;
- destination-node aggregation;
- residual node updates;
- five processor blocks in the supplied full configuration;
- two-output normalized increment decoder;
- optional gradient checkpointing;
- forward-pass API.

The full model still uses hidden width 64 and five message-passing blocks. Removing one input channel changes only the configured `input_dim_nodes` from 16 to 15, reducing the parameter count from **214,210 to 214,146**.

### `graph.py`

No changes to `GraphData`, `GraphBatch`, `batch_graphs`, `collate_graphs`, or `scatter_aggregate`.

Aji-4’s larger graph comes from its data, not new code: 15,990 nodes and four neighbors per node produce 63,960 directed edges.

### `losses.py`

The objective remains:

```text
0.5 × [MSE(normalized Δdepth) + MSE(normalized Δvolume)]
```

No new dam, gate, reservoir, conservation, or time feature loss was introduced.

### `metrics.py`

Physical-unit evaluation is unchanged. Aji-4 evaluation adds baseline bookkeeping in `evaluate.py`, not new definitions in `metrics.py`.

### `__init__.py`

Public imports and `__all__` are unchanged.

## 2. `dataset.py` changes

### 2.1 Static inputs

Main White River layout has ten static channels: XY plus eight terrain/land-cover attributes, including infiltration.

This version loads nine: XY plus seven attributes, omitting `M80_IP` because Aji-4 infiltration is identically zero. A guard prevents division by zero for constant static channels.

### 2.2 Dynamic inputs

Main uses:

```text
[inflow_hydrograph, precipitation]
```

This version uses:

```text
[release, reservoir_inflow]
```

- `release` comes from `M80_US_InF_<ID>.txt`: dam → downstream valley.
- `inflow` comes from `reservoir_inflow_hourly.csv`: catchment → Aji-4 reservoir.
- `M80_Pr_<ID>.txt` is not read because precipitation is zero.
- Reservoir inflow is interpolated from hourly to 30-minute rows.

There is **no elapsed-hour channel**.

### 2.3 Exact 15-feature layout

At `K=2`:

```text
Main: 10 static + 2 forcing + 4 history = 16
Here:  9 static + 2 forcing + 4 history = 15
```

Depth history now starts at column **11 instead of 12**. Volume history starts at `11+K`.

### 2.4 Event windows

Main skips 72 original rows and truncates at 25 rows after maximum inflow.

Aji-4 defaults to per-flood startup-burst detection over rows 8–40, then retains through the end. This avoids selecting the artificial startup release as the flood maximum. Integer fixed-row skipping remains supported.

### 2.5 Split handling

An optional second ID file was added so training can pool `train.txt + val.txt`. Explicit missing ID lists now raise clear errors. Deduplication preserves order.

### 2.6 Memory management

Main eagerly normalizes and stores all dynamic runs and uses per-flood NPZ caches.

This version:

- stores float32 NPY arrays;
- memory-maps large WD/volume arrays;
- keeps a small LRU of open flood maps;
- normalizes only requested windows;
- streams any missing normalization statistics in chunks;
- stores only ID, retained row count, and related metadata in `dynamic_data`.

### 2.7 Normalization consistency and cache identity

The original Aji-4 JSON may contain White River-style `inflow_hydrograph` but not the two required `release`/`inflow` keys. The first training run computes missing fields and writes a complete training mapping. Test data reuses that mapping instead of deriving test normalization.

Cache keys include onset mode, interval, and `delta_t` so changing time spacing does not silently reuse incompatible inflow interpolation.

### 2.8 What stayed the same

- fast text parsing;
- z-score normalization convention;
- normalized-XY four-nearest-neighbor graph;
- three edge features `[Δx, Δy, distance]`;
- `K` previous depth and volume states;
- target-row forcing alignment;
- normalized two-component increment target;
- random-walk/last-step/no-noise options;
- drift-corrected noise targets;
- train-window indexing and rollout packaging logic apart from forcing keys.

## 3. `train.py` changes

Necessary Aji-4 changes:

- default config is `configs/aji4_15.yaml`;
- `--data_dir` accepts a G1O/G12O/G123O folder;
- optional `--test_dir`; otherwise test IDs come from the same folder;
- all listed floods are used by default;
- `val.txt` is pooled into optimization while `test.txt` remains separate;
- onset mode, `delta_t`, and mmap size are passed to datasets;
- startup logging reports 15 features and Aji-4 onset/window counts;
- evaluation defaults use 1,800 seconds per row.

Full-run reliability changes, without altering model mathematics:

- batch 1 + gradient accumulation 8 is the default memory-conservative effective batch;
- scaler and non-scaler AMP paths both accumulate correctly;
- cosine scheduler counts optimizer updates rather than microbatches;
- a final partial accumulation group is stepped rather than discarded;
- `best.pt` compares only test-rollout RMSE values, never training loss against RMSE;
- processor checkpointing is enabled in the config to reduce activation memory.

The following remain from Main: seed setup, device selection, DDP structure, DataLoader/collation, AdamW, AMP selection, gradient clipping, resume structure, epoch history, and periodic `evaluate_dataset` calls.

## 4. `evaluate.py` changes

- Reads `release` and `inflow`; no precipitation and no hour sequence.
- Constructs two forcing columns rather than Main’s two semantically different White River columns.
- Static slice changes from `:10` to `:9`.
- History slices change from `12...` to `11...`.
- Rollout length is obtained from the release sequence.
- Default `delta_t` is 1,800 seconds.
- Adds persistence and zero-depth RMSE baselines and reference lines on the RMSE plot.
- Supports portable `--data_dir`/`--test_dir` overrides.
- `--split train` now actually selects `train.txt`; rollout packaging still uses dataset test mode.

The actual recurrence is unchanged:

```text
predict normalized increments
→ add to latest depth/volume
→ clamp at physical zero
→ shift history
→ repeat
→ denormalize and evaluate
```

PNG/GIF generation remains essentially Main’s implementation.

## 5. Configuration and dependency changes

`configs/aji4_15.yaml` is a complete recipe:

- 150 full epochs;
- no maximum-steps cap;
- all train+val/test IDs;
- input width 15;
- original hidden width/processor/loss/noise settings;
- 30-minute time step and 60-step/30-hour evaluation;
- conservative memory settings that can be changed for larger GPUs.

`requests` and `remotezip` were removed because the White River downloader was removed. Torch, NumPy, SciPy, YAML, tqdm, and Matplotlib remain.

## 6. Explicitly not included

- subset/sample configs;
- subset-building tools;
- elapsed-hour inputs;
- precipitation or infiltration inputs;
- KAN layers or PhysicsNeMo dependencies;
- architecture changes;
- new physics losses;
- raw datasets or generated caches;
- old White River checkpoints/results.

The zip is source code and documentation for running on your own complete Aji-4 data and compute.
