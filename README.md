# HydroGraphNet — Aji-4, full-data 15-feature edition

A minimal-change adaptation of [`sidhu2690/HydroGraphNet`](https://github.com/sidhu2690/HydroGraphNet) for the Aji-4 dam HEC-RAS datasets.

- Same MLP-MeshGraphNet architecture, graph backend, loss, and metrics as the main repository.
- **15 node inputs:** 9 static + 2 forcings + 4 history values.
- **No elapsed-hour feature.**
- **No precipitation or infiltration input.** Both are zero/unused in Aji-4.
- Uses both dam release (`US_InF`) and reservoir inflow (`reservoir_inflow_hourly.csv`).
- Full-data recipe: all `train.txt + val.txt` floods, no subset and no per-epoch step cap.
- Works with G1O, G12O, or G123O by changing `--data_dir`.

See [`changed.md`](changed.md) for the exact main-repository comparison and [`DATASET.md`](DATASET.md) for the data layout.

## 1. Install

Python 3.10+ is recommended.

```bash
cd HydroGraphNet_Aji4_15_Full
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your GPU if required. The package itself does not require PhysicsNeMo, DGL, PyG, `torch_scatter`, or other graph extensions.

## 2. Data folder

Keep the extracted gate-case folder intact. For example:

```text
/path/to/aji4_flood_data_G1O/
├── M80_XY.txt
├── M80_CA.txt
├── M80_CE.txt
├── ...
├── M80_WD_R001.txt
├── M80_V_R001.txt
├── M80_US_InF_R001.txt
├── reservoir_inflow_hourly.csv
├── train.txt
├── val.txt
└── test.txt
```

You do **not** need to rename the `M80` files. That prefix is retained for compatibility with the original loader.

## 3. Full training

### G1O — one gate group open

```bash
python train.py --data_dir /path/to/aji4_flood_data_G1O
```

### G12O — two gate groups open

Only the folder changes:

```bash
python train.py --data_dir /path/to/aji4_flood_data_G12O
```

### G123O — all gate groups open

```bash
python train.py --data_dir /path/to/aji4_flood_data_G123O
```

By default, training uses:

- all 195 `train.txt` + 39 `val.txt` floods = **234 optimization floods**;
- all 38 `test.txt` floods for periodic rollout evaluation;
- **150 complete epochs**, with `max_steps_per_epoch: null`;
- the full 64-wide, five-processor-block, **214,146-parameter** model;
- effective batch 8: `batch_size: 1 × grad_accum_steps: 8`;
- CUDA AMP when available;
- automatic CUDA/MPS/CPU device selection.

The first run creates `.cache_npy/` inside the gate-case folder. Keep enough disk space for the raw package plus roughly 5–7 GB of cache. Subsequent runs reuse it.

### Multi-GPU

```bash
torchrun --nproc_per_node=4 train.py --data_dir /path/to/aji4_flood_data_G1O
```

### Resume

```bash
python train.py --data_dir /path/to/aji4_flood_data_G1O --resume
```

Checkpoints:

```text
checkpoints_aji4_15/latest.pt   # latest model/optimizer/scheduler state
checkpoints_aji4_15/best.pt     # lowest evaluated test-rollout RMSE
```

Outputs and epoch history go to `outputs_aji4_15/`.

## 4. Evaluate

The default evaluates every test flood for 60 steps = **30 hours** at 30 minutes per step:

```bash
python evaluate.py \
  --data_dir /path/to/aji4_flood_data_G1O \
  --ckpt checkpoints_aji4_15/best.pt
```

To evaluate G12O, change both paths consistently:

```bash
python evaluate.py \
  --data_dir /path/to/aji4_flood_data_G12O \
  --ckpt checkpoints_aji4_15/best.pt
```

Optional 20-hour evaluation:

```bash
python evaluate.py --data_dir /path/to/aji4_flood_data_G1O --rollout_steps 40
```

Add `--save_gif` for animations. `--split train` evaluates `train.txt` IDs in rollout mode.

Outputs:

- `evaluation_metrics.json` — aggregate, per-flood, per-step, persistence, and zero-forecast metrics;
- `rmse_over_time.png` — mean rollout RMSE by lead time;
- `rollout_<ID>.png` — prediction, HEC-RAS ground truth, absolute error, and RMSE curve for every evaluated flood;
- optional `animation_<ID>.gif`.

## 5. Model input/output summary

At `K=2` history frames:

```text
x[:, 0:9]    normalized static cell attributes
x[:, 9]      normalized dam release at target row
x[:, 10]     normalized reservoir inflow at target row
x[:, 11:13]  normalized water-depth history
x[:, 13:15]  normalized cell-volume history

y[:, 0]      normalized next water-depth increment
y[:, 1]      normalized next cell-volume increment
```

The model is autoregressive only in water depth and volume. Future release and reservoir-inflow forcings are read from the dataset.

## 6. Resource tuning

The default is deliberately conservative for a 15,990-node/63,960-edge graph.

- More VRAM: increase `batch_size`, reduce `grad_accum_steps`, and optionally set `checkpoint_segments: 0`.
- Less VRAM: keep batch 1, increase `grad_accum_steps`, and retain `checkpoint_segments: 1`.
- `num_workers: 0` can help memory-constrained systems.
- Keep `batch_size × grad_accum_steps` constant when comparing effective-batch experiments.

This package contains no sampling configuration or subset-building tool; the supplied recipe is for complete Aji-4 folders.
