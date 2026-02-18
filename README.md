# Deep Learning Trigger for Hyper-Kamiokande

Code accompanying the paper *Deep-learning-based low-energy trigger algorithms\\for the Hyper-Kamiokande experiment*.

Two complementary approaches for event classification in a large water-Cherenkov detector:

- **Supervised transformer classifier** — trained on labelled simulation to distinguish signal from background events.
- **Anomaly detection (MPDR)** — Manifold Projection–Diffusion Recovery, an energy-based model that learns the background distribution and flags anomalous events without requiring signal labels during training.

---

## Repository Structure

```
hk-ml-trigger/
├── supervised/                 # Supervised transformer classifier
│   ├── model/
│   │   ├── model.py            # TransformerClassifier with relative positional attention
│   │   ├── lightning_model.py  # PyTorch Lightning wrapper
│   │   └── __init__.py
│   ├── dataset/
│   │   ├── dataset.py          # HKDataset with Fourier feature encoding
│   │   └── __init__.py
│   ├── utils/
│   │   ├── scheduler.py        # Warmup + cosine annealing schedulers
│   │   └── __init__.py
│   ├── train/
│   │   └── train.py            # Training entry point
│   ├── train_event_level.sh    # Event-level classification
│   └── train_hit_level.sh      # Hit-level (per-PMT) classification
│
├── anomaly_detection/          # MPDR anomaly detection
│   ├── models/
│   │   ├── ae.py               # Transformer autoencoder (encoder + decoder)
│   │   ├── mpdr_wrapper.py     # MPDR-S and MPDR-R wrappers
│   │   ├── mcmc.py             # Langevin Monte Carlo samplers
│   │   ├── sampler.py          # Empirical K-sampler for variable-length data
│   │   ├── loss.py             # Multi-scale probabilistic Chamfer distance
│   │   ├── lightning_modules.py
│   │   ├── utils.py            # Weight norm utility
│   │   └── __init__.py
│   ├── dataset/
│   │   ├── dataset.py          # NAEDatasetSimple with unit-sphere projection
│   │   └── __init__.py
│   ├── configs/                # YAML configuration files
│   │   ├── mpdr_simple.yaml
│   │   ├── mpdr_simple_signal.yaml
│   │   ├── mpdr_recovery.yaml
│   │   └── mpdr_recovery_signal.yaml
│   ├── scripts/                # Training shell scripts
│   │   ├── train_phase1.sh
│   │   ├── train_mpdr_simple.sh
│   │   └── train_mpdr_recovery.sh
│   ├── config.py               # Dataclass-based experiment configuration
│   ├── train_phase1.py         # Phase 1: autoencoder pre-training
│   ├── train_phase2.py         # Phase 2: energy function training
│   └── utils.py                # Collate function and utilities
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

Python >= 3.10 with CUDA-capable GPU (tested on NVIDIA A100 and H100).

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.5.0
- PyTorch Lightning >= 2.4.0
- timm >= 1.0.0
- scikit-learn >= 1.6.0

---

## Data Preparation

Both approaches expect WCSim simulation output stored as per-event `.npz` files with the following arrays:

| Key        | Shape  | Description                          |
|------------|--------|--------------------------------------|
| `fPosX`    | (N,)   | PMT x-coordinates (mm)              |
| `fPosY`    | (N,)   | PMT y-coordinates (mm)              |
| `fPosZ`    | (N,)   | PMT z-coordinates (mm)              |
| `fCharge`  | (N,)   | Measured charge (p.e.)               |
| `fTime`    | (N,)   | Hit time (ns)                        |
| `fLoc`     | (N,)   | PMT location index                   |
| `fPMTFlag` | (N,)   | 0 = background (dark noise), 1 = signal |

Organise files under a single root directory; the data loaders search recursively for `*.npz`.

---

## Supervised Classifier

A transformer encoder with relative positional self-attention operating on cylindrical PMT coordinates encoded via Fourier features. Two classification modes are supported:

- **Event-level** (default) — classifies the entire event as signal or background using the CLS token representation.
- **Hit-level** (`--token_level`) — produces a per-PMT-hit label, classifying each hit individually as signal (Cherenkov) or background (dark noise).

### Training

```bash
cd supervised

# Event-level classifier
python train/train.py \
    --root_path /path/to/wcsim_numpy \
    --feature_mode no_time_no_charge \
    --batch_size 256 \
    --epochs 50 \
    --gpus 0 1

# Hit-level classifier
python train/train.py \
    --root_path /path/to/wcsim_numpy \
    --feature_mode no_time_no_charge \
    --batch_size 256 \
    --epochs 50 \
    --gpus 0 1 \
    --token_level
```

#### Arguments

| Argument           | Default              | Description                                              |
|--------------------|---------------------|----------------------------------------------------------|
| `--root_path`      | *(required)*         | Root directory containing `.npz` data files              |
| `--feature_mode`   | `no_time_no_charge`  | Feature set: `all`, `no_time`, `no_charge`, `no_time_no_charge` |
| `--token_level`    | `False`              | Enable hit-level classification (per-PMT labels)         |
| `--batch_size`     | `256`                | Batch size per GPU                                       |
| `--epochs`         | `50`                 | Maximum training epochs                                  |
| `--gpus`           | `0`                  | GPU IDs (multiple GPUs enable DDP)                       |
| `--lr`             | `1e-4`               | Learning rate (scaled by effective batch size)            |
| `--d_model`        | `192`                | Transformer embedding dimension                          |
| `--nhead`          | `12`                 | Number of attention heads                                |
| `--num_layers`     | `12`                 | Number of transformer layers                             |
| `--dropout`        | `0.1`                | Dropout rate                                             |
| `--num_workers`    | `16`                 | Data loader workers                                      |
| `--train_split`    | `0.95`               | Fraction of data used for training                       |
| `--warmup_steps`   | `1`                  | Warmup epochs                                            |
| `--compile`        | `False`              | Use `torch.compile` (PyTorch 2.0+)                       |

### Quick Start

```bash
# Event-level classifier
bash train_event_level.sh

# Hit-level classifier
bash train_hit_level.sh
```

---

## Anomaly Detection (MPDR)

Training proceeds in two phases:

1. **Phase 1** — Pre-train a normalising autoencoder that maps variable-length point clouds to a fixed-dimensional latent space on the unit hypersphere.
2. **Phase 2** — Train an energy function via contrastive divergence using Manifold Projection–Diffusion Recovery (MPDR).

Two energy-function variants are supported:
- **MPDR-S** (Simple): a separate scalar energy network.
- **MPDR-R** (Recovery): reconstruction-based energy from a second autoencoder.

### Phase 1: Autoencoder Pre-training

```bash
cd anomaly_detection

python train_phase1.py \
    --config configs/mpdr_simple.yaml \
    --name ae_phase1 \
    --output_dir experiments \
    --gpus 0,1
```

#### Phase 1 Arguments

| Argument         | Default        | Description                                    |
|------------------|---------------|------------------------------------------------|
| `--config`       | `None`         | Path to YAML configuration file                |
| `--name`         | auto-generated | Experiment name                                |
| `--output_dir`   | `experiments`  | Output directory for checkpoints and logs       |
| `--feature_mode` | from config    | Override feature mode                           |
| `--batch_size`   | from config    | Override batch size                             |
| `--epochs`       | from config    | Override number of epochs                       |
| `--device`       | from config    | `cuda` or `cpu`                                 |
| `--gpus`         | from config    | Comma-separated GPU IDs (e.g. `0,1`)           |

### Phase 2: Energy Function Training

```bash
python train_phase2.py \
    --variant simple \
    --config configs/mpdr_simple.yaml \
    --ae_checkpoint experiments/ae_phase1/checkpoints/last.ckpt \
    --name mpdr_simple \
    --output_dir experiments
```

#### Phase 2 Arguments

| Argument           | Default        | Description                                      |
|--------------------|---------------|--------------------------------------------------|
| `--config`         | `None`         | Path to YAML configuration file                  |
| `--variant`        | `simple`       | `simple` (MPDR-S) or `recovery` (MPDR-R)        |
| `--ae_checkpoint`  | *(required)*   | Path to pre-trained AE checkpoint from Phase 1   |
| `--name`           | auto-generated | Experiment name                                  |
| `--output_dir`     | `experiments`  | Output directory                                 |
| `--feature_mode`   | from config    | Override feature mode                            |
| `--batch_size`     | from config    | Override batch size                              |
| `--epochs`         | from config    | Override number of epochs                        |
| `--device`         | from config    | `cuda` or `cpu`                                  |
| `--max_steps`      | `None`         | Limit training steps (for parameter sweeps)      |

### Configuration

All hyperparameters are specified in YAML files under `anomaly_detection/configs/`. The four provided configurations are:

| File                         | Variant | Training mode | Description                       |
|------------------------------|---------|---------------|-----------------------------------|
| `mpdr_simple.yaml`           | MPDR-S  | background    | Train on background, detect signal as anomaly |
| `mpdr_simple_signal.yaml`    | MPDR-S  | signal        | Train on signal, detect background as anomaly |
| `mpdr_recovery.yaml`         | MPDR-R  | background    | Reconstruction-based energy (background)      |
| `mpdr_recovery_signal.yaml`  | MPDR-R  | signal        | Reconstruction-based energy (signal)          |

Update `data.data_path` in the YAML files to point to your data directory.

### Quick Start

```bash
cd anomaly_detection

# Phase 1: train autoencoder
bash scripts/train_phase1.sh

# Phase 2: train MPDR-S energy function
bash scripts/train_mpdr_simple.sh experiments/ae_phase1_*/checkpoints/last.ckpt

# Or MPDR-R
bash scripts/train_mpdr_recovery.sh experiments/ae_phase1_*/checkpoints/last.ckpt
```

---

## Monitoring

Both approaches log to TensorBoard:

```bash
tensorboard --logdir experiments/
```

---

## Citation

```bibtex
@article{hk_ml_trigger,
    title   = {Deep-learning-based low-energy trigger algorithms\\for the Hyper-Kamiokande experiment},
    year    = {2026},
}
```

## License

This project is released under the [MIT License](LICENSE).
