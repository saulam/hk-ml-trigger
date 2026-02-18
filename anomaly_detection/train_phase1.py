"""Phase 1: Train autoencoder (shared for both MPDR-S and MPDR-R)."""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from config import ExperimentConfig
from dataset.dataset import NAEDatasetSimple
from models.ae import TransformerEncoder, TransformerDecoder
from models.mpdr_wrapper import VariableLengthAE
from models.lightning_modules import AutoencoderModule
from utils import collate_variable_length

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True
        return bar


def create_dataloaders(config):
    """Create train/val dataloaders with separate augmentation policies."""
    full_dataset = NAEDatasetSimple(
        path=config.data.data_path, mode=config.data.train_mode,
        time_range=config.data.time_range, augmentations=False,
    )
    full_dataset.set_feature_mode(config.data.feature_mode)

    indices = list(range(len(full_dataset)))
    random.seed(config.training.seed)
    random.shuffle(indices)
    train_size = int(config.data.train_split * len(full_dataset))
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_ds = NAEDatasetSimple(
        path=config.data.data_path, mode=config.data.train_mode,
        time_range=config.data.time_range, augmentations=config.data.augmentations,
    )
    train_ds.set_feature_mode(config.data.feature_mode)

    val_ds = NAEDatasetSimple(
        path=config.data.data_path, mode=config.data.train_mode,
        time_range=config.data.time_range, augmentations=False,
    )
    val_ds.set_feature_mode(config.data.feature_mode)

    common = dict(
        collate_fn=collate_variable_length, pin_memory=True,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.num_workers > 0,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=config.data.batch_size,
        shuffle=True, drop_last=True, **common,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=config.data.batch_size,
        shuffle=False, **common,
    )
    return train_loader, val_loader


def create_autoencoder(config, use_compile=True):
    """Instantiate the VariableLengthAE from config."""
    dummy = NAEDatasetSimple(
        path=config.data.data_path, mode="background", augmentations=False,
    )
    dummy.set_feature_mode(config.data.feature_mode)
    feat_dim = dummy.get_feature_size()

    encoder = TransformerEncoder(
        input_dim=feat_dim, latent_dim=config.model.encoder_latent_dim,
        num_heads=config.model.encoder_num_heads,
        num_layers=config.model.encoder_num_layers,
        dropout=config.model.encoder_dropout,
        energy_mode=False,
        energy_mlp_hidden=config.model.encoder_energy_mlp_hidden,
        pool="cls",
    )
    decoder = TransformerDecoder(
        latent_dim=config.model.encoder_latent_dim, output_dim=feat_dim,
        num_output_points=config.model.max_output_points,
        num_heads=config.model.decoder_num_heads,
        num_layers=config.model.decoder_num_layers,
        dropout=config.model.decoder_dropout,
        memory_tokens=config.model.decoder_memory_tokens,
    )

    repulsion_weight = 0.0 if config.data.train_mode == "signal" else config.model.repulsion_weight

    ae = VariableLengthAE(
        encoder=encoder, decoder=decoder,
        spherical=config.model.spherical,
        encoding_noise=config.model.encoding_noise,
        loss=config.model.ae_loss,
        alphas=tuple(config.training.alphas),
        repulsion_weight=repulsion_weight,
    )

    if use_compile:
        ae = torch.compile(ae, mode="max-autotune")
    return ae


def main(args):
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        from config import create_mpdr_simple_config
        config = create_mpdr_simple_config()

    gpu_ids = config.training.gpu_ids
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
        config.training.gpu_ids = gpu_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    n_gpus = len(gpu_ids)

    if args.feature_mode:
        config.data.feature_mode = args.feature_mode
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.device:
        config.training.device = args.device
    if args.epochs:
        config.training.phase1_epochs = args.epochs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.name = args.name or f"ae_phase1_{config.data.feature_mode}_{timestamp}"
    if args.output_dir:
        config.output_dir = args.output_dir

    pl.seed_everything(config.training.seed)
    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    use_compile = n_gpus == 1
    ae = create_autoencoder(config, use_compile=use_compile)
    print(f"AE parameters: {sum(p.numel() for p in ae.parameters()):,}")

    if n_gpus > 1:
        config.training.ae_lr *= n_gpus
        print(f"Scaled LR for {n_gpus} GPUs: {config.training.ae_lr}")

    module = AutoencoderModule(ae, config)

    trainer = pl.Trainer(
        max_epochs=config.training.phase1_epochs,
        accelerator="gpu" if "cuda" in config.training.device else "cpu",
        devices=n_gpus,
        strategy="ddp" if n_gpus > 1 else "auto",
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename="ae-{epoch:02d}-{val/loss:.4f}",
                monitor="val/loss", mode="min", save_top_k=3, save_last=True,
            ),
            EarlyStopping(
                monitor="val/loss", patience=config.training.early_stopping_patience,
                mode="min", verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            CustomProgressBar(),
        ],
        logger=TensorBoardLogger(save_dir=output_dir, name="logs", version=""),
        gradient_clip_val=config.training.clip_grad if config.training.clip_grad else None,
        log_every_n_steps=50,
        precision="bf16-mixed",
        deterministic=False,
    )

    print(f"\n{'=' * 80}\nTRAINING AUTOENCODER (Phase 1)\n{'=' * 80}")
    trainer.fit(module, train_loader, val_loader)
    print(f"\nBest model: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Train Autoencoder")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--feature_mode", type=str, default=None,
                        choices=["all", "no_time", "no_charge", "no_time_no_charge"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1'")
    main(parser.parse_args())
