"""Phase 2: Train MPDR energy function (MPDR-S or MPDR-R) with pre-trained AE."""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from config import (
    ExperimentConfig, create_mpdr_simple_config, create_mpdr_recovery_config,
)
from dataset.dataset import NAEDatasetSimple
from models.ae import TransformerEncoder, TransformerDecoder
from models.mpdr_wrapper import VariableLengthAE, MPDR_VariableLength
from models.lightning_modules import MPDRModule
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

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.ascii = True
        return bar


def create_dataloaders(config):
    """Create train, validation, and test dataloaders."""
    common = dict(
        collate_fn=collate_variable_length, pin_memory=True,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.num_workers > 0,
    )

    train_ds = NAEDatasetSimple(
        path=config.data.data_path, mode=config.data.train_mode,
        time_range=config.data.time_range, augmentations=config.data.augmentations,
    )
    train_ds.set_feature_mode(config.data.feature_mode)

    val_ds = NAEDatasetSimple(
        path=config.data.data_path, mode=config.data.test_mode,
        time_range=config.data.time_range, augmentations=False,
    )
    val_ds.set_feature_mode(config.data.feature_mode)

    val_size = min(len(val_ds), len(train_ds) // 5)
    val_idx = torch.randperm(
        len(val_ds), generator=torch.Generator().manual_seed(config.training.seed),
    )[:val_size]
    val_subset = torch.utils.data.Subset(val_ds, val_idx)

    test_ds = NAEDatasetSimple(
        path=config.data.data_path, mode=config.data.test_mode,
        time_range=config.data.time_range, augmentations=False,
    )
    test_ds.set_feature_mode(config.data.feature_mode)

    train_loader = DataLoader(
        train_ds, batch_size=config.data.batch_size, shuffle=True, **common,
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.data.batch_size, shuffle=False, **common,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.data.batch_size, shuffle=False, **common,
    )
    return train_loader, val_loader, test_loader


def create_mpdr_model(config, device):
    """Build the full MPDR model from config."""
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
        spherical=config.model.spherical, encoding_noise=None,
        loss=config.model.ae_loss,
        alphas=tuple(config.mpdr.alphas),
        repulsion_weight=repulsion_weight,
    )

    if config.mpdr.variant == "simple":
        energy_net = TransformerEncoder(
            input_dim=feat_dim, latent_dim=config.model.energy_hidden_dim,
            num_heads=config.model.energy_num_heads,
            num_layers=config.model.energy_num_layers,
            dropout=config.model.energy_dropout,
            energy_mode=True,
            energy_mlp_hidden=config.model.encoder_energy_mlp_hidden,
            pool="topk_logmeanexp", pool_tau=0.1,
        )
        energy_ae = None
    else:
        energy_encoder = TransformerEncoder(
            input_dim=feat_dim, latent_dim=config.model.encoder_latent_dim,
            num_heads=config.model.encoder_num_heads,
            num_layers=config.model.encoder_num_layers,
            dropout=config.model.encoder_dropout,
            energy_mode=False,
            energy_mlp_hidden=config.model.encoder_energy_mlp_hidden,
            pool="cls",
        )
        energy_decoder = TransformerDecoder(
            latent_dim=config.model.encoder_latent_dim, output_dim=feat_dim,
            num_output_points=config.model.max_output_points,
            num_heads=config.model.decoder_num_heads,
            num_layers=config.model.decoder_num_layers,
            dropout=config.model.decoder_dropout,
            memory_tokens=config.model.decoder_memory_tokens,
        )
        energy_ae = VariableLengthAE(
            encoder=energy_encoder, decoder=energy_decoder,
            spherical=config.model.spherical, encoding_noise=None,
            loss=config.model.ae_loss,
            alphas=tuple(config.mpdr.alphas), repulsion_weight=0.0,
        )
        energy_net = None

    mpdr = MPDR_VariableLength(
        ae=ae, net_x=energy_net, energy_ae=energy_ae,
        variant=config.mpdr.variant,
        physics_noise_ratio=config.mpdr.physics_noise_ratio,
        alphas=tuple(config.mpdr.alphas),
        proj_mode=config.mpdr.proj_mode,
        proj_noise_start=config.mpdr.proj_noise_start,
        proj_noise_end=config.mpdr.proj_noise_end,
        proj_const=config.mpdr.proj_const,
        proj_const_omi=config.mpdr.proj_const_omi,
        proj_dist=config.mpdr.proj_dist,
        mcmc_n_step_omi=config.mpdr.mcmc_n_step_omi,
        mcmc_stepsize_omi=config.mpdr.mcmc_stepsize_omi,
        mcmc_noise_omi=config.mpdr.mcmc_noise_omi,
        mcmc_normalize_omi=config.mpdr.mcmc_normalize_omi,
        mh_omi=config.mpdr.mh_omi,
        mcmc_n_step_x=config.mpdr.mcmc_n_step_x,
        mcmc_stepsize_x=config.mpdr.mcmc_stepsize_x,
        mcmc_noise_x=config.mpdr.mcmc_noise_x,
        mcmc_bound_x=(
            tuple(config.mpdr.mcmc_bound_x)
            if isinstance(config.mpdr.mcmc_bound_x, list)
            else config.mpdr.mcmc_bound_x
        ),
        mcmc_custom_stepsize=config.mpdr.mcmc_custom_stepsize,
        mh_x=config.mpdr.mh_x,
        temperature=config.mpdr.temperature,
        temperature_omi=config.mpdr.temperature_omi,
        gamma_vx=config.mpdr.gamma_vx,
        gamma_neg_recon=config.mpdr.gamma_neg_recon,
        l2_norm_reg_netx=config.mpdr.l2_norm_reg_netx,
        lambda_center=config.mpdr.lambda_center,
        tau=config.mpdr.tau,
        energy_method=config.mpdr.energy_method,
        grad_clip_omi=config.mpdr.grad_clip_omi,
        grad_clip_off=config.mpdr.grad_clip_off,
    )
    return mpdr.to(device)


def main(args):
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    elif args.variant == "simple":
        config = create_mpdr_simple_config()
    else:
        config = create_mpdr_recovery_config()

    if args.feature_mode:
        config.data.feature_mode = args.feature_mode
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.device:
        config.training.device = args.device
    if args.epochs:
        config.training.phase2_epochs = args.epochs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.name = args.name or config.name or f"mpdr_{args.variant}_{timestamp}"
    if args.output_dir:
        config.output_dir = args.output_dir

    pl.seed_everything(config.training.seed)
    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    device = torch.device(config.training.device)
    mpdr = create_mpdr_model(config, device)

    ae_params = sum(p.numel() for p in mpdr.ae.parameters())
    e_params = (
        sum(p.numel() for p in mpdr.net_x.parameters())
        if args.variant == "simple"
        else sum(p.numel() for p in mpdr.energy_ae.parameters())
    )
    print(f"AE params: {ae_params:,}, Energy params: {e_params:,}")

    ae_ckpt = args.ae_checkpoint if args.ae_checkpoint else None
    if ae_ckpt is None:
        print("WARNING: No AE checkpoint provided. Using random initialisation.")
    module = MPDRModule(mpdr, config, ae_checkpoint_path=ae_ckpt)

    callbacks = []
    if args.max_steps is None:
        callbacks.extend([
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename="mpdr-{epoch:02d}-{val/auroc:.4f}",
                monitor="val/auroc", mode="max", save_top_k=20, save_last=True,
            ),
            EarlyStopping(
                monitor="val/auroc",
                patience=config.training.early_stopping_patience,
                mode="max", verbose=True,
            ),
        ])
    else:
        callbacks.append(ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="mpdr-{step:05d}", save_last=True,
            every_n_train_steps=args.max_steps,
        ))
    callbacks.extend([
        LearningRateMonitor(logging_interval="epoch"),
        CustomProgressBar(),
    ])

    trainer_kwargs = dict(
        max_epochs=config.training.phase2_epochs,
        accelerator="gpu" if "cuda" in config.training.device else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=output_dir, name="logs", version=""),
        gradient_clip_val=config.training.clip_grad if config.training.clip_grad else None,
        gradient_clip_algorithm="norm",
        log_every_n_steps=50,
        precision="bf16-mixed",
        deterministic=False,
    )
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps

    trainer = pl.Trainer(**trainer_kwargs)

    print(f"\n{'=' * 80}\nTRAINING MPDR-{args.variant.upper()} (Phase 2)\n{'=' * 80}")

    if args.max_steps is not None:
        trainer.fit(module, train_loader)
    else:
        trainer.fit(module, train_loader, val_loader)
        print(f"\n{'=' * 80}\nFINAL EVALUATION\n{'=' * 80}")
        trainer.test(module, test_loader, ckpt_path="best")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Train MPDR Energy Function")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--variant", type=str, default="simple",
                        choices=["simple", "recovery"])
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--ae_checkpoint", type=str, required=True,
                        help="Path to pre-trained AE checkpoint from Phase 1")
    parser.add_argument("--feature_mode", type=str, default=None,
                        choices=["all", "no_time", "no_charge", "no_time_no_charge"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max training steps (for parameter sweeps)")
    main(parser.parse_args())
