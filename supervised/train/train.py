import os
import sys
import copy
import torch
import pytorch_lightning as pl
import argparse
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset import HKDataset
from model import LitTransformerClassifier
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True
        return bar


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Classifier")

    parser.add_argument("--root_path", type=str, required=True,
                        help="Root path to the dataset")
    parser.add_argument("--data_pattern", type=str, default="/*signoise*",
                        help="Glob pattern for data files")

    parser.add_argument("--log_dir", type=str, default="logs/logs",
                        help="Directory for CSV logs")
    parser.add_argument("--tb_log_dir", type=str, default="logs/tb_logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--log_name", type=str, default="hk_run",
                        help="Name for logging (appended with feature_mode)")

    parser.add_argument("--feature_mode", type=str, default="no_time_no_charge",
                        choices=["all", "no_time", "no_charge", "no_time_no_charge"],
                        help="Feature mode to use")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--train_split", type=float, default=0.95)
    parser.add_argument("--warmup_steps", type=int, default=1)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--gpus", nargs="*", default=[0],
                        help="List of GPUs to use (multiple GPUs enable DDP)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--token_level", action="store_true", default=False,
                        help="Per-hit classification instead of per-event")

    return parser.parse_args()


def collate_fn(batch):
    """Pad variable-length hits to the maximum length in the batch."""
    feats_list = [item["feats"] for item in batch]
    locs_list = [item["loc"] for item in batch]
    coords_list = [item["coords"] for item in batch]
    times_list = [item["times"] for item in batch]
    pmt_flag_list = [item["pmt_flag"] for item in batch]
    lengths = [f.size(0) for f in feats_list]
    labels = torch.stack([item["label"] for item in batch], dim=0)

    padded_feats = pad_sequence(feats_list, batch_first=True, padding_value=0.0)
    padded_locs = pad_sequence(locs_list, batch_first=True, padding_value=0)
    padded_coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    padded_times = pad_sequence(times_list, batch_first=True, padding_value=0.0)
    padded_pmt_flag = pad_sequence(pmt_flag_list, batch_first=True, padding_value=0)

    B, L, _ = padded_feats.shape
    mask = torch.zeros((B, L + 2), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, length + 2:] = True

    return {
        "feats": padded_feats,
        "loc": padded_locs,
        "coords": padded_coords,
        "times": padded_times,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "mask": mask,
        "label": labels.squeeze(1),
        "pmt_flag": padded_pmt_flag,
    }


def main():
    args = parse_args()

    nb_gpus = len(args.gpus)
    gpus = ", ".join(str(g) for g in args.gpus)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    dataset_path = args.root_path + args.data_pattern
    dataset = HKDataset(dataset_path)
    dataset.set_feature_mode(args.feature_mode)
    feature_size = dataset.get_feature_size()
    print(f"Feature mode: {args.feature_mode}, feature size: {feature_size}")

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    def extract_files(indices):
        return [dataset.data_files[i] for i in indices]

    train_set, val_set = (copy.deepcopy(dataset) for _ in range(2))
    train_set.data_files = extract_files(train_split.indices)
    train_set.augmentations = True
    train_set.set_feature_mode(args.feature_mode)
    val_set.data_files = extract_files(val_split.indices)
    val_set.set_feature_mode(args.feature_mode)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    accum_grad_batches = 1
    warmup_steps = args.warmup_steps
    scheduler_steps = args.epochs - warmup_steps

    log_name = f"{args.log_name}_{args.feature_mode}"
    logger = CSVLogger(save_dir=args.log_dir, name=log_name)
    tb_logger = TensorBoardLogger(save_dir=args.tb_log_dir, name=log_name)
    progress_bar = CustomProgressBar()
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, log_name),
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        save_last=True,
    )

    denom = accum_grad_batches * nb_gpus
    scheduler_steps = len(train_loader) * scheduler_steps // denom
    lr = args.lr * (args.batch_size * denom) / 256.0

    model = LitTransformerClassifier(
        dropout=args.dropout,
        feature_size=feature_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        lr=lr,
        betas=(0.9, 0.95),
        warmup_steps=len(train_loader) * warmup_steps // denom,
        start_cosine_step=(len(train_loader) * args.epochs // denom) - scheduler_steps,
        cosine_annealing_steps=scheduler_steps,
        token_level=args.token_level,
        feature_mode=args.feature_mode,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            model.model = torch.compile(model.model, mode="default")
            print("  Compilation successful")
        except Exception as e:
            print(f"  Compilation failed: {e}. Continuing without compilation.")

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=nb_gpus,
        precision="bf16-mixed",
        strategy="ddp" if nb_gpus > 1 else None,
        logger=[logger, tb_logger],
        callbacks=[checkpoint, progress_bar],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
