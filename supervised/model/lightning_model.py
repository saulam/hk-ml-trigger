import torch
import torch.nn as nn
import pytorch_lightning as pl
from model import TransformerClassifier
from utils import CustomLambdaLR, CombinedScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class LitTransformerClassifier(pl.LightningModule):
    """PyTorch Lightning module for training the TransformerClassifier."""

    def __init__(
        self,
        feature_size: int = 4,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 5,
        num_classes: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        warmup_steps=-1,
        start_cosine_step=-1,
        cosine_annealing_steps=-1,
        token_level: bool = False,
        feature_mode: str = "all",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerClassifier(
            feature_size=feature_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            token_level=token_level,
            feature_mode=feature_mode,
        )
        reduction = "none" if token_level else "mean"
        if num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, feats, coords, times, loc, lengths, mask):
        return self.model(feats, coords, times, loc, lengths, src_key_padding_mask=mask)

    def _shared_step(self, batch):
        feats, mask = batch["feats"], batch["mask"]
        loc, coords, times = batch["loc"], batch["coords"], batch["times"]
        lengths = batch["lengths"]

        logits = self(feats, coords, times, loc, lengths, mask)

        if self.hparams.token_level:
            logits = logits.squeeze(-1)
            token_mask = mask[:, 2:]
            labels = batch["pmt_flag"]
            B, L = labels.shape
            loss_raw = self.criterion(logits.view(B, L), labels.float())
            loss = (loss_raw * (~token_mask).float()).sum() / (~token_mask).float().sum()
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(-1)
            acc = (preds[~token_mask] == labels[~token_mask]).float().mean()
        else:
            labels = batch["label"]
            if logits.size(-1) == 1:
                loss = self.criterion(logits.view(-1), labels.float())
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
            else:
                loss = self.criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", lr, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        no_decay_names = set()
        if hasattr(self.model, "no_weight_decay"):
            no_decay_names = set(self.model.no_weight_decay())

        decay_params, no_decay_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name in no_decay_names:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            betas=self.hparams.betas,
            lr=self.hparams.lr,
            eps=self.hparams.eps,
        )

        warmup_scheduler = CustomLambdaLR(optimizer, self.hparams.warmup_steps)
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.hparams.cosine_annealing_steps,
            eta_min=0,
        )
        combined_scheduler = CombinedScheduler(
            optimizer=optimizer,
            scheduler1=warmup_scheduler,
            scheduler2=cosine_scheduler,
            warmup_steps=self.hparams.warmup_steps,
            start_cosine_step=self.hparams.start_cosine_step,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": combined_scheduler, "interval": "step"},
        }
