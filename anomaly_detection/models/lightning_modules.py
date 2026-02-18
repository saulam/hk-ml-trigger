"""PyTorch Lightning modules for MPDR training.

Phase 1: AutoencoderModule  — pre-train the autoencoder
Phase 2: MPDRModule          — train the energy function (MPDR-S or MPDR-R)
"""

import torch
import pytorch_lightning as pl
from pathlib import Path


class AutoencoderModule(pl.LightningModule):
    """Phase 1: autoencoder pre-training."""

    def __init__(self, ae_model, config):
        super().__init__()
        self.ae = ae_model
        self.config = config
        self.save_hyperparameters(ignore=["ae_model"])

    def forward(self, feats, mask=None):
        return self.ae(feats, mask=mask)

    def training_step(self, batch, batch_idx):
        feats, mask = batch["feats"], batch["mask"]
        bs = feats.size(0)
        loss, d_train = self.ae.compute_training_loss(feats, mask)

        for key, value in d_train.items():
            if isinstance(value, (int, float)):
                self.log(f"train/{key}", value, on_step=False, on_epoch=True,
                         prog_bar=(key == "loss"), batch_size=bs, sync_dist=True)

        if self.ae.loss == "chamfer":
            raw_dist_mean = (d_train["raw_dist_a"] + d_train["raw_dist_b"]) / 2
            self.log("train/raw_dist", raw_dist_mean, on_step=False, on_epoch=True,
                     batch_size=bs, sync_dist=True)
            for i, alpha in enumerate(self.ae.alphas, 1):
                self.log(f"train/alpha_{i}", alpha, on_step=False, on_epoch=True,
                         batch_size=bs, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, on_epoch=True, prog_bar=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        feats, mask = batch["feats"], batch["mask"]
        bs = feats.size(0)

        if self.ae.loss == "chamfer":
            (recon_error, term_a, term_b, count_loss, extra_loss,
             rep_loss, raw_dist_a, raw_dist_b, beta) = self.ae.recon_error(
                feats, mask=mask, noise=False, return_details=True)
            val_loss = recon_error.mean()
            self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=bs, sync_dist=True)
            for name, val in [("term_a", term_a), ("term_b", term_b),
                              ("count_loss", count_loss), ("extra_loss", extra_loss),
                              ("rep_loss", rep_loss), ("raw_dist_a", raw_dist_a),
                              ("raw_dist_b", raw_dist_b), ("beta", beta)]:
                self.log(f"val/{name}", val.mean(), on_step=False, on_epoch=True,
                         batch_size=bs, sync_dist=True)
            raw_dist = (raw_dist_a.mean() + raw_dist_b.mean()) / 2
            self.log("val/raw_dist", raw_dist, on_step=False, on_epoch=True,
                     batch_size=bs, sync_dist=True)
        else:
            val_loss = self.ae.recon_error(feats, mask=mask, noise=False).mean()
            self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=bs, sync_dist=True)
        return {"val_loss": val_loss}

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []
        no_decay_keywords = ("bias", "norm", "cls_token", "anchor",
                             "pos_emb", "positional", "global_token")
        for name, param in self.ae.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name.lower() for kw in no_decay_keywords) or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.config.training.ae_weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.config.training.ae_lr)

        schedulers, milestones = [], []
        if self.config.training.use_warmup:
            from torch.optim.lr_scheduler import LambdaLR
            warmup_sched = LambdaLR(
                optimizer,
                lr_lambda=lambda ep: min(1.0, (ep + 1) / self.config.training.warmup_epochs),
            )
            schedulers.append(warmup_sched)
            milestones.append(self.config.training.warmup_epochs)

        from torch.optim.lr_scheduler import CosineAnnealingLR
        warmup_ep = self.config.training.warmup_epochs if self.config.training.use_warmup else 0
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=self.config.training.phase1_epochs - warmup_ep, eta_min=0.0,
        )
        schedulers.append(cosine_sched)

        if len(schedulers) > 1:
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
        else:
            scheduler = schedulers[0]

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


class MPDRModule(pl.LightningModule):
    """Phase 2: MPDR training (works for both MPDR-S and MPDR-R)."""

    def __init__(self, mpdr_model, config, ae_checkpoint_path=None):
        super().__init__()
        self.mpdr = mpdr_model
        self.config = config
        self.save_hyperparameters(ignore=["mpdr_model"])

        self.validation_step_outputs = []
        self.test_step_outputs = []

        if ae_checkpoint_path is not None:
            self._load_ae_checkpoint(ae_checkpoint_path)

        for param in self.mpdr.ae.parameters():
            param.requires_grad = False
        self.mpdr.ae.eval()

    def _load_ae_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: AE checkpoint not found at {checkpoint_path}")
            return

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "state_dict" in ckpt:
            ae_sd = {k.replace("ae.", ""): v for k, v in ckpt["state_dict"].items()
                     if k.startswith("ae.")}
        else:
            ae_sd = ckpt.get("model_state_dict", ckpt)

        ae_sd = {k.replace("_orig_mod.", ""): v for k, v in ae_sd.items()}
        self.mpdr.ae.load_state_dict(ae_sd, strict=True)
        print(f"Loaded pre-trained AE from {checkpoint_path}")

        if self.config.mpdr.variant == "recovery" and self.mpdr.energy_ae is not None:
            self.mpdr.energy_ae.encoder.load_state_dict(
                self.mpdr.ae.encoder.state_dict(), strict=True)
            self.mpdr.energy_ae.decoder.load_state_dict(
                self.mpdr.ae.decoder.state_dict(), strict=True)
            print("Initialised energy_ae from pre-trained AE for MPDR-R")

    def forward(self, feats, mask=None):
        return self.mpdr(feats, mask=mask)

    def training_step(self, batch, batch_idx):
        feats, mask = batch["feats"], batch["mask"]
        bs = feats.size(0)
        loss, d_train = self.mpdr.compute_training_loss(feats, mask)

        for key, value in d_train.items():
            if key in ("x_neg", "x_init", "d_mcmc", "d_sample", "l_grad_y"):
                continue
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value
            if isinstance(value, (int, float)):
                self.log(f"train/{key}", value, on_step=True, on_epoch=True,
                         prog_bar=(key == "loss"), batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        scores = self.mpdr(batch["feats"], mask=batch["mask"])
        output = {"scores": scores, "is_background": batch["is_background"]}
        self.validation_step_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        scores = self.mpdr(batch["feats"], mask=batch["mask"])
        output = {"scores": scores, "is_background": batch["is_background"],
                  "lengths": batch["lengths"]}
        self.test_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        all_scores = torch.cat([x["scores"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["is_background"] for x in self.validation_step_outputs])

        if self.config.data.train_mode == "signal":
            anomaly_labels = all_labels
        else:
            anomaly_labels = 1 - all_labels

        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(
                anomaly_labels.float().cpu().numpy(),
                all_scores.float().cpu().numpy(),
            )
            self.log("val/auroc", auroc, prog_bar=True)
        except Exception as e:
            print(f"Could not compute AUROC: {e}")

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        all_scores = torch.cat([x["scores"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["is_background"] for x in self.test_step_outputs])

        if self.config.data.train_mode == "signal":
            anomaly_labels = all_labels
        else:
            anomaly_labels = 1 - all_labels

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
            scores_np = all_scores.float().cpu().numpy()
            labels_np = anomaly_labels.float().cpu().numpy()

            auroc = roc_auc_score(labels_np, scores_np)
            auprc = average_precision_score(labels_np, scores_np)
            fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
            idx_95tpr = (tpr >= 0.95).argmax()

            self.log("test/auroc", auroc)
            self.log("test/auprc", auprc)
            self.log("test/fpr_at_95tpr", fpr[idx_95tpr])
            self.log("test/threshold_95tpr", thresholds[idx_95tpr])

            print(f"\nTest Results:")
            print(f"  AUROC: {auroc:.4f}")
            print(f"  AUPRC: {auprc:.4f}")
            print(f"  FPR @ 95% TPR: {fpr[idx_95tpr]:.4f}")
        except Exception as e:
            print(f"Could not compute test metrics: {e}")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        if self.config.mpdr.variant == "simple":
            param_source = self.mpdr.net_x
        elif self.config.mpdr.variant == "recovery":
            param_source = self.mpdr.energy_ae
        else:
            raise ValueError(f"Unknown variant: {self.config.mpdr.variant}")

        decay_params, no_decay_params = [], []
        no_decay_keywords = ("bias", "norm", "cls_token", "anchor",
                             "pos_emb", "positional", "global_token")
        for name, param in param_source.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name.lower() for kw in no_decay_keywords) or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.config.training.energy_weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.config.training.energy_lr)

        if not self.config.training.use_scheduler:
            return optimizer

        schedulers, milestones = [], []
        if self.config.training.use_warmup:
            from torch.optim.lr_scheduler import LambdaLR
            warmup_sched = LambdaLR(
                optimizer,
                lr_lambda=lambda ep: min(1.0, (ep + 1) / self.config.training.warmup_epochs),
            )
            schedulers.append(warmup_sched)
            milestones.append(self.config.training.warmup_epochs)

        if self.config.training.lr_decay_epochs:
            from torch.optim.lr_scheduler import MultiStepLR
            step_sched = MultiStepLR(
                optimizer,
                milestones=self.config.training.lr_decay_epochs,
                gamma=self.config.training.lr_decay_factor,
            )
            schedulers.append(step_sched)
        else:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            warmup_ep = self.config.training.warmup_epochs if self.config.training.use_warmup else 0
            cosine_sched = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.phase2_epochs - warmup_ep,
                eta_min=self.config.training.energy_lr * 0.01,
            )
            schedulers.append(cosine_sched)

        if len(schedulers) > 1:
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
        else:
            scheduler = schedulers[0]

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def on_train_epoch_end(self):
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, on_epoch=True, prog_bar=False)
