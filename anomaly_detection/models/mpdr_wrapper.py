"""MPDR wrapper for variable-length point cloud sequences.

Handles padding and masking for point clouds with varying numbers of hits.
Supports two training modes:
- MPDR-S (Simple): scalar energy network
- MPDR-R (Recovery): reconstruction-based energy from a separate autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mcmc import sample_langevin_v2_proj
from models.utils import weight_norm
from models.loss import multi_scale_probabilistic_dcd_smooth
from models.sampler import EmpiricalKSampler, valid_count_from_mask, clamp_K


class VariableLengthAE(nn.Module):
    """Autoencoder wrapper for variable-length sequences with masking."""

    def __init__(self, encoder, decoder, spherical=False, eps=1e-6,
                 encoding_noise=None, loss="l2", alphas=(10.0, 50.0, 200.0),
                 repulsion_weight=0.02):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.spherical = spherical
        self.eps = eps
        self.encoding_noise = encoding_noise
        self.loss = loss
        self.alphas = alphas
        self.repulsion_weight = repulsion_weight

        if loss in ["l2", "l2_sum", "l1"]:
            import warnings
            warnings.warn(
                f"Loss '{loss}' ignores existence_p from the decoder. "
                f"Use loss='chamfer' for variable-length data.",
                UserWarning,
            )

    def _project(self, z):
        return z / (torch.norm(z, dim=1, keepdim=True) + self.eps)

    def encode(self, x, mask=None, noise=False):
        z = self.encoder(x, mask=mask)
        if noise and self.encoding_noise is not None:
            if self.spherical:
                z = self._project(z)
            z = z + torch.randn_like(z) * self.encoding_noise
        if self.spherical:
            z = self._project(z)
        return z

    def decode(self, z):
        return self.decoder(z)

    def recon_error(self, x, mask=None, noise=False, return_details=False, alphas=None):
        """Reconstruction error with proper masking."""
        z = self.encode(x, mask=mask, noise=noise)
        x_recon, existence_p = self.decode(z)

        if self.loss == "chamfer":
            alphas_value = alphas if alphas is not None else self.alphas
            dcd_results = multi_scale_probabilistic_dcd_smooth(
                x, x_recon, existence_p, mask_x=mask, mask_y=None,
                alphas=alphas_value, repulsion_weight=self.repulsion_weight,
            )
            error_per_sample = dcd_results[0]
            if return_details:
                return dcd_results
        elif self.loss in ["l2", "l2_sum", "l1"]:
            if self.loss == "l1":
                error = torch.abs(x - x_recon)
            else:
                error = (x - x_recon) ** 2

            if mask is not None:
                error = error * (~mask.unsqueeze(-1)).float()

            if self.loss == "l2":
                if mask is not None:
                    valid_counts = (~mask).sum(dim=1, keepdim=True).float()
                    error_per_sample = error.sum(dim=(1, 2)) / (valid_counts.squeeze() * x.size(-1) + self.eps)
                else:
                    error_per_sample = error.view(len(x), -1).mean(dim=1)
            elif self.loss == "l2_sum":
                error_per_sample = error.sum(dim=(1, 2))
            elif self.loss == "l1":
                if mask is not None:
                    valid_counts = (~mask).sum(dim=1, keepdim=True).float()
                    error_per_sample = error.sum(dim=(1, 2)) / (valid_counts.squeeze() * x.size(-1) + self.eps)
                else:
                    error_per_sample = error.view(len(x), -1).mean(dim=1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss}")

        return error_per_sample

    def project_diffuse(self, x, mask, proj_noise):
        """Encode and apply Gaussian noise in latent space."""
        z = self.encode(x, mask=mask, noise=False)
        z = z + torch.randn_like(z) * proj_noise.view(len(z), *[1] * len(z.shape[1:]))
        if self.spherical:
            z = self._project(z)
        decoded, existence_p = self.decode(z)
        return decoded, existence_p, z

    def compute_training_loss(self, x, mask, alphas=None):
        """Compute training loss (Lightning-compatible, no optimiser step)."""
        self.train()
        z = self.encode(x, mask=mask, noise=True)
        recon, existence_p = self.decode(z)

        if self.loss == "chamfer":
            alphas_value = alphas if alphas is not None else self.alphas
            dcd_out = multi_scale_probabilistic_dcd_smooth(
                x, recon, existence_p, mask_x=mask, mask_y=None,
                alphas=alphas_value, repulsion_weight=self.repulsion_weight,
            )
            loss = dcd_out[0].mean()
            metric_keys = [
                "term_a", "term_b", "count_loss", "extra_loss",
                "repulsion_loss", "raw_dist_a", "raw_dist_b", "beta",
            ]
            result = {k: v.mean().item() for k, v in zip(metric_keys, dcd_out[1:])}
            result["loss"] = loss.item()
        else:
            diff = x - recon
            if mask is not None:
                diff = diff * (~mask.unsqueeze(-1)).float()
                valid_counts = (~mask).sum(dim=1).float()
                loss = (diff ** 2).sum(dim=(1, 2)) / (valid_counts * x.size(-1) + self.eps)
                loss = loss.mean()
            else:
                loss = (diff ** 2).mean()
            result = {"loss": loss.item(), "recon_error": loss.item()}

        return loss, result


# ---------------------------------------------------------------------------
# Masking utilities for negative sample generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def pad_mask_topk(existence_p, K):
    """Keep top-K points by existence probability, mask the rest."""
    B, M = existence_p.shape
    device = existence_p.device
    K = K.to(device=device, dtype=torch.long).clamp(0, M)
    Kmax = int(K.max().item())
    if Kmax == 0:
        return torch.ones((B, M), device=device, dtype=torch.bool)
    idx = existence_p.topk(Kmax, dim=1).indices
    ar = torch.arange(Kmax, device=device).unsqueeze(0).expand(B, Kmax)
    take = ar < K.unsqueeze(1)
    keep = torch.zeros((B, M), device=device, dtype=torch.bool)
    b = torch.arange(B, device=device).unsqueeze(1).expand(B, Kmax)
    keep[b[take], idx[take]] = True
    return ~keep


@torch.no_grad()
def pad_mask_samplek(existence_p, K, eps=1e-6):
    """Sample K indices without replacement weighted by existence probability."""
    B, M = existence_p.shape
    device = existence_p.device
    K = K.to(device=device, dtype=torch.long).clamp(0, M)
    w = existence_p.clamp_min(eps)
    w = w / w.sum(dim=1, keepdim=True)
    keep = torch.zeros((B, M), device=device, dtype=torch.bool)
    for b in range(B):
        k = int(K[b].item())
        if k == 0:
            continue
        idx = torch.multinomial(w[b], num_samples=k, replacement=False)
        keep[b, idx] = True
    return ~keep


@torch.no_grad()
def pad_mask_uniform(K, M, device):
    """Uniformly sample K indices without replacement."""
    B = K.shape[0]
    K = K.to(device=device, dtype=torch.long).clamp(0, M)
    keep = torch.zeros((B, M), device=device, dtype=torch.bool)
    for b in range(B):
        k = int(K[b].item())
        if k == 0:
            continue
        idx = torch.randperm(M, device=device)[:k]
        keep[b, idx] = True
    return ~keep


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def reflect_to_interval(u, low=-1.0, high=1.0):
    """Map any real value to [low, high] by reflective wrapping."""
    width = high - low
    period = 2.0 * width
    t = torch.remainder(u - low, period)
    return torch.where(t < width, low + t, high - (t - width))


def project_raw(x_raw, eps=1e-8):
    """Project xyz to unit sphere and reflect auxiliary features to [-1, 1]."""
    xyz = F.normalize(x_raw[..., :3], p=2, dim=-1, eps=eps)
    if x_raw.size(-1) == 3:
        return xyz
    feats = reflect_to_interval(x_raw[..., 3:], low=-1.0, high=1.0)
    return torch.cat([xyz, feats], dim=-1)


def raw_to_model_space(x_raw, eps=1e-8):
    """Normalise xyz to the unit sphere; keep auxiliary features as-is."""
    xyz = F.normalize(x_raw[..., :3], p=2, dim=-1, eps=eps)
    if x_raw.size(-1) == 3:
        return xyz
    return torch.cat([xyz, x_raw[..., 3:]], dim=-1)


def strip_to_scalars(obj):
    """Recursively extract scalar values suitable for logging."""
    if torch.is_tensor(obj):
        if obj.is_cuda:
            obj = obj.detach().cpu()
        if obj.numel() == 1:
            return obj.item()
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            vv = strip_to_scalars(v)
            if vv is not None:
                out[k] = vv
        return out
    if isinstance(obj, (list, tuple)):
        return [v for v in (strip_to_scalars(x) for x in obj) if v is not None]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return None


# ---------------------------------------------------------------------------
# MPDR model
# ---------------------------------------------------------------------------

class MPDR_VariableLength(nn.Module):
    """MPDR for variable-length point cloud sequences.

    Two variants (both use Langevin dynamics for sampling):
    - MPDR-S (Scalar):       E_theta(x) from a scalar energy network
    - MPDR-R (Reconstruction): E_theta(x) = ||x - g_d(g_e(x))||^2
      from a separate autoencoder (g_e, g_d)
    """

    def __init__(
        self,
        ae,
        net_x=None,
        energy_ae=None,
        variant="simple",
        physics_noise_ratio=0.3,
        alphas=(10.0, 50.0, 200.0),
        proj_mode="constant",
        proj_noise_start=0.1,
        proj_noise_end=0.1,
        proj_const=1.0,
        proj_const_omi=None,
        proj_dist="geodesic",
        mcmc_n_step_omi=0,
        mcmc_stepsize_omi=0.01,
        mcmc_noise_omi=0.01,
        mcmc_normalize_omi=False,
        mh_omi=False,
        mcmc_n_step_x=10,
        mcmc_stepsize_x=0.01,
        mcmc_noise_x=0.01,
        mcmc_bound_x=None,
        mcmc_custom_stepsize=False,
        mh_x=False,
        temperature=1.0,
        temperature_omi=1.0,
        gamma_vx=None,
        gamma_neg_recon=None,
        l2_norm_reg_netx=None,
        lambda_center=1e-4,
        tau=10.0,
        energy_method=None,
        eps=1e-6,
        grad_clip_omi=None,
        grad_clip_off=None,
    ):
        super().__init__()
        self.ae = ae
        self.variant = variant

        max_M = int(self.ae.decoder.num_output_points)
        self.k_sampler = EmpiricalKSampler(
            max_M=max_M, ema_decay=0.995, use_ema=True,
            prior_count=1e-3, min_k=1,
        )

        if variant == "simple":
            if net_x is None:
                raise ValueError("MPDR-S requires net_x (scalar energy network)")
            self.net_x = net_x
            self.energy_ae = None
        elif variant == "recovery":
            if energy_ae is None:
                raise ValueError("MPDR-R requires energy_ae (separate autoencoder)")
            self.net_x = None
            self.energy_ae = energy_ae
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.physics_noise_ratio = physics_noise_ratio
        self.alphas = alphas

        self.proj_mode = proj_mode
        self.proj_noise_start = proj_noise_start
        self.proj_noise_end = proj_noise_end
        self.proj_const = proj_const
        self.proj_const_omi = proj_const_omi if proj_const_omi is not None else proj_const
        self.proj_dist = proj_dist

        self.mcmc_n_step_omi = mcmc_n_step_omi
        self.mcmc_stepsize_omi = mcmc_stepsize_omi
        self.mcmc_noise_omi = mcmc_noise_omi
        self.mcmc_normalize_omi = mcmc_normalize_omi
        self.mh_omi = mh_omi
        self.mcmc_n_step_x = mcmc_n_step_x
        self.mcmc_stepsize_x = mcmc_stepsize_x
        self.mcmc_noise_x = mcmc_noise_x
        self.mcmc_bound_x = mcmc_bound_x
        self.mcmc_custom_stepsize = mcmc_custom_stepsize
        self.mh_x = mh_x

        self.temperature = temperature
        self.temperature_omi = temperature_omi

        self.gamma_vx = gamma_vx
        self.gamma_neg_recon = gamma_neg_recon
        self.l2_norm_reg_netx = l2_norm_reg_netx
        self.lambda_center = lambda_center

        self.tau = tau
        self.energy_method = energy_method
        self.eps = eps
        self.grad_clip_omi = grad_clip_omi
        self.grad_clip_off = grad_clip_off

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def get_proj_noise(self, x):
        n_sample = len(x)
        device = x.device
        if self.proj_mode == "uniform":
            return (
                torch.rand(n_sample, device=device)
                * (self.proj_noise_start - self.proj_noise_end)
                + self.proj_noise_end
            )
        elif self.proj_mode == "constant":
            return self.proj_noise_start * torch.ones(n_sample, device=device)
        raise NotImplementedError(f"Unknown proj_mode: {self.proj_mode}")

    def vx(self, x, mask=None):
        """Scalar energy v_theta(x) (MPDR-S only)."""
        if self.variant != "simple":
            raise RuntimeError("vx() is only for MPDR-S variant")
        return self.net_x(x, mask=mask).flatten()

    def energy_x(self, x, mask=None, train=False):
        """Compute energy E_theta(x)."""
        d_out = {}
        if self.variant == "simple":
            vx = self.vx(x, mask=mask)
            d_out["vx"] = vx
            energy = torch.exp(vx) if self.energy_method == "exp" else vx
        elif self.variant == "recovery":
            recon_error = self.energy_ae.recon_error(
                x, mask=mask, noise=False, alphas=self.alphas,
            )
            energy = recon_error
            d_out["recon_error"] = recon_error

        if train:
            d_out["energy"] = energy
            return d_out
        return energy

    def forward(self, x, mask=None):
        """Inference: compute anomaly score."""
        self.eval()
        with torch.no_grad():
            return self.energy_x(x, mask=mask, train=False)

    @torch.no_grad()
    def score_with_count(self, x, mask, beta=0.2):
        """Combined anomaly score: energy + beta * NLL_noise(K)."""
        e = self.energy_x(x, mask=mask, train=False)
        K = (~mask).sum(dim=1)
        M = int(self.ae.decoder.num_output_points)
        nllK = self.k_sampler.upper_tail_nll(K, M=M).to(e.device)
        return e + beta * nllK

    def predict(self, x, mask=None):
        return self.forward(x, mask=mask)

    # ------------------------------------------------------------------
    # Log-probability for MCMC
    # ------------------------------------------------------------------

    def log_prob(self, x, mask, z0, proj_noise, proj_const=None):
        if proj_const is None:
            proj_const = self.proj_const
        e = self.energy_x(x, mask=mask)

        if z0 is None:
            recov = 0.0
        else:
            z_current = self.ae.encode(x, mask=mask)
            if self.proj_dist == "geodesic":
                eps = 1e-6
                sigma2 = proj_noise.squeeze() ** 2
                dot = (z_current * z0).sum(dim=1).clamp(-1 + eps, 1 - eps)
                theta = torch.acos(dot)
                recov = theta ** 2 / (2 * sigma2)
            elif self.proj_dist == "sum":
                recov = (
                    ((z_current - z0) ** 2).view(len(x), -1).sum(dim=1)
                    / 2 / (proj_noise ** 2)
                )
            else:
                raise ValueError(f"Unknown proj_dist: {self.proj_dist}")

        return -e / self.temperature - recov * proj_const

    # ------------------------------------------------------------------
    # MCMC sampling stages
    # ------------------------------------------------------------------

    def on_manifold_init(self, x_init, z_obs, proj_noise, mask=None, K_fixed=None):
        """On-manifold initialisation via latent-space Langevin."""
        if self.mcmc_n_step_omi == 0:
            return {"x_init": x_init, "existence_p_init": None}

        z = z_obs.detach().clone()
        proj_noise = proj_noise.reshape(len(proj_noise), *[1] * len(z.shape[1:]))

        if mask is not None:
            mask = mask.detach()
        if K_fixed is None and mask is not None:
            K_fixed = (~mask).sum(dim=1)
        if K_fixed is not None:
            K_fixed = K_fixed.to(z.device).long()

        def energy_fn(z_):
            x_, existence_p_ = self.ae.decode(z_)
            if K_fixed is not None:
                mask_step = pad_mask_topk(existence_p_.detach(), K_fixed)
            elif mask is not None:
                mask_step = mask
            else:
                mask_step = (existence_p_.detach() < 0.5)
            log_p = self.log_prob(
                x_, mask=mask_step, z0=z_obs,
                proj_noise=proj_noise, proj_const=self.proj_const_omi,
            )
            return -log_p

        d_mcmc = sample_langevin_v2_proj(
            z, energy_fn,
            stepsize=self.mcmc_stepsize_omi,
            n_step=self.mcmc_n_step_omi,
            noise_scale=self.mcmc_noise_omi,
            mh=self.mh_omi,
            temperature=self.temperature_omi,
            normalize_grad=self.mcmc_normalize_omi,
            clip_grad=self.grad_clip_omi,
            bound="spherical" if self.ae.spherical else None,
        )

        z_refined = d_mcmc["sample"]
        x_refined, existence_p_refined = self.ae.decode(z_refined)

        d_result = {
            "x_init": x_refined,
            "existence_p_init": existence_p_refined,
            "d_mcmc": d_mcmc,
        }
        if self.mcmc_n_step_omi > 0:
            drift = torch.stack(d_mcmc["l_drift"])
            drift_norm = drift.reshape(drift.size(0), drift.size(1), -1).norm(dim=-1)
            d_result["mpdr/omi_drift_norm_"] = drift_norm.mean().item()
            d_result["mpdr/omi_drift_norm_max_"] = drift_norm.max().item()
            d_result["mpdr/omi_grad_norm_est_"] = (drift_norm / self.mcmc_stepsize_omi).mean().item()
            if self.mh_omi:
                d_result["mcmc/omi_accept_"] = (
                    torch.cat(d_mcmc["l_accept"]).float().mean().item()
                )
        return d_result

    def p_sample_langevin_off(self, x_init, z_obs, proj_noise, mask):
        """Off-manifold Langevin sampling in visible space."""
        if self.mcmc_n_step_x == 0:
            return {"sample": x_init, "l_grad_y": [], "avg_grad_norm": torch.tensor(0.0)}

        mask = mask.detach()
        proj_noise = proj_noise.reshape(len(proj_noise), *[1] * len(z_obs.shape[1:]))

        noise_scale = self.mcmc_noise_x
        if self.mcmc_custom_stepsize:
            step_size = self.mcmc_stepsize_x
        else:
            step_size = self.mcmc_stepsize_x * (noise_scale ** 2) / 2

        def energy_fn(x_raw_):
            x_proj = raw_to_model_space(x_raw_)
            log_p = self.log_prob(x_proj, mask=mask, z0=z_obs, proj_noise=proj_noise)
            return -log_p

        x0 = x_init.clone().detach()
        d_sample = sample_langevin_v2_proj(
            x0, energy_fn,
            stepsize=step_size,
            n_step=self.mcmc_n_step_x,
            noise_scale=noise_scale,
            bound=self.mcmc_bound_x,
            mh=self.mh_x,
            temperature=None,
            clip_grad=self.grad_clip_off,
            project_fn=project_raw,
        )

        sample = raw_to_model_space(d_sample["sample"])

        d_result = {
            "sample": sample,
            "l_grad_y": d_sample["l_drift"],
            "d_sample": d_sample,
            "l_drift_rms_mean": d_sample["l_drift_rms_mean"],
            "l_noise_rms_mean": d_sample["l_noise_rms_mean"],
        }
        l_grad_y = d_sample["l_drift"]
        if len(l_grad_y) > 0:
            d_result["avg_grad_norm"] = (
                torch.stack(l_grad_y).reshape(len(l_grad_y), -1).norm(dim=1).mean()
            )
        else:
            d_result["avg_grad_norm"] = torch.tensor(0.0)
        return d_result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_training_loss(self, x, mask):
        """Contrastive divergence training (Lightning-compatible).

        Returns (loss, d_train) where d_train contains diagnostics.
        """
        self.eval()

        batch_size = x.shape[0]
        M = int(self.ae.decoder.num_output_points)

        with torch.no_grad():
            proj_noise = self.get_proj_noise(x)
            x_init_pre, existence_p_pre, z_obs = self.ae.project_diffuse(x, mask, proj_noise)
            K_pos = valid_count_from_mask(mask)
            K_neg = K_pos
            pad_mask_omi = pad_mask_topk(existence_p_pre, K_neg)
            self.k_sampler.update(K_pos, M_for_clip=M)

        d_omi = self.on_manifold_init(x_init_pre, z_obs, proj_noise, mask=pad_mask_omi, K_fixed=K_neg)
        x_init = d_omi["x_init"]
        existence_p_init = d_omi["existence_p_init"]

        if existence_p_init is None:
            pad_mask_neg = pad_mask_omi
        else:
            pad_mask_neg = pad_mask_topk(existence_p_init, K_neg)

        d_sample = self.p_sample_langevin_off(x_init, z_obs, proj_noise, pad_mask_neg)
        x_neg = d_sample["sample"]

        with torch.no_grad():
            z_x_neg = self.ae.encode(x_neg, mask=pad_mask_neg)

        d_train = {}

        if x_neg is not None:
            d_train["mpdr/off_avg_grad_norm_"] = d_sample["avg_grad_norm"].item()
            d_train["mpdr/neg_valid_count_"] = (~pad_mask_neg).sum(dim=1).float().mean().item()
            d_train["mpdr/pos_valid_count_"] = (~mask).sum(dim=1).float().mean().item()

            with torch.no_grad():
                valid = (~pad_mask_neg).float()
                delta_xyz = x_neg[..., :3] - x_init[..., :3]
                delta_norm = (delta_xyz.pow(2).sum(-1) + 1e-8).sqrt()
                mean_disp = (delta_norm * valid).sum(1) / valid.sum(1).clamp(min=1.0)
                max_disp = (delta_norm * valid + (1.0 - valid) * (-1e9)).max(1).values
                d_train["mpdr/mcmc_mean_disp_xyz_"] = mean_disp.mean().item()
                d_train["mpdr/mcmc_max_disp_xyz_"] = max_disp.mean().item()

                drift_rms = d_sample["l_drift_rms_mean"]
                noise_rms = d_sample["l_noise_rms_mean"]
                if torch.is_tensor(drift_rms):
                    drift_rms = drift_rms.detach()
                if torch.is_tensor(noise_rms):
                    noise_rms = noise_rms.detach()
                d_train["mpdr/drift_to_noise_rms_"] = (drift_rms / (noise_rms + 1e-12)).item()

        d_train["mpdr/z_inner_"] = (z_x_neg * z_obs).sum(dim=1).mean().item()

        with torch.no_grad():
            if self.variant == "simple":
                self.net_x.eval()
            e_init = self.energy_x(x_init, mask=pad_mask_neg, train=False)
            e_neg_after = self.energy_x(x_neg, mask=pad_mask_neg, train=False)
            d_train["mpdr/e_init_"] = e_init.mean().item()
            d_train["mpdr/e_neg_after_mcmc_"] = e_neg_after.mean().item()
            d_train["mpdr/e_drop_"] = (e_init - e_neg_after).mean().item()

        loss, d_loss = self.compute_energy_loss(x, x_neg, mask, pad_mask_neg)

        d_train = {**d_omi, **d_sample, **d_train, **d_loss}
        d_train = strip_to_scalars(d_train)
        return loss, d_train

    def compute_energy_loss(self, x_pos, x_neg, mask_pos, mask_neg):
        """Contrastive divergence energy loss."""
        if self.variant == "simple":
            self.net_x.train()
        else:
            self.energy_ae.train()

        d_pos = self.energy_x(x_pos, mask=mask_pos, train=True)
        d_neg = self.energy_x(x_neg, mask=mask_neg, train=True)
        e_pos = d_pos["energy"]
        e_neg = d_neg["energy"]

        loss_cd = self.tau * F.softplus((e_pos - e_neg) / self.tau).mean()
        loss = loss_cd

        d_loss = {
            "mpdr/loss_cd_": loss_cd.item(),
            "mpdr/e_pos_": e_pos.mean().item(),
            "mpdr/e_neg_": e_neg.mean().item(),
        }

        if self.variant == "simple":
            vx_pos = d_pos["vx"]
            vx_neg = d_neg["vx"]
            d_loss["mpdr/vx_pos_mean_"] = vx_pos.mean().item()
            d_loss["mpdr/vx_neg_mean_"] = vx_neg.mean().item()
            d_loss["mpdr/vx_pos_std_"] = vx_pos.std().item()
            d_loss["mpdr/vx_neg_std_"] = vx_neg.std().item()

            if self.gamma_vx is not None:
                reg_vx = (vx_pos ** 2).mean() + (vx_neg ** 2).mean()
                loss += self.gamma_vx * reg_vx
                d_loss["reg/gamma_vx_"] = reg_vx.item()
            if self.l2_norm_reg_netx is not None:
                netx_norm = self.weight_norm()
                loss += self.l2_norm_reg_netx * netx_norm
                d_loss["reg/netx_norm_"] = netx_norm.item()

            center = 0.5 * (vx_pos.mean() + vx_neg.mean())
            loss = loss + self.lambda_center * center.pow(2)
            d_loss["reg/lambda_center_"] = (self.lambda_center * center.pow(2)).item()

        elif self.variant == "recovery":
            d_loss["recon_error/recon_pos_"] = d_pos["recon_error"].mean().item()
            d_loss["recon_error/recon_neg_"] = d_neg["recon_error"].mean().item()
            if self.gamma_neg_recon is not None:
                reg_neg_recon = (d_neg["recon_error"] ** 2).mean()
                loss += self.gamma_neg_recon * reg_neg_recon
                d_loss["reg/gamma_neg_recon_"] = reg_neg_recon.item()

        d_loss["loss"] = loss.item()
        return loss, d_loss

    def weight_norm(self):
        if self.variant == "simple":
            return weight_norm(self.net_x)
        elif self.variant == "recovery":
            return weight_norm(self.energy_ae.encoder) + weight_norm(self.energy_ae.decoder)
        return torch.tensor(0.0)

    def sample_negatives(self, x, mask, n_step_x=None, n_step_omi=None):
        """Generate negative samples with their padding mask."""
        self.eval()
        old_x = self.mcmc_n_step_x
        old_omi = self.mcmc_n_step_omi
        if n_step_x is not None:
            self.mcmc_n_step_x = int(n_step_x)
        if n_step_omi is not None:
            self.mcmc_n_step_omi = int(n_step_omi)

        try:
            proj_noise = self.get_proj_noise(x)
            x_init_pre, existence_p_pre, z_obs = self.ae.project_diffuse(x, mask, proj_noise)
            K_neg = valid_count_from_mask(mask)
            mask_omi = pad_mask_topk(existence_p_pre, K_neg)

            d_omi = self.on_manifold_init(
                x_init_pre, z_obs, proj_noise, mask=mask_omi, K_fixed=K_neg,
            )
            x_init = d_omi["x_init"]
            existence_p_init = d_omi.get("existence_p_init", None)
            mask_neg = mask_omi if existence_p_init is None else pad_mask_topk(existence_p_init, K_neg)

            d_off = self.p_sample_langevin_off(x_init, z_obs, proj_noise, mask_neg)
            return d_off["sample"], mask_neg
        finally:
            self.mcmc_n_step_x = old_x
            self.mcmc_n_step_omi = old_omi
