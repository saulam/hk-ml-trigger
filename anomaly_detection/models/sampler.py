"""Empirical K-sampler for variable-length point cloud negative generation."""

import torch
import torch.nn as nn


@torch.no_grad()
def valid_count_from_mask(pad_mask):
    """Count valid (non-padding) points per sample."""
    return (~pad_mask).sum(dim=1)


@torch.no_grad()
def clamp_K(K, M, min_k=1):
    """Clamp K to [min_k, M]."""
    return K.to(dtype=torch.long).clamp(min=min_k, max=int(M))


class EmpiricalKSampler(nn.Module):
    """Maintains an empirical histogram p_noise(K) and samples K_neg from it.

    Supports DDP via all-reduce on batch histograms.
    """

    def __init__(self, max_M, ema_decay=0.995, use_ema=True,
                 prior_count=1e-3, min_k=1, device=None):
        super().__init__()
        self.max_M = int(max_M)
        self.ema_decay = float(ema_decay)
        self.use_ema = bool(use_ema)
        self.prior_count = float(prior_count)
        self.min_k = int(min_k)

        self.register_buffer("hist", torch.zeros(self.max_M + 1, dtype=torch.float32, device=device))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long, device=device))

    @torch.no_grad()
    def update_from_mask(self, pad_mask, M_for_clip=None):
        self.update(valid_count_from_mask(pad_mask), M_for_clip=M_for_clip)

    @torch.no_grad()
    def update(self, K_pos, M_for_clip=None):
        if K_pos.numel() == 0:
            return
        M_clip = self.max_M if M_for_clip is None else min(int(M_for_clip), self.max_M)
        k = K_pos.detach().to(device=self.hist.device, dtype=torch.long).clamp(min=0, max=M_clip)
        batch_hist = torch.bincount(k, minlength=self.max_M + 1).to(self.hist.dtype)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_hist, op=torch.distributed.ReduceOp.SUM)

        if self.use_ema:
            self.hist.mul_(self.ema_decay).add_(batch_hist, alpha=1.0 - self.ema_decay)
        else:
            self.hist.add_(batch_hist)
        self.n_updates.add_(1)

    @torch.no_grad()
    def sample(self, B, M, device=None, min_k=None, max_k=None):
        """Sample K_neg ~ p_noise(K) truncated to [min_k, max_k]."""
        device = device if device is not None else self.hist.device
        M_eff = min(int(M), self.max_M)
        if min_k is None:
            min_k = self.min_k
        min_k = max(int(min_k), 0)
        max_k = M_eff if max_k is None else min(int(max_k), M_eff)
        max_k = max(max_k, min_k)

        h = self.hist[:M_eff + 1].clone() + self.prior_count
        if min_k > 0:
            h[:min_k] = 0.0
        if max_k < M_eff:
            h[max_k + 1:] = 0.0

        if float(h.sum().item()) <= 0.0:
            k0 = max(min_k, min(max_k, max(1, int(0.5 * M_eff))))
            return torch.full((B,), k0, device=device, dtype=torch.long)

        probs = h / h.sum()
        return torch.multinomial(probs, num_samples=B, replacement=True).to(device=device, dtype=torch.long)

    @torch.no_grad()
    def get_probs(self, M=None):
        if M is None:
            M = self.max_M
        M_eff = min(int(M), self.max_M)
        h = self.hist[:M_eff + 1].clone() + self.prior_count
        return h / h.sum()

    @torch.no_grad()
    def nll(self, K, M):
        p = self.get_probs(M=M)
        k = K.to(device=p.device, dtype=torch.long).clamp(0, min(M, self.max_M))
        return (-torch.log(p[k].clamp_min(1e-12))).to(device=K.device)

    @torch.no_grad()
    def upper_tail_nll(self, K, M, eps=1e-12, clamp_below_ref=True):
        """One-sided penalty: -log P(K >= k)."""
        p = self.get_probs(M=M)
        S = torch.flip(torch.cumsum(torch.flip(p, dims=[0]), dim=0), dims=[0])
        k = K.to(device=p.device, dtype=torch.long).clamp(0, min(M, self.max_M))
        penalty = (-torch.log(S[k].clamp_min(eps))).to(device=K.device)

        if clamp_below_ref:
            ks = torch.arange(p.numel(), device=p.device, dtype=p.dtype)
            ref_k = int(torch.round((ks * p).sum()).item())
            penalty = torch.where(K > ref_k, penalty, torch.zeros_like(penalty))
        return penalty
