import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def _all_masked(mask, dim):
    return mask.all(dim=dim, keepdim=False)


def masked_mean_pool(tokens, mask, dim=1):
    if mask is None:
        return tokens.mean(dim=dim)
    valid = (~mask).to(tokens.dtype)
    if tokens.dim() == 3:
        valid = valid.unsqueeze(-1)
    denom = valid.sum(dim=dim).clamp(min=1.0)
    return (tokens * valid).sum(dim=dim) / denom


def masked_max_pool(tokens, mask, dim=1):
    if mask is None:
        return tokens.max(dim=dim).values
    neg_inf = -torch.finfo(tokens.dtype).max
    if tokens.dim() == 3:
        tokens_masked = tokens.masked_fill(mask.unsqueeze(-1), neg_inf)
        out = tokens_masked.max(dim=dim).values
        all_m = _all_masked(mask, dim=dim).unsqueeze(-1)
        return torch.where(all_m, torch.zeros_like(out), out)
    else:
        tokens_masked = tokens.masked_fill(mask, neg_inf)
        out = tokens_masked.max(dim=dim).values
        all_m = _all_masked(mask, dim=dim)
        return torch.where(all_m, torch.zeros_like(out), out)


def masked_logsumexp_pool(tokens, mask, dim=1, tau=1.0):
    if tau <= 0:
        raise ValueError("tau must be > 0")
    if mask is None:
        return tau * torch.logsumexp(tokens / tau, dim=dim)
    neg_inf = -torch.finfo(tokens.dtype).max
    if tokens.dim() == 3:
        x = (tokens / tau).masked_fill(mask.unsqueeze(-1), neg_inf)
        out = tau * torch.logsumexp(x, dim=dim)
        all_m = _all_masked(mask, dim=dim).unsqueeze(-1)
        return torch.where(all_m, torch.zeros_like(out), out)
    else:
        x = (tokens / tau).masked_fill(mask, neg_inf)
        out = tau * torch.logsumexp(x, dim=dim)
        all_m = _all_masked(mask, dim=dim)
        return torch.where(all_m, torch.zeros_like(out), out)


def masked_logmeanexp(e_tok, mask, tau=0.25, dim=1, eps=1e-9):
    valid = (~mask).float()
    K = valid.sum(dim=dim).clamp(min=1.0)
    e = e_tok.masked_fill(mask, float("-inf"))
    lse = tau * torch.logsumexp(e / tau, dim=dim)
    return lse - tau * torch.log(K + eps)


def masked_topk_mean(e_tok, mask, k=10):
    e = e_tok.masked_fill(mask, float("-inf"))
    topk, _ = torch.topk(e, k=min(k, e.size(1)), dim=1)
    finite = torch.isfinite(topk).float()
    return (topk * finite).sum(dim=1) / finite.sum(dim=1).clamp(min=1.0)


def masked_topk_logmeanexp(e_tok, mask, k=10, tau=0.25, eps=1e-9):
    e = e_tok.masked_fill(mask, float("-inf"))
    kk = min(k, e.size(1))
    topk, _ = torch.topk(e, k=kk, dim=1)
    lse = tau * torch.logsumexp(topk / tau, dim=1)
    return lse - tau * torch.log(torch.tensor(float(kk), device=e.device) + eps)


def masked_pool(tokens, mask, pool, dim=1, tau=1.0, k=10):
    pool = pool.lower()
    if pool == "mean":
        return masked_mean_pool(tokens, mask, dim=dim)
    if pool == "max":
        return masked_max_pool(tokens, mask, dim=dim)
    if pool in ("logsumexp", "lse"):
        return masked_logsumexp_pool(tokens, mask, dim=dim, tau=tau)
    if pool in ("logmeanexp", "lme"):
        return masked_logmeanexp(tokens, mask, tau=tau, dim=dim)
    if pool in ("topk_mean", "tkm"):
        return masked_topk_mean(tokens, mask, k=k)
    if pool in ("topk_logmeanexp", "tk_lme"):
        return masked_topk_logmeanexp(tokens, mask, k=k, tau=tau)
    raise ValueError(f"Unknown pool '{pool}'.")


class TransformerEncoder(nn.Module):
    """Transformer encoder for latent embedding or scalar energy output.

    If energy_mode is True, outputs a scalar energy per sample via a
    spectrally normalised head. Otherwise outputs a latent vector.
    """

    def __init__(
        self,
        input_dim=5,
        latent_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        energy_mode=False,
        energy_mlp_hidden=None,
        pool="cls",
        pool_tau=0.25,
    ):
        super().__init__()
        assert input_dim >= 3
        self.latent_dim = latent_dim
        self.energy_mode = energy_mode
        self.pool = pool.lower()
        self.pool_tau = float(pool_tau)
        self.include_cls = (self.pool == "cls")

        self.embedding = nn.Linear(input_dim, latent_dim)

        if self.energy_mode:
            def transformer_act(t):
                return F.leaky_relu(t, negative_slope=0.2)
        else:
            transformer_act = "gelu"

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation=transformer_act,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if self.include_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        if self.energy_mode:
            if energy_mlp_hidden is None:
                self.energy_head = nn.Linear(latent_dim, 1)
            else:
                self.energy_head = nn.Sequential(
                    nn.Linear(latent_dim, energy_mlp_hidden),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(energy_mlp_hidden, 1),
                )
            self.energy_head = spectral_norm(self.energy_head)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask, return_tokens=False):
        B, N, _ = x.shape
        tok = self.embedding(x)

        if self.include_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
            mask_tf = torch.cat(
                [torch.zeros((B, 1), device=mask.device, dtype=torch.bool), mask], dim=1
            )
        else:
            mask_tf = mask

        out = self.transformer(tok, src_key_padding_mask=mask_tf)

        if self.include_cls:
            cls_out = out[:, 0]
            tok_out = out[:, 1:]
            tok_mask = mask
        else:
            cls_out = None
            tok_out = out
            tok_mask = mask_tf

        if self.energy_mode:
            if self.pool == "cls":
                e = self.energy_head(cls_out).squeeze(-1)
            else:
                e_tok = self.energy_head(tok_out).squeeze(-1)
                e = masked_pool(e_tok, tok_mask, pool=self.pool, dim=1, tau=self.pool_tau, k=10)
            return (e, tok_out) if return_tokens else e

        if self.pool == "cls":
            z = cls_out
        else:
            z = masked_pool(tok_out, tok_mask, pool=self.pool, dim=1, tau=self.pool_tau)
        return (z, tok_out) if return_tokens else z


class TransformerDecoder(nn.Module):
    """Cross-attention decoder with learnable query anchors.

    Outputs a fixed-size reconstructed point cloud with per-point
    existence probabilities.
    """

    def __init__(
        self,
        latent_dim=128,
        num_output_points=192,
        output_dim=5,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        memory_tokens=4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output_points = num_output_points
        self.output_dim = output_dim
        self.memory_tokens = memory_tokens

        self.anchors = nn.Parameter(torch.randn(1, num_output_points, latent_dim) * 0.02)
        self.to_memory = nn.Linear(latent_dim, memory_tokens * latent_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.regressor = nn.Linear(latent_dim, output_dim)
        self.existence_head = nn.Linear(latent_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        B = z.size(0)
        tgt = self.anchors.expand(B, -1, -1)
        mem = self.to_memory(z).view(B, self.memory_tokens, self.latent_dim)
        out = self.decoder(tgt=tgt, memory=mem)

        raw_out = self.regressor(out)
        existence_logits = self.existence_head(out).squeeze(-1)

        raw_spatial = raw_out[..., :3]
        raw_feats = raw_out[..., 3:]
        proj_spatial = F.normalize(raw_spatial, p=2, dim=-1, eps=1e-8)
        proj_feats = torch.tanh(raw_feats) if raw_feats.numel() > 0 else raw_feats

        x_recon = (
            torch.cat([proj_spatial, proj_feats], dim=-1)
            if raw_feats.numel() > 0
            else proj_spatial
        )
        existence_p = torch.sigmoid(existence_logits)
        return x_recon, existence_p


class PointCloudAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask=None):
        z = self.encoder(x, mask)
        x_recon, existence_p = self.decoder(z)
        return x_recon, existence_p, z
