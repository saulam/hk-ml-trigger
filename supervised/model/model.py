import torch
import torch.nn as nn
try:
    from timm.layers import trunc_normal_
except ImportError:
    trunc_normal_ = None


class RelPosSelfAttention(nn.Module):
    """Multi-head self-attention with learnt relative positional bias."""

    def __init__(self, d_model, nhead, use_cartesian=False, use_time=True):
        super().__init__()
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5
        self.use_cartesian = use_cartesian
        self.use_time = use_time
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model)
        bias_input_size = 4 if use_time else 3
        self.bias_alpha = nn.Parameter(torch.zeros(1))
        self.bias_mlp = nn.Sequential(
            nn.Linear(bias_input_size, 32), nn.ReLU(), nn.Linear(32, nhead)
        )

    def forward(self, x, coords, times, src_key_padding_mask=None):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.nhead, -1).transpose(1, 2) for t in qkv]

        ci, cj = coords.unsqueeze(2), coords.unsqueeze(1)
        dpos = ci - cj
        dz = dpos[..., 2:3]

        if self.use_cartesian:
            dx = dpos[..., 0:1]
            dy = dpos[..., 1:2]
            if self.use_time:
                dt = (times.unsqueeze(2) - times.unsqueeze(1)).abs().unsqueeze(-1)
                e_ij = torch.cat([dx, dy, dz, dt], dim=-1)
            else:
                e_ij = torch.cat([dx, dy, dz], dim=-1)
        else:
            r_i = torch.sqrt(ci[..., 0:1] ** 2 + ci[..., 1:2] ** 2)
            r_j = torch.sqrt(cj[..., 0:1] ** 2 + cj[..., 1:2] ** 2)
            dr = r_i - r_j
            phi_i = torch.atan2(ci[..., 1], ci[..., 0])
            phi_j = torch.atan2(cj[..., 1], cj[..., 0])
            raw = phi_i - phi_j
            dphi = (raw + torch.pi) % (2 * torch.pi) - torch.pi
            if self.use_time:
                dt = (times.unsqueeze(2) - times.unsqueeze(1)).abs().unsqueeze(-1)
                e_ij = torch.cat([dr, dphi.unsqueeze(-1), dz, dt], dim=-1)
            else:
                e_ij = torch.cat([dr, dphi.unsqueeze(-1), dz], dim=-1)

        bias = self.bias_mlp(e_ij).permute(0, 3, 1, 2)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        logits = dots + self.bias_alpha * bias
        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(key_mask, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        out = (attn @ v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out)


class RelPosEncoderLayer(nn.Module):
    """Transformer encoder layer with relative positional attention."""

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1,
                 use_cartesian=True, use_time=True):
        super().__init__()
        self.self_attn = RelPosSelfAttention(
            d_model, nhead, use_cartesian=False, use_time=use_time
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, src, coords, times, src_key_padding_mask=None):
        src2 = self.self_attn(src, coords, times, src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout(src2)
        return self.norm2(src)


class TransformerClassifier(nn.Module):
    """
    Set-input transformer classifier for variable-length PMT hit sequences.

    Uses relative positional attention in cylindrical coordinates and separate
    linear projections for barrel and endcap PMTs.
    """

    def __init__(
        self,
        feature_size: int = 24,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 5,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_cartesian: bool = False,
        token_level: bool = False,
        feature_mode: str = "all",
    ):
        super().__init__()
        self.token_level = token_level
        self.feature_mode = feature_mode
        use_time = feature_mode not in ["no_time", "no_time_no_charge"]

        self.proj_wall = nn.Linear(feature_size, d_model)
        self.proj_cap = nn.Linear(feature_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.len_token_proj = nn.Linear(1, d_model)
        self.token_type_emb = nn.Embedding(5, d_model)

        self.layers = nn.ModuleList([
            RelPosEncoderLayer(
                d_model, nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                use_cartesian=use_cartesian,
                use_time=use_time,
            )
            for _ in range(num_layers)
        ])

        self.head_dropout = nn.Dropout(dropout)
        if token_level:
            self.token_head = nn.Linear(d_model, num_classes)
        else:
            self.classifier = nn.Linear(d_model, num_classes)

        if trunc_normal_ is not None:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def no_weight_decay(self):
        return {
            "cls_token",
            "len_token_proj.weight",
            "token_type_emb.weight",
            "bias_alpha",
        }

    def forward(self, feats, coords, times, loc, lengths, src_key_padding_mask):
        B, L, _ = feats.shape

        mask_wall = (loc == 1).unsqueeze(-1)
        x_wall = self.proj_wall(feats)
        x_cap = self.proj_cap(feats)
        x = x_wall * mask_wall + x_cap * (~mask_wall)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        len_vals = lengths.float().unsqueeze(1)
        len_tokens = self.len_token_proj(len_vals).unsqueeze(1)
        x = torch.cat([cls_tokens, len_tokens, x], dim=1)

        prefix = torch.zeros(B, 2, dtype=torch.long, device=loc.device)
        type_ids = torch.cat([prefix, loc], dim=1)
        x = x + self.token_type_emb(type_ids)

        x = self.input_norm(x)
        x = self.input_dropout(x)

        coords_pad = torch.zeros(B, 2, 3, device=coords.device)
        coords = torch.cat([coords_pad, coords], dim=1)
        times_pad = torch.zeros(B, 2, device=times.device)
        times = torch.cat([times_pad, times], dim=1)

        for layer in self.layers:
            x = layer(x, coords, times, src_key_padding_mask)

        if self.token_level:
            token_repr = x[:, 2:, :]
            logits = self.token_head(self.head_dropout(token_repr))
        else:
            cls_repr = x[:, 0, :]
            logits = self.classifier(self.head_dropout(cls_repr))
        return logits
