import torch
import torch.nn.functional as F


def _masked_softmax(logits, mask, dim):
    if mask is None:
        return torch.softmax(logits, dim=dim)
    neg_inf = torch.finfo(logits.dtype).min
    logits_masked = logits.masked_fill(mask, neg_inf)
    all_masked = mask.all(dim=dim, keepdim=True)
    probs = torch.softmax(logits_masked, dim=dim)
    probs = torch.where(all_masked, torch.zeros_like(probs), probs)
    return probs


def _repulsion_loss(y_xyz, y_probs, mask_y, gamma=400.0, eps=1e-6):
    B, M, _ = y_xyz.shape
    dtype = y_xyz.dtype
    device = y_xyz.device

    valid_y = (~mask_y).to(dtype=dtype)
    y_probs = y_probs * valid_y
    soft_count = y_probs.sum(dim=1).clamp(min=eps)

    diff = y_xyz.unsqueeze(2) - y_xyz.unsqueeze(1)
    d2 = (diff * diff).sum(dim=-1)

    diag = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0)
    invalid = diag | mask_y.unsqueeze(2) | mask_y.unsqueeze(1)
    d2 = d2.masked_fill(invalid, float("inf"))

    nn_d2, _ = d2.min(dim=2)
    nn_d2 = torch.where(torch.isfinite(nn_d2), nn_d2, torch.zeros_like(nn_d2))

    pen = torch.exp(-gamma * nn_d2)
    rep = (pen * y_probs).sum(dim=1) / soft_count
    return rep


def _weighted_multiscale_kernel(d2, alphas, alpha_weights=None):
    device = d2.device
    dtype = d2.dtype
    alphas_t = torch.as_tensor(alphas, device=device, dtype=dtype).view(1, 1, 1, -1)
    ks = torch.exp(-d2.unsqueeze(-1) * alphas_t)
    if alpha_weights is None:
        return ks.mean(dim=-1)
    w = torch.as_tensor(alpha_weights, device=device, dtype=dtype).view(1, 1, 1, -1)
    w = w.clamp(min=0.0)
    wsum = w.sum().clamp(min=torch.finfo(dtype).eps)
    return (ks * w).sum(dim=-1) / wsum


def _auto_assign_beta_from_nn(d2, pair_mask, mask_x, target=0.5,
                               beta_min=5.0, beta_max=100.0, eps=1e-6):
    B, N, M = d2.shape
    device = d2.device
    dtype = d2.dtype

    d2_valid = d2.masked_fill(pair_mask, float("inf"))
    min_d2_x, _ = d2_valid.min(dim=2)
    valid_x = (~mask_x) & torch.isfinite(min_d2_x)
    min_d2_x_f = torch.where(valid_x, min_d2_x, torch.full_like(min_d2_x, float("inf")))
    sorted_vals, _ = min_d2_x_f.sort(dim=1)
    counts = valid_x.sum(dim=1)
    mid = ((counts - 1) // 2).clamp(min=0, max=N - 1)
    median_min_d2 = sorted_vals.gather(1, mid.view(B, 1)).squeeze(1)
    median_min_d2 = torch.where(counts > 0, median_min_d2, torch.ones_like(median_min_d2))
    median_min_d2 = median_min_d2.detach().clamp(min=eps)

    t = torch.as_tensor(target, device=device, dtype=dtype)
    beta = (-torch.log(t)) / median_min_d2
    beta = beta.clamp(min=beta_min, max=beta_max)
    return beta


def multi_scale_probabilistic_dcd_smooth(
    x, y, y_probs,
    mask_x=None, mask_y=None,
    alphas=(2.0, 10.0, 40.0),
    alpha_weights=(0.6, 0.3, 0.1),
    n_lambda=1.0,
    count_loss_weight=0.1,
    extra_feat_weight=0.5,
    extra_feat_loss="l1",
    huber_delta=0.1,
    assign_beta=40.0,
    assign_beta_mode="auto",
    assign_beta_target=0.5,
    assign_beta_min=5.0,
    assign_beta_max=200.0,
    repulsion_weight=0.02,
    repulsion_gamma=400.0,
    eps=1e-6,
    return_beta=True,
):
    """Multi-scale probabilistic dense correspondence distance.

    Combines soft-assignment matching with existence-probability weighting
    for variable-length point cloud reconstruction.
    """
    device = x.device
    dtype = x.dtype
    B, N, D = x.shape
    _, M, Dy = y.shape
    assert D == Dy

    if mask_x is None:
        mask_x = torch.zeros((B, N), device=device, dtype=torch.bool)
    if mask_y is None:
        mask_y = torch.zeros((B, M), device=device, dtype=torch.bool)

    y_probs = y_probs.to(dtype=dtype).masked_fill(mask_y, 0.0)

    x_xyz = x[..., :3]
    y_xyz = y[..., :3]
    d2 = ((x_xyz.unsqueeze(2) - y_xyz.unsqueeze(1)) ** 2).sum(dim=-1)
    pair_mask = mask_x.unsqueeze(2) | mask_y.unsqueeze(1)

    k_ij = _weighted_multiscale_kernel(d2, alphas=alphas, alpha_weights=alpha_weights)
    k_ij = k_ij.masked_fill(pair_mask, 0.0)

    if assign_beta_mode == "auto" or (isinstance(assign_beta, str) and assign_beta == "auto"):
        beta_b = _auto_assign_beta_from_nn(
            d2=d2, pair_mask=pair_mask, mask_x=mask_x,
            target=assign_beta_target, beta_min=assign_beta_min,
            beta_max=assign_beta_max, eps=eps,
        )
        beta_eff = beta_b.view(B, 1, 1)
    else:
        beta_eff = torch.as_tensor(assign_beta, device=device, dtype=dtype).view(1, 1, 1)
        beta_b = beta_eff.view(-1).expand(B)

    logits_xy = -beta_eff * d2
    w_xy = _masked_softmax(logits_xy, mask=pair_mask, dim=2)
    w_xy = w_xy.masked_fill(mask_x.unsqueeze(2), 0.0)

    logits_yx = -beta_eff * d2.transpose(1, 2)
    pair_mask_T = pair_mask.transpose(1, 2)
    w_yx = _masked_softmax(logits_yx, mask=pair_mask_T, dim=2)
    w_yx = w_yx.masked_fill(mask_y.unsqueeze(2), 0.0)

    count1_y = w_xy.sum(dim=1).detach()
    weight_y = (count1_y.clamp(min=1.0)).pow(-n_lambda)
    weight_y = weight_y.masked_fill(mask_y, 0.0)

    count2_x = torch.sum(w_yx * y_probs.unsqueeze(-1), dim=1)
    weight_x = (count2_x.clamp(min=1.0)).pow(-n_lambda)
    weight_x = weight_x.masked_fill(mask_x, 0.0)

    soft_count = y_probs.sum(dim=1).clamp(min=eps)

    # Term A (recall): x -> y
    coverage = torch.sum(
        w_xy * k_ij * weight_y.unsqueeze(1) * y_probs.unsqueeze(1), dim=2
    )
    coverage = coverage.masked_fill(mask_x, 0.0)
    valid_x = (~mask_x).sum(dim=1).float().clamp(min=1.0)
    valid = (~mask_x).to(dtype)
    term_a = ((1.0 - coverage) * valid).sum(dim=1) / valid_x

    # Term B (precision): y -> x
    k_T = k_ij.transpose(1, 2)
    prec = torch.sum(w_yx * k_T * weight_x.unsqueeze(1), dim=2)
    term_b = ((1.0 - prec) * y_probs).sum(dim=1) / soft_count

    # Count regulariser
    target_count = (~mask_x).sum(dim=1).float()
    count_loss = F.smooth_l1_loss(soft_count, target_count, beta=5.0, reduction="none")

    # Extra feature loss (soft correspondences)
    extra_loss = torch.zeros_like(term_a)
    if (x.size(-1) > 3) and (extra_feat_loss is not None):
        x_extra = x[..., 3:]
        y_extra = y[..., 3:]

        w_a = (w_xy * k_ij.detach()).masked_fill(pair_mask, 0.0)
        denom_a = w_a.sum(dim=2, keepdim=True).clamp(min=eps)
        y_extra_matched = torch.matmul(w_a / denom_a, y_extra)

        if extra_feat_loss == "l1":
            err_a = torch.abs(x_extra - y_extra_matched).mean(dim=-1)
        elif extra_feat_loss == "l2":
            err_a = ((x_extra - y_extra_matched) ** 2).mean(dim=-1)
        elif extra_feat_loss == "huber":
            err_a = F.huber_loss(
                y_extra_matched, x_extra, reduction="none", delta=huber_delta
            ).mean(dim=-1)
        else:
            raise ValueError(f"Unknown extra_feat_loss: {extra_feat_loss}")

        err_a = err_a.masked_fill(mask_x, 0.0)
        feat_term_a = err_a.sum(dim=1) / valid_x

        w_b = (w_yx * k_T.detach()).masked_fill(pair_mask_T, 0.0)
        denom_b = w_b.sum(dim=2, keepdim=True).clamp(min=eps)
        x_extra_matched = torch.matmul(w_b / denom_b, x_extra)

        if extra_feat_loss == "l1":
            err_b = torch.abs(y_extra - x_extra_matched).mean(dim=-1)
        elif extra_feat_loss == "l2":
            err_b = ((y_extra - x_extra_matched) ** 2).mean(dim=-1)
        elif extra_feat_loss == "huber":
            err_b = F.huber_loss(
                x_extra_matched, y_extra, reduction="none", delta=huber_delta
            ).mean(dim=-1)

        err_b = err_b.masked_fill(mask_y, 0.0)
        feat_term_b = (err_b * y_probs).sum(dim=1) / soft_count

        extra_loss = 0.5 * (feat_term_a + feat_term_b)

    # Repulsion
    if repulsion_weight > 0.0:
        rep_loss = _repulsion_loss(
            y_xyz=y[..., :3], y_probs=y_probs, mask_y=mask_y,
            gamma=repulsion_gamma, eps=eps,
        )
    else:
        rep_loss = torch.zeros_like(term_a)

    # Raw distance metrics
    dist_a = torch.sum(w_xy * d2.masked_fill(pair_mask, 0.0), dim=2).masked_fill(mask_x, 0.0)
    raw_mean_dist_a_xyz = dist_a.sum(dim=1) / valid_x
    dist_b = torch.sum(
        w_yx * d2.transpose(1, 2).masked_fill(pair_mask_T, 0.0), dim=2
    ).masked_fill(mask_y, 0.0)
    raw_mean_dist_b_xyz = (dist_b * y_probs).sum(dim=1) / soft_count

    total_loss = (
        term_a
        + term_b
        + count_loss_weight * count_loss
        + extra_feat_weight * extra_loss
        + repulsion_weight * rep_loss
    )

    out = (
        total_loss, term_a, term_b, count_loss,
        extra_loss, rep_loss, raw_mean_dist_a_xyz, raw_mean_dist_b_xyz,
    )
    if return_beta:
        out = out + (beta_b,)
    return out
