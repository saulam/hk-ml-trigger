"""Langevin Monte Carlo samplers for MPDR negative generation."""

import numpy as np
import torch
import torch.autograd as autograd


def orthogonalize_batch_v2(v, p):
    """Project v onto the plane orthogonal to unit vector p."""
    return v - (v * p).sum(dim=1, keepdim=True) * p


def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * (
        (norm < max_norm).to(torch.float)
        + (norm > max_norm).to(torch.float) * max_norm / norm
        + 1e-6
    )
    return x


def sample_langevin_v2_proj(
    x, model, stepsize, n_step, noise_scale=None,
    bound=None, clip_grad=None, reject_boundary=False,
    noise_anneal=None, noise_anneal_v2=None,
    mh=False, temperature=None, normalize_grad=False,
    project_fn=None, store_traj=False,
):
    """Langevin Monte Carlo with optional projection after each step.

    Args:
        x: Initial points.
        model: Energy function returning scalar energy per sample.
        stepsize: Gradient step size.
        n_step: Number of MCMC steps.
        noise_scale: Noise standard deviation (default: sqrt(2 * stepsize)).
        bound: Domain constraint ('spherical' or (low, high) tuple).
        clip_grad: Gradient clipping value.
        mh: Use Metropolis-Hastings acceptance.
        temperature: Sampling temperature.
        normalize_grad: Normalise gradients to unit norm.
        project_fn: Optional projection applied after each step.
        store_traj: Store full trajectory for diagnostics.
    """
    assert not (stepsize is None and noise_scale is None)
    if mh and project_fn is not None:
        raise ValueError("MH is not valid with project_fn.")

    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    if stepsize is None:
        stepsize = (noise_scale ** 2) / 2
    noise_scale_ = noise_scale
    stepsize_ = stepsize
    if temperature is None:
        temperature = 1.0

    def _tangent_project_xyz(v_xyz, x_xyz):
        return v_xyz - (v_xyz * x_xyz).sum(dim=-1, keepdim=True) * x_xyz

    def _tangent_project_xyz_inplace(tensor, ref_xyz):
        if tensor.dim() == 3 and tensor.size(-1) >= 3:
            with torch.no_grad():
                xyz_proj = _tangent_project_xyz(tensor[..., :3], ref_xyz)
            if tensor.size(-1) == 3:
                return xyz_proj
            return torch.cat([xyz_proj, tensor[..., 3:]], dim=-1)
        return tensor

    if project_fn is not None:
        x = project_fn(x)
    x = x.detach().requires_grad_(True)

    E_x = model(x)
    grad_E_x = autograd.grad(E_x.sum(), x, only_inputs=True)[0]
    if bound == "spherical":
        grad_E_x = orthogonalize_batch_v2(grad_E_x, x)
    if clip_grad is not None:
        grad_E_x.clamp_(-clip_grad, clip_grad)
    if normalize_grad:
        grad_E_x = grad_E_x / grad_E_x.norm(dim=1, keepdim=True)
    if x.dim() == 3 and x.size(-1) >= 3:
        x_xyz_det = x[..., :3].detach()
        grad_E_x = _tangent_project_xyz_inplace(grad_E_x, x_xyz_det)

    if store_traj:
        l_sample = [x.detach().cpu()]
        l_E = [E_x.detach().cpu()]
        l_dynamics, l_diffusion = [], []
    else:
        l_sample = l_E = None
        l_dynamics = l_diffusion = None
    l_drift = []
    l_accept = []
    l_drift_rms = []
    l_noise_rms = []

    for i_step in range(n_step):
        drift = -stepsize_ * grad_E_x / temperature
        noise = torch.randn_like(x) * noise_scale_
        if x.dim() == 3 and x.size(-1) >= 3:
            x_xyz_det = x[..., :3].detach()
            noise = _tangent_project_xyz_inplace(noise, x_xyz_det)
        dynamics = drift + noise
        y = x + dynamics

        with torch.no_grad():
            l_drift_rms.append(drift.pow(2).mean().sqrt().cpu())
            l_noise_rms.append(noise.pow(2).mean().sqrt().cpu())

        if bound == "spherical":
            y = y / y.norm(dim=1, p=2, keepdim=True)
        elif bound is not None:
            if reject_boundary:
                accept = ((y >= bound[0]) & (y <= bound[1])).view(len(x), -1).all(dim=1)
                reject = ~accept
                y[reject] = x[reject]
            else:
                y = torch.clamp(y, bound[0], bound[1])

        if project_fn is not None:
            y = project_fn(y)
        y = y.detach().requires_grad_(True)

        E_y = model(y)
        grad_E_y = autograd.grad(E_y.sum(), y, only_inputs=True)[0]
        if bound == "spherical":
            grad_E_y = orthogonalize_batch_v2(grad_E_y, y)
        if clip_grad is not None:
            grad_E_y.clamp_(-clip_grad, clip_grad)
        if normalize_grad:
            grad_E_y = grad_E_y / grad_E_y.norm(dim=1, p=2, keepdim=True)
        if y.dim() == 3 and y.size(-1) >= 3:
            y_xyz_det = y[..., :3].detach()
            grad_E_y = _tangent_project_xyz_inplace(grad_E_y, y_xyz_det)

        l_drift.append(drift.detach().cpu())

        if mh:
            y_to_x = (
                -0.5
                * (x - (y - stepsize_ * grad_E_y / temperature))
                .view(len(x), -1)
                .pow(2)
                .sum(dim=1)
                / noise_scale_ ** 2
            )
            x_to_y = (
                -0.5
                * (y - (x - stepsize_ * grad_E_x / temperature))
                .view(len(x), -1)
                .pow(2)
                .sum(dim=1)
                / noise_scale_ ** 2
            )
            transition = y_to_x - x_to_y
            prob = (-E_y + E_x).flatten() / temperature
            accept_prob = torch.exp(transition + prob)
            reject = torch.rand_like(accept_prob) > accept_prob
            y[reject] = x[reject]
            E_y[reject] = E_x[reject]
            grad_E_y[reject] = grad_E_x[reject]
            l_accept.append((~reject).detach().cpu())

        x = y
        E_x = E_y
        grad_E_x = grad_E_y

        if noise_anneal is not None:
            noise_scale_ = noise_scale / (1 + i_step)
        if noise_anneal_v2 is not None:
            noise_scale_ = noise_scale / (1 + i_step)
            stepsize_ = stepsize / ((1 + i_step) ** 2)

        if store_traj:
            l_dynamics.append(dynamics.detach().cpu())
            l_diffusion.append(noise.detach().cpu())
            l_sample.append(x.detach().cpu())
            l_E.append(E_x.detach().cpu())

    d_result = {
        "sample": x.detach(),
        "l_drift": l_drift,
        "l_accept": l_accept,
        "l_drift_rms_mean": (
            torch.stack(l_drift_rms).mean() if l_drift_rms else torch.tensor(0.0)
        ),
        "l_noise_rms_mean": (
            torch.stack(l_noise_rms).mean() if l_noise_rms else torch.tensor(0.0)
        ),
    }
    if store_traj:
        d_result.update({
            "l_sample": torch.stack(l_sample),
            "l_dynamics": l_dynamics,
            "l_diffusion": l_diffusion,
            "l_E": torch.stack(l_E),
        })
    return d_result
