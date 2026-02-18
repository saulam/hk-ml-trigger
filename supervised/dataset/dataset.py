import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from glob import glob
from torch.utils.data import Dataset


class HKDataset(Dataset):
    """Dataset for Hyper-Kamiokande PMT hit sequences.

    Each sample is a variable-length set of PMT hits with spatial coordinates,
    timing, and charge information. Hits are represented in cylindrical
    coordinates with Fourier feature encoding.
    """

    def __init__(self, path, time_range=None):
        self.root_dir = path
        self.data_files = sorted(glob(f"{self.root_dir}/*.npz"))

        # Detector geometry
        self.z_min = -3296.4712
        self.z_max = 3296.4712
        self.radius = 3242.766
        self.t_min = 0
        self.t_max = 400
        self.K_z = 4
        self.K_r = 4
        self.K_t = 6
        self.q_mean = 3.780056
        self.q_std = 0.95529866

        self.z_range = self.z_max - self.z_min
        self.t_range = self.t_max - self.t_min

        self.time_range = time_range
        self.augmentations = False
        self.feature_mode = "all"

    def set_feature_mode(self, mode):
        valid_modes = ["all", "no_time", "no_charge", "no_time_no_charge"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid feature_mode '{mode}'. Must be one of {valid_modes}")
        self.feature_mode = mode

    def get_feature_size(self):
        """Return the number of features per hit for the current mode."""
        base_size = 3
        spatial_size = 2 * self.K_z
        time_size = 2 * self.K_t
        charge_size = 1

        feature_size = base_size + spatial_size
        if self.feature_mode not in ["no_time", "no_time_no_charge"]:
            feature_size += time_size
        if self.feature_mode not in ["no_charge", "no_time_no_charge"]:
            feature_size += charge_size
        return feature_size

    def mirror(self, coords, selected_axes=("x", "y", "z")):
        axes = ["x", "y", "z"]
        for axis in range(3):
            if axes[axis] in selected_axes and np.random.choice([True, False]):
                coords[:, axis] *= -1
        return coords

    def fourier_feats_np(self, vals, K):
        feats = [np.sin(2 * np.pi * k * vals) for k in range(1, K + 1)]
        feats += [np.cos(2 * np.pi * k * vals) for k in range(1, K + 1)]
        return np.stack(feats, axis=1)

    def rotate(self, coords, selected_axes=("x", "y", "z")):
        reference_point = np.array([0, 0, 0])
        shifted_coords = coords - reference_point
        for axis in selected_axes:
            angle_deg = np.random.uniform(0, 360)
            rotation = R.from_euler(axis, angle_deg, degrees=True)
            shifted_coords = rotation.apply(shifted_coords)
        return shifted_coords + reference_point

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])

        x = data["fPosX"]
        y = data["fPosY"]
        z = data["fPosZ"]
        q = data["fCharge"]
        t = data["fTime"]
        l = data["fLoc"]
        pmt_flag = data["fPMTFlag"]

        mask = (l >= 0) & (q > 0.0)
        x, y, z, q, t, l, pmt_flag = (
            x[mask], y[mask], z[mask], q[mask], t[mask], l[mask], pmt_flag[mask]
        )

        if self.time_range is not None:
            start, end = self.time_range
            mask = (t >= start) & (t < end)
            x, y, z, q, t, l, pmt_flag = (
                x[mask], y[mask], z[mask], q[mask], t[mask], l[mask], pmt_flag[mask]
            )
            t -= start

        if self.augmentations:
            coords = np.stack((x, y, z), axis=1)
            coords = self.rotate(coords, selected_axes=["z"])
            coords = self.mirror(coords, selected_axes=["x", "y", "z"])
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        phi = np.arctan2(y, x)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        z_norm = np.clip((z - self.z_min) / self.z_range, 0, 1)
        r = np.where(l == 1, self.radius, np.sqrt(x ** 2 + y ** 2))
        r_norm = np.where(l == 1, 1.0, np.clip(r / self.radius, 0, 1))
        t_norm = np.clip((t - self.t_min) / self.t_range, 0, 1)

        fz = self.fourier_feats_np(z_norm, self.K_z)
        fr = self.fourier_feats_np(r_norm, self.K_r)
        ft = self.fourier_feats_np(t_norm, self.K_t)

        q_log = np.log1p(q)

        barrel_base = np.stack([sin_phi, cos_phi, z_norm], axis=1)
        cap_base = np.stack([r_norm, sin_phi, cos_phi], axis=1)
        mask_wall = (l == 1)[:, None]
        base = np.where(mask_wall, barrel_base, cap_base)
        spatial_feats = np.where(mask_wall, fz, fr)

        feature_list = [
            base.astype(np.float32),
            spatial_feats.astype(np.float32),
        ]
        if self.feature_mode not in ["no_time", "no_time_no_charge"]:
            feature_list.append(ft.astype(np.float32))
        if self.feature_mode not in ["no_charge", "no_time_no_charge"]:
            feature_list.append(q_log[:, None].astype(np.float32))

        feats_np = np.concatenate(feature_list, axis=1)

        feats = torch.from_numpy(feats_np)
        loc = torch.from_numpy(l).long()
        coords_t = torch.from_numpy(np.stack([x, y, z], axis=1)).float()
        times_t = torch.from_numpy(t).float()
        pmt_flag_t = torch.from_numpy(pmt_flag).float()
        label = torch.tensor([1.0]) if pmt_flag_t.sum() > 0 else torch.tensor([0.0])

        del data
        return {
            "feats": feats,
            "loc": loc,
            "coords": coords_t,
            "times": times_t,
            "pmt_flag": pmt_flag_t,
            "label": label,
        }
