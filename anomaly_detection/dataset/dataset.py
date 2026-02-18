"""Dataset for anomaly detection with variable-length point cloud events."""

import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class NAEDatasetSimple(Dataset):
    """Point cloud dataset for normalising-autoencoder anomaly detection.

    Each event is a variable-length set of PMT hits with spatial coordinates
    projected onto the unit sphere, plus optional timing and charge features.
    """

    FEATURE_DIMS = {
        "all": 5,              # x, y, z, time, charge
        "no_time": 4,          # x, y, z, charge
        "no_charge": 4,        # x, y, z, time
        "no_time_no_charge": 3,  # x, y, z
    }

    # Detector geometry (Hyper-Kamiokande)
    Z_MIN, Z_MAX = -3296.4712, 3296.4712
    RADIUS = 3242.766
    T_MIN, T_MAX = 0, 400
    Q_MIN, Q_MAX = 0, 225.3

    def __init__(self, path, mode="background", time_range=None, augmentations=False):
        """
        Args:
            path: glob root for .npz data files
            mode: 'background', 'signal', or 'mixed'
            time_range: optional (start, end) time window
            augmentations: apply rotation / mirroring
        """
        self.root_dir = path
        self.mode = mode
        self.time_range = time_range
        self.augmentations = augmentations
        self.feature_mode = "all"

        self.z_range = self.Z_MAX - self.Z_MIN
        self._load_data_files()

    def set_feature_mode(self, mode):
        if mode not in self.FEATURE_DIMS:
            raise ValueError(f"Invalid feature_mode '{mode}'. "
                             f"Must be one of {list(self.FEATURE_DIMS)}")
        self.feature_mode = mode

    def get_feature_size(self):
        return self.FEATURE_DIMS[self.feature_mode]

    def _load_data_files(self):
        self.data_files = sorted(glob(f"{self.root_dir}/**/*.npz", recursive=True))
        if not self.data_files:
            raise ValueError(f"No .npz files found in {self.root_dir}")

    def __len__(self):
        return len(self.data_files)

    # ------------------------------------------------------------------
    # Augmentations
    # ------------------------------------------------------------------

    @staticmethod
    def _mirror(coords, axes="xyz"):
        for i, a in enumerate("xyz"):
            if a in axes and np.random.choice([True, False]):
                coords[:, i] *= -1
        return coords

    @staticmethod
    def _rotate(coords, axes="z"):
        for axis in axes:
            angle = np.random.uniform(0, 360)
            coords = R.from_euler(axis, angle, degrees=True).apply(coords)
        return coords

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _to_unit_sphere(coords, eps=1e-8):
        """Project cylinder-surface coordinates to the unit sphere."""
        n = np.linalg.norm(coords, axis=1, keepdims=True)
        return coords / (n + eps)

    # ------------------------------------------------------------------
    # Item loading
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        with np.load(self.data_files[idx]) as data:
            x = data["fPosX"].copy()
            y = data["fPosY"].copy()
            z = data["fPosZ"].copy()
            q = data["fCharge"].copy()
            t = data["fTime"].copy()
            loc = data["fLoc"].copy()
            pmt_flag = data["fPMTFlag"].copy()

        # Filter invalid hits
        valid = (loc >= 0) & (q > 0.0)
        x, y, z, q, t, loc, pmt_flag = (
            x[valid], y[valid], z[valid], q[valid],
            t[valid], loc[valid], pmt_flag[valid],
        )

        event_has_signal = (pmt_flag == 1).any()

        # Mode filtering
        if self.mode == "signal":
            if not event_has_signal:
                x, y, z, q, t, loc, pmt_flag = (
                    x[:0], y[:0], z[:0], q[:0], t[:0], loc[:0], pmt_flag[:0],
                )
        elif self.mode == "background":
            bg = pmt_flag == 0
            x, y, z, q, t, loc, pmt_flag = (
                x[bg], y[bg], z[bg], q[bg], t[bg], loc[bg], pmt_flag[bg],
            )

        # Time window
        if self.time_range is not None:
            lo, hi = self.time_range
            tw = (t >= lo) & (t < hi)
            x, y, z, q, t, loc, pmt_flag = (
                x[tw], y[tw], z[tw], q[tw], t[tw], loc[tw], pmt_flag[tw],
            )
            t = t - lo

        # Augmentations
        if self.augmentations:
            coords = np.stack((x, y, z), axis=1)
            coords = self._rotate(coords, axes="z")
            coords = self._mirror(coords, axes="xyz")
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # Normalise to [-1, 1]
        x_n = np.clip(x / self.RADIUS, -1, 1)
        y_n = np.clip(y / self.RADIUS, -1, 1)
        z_n = np.clip(2 * (z - self.Z_MIN) / self.z_range - 1, -1, 1)
        t_n = np.clip(2 * (t - self.T_MIN) / (self.T_MAX - self.T_MIN) - 1, -1, 1)
        q_n = np.clip(2 * (q - self.Q_MIN) / (self.Q_MAX - self.Q_MIN) - 1, -1, 1)

        # Project to unit sphere
        sphere = self._to_unit_sphere(np.stack((x_n, y_n, z_n), axis=1))

        # Build feature vector
        feats_list = [sphere[:, 0], sphere[:, 1], sphere[:, 2]]
        if self.feature_mode not in ("no_time", "no_time_no_charge"):
            feats_list.append(t_n.astype(np.float32))
        if self.feature_mode not in ("no_charge", "no_time_no_charge"):
            feats_list.append(q_n.astype(np.float32))

        feats = torch.from_numpy(np.stack(feats_list, axis=1).astype(np.float32))
        is_background = torch.tensor([0.0]) if event_has_signal else torch.tensor([1.0])

        return {
            "feats": feats,
            "loc": torch.from_numpy(loc).long(),
            "coords": torch.from_numpy(np.stack([x_n, y_n, z_n], axis=1)).float(),
            "times": torch.from_numpy(t).float(),
            "is_background": is_background,
            "pmt_flag": torch.from_numpy(pmt_flag).float(),
        }
