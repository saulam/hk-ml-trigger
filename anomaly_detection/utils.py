"""Utility functions for MPDR training."""

import torch
import numpy as np


def collate_variable_length(batch):
    """Collate variable-length point cloud events into a padded batch.

    Returns a dict with:
        feats:         (B, max_N, F) padded features
        mask:          (B, max_N)    True = padding
        lengths:       (B,)         original hit counts
        is_background: (B,)         event labels
    """
    batch = [item for item in batch if item["feats"].size(0) > 0]
    lengths = [item["feats"].size(0) for item in batch]
    max_len = max(max(lengths), 1)
    bs = len(batch)
    feat_dim = batch[0]["feats"].size(1)

    feats_padded = torch.zeros(bs, max_len, feat_dim)
    mask = torch.ones(bs, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = lengths[i]
        feats_padded[i, :n] = item["feats"]
        mask[i, :n] = False

    return {
        "feats": feats_padded,
        "mask": mask,
        "lengths": torch.tensor(lengths),
        "is_background": torch.stack([item["is_background"] for item in batch]),
        "loc": [item["loc"] for item in batch],
        "coords": [item["coords"] for item in batch],
        "times": [item["times"] for item in batch],
        "pmt_flag": [item["pmt_flag"] for item in batch],
    }
