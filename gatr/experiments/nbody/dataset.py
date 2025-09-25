# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class NBodyDatasetConfig:
    """Configuration for NBodyDataset."""

    use_gmcnn: bool = False
    input_channels: int = 7
    output_channels: int = 3


class NBodyDataset(torch.utils.data.Dataset):
    """N-body prediction dataset.

    Loads data generated with generate_nbody_dataset.py from disk.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the npz file with the dataset to be loaded.
    subsample : None or float
        If not None, defines the fraction of the dataset to be used. For instance, `subsample=0.1`
        uses just 10% of the samples in the dataset.
    keep_trajectories : bool
        Whether to keep the full particle trajectories in the dataset. They are neither needed
        for training nor evaluation, but can be useful for visualization.
    config : Optional[NBodyDatasetConfig]
        Configuration for the dataset
    """

    def __init__(
        self,
        filename,
        subsample=None,
        keep_trajectories=False,
        config: Optional[NBodyDatasetConfig] = None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        filename : Path
            Path to the data file
        subsample : Optional[float]
            Fraction of data to use
        keep_trajectories : bool
            Whether to keep full trajectories
        config : Optional[NBodyDatasetConfig]
            Configuration for the dataset
        """
        super().__init__()
        self.config = config  # Assign config to self.config

        # Handle the case where config might be None
        use_gmcnn = self.config.use_gmcnn if self.config is not None else False
        print(f"[NBodyDataset] Dataset config: use_gmcnn={use_gmcnn}")
        print(f"[NBodyDataset] Subsample fraction: {subsample}")
        print(f"[NBodyDataset] Keep trajectories: {keep_trajectories}")

        self.x, self.y, self.trajectories = self._load_data(
            filename, subsample, keep_trajectories=keep_trajectories
        )

        print(f"[NBodyDataset] Dataset initialized with {len(self.x)} samples")
        print(f"[NBodyDataset] Input shape: {self.x.shape}, Output shape: {self.y.shape}")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Returns the `idx`-th sample from the dataset."""
        x, y = self.x[idx], self.y[idx]

        if idx == 0:  # Only print for first sample to avoid spam
            print(
                f"[NBodyDataset] Getting sample {idx} - Original shapes: x={x.shape}, y={y.shape}"
            )

        return x, y

    @staticmethod
    def _load_data(filename, subsample=None, keep_trajectories=False):
        """Loads data from file and converts to input and output tensors."""
        print(f"[NBodyDataset] Loading data from {filename}")

        # Load data from file
        npz = np.load(filename, "r")
        print(f"[NBodyDataset] NPZ file keys: {list(npz.keys())}")
        m, x_initial, v_initial, x_final = (
            npz["m"],
            npz["x_initial"],
            npz["v_initial"],
            npz["x_final"],
        )

        print(f"[NBodyDataset] Raw data shapes:")
        print(f"[NBodyDataset]   masses: {m.shape}")
        print(f"[NBodyDataset]   x_initial: {x_initial.shape}")
        print(f"[NBodyDataset]   v_initial: {v_initial.shape}")
        print(f"[NBodyDataset]   x_final: {x_final.shape}")

        # Convert to tensors
        m = torch.from_numpy(m).to(torch.float32).unsqueeze(2)
        x_initial = torch.from_numpy(x_initial).to(torch.float32)
        v_initial = torch.from_numpy(v_initial).to(torch.float32)
        x_final = torch.from_numpy(x_final).to(torch.float32)

        print(f"[NBodyDataset] After tensor conversion:")
        print(f"[NBodyDataset]   masses: {m.shape}")
        print(f"[NBodyDataset]   x_initial: {x_initial.shape}")
        print(f"[NBodyDataset]   v_initial: {v_initial.shape}")
        print(f"[NBodyDataset]   x_final: {x_final.shape}")

        # Concatenate into inputs and outputs
        x = torch.cat((m, x_initial, v_initial), dim=2)  # (batchsize, num_objects, 7)
        y = x_final  # (batchsize, num_objects, 3)

        print(f"[NBodyDataset] Final concatenated shapes: x={x.shape}, y={y.shape}")

        # Optionally, keep raw trajectories around (for plotting)
        if keep_trajectories:
            trajectories = npz["trajectories"]
            print(f"[NBodyDataset] Kept trajectories with shape: {trajectories.shape}")
        else:
            trajectories = None
            print(f"[NBodyDataset] No trajectories kept")

        # Subsample
        if subsample is not None and subsample < 1.0:
            n_original = len(x)
            n_keep = int(round(subsample * n_original))
            assert 0 < n_keep <= n_original
            print(
                f"[NBodyDataset] Subsampling: {n_original} -> {n_keep} samples ({subsample*100:.1f}%)"
            )

            x = x[:n_keep]
            y = y[:n_keep]
            if trajectories is not None:
                trajectories = trajectories[:n_keep]
        else:
            print(f"[NBodyDataset] No subsampling applied, keeping all {len(x)} samples")

        print(f"[NBodyDataset] Data loading complete")
        return x, y, trajectories
