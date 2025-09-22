# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any


import torch
from torch import nn
from omegaconf import DictConfig
from torch.utils.data import Dataset

from gatr.experiments.base_experiment import BaseExperiment
from gatr.experiments.nbody.dataset import NBodyDataset, NBodyDatasetConfig


class NBodyExperiment(BaseExperiment):
    """Experiment manager for n-body prediction.

    Parameters
    ----------
    cfg : OmegaConf
        Experiment configuration. See the config folder in the repository for examples.
    """

    def __init__(self, cfg):
        print(
            f"[NBodyExperiment] Initializing N-Body experiment with run name: {cfg.get('run_name', 'unknown')}"
        )
        super().__init__(cfg)
        self._mse_criterion = torch.nn.MSELoss()
        self._mae_criterion = torch.nn.L1Loss(reduction="mean")

        # Create dataset config
        self.dataset_config = NBodyDatasetConfig(
            use_gmcnn=cfg.get("data", {}).get("use_gmcnn", False),  # False for now
            input_channels=cfg.get("model", {}).get("input_channels", 7),
            output_channels=cfg.get("model", {}).get("output_channels", 3),
        )

    def _load_dataset(self, tag: str):
        """Loads dataset.

        Parameters
        ----------
        tag : str
            Dataset tag, like "train", "val", or one of self._eval_tags.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Dataset.
        """
        print(f"[NBodyExperiment] Loading dataset: {tag}")

        if tag == "train":
            subsample_fraction = self.cfg.data.subsample
        else:
            subsample_fraction = None

        filename = Path(self.cfg.data.data_dir) / f"{tag}.npz"
        print(f"[NBodyExperiment] Dataset file: {filename}")
        print(f"[NBodyExperiment] Subsample fraction: {subsample_fraction}")
        keep_trajectories = tag == "val"
        dataset = NBodyDataset(
            filename,
            subsample=subsample_fraction,
            keep_trajectories=keep_trajectories,
            config=self.dataset_config,
        )

        print(f"[NBodyExperiment] Loaded {tag} dataset with {len(dataset)} samples")
        return dataset

    def _forward(self, *data):
        """Model forward pass.

        Parameters
        ----------
        data : tuple of torch.Tensor
            Data batch.

        Returns
        -------
        loss : torch.Tensor
            Loss
        metrics : dict with str keys and float values
            Additional metrics for logging
        """

        # Forward pass
        assert self.model is not None
        x, y = data
        y_pred, reg = self.model(x)

        # Compute loss
        mse = self._mse_criterion(y_pred, y)
        output_reg = torch.mean(reg)
        loss = mse + self.cfg.training.output_regularization * output_reg

        # Additional metrics
        mae = self._mae_criterion(y_pred, y)
        metrics = dict(
            mse=mse.item(), rmse=loss.item() ** 0.5, output_reg=output_reg.item(), mae=mae.item()
        )

        return loss, metrics

    @property
    def _eval_dataset_tags(self):
        """Eval dataset tags.

        Returns
        -------
        tags : iterable of str
            Eval dataset tags
        """

        # Only evaluate on object_generalization dataset when method supports variable token number
        assert self.model is not None
        if hasattr(self.model, "supports_variable_items") and self.model.supports_variable_items:
            eval_tags = {"eval", "e3_generalization", "object_generalization"}
        else:
            eval_tags = {"eval", "e3_generalization"}
        print(f"[NBodyExperiment] Evaluation dataset tags: {eval_tags}")
        return eval_tags
