#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

print("DEBUG: Starting nbody_experiment.py script...")

import hydra

print("DEBUG: Hydra imported successfully")

from gatr.experiments.nbody import NBodyExperiment

# Add at beginning of script
import torch

print("DEBUG: PyTorch imported successfully")

print("DEBUG: About to import NBodyExperiment...")
from gatr.experiments.nbody import NBodyExperiment

print("DEBUG: NBodyExperiment imported successfully")

print(f"DEBUG: CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"DEBUG: CUDA version: {torch.version.cuda}")
    print(f"DEBUG: GPU device: {torch.cuda.get_device_name(0)}")


@hydra.main(config_path="../config", config_name="nbody", version_base=None)
def main(cfg):
    """Entry point for n-body experiment."""
    print("DEBUG: Inside main function")
    print(f"DEBUG: Config received: {cfg.model}")
    print("DEBUG: About to create NBodyExperiment...")
    exp = NBodyExperiment(cfg)
    print("DEBUG: NBodyExperiment created successfully")
    exp()


if __name__ == "__main__":
    main()
