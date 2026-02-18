#!/bin/bash
# Phase 1: Train autoencoder on background data.
# Update --config to point to the desired YAML configuration.

python train_phase1.py \
    --config configs/mpdr_simple.yaml \
    --name "ae_phase1_background_$(date +%Y%m%d_%H%M%S)" \
    --output_dir experiments \
    --device cuda
