#!/bin/bash
# Phase 2: Train MPDR-R (Recovery) energy function.
# Requires a pre-trained autoencoder checkpoint from Phase 1.
#
# Usage:
#   ./scripts/train_mpdr_recovery.sh <path/to/ae_checkpoint.ckpt>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <ae_checkpoint_path>"
    echo ""
    echo "First train the autoencoder (Phase 1):"
    echo "  ./scripts/train_phase1.sh"
    echo ""
    echo "Then provide the checkpoint path:"
    echo "  $0 experiments/ae_phase1_*/checkpoints/last.ckpt"
    exit 1
fi

AE_CHECKPOINT=$1

python train_phase2.py \
    --variant recovery \
    --config configs/mpdr_recovery.yaml \
    --ae_checkpoint "${AE_CHECKPOINT}" \
    --name "mpdr_recovery_$(date +%Y%m%d_%H%M%S)" \
    --output_dir experiments \
    --device cuda
