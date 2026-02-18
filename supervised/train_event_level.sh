#!/bin/bash

# Train the event-level transformer classifier.
# Produces a single label per event (signal vs background).

ROOT_PATH="/path/to/wcsim_numpy"
DATA_PATTERN="/*signoise*"

LOG_DIR="logs/logs"
TB_LOG_DIR="logs/tb_logs"
CHECKPOINT_DIR="checkpoints"
LOG_NAME="event_level"

FEATURE_MODE="no_time_no_charge"
BATCH_SIZE=256
EPOCHS=50
NUM_WORKERS=16
TRAIN_SPLIT=0.95
WARMUP_STEPS=5
DROPOUT=0.1
D_MODEL=192
NHEAD=12
NUM_LAYERS=12
LR=1e-4
GPUS=(0 1)

python -m train.train \
    --root_path "$ROOT_PATH" \
    --data_pattern "$DATA_PATTERN" \
    --log_dir "$LOG_DIR" \
    --tb_log_dir "$TB_LOG_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_name "$LOG_NAME" \
    --feature_mode "$FEATURE_MODE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --num_workers "$NUM_WORKERS" \
    --train_split "$TRAIN_SPLIT" \
    --warmup_steps "$WARMUP_STEPS" \
    --dropout "$DROPOUT" \
    --d_model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num_layers "$NUM_LAYERS" \
    --lr "$LR" \
    --gpus "${GPUS[@]}"
