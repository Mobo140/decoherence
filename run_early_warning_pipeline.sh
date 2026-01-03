#!/bin/bash
# Pipeline script for running complete early warning experiments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Early Warning System Pipeline${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Parse arguments
EXPERIMENT=${1:-E5}
SKIP_DATASET=${2:-false}
SKIP_TRAIN=${3:-false}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Experiment: $EXPERIMENT"
echo "  Skip dataset generation: $SKIP_DATASET"
echo "  Skip training: $SKIP_TRAIN"
echo ""

# Paths
CONFIG="configs/experiments/${EXPERIMENT}.yaml"
DATASET_PATH="data/${EXPERIMENT}_dataset.npz"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT}"
OUTPUT_DIR="figures/${EXPERIMENT}"
RUNS_DIR="runs/${EXPERIMENT}"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    echo "Available experiments: E5, E6, E7"
    exit 1
fi

# Step 1: Generate dataset
if [ "$SKIP_DATASET" = "false" ]; then
    echo -e "${GREEN}[1/3] Generating dataset...${NC}"
    python3 -m src.data.generate_dataset \
        --config "$CONFIG" \
        --sliding-window
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dataset generated successfully${NC}"
    else
        echo -e "${RED}✗ Dataset generation failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[1/3] Skipping dataset generation${NC}"
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo -e "${RED}Error: Dataset not found: $DATASET_PATH${NC}"
        echo "Run without --skip-dataset to generate it"
        exit 1
    fi
fi

echo ""

# Step 2: Train model
if [ "$SKIP_TRAIN" = "false" ]; then
    echo -e "${GREEN}[2/3] Training model...${NC}"
    python -m src.train.train_multitask \
        --config "$CONFIG" \
        --output-dir "$RUNS_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed successfully${NC}"
    else
        echo -e "${RED}✗ Training failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[2/3] Skipping training${NC}"
    
    # Check if checkpoint exists
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo -e "${RED}Error: Checkpoint directory not found: $CHECKPOINT_DIR${NC}"
        echo "Run without --skip-train to train the model"
        exit 1
    fi
fi

echo ""

# Step 3: Evaluate and visualize
echo -e "${GREEN}[3/3] Evaluating model and generating visualizations...${NC}"
python3 -m src.train.eval_multitask \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --split test

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluation completed successfully${NC}"
else
    echo -e "${RED}✗ Evaluation failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Results:"
echo "  Dataset: $DATASET_PATH"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Training logs: $RUNS_DIR"
echo "  Figures & metrics: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  - Metrics: cat $OUTPUT_DIR/metrics.json"
echo "  - Plots: open $OUTPUT_DIR/*.png"
echo ""

# If E7 (control experiment), show additional instructions
if [ "$EXPERIMENT" = "E7" ]; then
    echo -e "${YELLOW}Note: For E7 control baseline comparison:${NC}"
    echo "  Run the control evaluation script to compare with/without control"
    echo "  See README_EARLY_WARNING.md for details"
    echo ""
fi
