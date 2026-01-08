#!/bin/bash

# ==========================================
# Dynamic AlphaSteer Pipeline
# Combines null-space projection with adaptive λ(x)
# ==========================================

set -e  # Exit on error

cd "$(dirname "$0")/.." || exit

echo "====================================================="
echo "Starting Dynamic AlphaSteer Pipeline"
echo "Working Directory: $(pwd)"
echo "====================================================="

# =========================
# Configuration
# =========================

TRAIN_VAL_DIR=data/instructions/train_val
EMBEDDING_DIR=data/embeddings/llama3.1
NICKNAME=llama3.1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
DEVICE=cuda:0

STEERING_MATRIX_PATH=data/steering_matrix/steering_matrix_${NICKNAME}.pt
GATING_DIR=data/gating_networks/${NICKNAME}
GENERATE_CONFIG_DIR=config/llama3.1

# =========================
# Step 1: Extract Embeddings (if not exists)
# =========================

echo ""
echo "====================================================="
echo "Step 1: Extract Embeddings (if needed)"
echo "====================================================="

mkdir -p $EMBEDDING_DIR

for file in $TRAIN_VAL_DIR/*.json; do
    filename=$(basename "$file" .json)
    
    if [ ! -f "$EMBEDDING_DIR/embeds_$filename.pt" ]; then
        echo "[1/4] Extracting embeddings for $file"
        
        if [[ "$filename" == *"coconot"* ]]; then
            prompt_column="prompt"
        else
            prompt_column="query"
        fi
        
        python src/extract_embeddings.py --model_name $MODEL_NAME \
                                        --input_file $file \
                                        --prompt_column "$prompt_column" \
                                        --output_file $EMBEDDING_DIR/embeds_$filename.pt \
                                        --batch_size 16 \
                                        --device $DEVICE
    else
        echo "Embeddings for $filename already exist. Skipping."
    fi
done

# =========================
# Step 2: Calculate Steering Matrix (if not exists)
# =========================

echo ""
echo "====================================================="
echo "Step 2: Calculate Steering Matrix (if needed)"
echo "====================================================="

mkdir -p $(dirname $STEERING_MATRIX_PATH)

if [ ! -f "$STEERING_MATRIX_PATH" ]; then
    echo "[2/4] Calculating steering matrix for $NICKNAME"
    python src/calc_steering_matrix.py --model_name $NICKNAME \
                                        --embedding_dir $EMBEDDING_DIR \
                                        --device $DEVICE \
                                        --save_path $STEERING_MATRIX_PATH
else
    echo "Steering matrix already exists: $STEERING_MATRIX_PATH"
fi

# =========================
# Step 3: Train Gating Networks (NEW!)
# =========================

echo ""
echo "====================================================="
echo "Step 3: Train Gating Networks (Dynamic λ prediction)"
echo "====================================================="

mkdir -p $GATING_DIR

# Check if gating networks already exist
GATING_EXISTS=false
if [ -d "$GATING_DIR" ] && [ "$(ls -A $GATING_DIR/*.pt 2>/dev/null)" ]; then
    GATING_EXISTS=true
fi

if [ "$GATING_EXISTS" = false ]; then
    echo "[3/4] Training gating networks..."
    python src/train_gating_networks.py --model_name $NICKNAME \
                                        --embedding_dir $EMBEDDING_DIR \
                                        --device $DEVICE \
                                        --save_dir $GATING_DIR \
                                        --hidden_dim 512 \
                                        --num_epochs 100 \
                                        --batch_size 256
else
    echo "Gating networks already exist in $GATING_DIR"
fi

# =========================
# Step 4: Generate Responses with Dynamic Steering
# =========================

echo ""
echo "====================================================="
echo "Step 4: Generate Responses with Dynamic λ(x)"
echo "====================================================="

mkdir -p data/responses/$NICKNAME

if [ -d "$GENERATE_CONFIG_DIR" ]; then
    for file in $GENERATE_CONFIG_DIR/*.yaml; do
        filename=$(basename "$file" .yaml)
        echo "[4/4] Generating dynamic responses for $filename"
        
        # Create temporary config with gating_dir (properly formatted YAML)
        temp_config=$(mktemp --suffix=.yaml)
        cat $file > $temp_config
        # Ensure there's a newline before adding new field
        echo "" >> $temp_config
        echo "gating_dir: \"$GATING_DIR\"" >> $temp_config
        
        python src/generate_dynamic_response.py --config_path $temp_config
        
        rm $temp_config
    done
else
    echo "Error: Config directory $GENERATE_CONFIG_DIR does not exist!"
    exit 1
fi

echo ""
echo "====================================================="
echo "Dynamic AlphaSteer Pipeline Complete!"
echo "Steering: $STEERING_MATRIX_PATH"
echo "Gating Networks: $GATING_DIR"
echo "Responses: data/responses/$NICKNAME/"
echo "====================================================="