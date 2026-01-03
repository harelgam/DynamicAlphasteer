#!/bin/bash

# ==========================================
# 1. Hugging Face Token Setup
# ==========================================

cd /gpfs0/bgu-ataitler/users/harelgam/AlphaSteer || exit

# =========================
# Configuration for Llama 3.1
# =========================

TRAIN_VAL_DIR=data/instructions/train_val
EMBEDDING_DIR=data/embeddings/llama3.1
NICKNAME=llama3.1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
DEVICE=cuda:0

# יצירת תיקיות
mkdir -p $EMBEDDING_DIR
mkdir -p data/steering_matrix
mkdir -p data/responses/$NICKNAME

echo "====================================================="
echo "Starting AlphaSteer Pipeline for $NICKNAME"
echo "Working Directory: $(pwd)"
echo "====================================================="

# ---------------------------------------------------------
# Step 1: Extract Embeddings
# ---------------------------------------------------------
for file in $TRAIN_VAL_DIR/*.json; do
    filename=$(basename "$file" .json)
    
    if [ ! -f "$EMBEDDING_DIR/embeds_$filename.pt" ]; then
        echo "[1/3] Extracting embeddings for $file"
        
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

# ---------------------------------------------------------
# Step 2: Calculate Steering Matrix
# ---------------------------------------------------------
STEERING_SAVE_PATH=data/steering_matrix/steering_matrix_${NICKNAME}.pt
echo "[2/3] Calculating steering matrix for $NICKNAME"

python src/calc_steering_matrix.py --model_name $NICKNAME \
                                    --embedding_dir $EMBEDDING_DIR \
                                    --device $DEVICE \
                                    --save_path $STEERING_SAVE_PATH

# ---------------------------------------------------------
# Step 3: Generation (Steering)
# ---------------------------------------------------------
GENERATE_CONFIG_DIR=config/llama3.1
echo "[3/3] Generating responses for $NICKNAME"

if [ -d "$GENERATE_CONFIG_DIR" ]; then
    for file in $GENERATE_CONFIG_DIR/*.yaml; do
        filename=$(basename "$file" .yaml)
        echo "Generating response for $file"
        python src/generate_response.py --config_path $file
    done
else
    echo "Error: Config directory $GENERATE_CONFIG_DIR does not exist!"
fi

echo "Pipeline finished successfully!"