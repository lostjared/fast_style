#!/bin/bash

TYLE_IMAGE=$1
if [ -z "$STYLE_IMAGE" ]; then
    echo "Usage: ./glitch_factory.sh [path_to_style_image.jpg]"
    exit 1
fi

MODEL_DIR="./my_glitch_models"
DATASET_PATH="/home/jared/Drives/crucial/datasets/coco" #
TEMP_EXPORT="temp_export.onnx"
FINAL_ONNX="$2"

mkdir -p $(MODEL_DIR)
cho "Setting GPU power limits and persistence mode..."
ource glitch_env/bin/activate
cho "Starting training against $STYLE_IMAGE..."
python neural_style.py train \
    --dataset "$DATASET_PATH" \
    --style-image "$STYLE_IMAGE" \
    --save-model-dir "$MODEL_DIR" \
    --epochs 2 \
    --accel \
    --batch-size 4 \
    --log-interval 100

ATEST_MODEL=$(ls -t $MODEL_DIR/*.model | head -1)
echo "Training complete. Using model: $LATEST_MODEL"

echo "Exporting to ONNX..."
python neural_style.py eval \
    --model "$LATEST_MODEL" \
    --content-image "output.jpg" \
    --output-image "factory_test.jpg" \
    --export_onnx "$TEMP_EXPORT" \
    --accel

echo "Simplifying ONNX model for ACMX2..."
if command -v onnxsim &> /dev/null; then
    python -m onnxsim "$TEMP_EXPORT" "$FINAL_ONNX"
    rm "$TEMP_EXPORT"
    echo "Success! Final model ready at: $FINAL_ONNX"
else
    echo "Warning: onnxsim not found. Move $TEMP_EXPORT to your models folder."
fi

echo "Pipeline finished. Check factory_test.jpg for style preview."
