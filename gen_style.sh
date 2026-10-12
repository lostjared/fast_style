#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./gen_style.sh --style-image FILE --dataset DIR --content-image FILE --output-onnx FILE [options]

Required:
  --style-image FILE    Image whose style will be learned
  --dataset DIR         ImageFolder dataset (for example, DIR/images/*.jpg)
  --content-image FILE  Image used for the preview and ONNX export
  --output-onnx FILE    Destination for the exported model

Options:
  --model-dir DIR       Training output directory (default: ./my_glitch_models)
  --preview-image FILE  Preview destination (default: ./factory_test.jpg)
  --accel               Use an available PyTorch accelerator
  -h, --help            Show this help
EOF
}

STYLE_IMAGE=""
DATASET_PATH=""
CONTENT_IMAGE=""
FINAL_ONNX=""
MODEL_DIR="./my_glitch_models"
PREVIEW_IMAGE="./factory_test.jpg"
ACCEL_ARGS=()

while (($#)); do
    case "$1" in
        --style-image|--dataset|--content-image|--output-onnx|--model-dir|--preview-image)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            case "$1" in
                --style-image) STYLE_IMAGE="$2" ;;
                --dataset) DATASET_PATH="$2" ;;
                --content-image) CONTENT_IMAGE="$2" ;;
                --output-onnx) FINAL_ONNX="$2" ;;
                --model-dir) MODEL_DIR="$2" ;;
                --preview-image) PREVIEW_IMAGE="$2" ;;
            esac
            shift 2
            ;;
        --accel)
            ACCEL_ARGS=(--accel)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$STYLE_IMAGE" || -z "$DATASET_PATH" || -z "$CONTENT_IMAGE" || -z "$FINAL_ONNX" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -f "$STYLE_IMAGE" || ! -d "$DATASET_PATH" || ! -f "$CONTENT_IMAGE" ]]; then
    echo "Style image, dataset directory, or content image does not exist." >&2
    exit 2
fi
if [[ "$FINAL_ONNX" != *.onnx ]]; then
    echo "--output-onnx must end in .onnx" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p -- "$MODEL_DIR" "$(dirname -- "$FINAL_ONNX")" "$(dirname -- "$PREVIEW_IMAGE")"

echo "Training with style image: $STYLE_IMAGE"
python "$SCRIPT_DIR/neural_style.py" train \
    --dataset "$DATASET_PATH" \
    --style-image "$STYLE_IMAGE" \
    --save-model-dir "$MODEL_DIR" \
    --epochs 2 \
    --batch-size 4 \
    --log-interval 100 \
    "${ACCEL_ARGS[@]}"

shopt -s nullglob
models=("$MODEL_DIR"/*.model)
if ((${#models[@]} == 0)); then
    echo "Training finished without saving a model in $MODEL_DIR" >&2
    exit 1
fi
LATEST_MODEL="$(ls -t -- "${models[@]}" | sed -n '1p')"
echo "Training complete. Using model: $LATEST_MODEL"

python "$SCRIPT_DIR/neural_style.py" eval \
    --model "$LATEST_MODEL" \
    --content-image "$CONTENT_IMAGE" \
    --output-image "$PREVIEW_IMAGE" \
    --export_onnx "$FINAL_ONNX" \
    "${ACCEL_ARGS[@]}"

if python -m onnxsim --help >/dev/null 2>&1; then
    SIMPLIFIED_ONNX="$(mktemp "${FINAL_ONNX%.onnx}.XXXXXX.onnx")"
    trap 'rm -f -- "$SIMPLIFIED_ONNX"' EXIT
    python -m onnxsim "$FINAL_ONNX" "$SIMPLIFIED_ONNX"
    mv -- "$SIMPLIFIED_ONNX" "$FINAL_ONNX"
    trap - EXIT
fi

echo "Model ready at: $FINAL_ONNX"
echo "Preview saved at: $PREVIEW_IMAGE"
