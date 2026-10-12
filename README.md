# fast_style

A small command-line project for training a fast neural style transfer model and applying it to images. It can also export a trained PyTorch model to ONNX and run an ONNX model with ONNX Runtime.

## Install

Create a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch torchvision pillow onnx onnxruntime onnxsim
```

For GPU acceleration, install the PyTorch and TorchVision builds appropriate for your system before installing the other packages.

## Use

Train a model with an image dataset arranged for TorchVision's `ImageFolder` (for example, `dataset/images/*.jpg`):

```bash
python neural_style.py train --dataset dataset --style-image style.jpg --save-model-dir models
```

Apply a trained model to an image:

```bash
python neural_style.py eval --model models/example.model --content-image input.jpg --output-image output.jpg
```

Add `--accel` to either command to use an available accelerator. To export during evaluation, add `--export_onnx model.onnx`. To use an exported ONNX model, pass its path to `--model`.

The helper modules `utils.py`, `transformer_net.py`, and `vgg.py` come from the PyTorch fast neural style example. Their BSD 3-Clause license is included in `LICENSE-PYTORCH-EXAMPLES`.
