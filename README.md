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

To train, preview, and export in one step, pass the input paths to `gen_style.sh`:

```bash
./gen_style.sh --style-image style.jpg --dataset dataset --content-image input.jpg --output-onnx models/style.onnx
```

All four arguments are required:

| Argument | What to provide |
| --- | --- |
| `--style-image FILE` | Path to an existing image whose style the model will learn. |
| `--dataset DIR` | Path to an existing training dataset in TorchVision `ImageFolder` layout, such as `dataset/images/*.jpg`. The directory needs at least one class subfolder containing images. |
| `--content-image FILE` | Path to an existing image used to make the preview and export the ONNX model. |
| `--output-onnx FILE` | Destination path for the exported model; it must end in `.onnx`. The output file and its parent directory can be created by the script. |

The script saves a PyTorch model in `./my_glitch_models` and a preview in `./factory_test.jpg` by default. Use `--model-dir DIR` and `--preview-image FILE` to change those paths, and `--accel` to request an available accelerator. Run `./gen_style.sh --help` for the full usage text.

The helper modules `utils.py`, `transformer_net.py`, and `vgg.py` come from the PyTorch fast neural style example. Their BSD 3-Clause license is included in `LICENSE-PYTORCH-EXAMPLES`.
