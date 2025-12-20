#!/bin/bash

set -e

echo "Setting up environment..."

pip install --upgrade pip

pip install "torch>=2.2" "torchvision" --index-url https://download.pytorch.org/whl/cu121

if [ ! -d "diffusers" ]; then
    echo "Cloning diffusers repository..."
    git clone https://github.com/huggingface/diffusers
fi

cd diffusers
pip install -e .
cd ..

pip install -r requirements.txt

cd diffusers/examples/text_to_image
pip install -r requirements.txt
if [ -f "requirements_sdxl.txt" ]; then
    pip install -r requirements_sdxl.txt
fi
cd ../../..

echo "Setup complete!"
echo "Next steps:"
echo "1. Run 'python -m src.data.download' to download the dataset"
echo "2. Run 'python -m src.models.prepare_flux' to download Flux model"
echo "3. Run 'python -m src.models.prepare_sdxl' to download SDXL models"

