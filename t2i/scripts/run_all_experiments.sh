#!/bin/bash

set -e

echo "=== Running All Experiments ==="
echo ""

echo "--- Training Flux LoRA (base) ---"
./scripts/train_flux.sh base 2>&1 | tee experiments/logs/flux/train_base.log

echo ""
echo "--- Training Flux LoRA (textenc) ---"
./scripts/train_flux.sh textenc 2>&1 | tee experiments/logs/flux/train_textenc.log

echo ""
echo "--- Training SDXL LoRA (base) ---"
./scripts/train_sdxl.sh base 2>&1 | tee experiments/logs/sdxl/train_base.log

echo ""
echo "--- Training SDXL LoRA (textenc) ---"
./scripts/train_sdxl.sh textenc 2>&1 | tee experiments/logs/sdxl/train_textenc.log

echo ""
echo "=== All experiments completed ==="

