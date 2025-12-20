#!/bin/bash

set -e

MODEL=${1:-both}
LORA_DIR=${2:-}

if [ "$MODEL" = "both" ] || [ "$MODEL" = "flux" ]; then
    echo "Generating Flux images..."
    if [ -n "$LORA_DIR" ]; then
        python -m src.inference.generate flux "$LORA_DIR"
    else
        python -m src.inference.generate flux
    fi
fi

if [ "$MODEL" = "both" ] || [ "$MODEL" = "sdxl" ]; then
    echo "Generating SDXL images..."
    if [ -n "$LORA_DIR" ]; then
        python -m src.inference.generate sdxl "$LORA_DIR"
    else
        python -m src.inference.generate sdxl
    fi
fi

