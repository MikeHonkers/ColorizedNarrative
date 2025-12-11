# LoRA Fine-tuning

Fine-tune Flux and SDXL models using LoRA.

## Setup

```bash
bash setup.sh
python -m src.data.download
python -m src.models.prepare_flux
python -m src.models.prepare_sdxl
```

## Training

```bash
./scripts/train_flux.sh base
./scripts/train_flux.sh textenc
./scripts/train_sdxl.sh base
./scripts/train_sdxl.sh textenc
```

## Inference

```bash
./scripts/generate_images.sh
./scripts/generate_images.sh flux [lora_dir]
./scripts/generate_images.sh sdxl [lora_dir]
```

## Configuration

- `config/models.yaml` - model paths
- `config/training/*.yaml` - training hyperparameters
- `config/inference.yaml` - inference settings

## Output

- Models: `experiments/runs/`
- Images: `experiments/images/`
- Logs: `experiments/logs/`
