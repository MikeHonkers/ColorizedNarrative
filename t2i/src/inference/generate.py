import torch
from pathlib import Path
from diffusers import FluxPipeline, StableDiffusionXLPipeline, AutoencoderKL
from src.config import MODELS_CONFIG, INFERENCE_CONFIG, get_path, ensure_dir


def generate_flux_images(lora_dir: str = None):
    model_config = MODELS_CONFIG['models']['flux']
    gen_config = INFERENCE_CONFIG['generation']['flux']
    prompts = INFERENCE_CONFIG['prompts']
    
    base_path = get_path(model_config['local_path'])
    output_dir = ensure_dir(INFERENCE_CONFIG['output']['flux_dir'])
    dtype = torch.bfloat16
    
    print(f"Loading Flux model from {base_path}...")
    pipe = FluxPipeline.from_pretrained(
        str(base_path),
        torch_dtype=dtype,
    ).to("cuda")
    
    print("\nGenerating images with BASE model...")
    for lang, prompt in prompts.items():
        image = pipe(
            prompt=prompt,
            num_inference_steps=gen_config['num_inference_steps'],
            guidance_scale=gen_config['guidance_scale'],
        ).images[0]
        
        fname = f"{lang}_base_flux.png"
        image.save(output_dir / fname)
        print(f"Saved {fname}")
    
    if lora_dir:
        lora_path = get_path(lora_dir)
        print(f"\nLoading LoRA weights from {lora_path}...")
        pipe.load_lora_weights(
            str(lora_path),
            weight_name="pytorch_lora_weights.safetensors",
        )
        
        print("\nGenerating images with BASE + LoRA model...")
        for lang, prompt in prompts.items():
            image = pipe(
                prompt=prompt,
                num_inference_steps=gen_config['num_inference_steps'],
                guidance_scale=gen_config['guidance_scale'],
            ).images[0]
            
            fname = f"{lang}_lora_flux.png"
            image.save(output_dir / fname)
            print(f"Saved {fname}")


def generate_sdxl_images(lora_dir: str = None):
    model_config = MODELS_CONFIG['models']['sdxl']
    gen_config = INFERENCE_CONFIG['generation']['sdxl']
    prompts = INFERENCE_CONFIG['prompts']
    
    base_path = get_path(model_config['local_path'])
    vae_path = get_path(model_config['vae_local_path'])
    output_dir = ensure_dir(INFERENCE_CONFIG['output']['sdxl_dir'])
    dtype = torch.float16
    
    print(f"Loading SDXL model from {base_path}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(base_path),
        vae=AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=dtype),
        torch_dtype=dtype,
    ).to("cuda")
    
    print("\nGenerating images with BASE model...")
    for lang, prompt in prompts.items():
        image = pipe(
            prompt=prompt,
            num_inference_steps=gen_config['num_inference_steps'],
            guidance_scale=gen_config['guidance_scale'],
        ).images[0]
        
        fname = f"{lang}_base_sdxl.png"
        image.save(output_dir / fname)
        print(f"Saved {fname}")
    
    if lora_dir:
        lora_path = get_path(lora_dir)
        print(f"\nLoading LoRA weights from {lora_path}...")
        pipe.load_lora_weights(
            str(lora_path),
            weight_name="pytorch_lora_weights.safetensors",
        )
        
        print("\nGenerating images with BASE + LoRA model...")
        for lang, prompt in prompts.items():
            image = pipe(
                prompt=prompt,
                num_inference_steps=gen_config['num_inference_steps'],
                guidance_scale=gen_config['guidance_scale'],
            ).images[0]
            
            fname = f"{lang}_lora_sdxl.png"
            image.save(output_dir / fname)
            print(f"Saved {fname}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.generate <model> [lora_dir]")
        sys.exit(1)
    
    model = sys.argv[1]
    lora_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if model == "flux":
        generate_flux_images(lora_dir)
    elif model == "sdxl":
        generate_sdxl_images(lora_dir)
    else:
        print(f"Unknown model: {model}")
        sys.exit(1)

