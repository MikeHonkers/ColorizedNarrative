#!/usr/bin/env python3

from flask import Flask, request, jsonify
import os
import io
import base64
from PIL import Image
import traceback
import torch
from diffusers import StableDiffusionXLPipeline

app = Flask(__name__)

NUM_INFERENCE_STEPS = int(os.getenv("NUM_STEPS", "30"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))
HEIGHT = int(os.getenv("HEIGHT", "1024"))
WIDTH = int(os.getenv("WIDTH", "1024"))

print(f"[T2I] steps={NUM_INFERENCE_STEPS} guidance={GUIDANCE_SCALE} size={HEIGHT}x{WIDTH}")

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
print(f"[T2I] Loading {model_id}")

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    print(f"[T2I] GPU: {torch.cuda.get_device_name(0)}")

print("[T2I] Ready")


def generate_image(prompt: str, seed: int = 42) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route('/t2i', methods=['POST'])
def t2i():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        payload = request.get_json()
        speakers = payload.get("speakers", [])
        
        if not speakers:
            return jsonify({"error": "No speakers"}), 400
        
        prompts = [p.get("visual_prompt", "") for p in speakers]
        print(f"[T2I] Generating {len(prompts)} images")
        
        images_base64 = []
        for i, prompt in enumerate(prompts, 1):
            if not prompt:
                continue
            print(f"[T2I] {i}/{len(prompts)}: {prompt[:40]}...")
            img = generate_image(prompt, seed=42 + i)
            images_base64.append(image_to_base64(img))
        
        print(f"[T2I] Done: {len(images_base64)} images")
        return jsonify({"images": images_base64}), 200
    
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "t2i"}), 200


if __name__ == '__main__':
    print("[T2I] http://0.0.0.0:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)
