from flask import Flask, request, jsonify
import os
import io
import base64
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from typing import List, Dict
import traceback
from transformers import CLIPTokenizer

app = Flask(__name__)

TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")

TEXT_ENCODER_1 = "text_encoder"
TEXT_ENCODER_2 = "text_encoder_2"
UNET = "unet"
VAE_DECODER = "vae_decoder"

QWEN_MODEL = "qwen_onnx"

client = httpclient.InferenceServerClient(url=TRITON_URL)

NUM_INFERENCE_STEPS = int(os.getenv("NUM_STEPS", "30"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "5.0"))
HEIGHT = int(os.getenv("HEIGHT", "1024"))
WIDTH = int(os.getenv("WIDTH", "1024"))


tokenizer_1 = CLIPTokenizer.from_pretrained("aamocualg-hse/sdxl-onnx/tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained("aamocualg-hse/sdxl-onnx/tokenizer_2")

def load_sdxl_models():
    for model in [TEXT_ENCODER_1, TEXT_ENCODER_2, UNET, VAE_DECODER]:
        if not client.is_model_ready(model):
            print(f"Loading SDXL model: {model}")
            client.load_model(model)

def unload_sdxl_models():
    for model in [TEXT_ENCODER_1, TEXT_ENCODER_2, UNET, VAE_DECODER]:
        if client.is_model_ready(model):
            print(f"Unloading SDXL model: {model}")
            client.unload_model(model)

def encode_prompt(prompt: str):
    tokens_1 = tokenizer_1(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np"
    )
    tokens_2 = tokenizer_2(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np"
    )
    return tokens_1.input_ids.astype(np.int64), tokens_2.input_ids.astype(np.int64)

def generate_image(visual_prompt: str, seed: int = 42) -> Image.Image:
    load_sdxl_models()

    np.random.seed(seed)

    input_ids_1, input_ids_2 = encode_prompt(visual_prompt)

    te1_result = client.infer(
        model_name=TEXT_ENCODER_1,
        inputs=[httpclient.InferInput("input_ids", input_ids_1.shape, "INT64")
                .set_data_from_numpy(input_ids_1)]
    )
    encoder_hidden_states_1 = te1_result.as_numpy("last_hidden_state")

    te2_result = client.infer(
        model_name=TEXT_ENCODER_2,
        inputs=[httpclient.InferInput("input_ids", input_ids_2.shape, "INT64")
                .set_data_from_numpy(input_ids_2)]
    )
    encoder_hidden_states_2 = te2_result.as_numpy("last_hidden_state")
    pooled_prompt_embeds = te2_result.as_numpy("pooler_output") if "pooler_output" in te2_result.get_response()["outputs"] else encoder_hidden_states_2[:, -1:, :]

    encoder_hidden_states = np.concatenate([encoder_hidden_states_1, encoder_hidden_states_2], axis=-1)

    latents = np.random.randn(1, 4, HEIGHT // 8, WIDTH // 8).astype(np.float32)

    for i in range(NUM_INFERENCE_STEPS):
        t = np.array([999 - i * (1000 // NUM_INFERENCE_STEPS)], dtype=np.int64)

        unet_inputs = [
            httpclient.InferInput("sample", latents.shape, "FP32").set_data_from_numpy(latents),
            httpclient.InferInput("timestep", t.shape, "INT64").set_data_from_numpy(t),
            httpclient.InferInput("encoder_hidden_states", encoder_hidden_states.shape, "FP32")
                .set_data_from_numpy(encoder_hidden_states),
        ]
        if pooled_prompt_embeds is not None:
            unet_inputs.append(
                httpclient.InferInput("text_embeds", pooled_prompt_embeds.shape, "FP32")
                    .set_data_from_numpy(pooled_prompt_embeds)
            )

        unet_result = client.infer(model_name=UNET, inputs=unet_inputs)
        noise_pred = unet_result.as_numpy("latent")

        latents = latents - noise_pred / NUM_INFERENCE_STEPS

    vae_input = httpclient.InferInput("latent_sample", latents.shape, "FP32")
    vae_input.set_data_from_numpy(latents * (1 / 0.18215))

    vae_result = client.infer(model_name=VAE_DECODER, inputs=[vae_input])
    image_np = vae_result.as_numpy("images")[0]

    image_np = (image_np / 2 + 0.5).clip(0, 1) * 255
    image_np = image_np.transpose(1, 2, 0).astype(np.uint8)

    return Image.fromarray(image_np)

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

        prompts = [p["visual_prompt"] for p in payload["speakers"]]
        
        images_base64 = []
        for prompt in prompts:
            img = generate_image(prompt)
            images_base64.append(image_to_base64(img))

        return jsonify({"images": images_base64}), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        unload_sdxl_models()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)