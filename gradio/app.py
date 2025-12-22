import gradio as gr
from PIL import Image
import base64
import requests
import io
import json

FLASK_URL = "http://localhost:5000/asr"

def base64_to_pil(base64_str: str) -> Image.Image:
    """Конвертирует base64-строку изображения в PIL.Image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def generate_images_from_payload(audio):
    if audio is None:
        return "Аудио не загружено.", None

    try:
        with open(audio, "rb") as f:
            files = {"audio": ("audio.wav", f, "audio/wav")}
            response = requests.post(FLASK_URL, files=files, timeout=300)

        if response.status_code != 200:
            error_msg = response.json().get("error", "Неизвестная ошибка")
            return f"Ошибка сервера: {error_msg}", None

        data = response.json()
        images_base64 = data.get("images", [])

        if not images_base64:
            return "Сервер вернул пустой список изображений.", None

        pil_images = []
        for b64 in images_base64:
            try:
                pil_images.append(base64_to_pil(b64))
            except Exception as e:
                print(f"Error decoding image: {e}")

        return f"Сгенерировано {len(pil_images)} изображений!", pil_images

    except requests.exceptions.Timeout:
        return "Таймаут запроса к серверу.", None
    except requests.exceptions.ConnectionError:
        return "Не удалось подключиться к серверу. Убедитесь, что он запущен.", None
    except json.JSONDecodeError:
        return "Сервер вернул некорректный JSON.", None
    except Exception as e:
        return f"Неожиданная ошибка: {str(e)}", None


with gr.Blocks(title="ColorizedNarrative") as demo:
    gr.Markdown("# 🎨 ColorizedNarrative")
    gr.Markdown("Загрузите аудио → получите изображения на основе речи")

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                label="Аудио",
                sources=["upload", "microphone"],
                type="filepath"
            )
            btn = gr.Button("🚀 Сгенерировать", variant="primary", size="lg")
            status = gr.Textbox(label="Статус", lines=2)

        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Сгенерированные изображения",
                columns=2,
                height=500,
                object_fit="contain"
            )

    btn.click(
        fn=generate_images_from_payload,
        inputs=audio_in,
        outputs=[status, gallery]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
