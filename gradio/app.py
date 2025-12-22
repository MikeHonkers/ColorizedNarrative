import gradio as gr
from PIL import Image, ImageDraw
import base64
import requests
import io
import json

FLASK_URL = "http://localhost:5000/t2i"

def base64_to_pil(base64_str: str) -> Image.Image:
    """Конвертирует base64-строку изображения в PIL.Image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def generate_images_from_payload(audio):
    if audio is None:
        return "Аудио не загружено.", None

    with open(audio, "rb") as f:
        files = {
            "audio": f
        }

    try:
        response = requests.post(FLASK_URL, files=files, timeout=180)

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
                pil_images.append(None)

        return "Изображения успешно сгенерированы!", pil_images

    except requests.exceptions.Timeout:
        return "Таймаут запроса к серверу.", None
    except requests.exceptions.ConnectionError:
        return "Не удалось подключиться к Flask-серверу. Убедитесь, что он запущен.", None
    except json.JSONDecodeError:
        return "Сервер вернул некорректный JSON.", None
    except Exception as e:
        return f"Неожиданная ошибка: {str(e)}", None


with gr.Blocks() as demo:
    gr.Markdown("# Audio → Image Demo")

    with gr.Column():
        audio_in = gr.Audio(
            label="Загрузите аудио или запишите с микрофона",
            sources=["upload", "microphone"],
            type="filepath"
        )

        btn = gr.Button("Сгенерировать изображения", variant="primary")

        status = gr.Textbox(label="Статус", lines=1)

        carousel = gr.Carousel(
            label="Сгенерированные изображения",
            height=600
        )

    btn.click(
        fn=generate_images_from_payload,
        inputs=audio_in,
        outputs=[status, carousel]
    )

with gr.Blocks() as demo:
    gr.Markdown("# Audio → Множественные изображения (карусель)")

    with gr.Column():

        carousel = gr.Carousel(
            label="Сгенерированные изображения",
            height=600
        )

    btn.click(
        fn=generate_images_from_payload,
        inputs=audio_in,
        outputs=[status, carousel]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)