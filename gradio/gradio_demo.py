import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import random

def audio_to_image(audio):
    """
    audio: (sample_rate, np.ndarray) при type="numpy"
    возвращаем PIL.Image
    """
    if audio is None:
        return "Аудио не загружено.", None

    sr, y = audio
    duration = y.shape[0] / float(sr)


    img = Image.new("RGB", (512, 512), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    txt = f"Audio OK\nSR: {sr}\nDur: {duration:.2f}s"
    draw.text((40, 40), txt, fill=(200, 200, 200))

    return "Картинка сгенерирована", img


with gr.Blocks() as demo:
    gr.Markdown("# Audio → Image Demo")

    with gr.Column():
        audio_in = gr.Audio(
            label="Загрузите аудио",
            sources=["upload", "microphone"],
            type="numpy"
        )

        btn = gr.Button("Сгенерировать картинку")

        status = gr.Textbox(label="Статус", lines=1)

        img_out = gr.Image(
            label="Сгенерированное изображение",
            type="pil"
        )

    btn.click(
        audio_to_image,
        inputs=audio_in,
        outputs=[status, img_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=None, share=False)