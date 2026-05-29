from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.inference import predict_for_ui  # noqa: E402


CSS = """
.app-title h1 {
    margin-bottom: 0.25rem;
}
.compact-table table {
    font-size: 0.92rem;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="FoodProject") as demo:
        gr.Markdown("# FoodProject", elem_classes=["app-title"])

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                image = gr.Image(label="Изображение", type="pil", height=360)
                text = gr.Textbox(label="Описание", lines=2, placeholder="например: pizza, sushi, caesar salad")
                show_intermediate = gr.Checkbox(label="Промежуточные результаты", value=True)
                submit = gr.Button("Оценить", variant="primary")

            with gr.Column(scale=5, min_width=360):
                summary = gr.Textbox(label="Итог", lines=5, interactive=False)
                nutrition = gr.Dataframe(
                    label="Масса и БЖУ",
                    headers=["Метрика", "Значение", "Единица"],
                    interactive=False,
                    elem_classes=["compact-table"],
                )
                classes = gr.Dataframe(
                    label="Top-k классов",
                    headers=["Класс", "Уверенность"],
                    interactive=False,
                    elem_classes=["compact-table"],
                )

        with gr.Tabs():
            with gr.Tab("Маски"):
                with gr.Row():
                    food_overlay = gr.Image(label="Еда", type="pil")
                    plate_overlay = gr.Image(label="Тарелка", type="pil")
            with gr.Tab("Глубина"):
                depth_image = gr.Image(label="Depth map", type="pil")
            with gr.Tab("Признаки"):
                segments = gr.Dataframe(
                    label="Сегменты",
                    headers=["Метка", "Уверенность", "Доля площади", "Группа плотности", "Учитывать"],
                    interactive=False,
                    elem_classes=["compact-table"],
                )
                features = gr.Dataframe(
                    label="Численные признаки",
                    headers=["Признак", "Значение"],
                    interactive=False,
                    elem_classes=["compact-table"],
                )
            with gr.Tab("Статус"):
                warnings = gr.Markdown(label="Предупреждения")
                status = gr.Dataframe(
                    label="Модели",
                    headers=["Модуль", "Статус"],
                    interactive=False,
                    elem_classes=["compact-table"],
                )

        submit.click(
            fn=predict_for_ui,
            inputs=[image, text, show_intermediate],
            outputs=[
                summary,
                classes,
                nutrition,
                food_overlay,
                plate_overlay,
                depth_image,
                segments,
                features,
                warnings,
                status,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        css=CSS,
    )
