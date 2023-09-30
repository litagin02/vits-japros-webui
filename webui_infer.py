import os
import sys
from typing import Tuple

import gradio as gr
import numpy as np

from model import VITSJaProsModel
from text import g2p

python = sys.executable

model_root = "weights"
models = [
    d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
]
if len(models) == 0:
    raise ValueError(f"{model_root}ディレクトリにディレクトリがありません。")
models.sort()


def update_model_list():
    global models
    models = [
        d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
    ]
    models.sort()
    return gr.Dropdown.update(choices=models)


def load_model(model_name: str, device: str = "cpu"):
    global model
    model_dir = os.path.join(model_root, model_name)
    pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if len(pth_files) == 0:
        raise ValueError(f"`{model_dir}`に`pth`ファイルがありません。")
    elif len(pth_files) > 1:
        raise ValueError(f"`{model_dir}`に`pth`ファイルが複数あります。")
    yaml_file = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(yaml_file):
        yaml_file = None
    model_path = os.path.join(model_dir, pth_files[0])
    model = VITSJaProsModel(model_name, model_path, yaml_file, device=device)
    return gr.Dropdown.update()


def inference(
    model_name: str,
    p: str,
    speed_scale: float,
    pitch_scale: float,
    intonation_scale: float,
    noise_scale: float,
    noise_scale_dur: float,
    device: str = "cpu",
) -> Tuple[int, np.ndarray]:
    try:
        if model_name != model.name:
            load_model(model_name, device)
    except NameError:
        load_model(model_name, device)
    return model.p2speech(
        p, speed_scale, pitch_scale, intonation_scale, noise_scale, noise_scale_dur
    )


accent_guide = """
カタカナで実際の読み方を表し、記号を用いてアクセント等を制御します。

| 記号 | 役割 | 例 |
| --- | --- | --- |
| `[` | ここからアクセントが上昇（➚のイメージ） | こんにちは → `コ[ンニチワ` |
| `]` | ここからアクセントが下降（➘のイメージ） | 京都 → `キョ]オト` |
| ` `（半角スペース） | アクセント句（何となくひとまとまりの箇所）の切れ目 | `ソ[レワ ム[ズカシ]イ` |
| `、` | ポーズ（息継ぎ）。短いポーズを入れたいときに使います。 | `ハ]イ、ソ[オ オ[モイマ]ス` |
| `?` | 疑問文の終わりにつけます。 | `キ[ミワ ダ]レ?` |

カタカナ・上記記号以外を入れるとエラーになります。
"""

with gr.Blocks(title="VITS-JaPros-WebUI 音声合成") as app:
    gr.Markdown("# VITS-JaPros-WebUI 音声合成")
    radio_device = gr.Radio(
        ["cpu", "gpu"],
        label="使用するデバイス",
        info="GPUのほうが速度が早いですが、学習中の途中観察にはCPUを使ってください",
        value="cpu",
    )
    with gr.Row():
        model_drop = gr.Dropdown(label="モデル", choices=models, value=models[0])
        model_drop.select(
            fn=load_model, inputs=[model_drop, radio_device], outputs=[model_drop]
        )
        refresh_button = gr.Button("モデル一覧を更新", scale=0)
        refresh_button.click(fn=update_model_list, inputs=[], outputs=[model_drop])
        reload_button = gr.Button("モデルを再読み込み", scale=0)
        reload_button.click(
            fn=load_model, inputs=[model_drop, radio_device], outputs=[model_drop]
        )
    with gr.Row():
        text = gr.Textbox(label="テキストを入力してください。", value="これは音声合成のテストです。")
        button_2p = gr.Button(value="アクセント解析\n(Enter可)", variant="primary", scale=0)
    with gr.Column():
        p = gr.Textbox(
            label="解析結果",
            info="必要に応じて、記法ルールを見ながら正しいアクセントになるように修正してください。（直接ここに内容を入力することもできます。）",
        )
        with gr.Accordion("アクセント等の記法ルール", open=False):
            gr.Markdown(accent_guide)
    button_2p.click(fn=g2p, inputs=[text], outputs=[p], api_name="g2p")
    text.submit(fn=g2p, inputs=[text], outputs=[p])
    with gr.Accordion("設定", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("音程・抑揚は1以外だと音質劣化の可能性があります。")
                speed_scale = gr.Slider(
                    label="話速", minimum=0.5, maximum=2.0, value=1.0, step=0.1
                )
                pitch_scale = gr.Slider(
                    label="音程", minimum=0.85, maximum=1.15, value=1, step=0.01
                )
                intonation_scale = gr.Slider(
                    label="抑揚", minimum=0, maximum=2, value=1.0, step=0.1
                )
            with gr.Column():
                gr.Markdown("詳細設定（0以外にすると毎回結果が変わるみたいです。）")
                noise_scale = gr.Slider(
                    label="noise_scale (flowのゆらぎ?)",
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.01,
                )
                noise_scale_dur = gr.Slider(
                    label="noise_scale_dur (stochastic duration predictorのゆらぎ?)",
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.01,
                )
    button_infer = gr.Button(value="音声合成！（Enter可）", variant="primary")
    output_audio = gr.Audio(label="結果")
    button_infer.click(
        fn=inference,
        inputs=[
            model_drop,
            p,
            speed_scale,
            pitch_scale,
            intonation_scale,
            noise_scale,
            noise_scale_dur,
            radio_device,
        ],
        outputs=[output_audio],
        api_name="inference",
    )
    p.submit(
        fn=inference,
        inputs=[
            model_drop,
            p,
            speed_scale,
            pitch_scale,
            intonation_scale,
            noise_scale,
            noise_scale_dur,
            radio_device,
        ],
        outputs=[output_audio],
    )


def is_colab():
    import sys

    return "google.colab" in sys.modules


if __name__ == "__main__":
    app.launch(inbrowser=True, share=is_colab())
