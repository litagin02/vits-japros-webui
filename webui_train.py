import os
import subprocess
import sys
import time
import webbrowser

import gradio as gr

from train import run_train

python = sys.executable
data_dir = "data"


def run_transcribe(method: str) -> str:
    if method == "通常":
        result = subprocess.run(
            [python, "transcribe.py"],
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            text=True,
        )
    elif method == "分割":
        result = subprocess.run(
            [python, "transcribe_split.py"],
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        raise ValueError(f"Invalid method: {method}")
    if result.stderr:
        print(result.stderr)
        return f"Error: {result.stderr}"
    return "書き起こし処理が完了しました。`data/transcript_utf8.txt`を確認して、必要なら修正してください。"


def run_preprocess(model_name: str, method: str = "通常") -> str:
    if method == "分割":
        wavs_dir = os.path.join(data_dir, "split_wavs")
    elif method == "通常":
        wavs_dir = os.path.join(data_dir, "wavs")
    result = subprocess.run(
        [python, "preprocess.py", model_name, wavs_dir],
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stderr:
        return f"Error: {result.stderr}"
    return "前処理が完了しました。"


def run_tensorboard(model_name: str):
    tensorboard_cmd = [python, "-m", "tensorboard.main"]
    log_path = os.path.join("outputs", model_name, "checkpoints", "tensorboard")
    tensorboard_cmd.extend(["--logdir", log_path])

    subprocess.Popen(tensorboard_cmd, stdout=sys.stdout, stderr=sys.stdout)

    time.sleep(1)
    webbrowser.open(
        "http://127.0.0.1:6006/?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22generator_mel_loss%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22generator_loss%22%7D%5D"
    )


train_initial_md = """
# VITS-JaPros-WebUI 学習

VITSのJaProsモデルを学習します。以下のStep 0からStep 3に従ってください。

- Step 0: データの準備（wavデータを適切に配置）
- Step 1: 文字の書き起こし（音声データから自動的に書き起こし）・結果の手動修正
- Step 2: 学習前の事前準備
- Step 3: 学習の開始

補足

- 音声データのうち5ファイルは学習データとして使われず、検証データとして使われます。
- どれだけの音声データがあれば質が良くなるか等は分かりません、実験してください。Tensorboardの`generator_mel_loss`がいい指標かもしれません。
- 学習を途中で中断したい場合は単にターミナルを閉じてください。学習を再開したいときは、最後のステップ以外を飛ばし、最後のステップを「同じモデル名」で実行すれば、`data/outputs/{model_name}/checkpoints`に保存されている最新エポック・最新状態から再開されます。
- 音声合成に使うには、`weights/{model_name}`フォルダを作って、`outputs/{model_name}/checkpoints`にある`{数字}epoch.pth`ファイルをコピーしてください。**学習中はグラボが競合しないように、音声合成はCPUモードを選んでください。**
"""

step_0_md = """
- `data/wavs/`フォルダ内に、学習に用いる音声wavファイルを入れてください。ファイル名は空白を含まない半角英数字にしてください。また**44.1kHzでモノラル**なことを前提とします。過程で自動的に44.1kHzに変換されますが、その際に音質が落ちる可能性があります。
- 既存コーパスなどでセリフ文章がすでにある場合は、`data/transcript_utf8.txt`ファイルに、音声ファイルのファイル名（拡張子以外）と、その音声のテキスト、半角コロン`:`区切りで書いてください。書き起こしが無い場合は、次のステップでの自動書き起こしを利用します。

例：

```txt
wav_file1:これは最初の音声です。
next_wav:これはもしかして、次のファイルの音声だったりする？
third:そうかもしれないにゃー。
...
```
"""

step_1_md = """
`data/transcript_utf8.txt`を作成済みの場合は、このステップは不要です。

[faster-whisper](https://github.com/guillaumekln/faster-whisper) を利用して、音声データから自動的に書き起こしを行います。

2つのオプションがあります。

1. 通常オプション: `data/wavs/`フォルダ内のwavファイル1つ1つをそのまま書き下します。
2. 分割オプション: `data/wavs/`フォルダ内のwavファイルそれぞれを、文が区切れている箇所（`。`等）で区切って、音声ファイルも分割し`data/split_wavs`に保存し、書き下しも分割します。音声ファイルの長さが長い場合に、学習の精度が上がるかもしれません。

結果は`data/transcript_utf8.txt`に保存されます。途中経過は開いているターミナルに表示されるはずです。

TIPS:

余裕があれば、手動で音声ファイルを聞きながら`data/transcript_utf8.txt`を修正しましょう：
- 不適切な音声（言葉にできない変な声・感情が激しすぎる声が入ってたり等）があれば、その行を削除する（wavファイルはそのままで構いません）
- 誤字脱字修正
- 語尾が上がる疑問口調は、その箇所にちゃんと`？`を入れる
- ポーズ位置に`、`や`。`を入れ、逆にポーズがないところには`、`等を削除する
- 読み方が複数あって曖昧なものは、ひらがな等にする（「何で→なんで」「行った→おこなった」）

が、こだわりすぎなくても大丈夫かもしれません、よく分かりません。
"""


with gr.Blocks(title="VITS-JaPros-WebUI 学習") as app:
    gr.Markdown(train_initial_md)
    with gr.Accordion("Step 0: データの準備", open=False):
        gr.Markdown(step_0_md)
    with gr.Accordion("Step 1: 文字の書き起こし", open=False):
        gr.Markdown(step_1_md)
        trans_choice = gr.Radio(
            label="書き起こし方法",
            choices=["通常", "分割"],
            value="通常",
        )
        with gr.Row():
            button_transcribe = gr.Button("Step 1の実行", variant="primary")
            result_transcribe = gr.Textbox(label="結果")
        button_transcribe.click(
            fn=run_transcribe, inputs=[trans_choice], outputs=[result_transcribe]
        )
    with gr.Accordion("Step 2: 学習前の事前準備", open=False):
        gr.Markdown("上のステップが正常に終了して、必要なら修正・確認したら、モデル名を入力してボタンを押してください。")
        with gr.Row():
            trans_choice_2 = gr.Radio(
                label="書き起こし方法",
                choices=["通常", "分割"],
                value="通常",
            )
            train_model_name = gr.Textbox(
                label="モデル名（空白を含まない半角英数字）", value="test_model"
            )
            button_preprocess = gr.Button("Step 2の実行", variant="primary")
            result_preprocess = gr.Textbox(label="結果")
            button_preprocess.click(
                fn=run_preprocess,
                inputs=[train_model_name, trans_choice_2],
                outputs=[result_preprocess],
            )
            trans_choice.change(
                lambda x: x,
                inputs=[trans_choice],
                outputs=[trans_choice_2],
            )
    with gr.Accordion("Step 3: 学習の開始", open=False):
        with gr.Row():
            train_model_name2 = gr.Textbox(
                label="モデル名（Step 2で入れた名前）", value="test_model"
            )
            batch_bins = gr.Slider(
                label="batch_bins",
                info="バッチサイズのようなもの? GPUのVRAMに収まるように調整してください",
                minimum=500000,
                maximum=2000000,
                value=1000000,
                step=100000,
            )
            max_epoch = gr.Textbox(
                label="何epochまで回すか",
                value="200",
            )
            button_train = gr.Button("学習の開始", variant="primary")
            result_train = gr.Textbox(label="結果")
            button_train.click(
                fn=run_train,
                inputs=[train_model_name2, max_epoch, batch_bins],
                outputs=[result_train],
            )
        gr.Markdown(
            "学習成果はエポックごとに`outputs/{model_name}/checkpoints`に保存されます。ただし直近の10エポックのみ保存され、後は自動削除されます。"
        )
        with gr.Row():
            gr.Markdown(
                "Tensorboardのグラフを見ながら、（RVCと同様に）generator_mel_lossが20を切ってどんどん下がっていけば行くほど良くなる気がします（過学習はどういう感じか知らない）。右のボタンでTensorbaordを開けます。"
            )
            button_tensorboard = gr.Button("Tensorboardを開く")
            button_tensorboard.click(
                fn=run_tensorboard, inputs=[train_model_name2], outputs=[]
            )
        gr.Markdown(
            "適宜pthファイルを`weights`フォルダ内のサブフォルダに移して、音声合成WebUIの方をCPUモードで立ち上げて、音声合成を試してみて質を確認してみるとよいでしょう。"
        )
    button_preprocess.click(
        fn=lambda x: x, inputs=[train_model_name], outputs=[train_model_name2]
    )


def is_colab():
    import sys

    return "google.colab" in sys.modules


if __name__ == "__main__":
    app.launch(inbrowser=True, share=is_colab())
