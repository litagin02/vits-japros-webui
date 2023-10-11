import argparse
import glob
import os
import sys

import jaconv
from faster_whisper import WhisperModel
from tqdm import tqdm


def load_whisper_model(model_size: str = "large-v2"):
    print("Whisperモデルをロード中...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("Whisperモデルをロードしました。")
    return model


def transcribe(
    model: WhisperModel, audio_path: str, initial_prompt: str, allow_multi_segment=True
):
    # print(f"{audio_path}を処理中...")
    segments, _ = model.transcribe(
        audio_path, beam_size=5, language="ja", initial_prompt=initial_prompt
    )
    texts = [segment.text for segment in segments]
    if len(texts) == 0:
        return None
    elif len(texts) > 1:
        # print("セグメントが複数あります：")
        # print(texts)
        if allow_multi_segment:
            result = "".join(texts)
        else:
            # print("セグメントが複数あるので、このファイルは無視します。")
            return None
    else:
        result = texts[0]

    result = jaconv.normalize(result)
    result = result.strip()
    result = result.replace(".", "。")
    result = result.replace(" ", "、")
    # print(f"結果：{result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, default="こんにちは。元気、ですかー？私はちゃんと元気だよ。")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    initial_prompt = args.prompt
    data_dir = args.data_dir
    wavs_dir = os.path.join(data_dir, "wavs")
    transcript_path = os.path.join(data_dir, "transcript_utf8.txt")

    wav_paths = sorted(glob.glob(wavs_dir + "/**/*.wav", recursive=True))
    print(f"wavファイルの数: {len(wav_paths)}")

    model = load_whisper_model()

    with open(transcript_path, "w", encoding="utf-8") as output:
        for wav_file in tqdm(wav_paths, file=sys.stdout):
            file_name = os.path.basename(wav_file)[:-4]
            transcription = transcribe(
                model, wav_file, initial_prompt, allow_multi_segment=True
            )
            if transcription is None:
                continue
            output.write(f"{file_name}:{transcription}\n")

    print("書き起こし処理が完了しました。`data/transcript_utf8.txt`を確認して、必要なら修正してください。")
    print("---")
