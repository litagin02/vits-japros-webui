import glob
import os
import sys

import jaconv
from faster_whisper import WhisperModel
from tqdm import tqdm

model_size = "large-v2"

print("---")
print("文字の書き起こしを開始します。")
print("Whisperモデルをロード中...")
model = WhisperModel(model_size, device="cuda", compute_type="float16")
print("Whisperモデルをロードしました。")


def transcribe(audio_path, initial_prompt=None, allow_multi_segment=False):
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


try:
    initial_prompt = sys.argv[1]
except IndexError:
    initial_prompt = "こんにちは。元気、ですかー？私はちゃんと元気だよ。"

wavs_dir = os.path.join("data", "wavs")
transcript_path = os.path.join("data", "transcript_utf8.txt")

wav_paths = sorted(glob.glob(wavs_dir + "/**/*.wav", recursive=True))
print(f"wavファイルの数: {len(wav_paths)}")

with open(transcript_path, "w", encoding="utf-8") as output:
    for wav_file in tqdm(wav_paths, file=sys.stdout):
        file_name = os.path.basename(wav_file)[:-4]
        transcription = transcribe(wav_file, initial_prompt, allow_multi_segment=True)
        if transcription is None:
            continue
        output.write(f"{file_name}:{transcription}\n")

print("書き起こし処理が完了しました。`data/transcript_utf8.txt`を確認して、必要なら修正してください。")
print("---")
