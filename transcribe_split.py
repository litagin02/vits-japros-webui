import argparse
import glob
import os
import sys
from tqdm import tqdm

import jaconv
import librosa
import soundfile as sf
from faster_whisper import WhisperModel


def load_whisper_model(model_size: str = "large-v2"):
    print("Whisperモデルをロード中...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("Whisperモデルをロードしました。")
    return model


def transcribe_and_split(
    model: WhisperModel,
    wav_path: str,
    split_wavs_dir: str,
    transcript_path: str,
    initial_prompt: str,
):
    wav_name = os.path.basename(wav_path)[:-4]
    os.makedirs(split_wavs_dir, exist_ok=True)

    segments, _ = model.transcribe(
        wav_path, language="ja", word_timestamps=True, initial_prompt=initial_prompt
    )

    segments = list(segments)
    if len(segments) == 0:
        return

    # wavを読み込む
    y, sr = librosa.load(wav_path, sr=None)

    # 分割した音声ファイルのリストと書き起こし結果のリストを初期化
    split_audios = []
    transcriptions = []

    # 現在のセグメントの開始サンプルと書き起こしのテキストを初期化
    current_start_sample = None
    current_transcription = ""

    # 区切り文字の前後に余裕を持たせるサンプル数を計算
    margin_samples = librosa.time_to_samples(0.15, sr=sr)

    def save_current_segment(end_sample):
        if current_start_sample is not None:
            start_with_margin = max(0, current_start_sample - margin_samples)
            end_with_margin = min(len(y), end_sample + margin_samples)
            split_audio = y[start_with_margin:end_with_margin]
            # split_audio = y[current_start_sample:end_with_margin]
            split_audios.append(split_audio)
            transcriptions.append(current_transcription)

    for segment in segments:
        for word in segment.words:
            if current_start_sample is None:
                current_start_sample = librosa.time_to_samples(word.start, sr=sr)

            current_transcription += word.word

            if word.word[-1] in ["。", "？", "！", "．", ".", "?", "!"]:
                end_sample = librosa.time_to_samples(word.end, sr=sr)
                save_current_segment(end_sample)

                # 初期化
                current_start_sample = None
                current_transcription = ""

    # 区切り文字で終わらなかったとき、残りのセグメントを保存
    if current_start_sample is not None:
        end_sample = librosa.time_to_samples(segments[-1].words[-1].end, sr=sr)
        save_current_segment(end_sample)

    # 分割した音声を保存する
    for idx, split_audio in enumerate(split_audios):
        sf.write(
            os.path.join(split_wavs_dir, f"{wav_name}-{idx}.wav"),
            split_audio,
            sr,
            subtype="PCM_16",
        )

    # 書き起こし結果をtranscribe.txtに追加保存
    with open(transcript_path, "a", encoding="utf-8") as f:
        for idx, transcription in enumerate(transcriptions):
            result = jaconv.normalize(transcription)
            result = result.strip()
            result = result.replace(".", "。")
            result = result.replace(" ", "、")
            f.write(f"{wav_name}-{idx}:{result}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="こんにちは。元気、ですかー？私はちゃんと元気だよ。")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    initial_prompt = args.prompt
    data_dir = args.data_dir

    wavs_dir = os.path.join(data_dir, "wavs")
    transcript_path = os.path.join(data_dir, "transcript_utf8.txt")
    split_wavs_dir = os.path.join(data_dir, "split_wavs")

    wav_paths = sorted(glob.glob(wavs_dir + "/**/*.wav", recursive=True))
    print(f"wavファイルの数: {len(wav_paths)}")

    model = load_whisper_model()

    if os.path.exists(transcript_path):
        print(f"{transcript_path}が既に存在するので削除しています...")
        os.remove(transcript_path)

    for wav_path in tqdm(wav_paths, file=sys.stdout):
        transcribe_and_split(
            model, wav_path, split_wavs_dir, transcript_path, initial_prompt
        )

    print(
        "書き起こし処理が完了しました。`data/split_wavs/`ディレクトリと`data/transcript_utf8.txt`を確認して、必要なら修正してください。"
    )
    print("---")
