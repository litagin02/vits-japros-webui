import argparse
import os
import sys

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody


sampling_rate = 44100

num_of_tokens = 47  # 使用される最大のトークン数、固定値


def dump_dir_of(model_name: str) -> str:
    return os.path.join(output_dir, model_name, "dump")


def normalize_wav(wav_path: str, output_dir: str):
    wav, sr = sf.read(wav_path)
    if len(wav.shape) == 2:
        print(f"{wav_path}はステレオです。モノラルに変換します。")
        wav = np.mean(wav, axis=1)  # ステレオをモノラルに変換
    if sr != sampling_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sampling_rate)
    wav = librosa.effects.trim(wav, top_db=30)[0]
    normalized_wav = librosa.util.normalize(wav) * 0.9
    sf.write(
        os.path.join(output_dir, os.path.basename(wav_path)),
        normalized_wav,
        sampling_rate,
        "PCM_16",
    )


def split_data_and_dump(
    model_name: str,
    transcript_path: str,
    wavs_dir: str,
    normalized_wavs_dir: str,
    valid_count: int = 5,
):
    dump_dir = dump_dir_of(model_name)
    for folder in ["train", "valid"]:
        os.makedirs(os.path.join(dump_dir, folder), exist_ok=True)

    # Parse the transcript file
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    transcripts = {
        line.split(":")[0].strip(): line.split(":")[1].strip() for line in lines
    }

    # Get common wavs
    wavs = [file[:-4] for file in os.listdir(wavs_dir) if file.endswith(".wav")]
    common_wavs = [wav for wav in wavs if wav in transcripts.keys()]
    common_wavs.sort()

    # Normalize common wavs
    for wav in tqdm(common_wavs, desc="wavファイルの正規化中...", file=sys.stdout):
        normalize_wav(os.path.join(wavs_dir, wav + ".wav"), normalized_wavs_dir)

    # Split into training and validation
    valid_wavs = common_wavs[:valid_count]
    train_wavs = common_wavs[valid_count:]

    # Write transcripts and wav.scp
    for folder, dataset in [("train", train_wavs), ("valid", valid_wavs)]:
        with open(
            os.path.join(dump_dir, folder, "text"),
            "w",
            encoding="utf-8",
        ) as f_text, open(
            os.path.join(dump_dir, folder, "wav.scp"),
            "w",
            encoding="utf-8",
        ) as f_wavscp:
            for wav in dataset:
                f_text.write(f"{wav} {transcripts[wav]}\n")
                f_wavscp.write(
                    f"{wav} {os.path.join(normalized_wavs_dir, wav + '.wav')}\n"
                )


def process_shapes(model_name: str):
    """wavとtextの長さを計算してファイルに書き込む"""
    stats_dir = os.path.join(output_dir, model_name, "stats")
    dump_dir = dump_dir_of(model_name)
    os.makedirs(stats_dir, exist_ok=True)
    for folder in ["train", "valid"]:
        path = os.path.join(stats_dir, folder)
        os.makedirs(path, exist_ok=True)

    for folder in ["train", "valid"]:
        with open(
            os.path.join(dump_dir, folder, "text"),
            "r",
            encoding="utf-8",
        ) as f_text:
            lines = f_text.readlines()

        # 各行に対してg2pを適用してトークンの長さを計算
        with open(
            os.path.join(stats_dir, folder, "text_shape.phn"), "w", encoding="utf-8"
        ) as f_out:
            for line in lines:
                wav, text = line.strip().split(" ", 1)
                tokens = pyopenjtalk_g2p_prosody(text)
                f_out.write(f"{wav} {len(tokens)},{num_of_tokens}\n")

        # 各wavに対してlibrosaを用いて長さを計算
        with open(
            os.path.join(dump_dir, folder, "wav.scp"), "r", encoding="utf-8"
        ) as f_wavscp, open(
            os.path.join(stats_dir, folder, "speech_shape"), "w", encoding="utf-8"
        ) as f_out:
            for line in f_wavscp:
                wav, path = line.strip().split(" ", 1)
                y, _ = librosa.load(path, sr=None)
                f_out.write(f"{wav} {len(y)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--wavs-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    model_name = args.model_name
    wavs_dir = args.wavs_dir

    transcript_path = os.path.join(data_dir, "transcript_utf8.txt")
    normalized_wavs_dir = os.path.join(output_dir, model_name, "normalized_wavs")

    os.makedirs(normalized_wavs_dir, exist_ok=True)

    print("---")
    print("wavファイルを正規化・データを分割しています...")
    split_data_and_dump(
        model_name=model_name,
        transcript_path=transcript_path,
        wavs_dir=wavs_dir,
        normalized_wavs_dir=normalized_wavs_dir,
        valid_count=5,
    )
    print("完了！")
    print("wavとtextの長さを計算しています...")
    process_shapes(model_name=model_name)
    print("完了！前処理が完了しました。")
