from typing import List, Optional, Tuple

import numpy as np
import pyworld
import torch
import yaml
from espnet2.bin.tts_inference import Text2Speech
from espnet2.text.token_id_converter import TokenIDConverter

from text import p2tokens


class VITSJaProsModel:
    def __init__(
        self,
        name: str,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "gpu",
    ):
        self.name = name
        if config_path is None:
            config_path = "conf/config.yaml"
        if torch.cuda.is_available() and device == "gpu":
            self.device = "cuda"
        elif device == "cpu":
            self.device = "cpu"
        else:
            raise ValueError("deviceはgpuかcpuである必要があります。")
        self.model = Text2Speech(
            train_config=config_path, model_file=model_path, device=self.device
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config["tts"] != "vits":
            raise ValueError("このモデルはVITSではありません。")
        if config["g2p"] != "pyopenjtalk_prosody":
            raise ValueError("このモデルのg2pはpyopenjtalk_prosodyではありません。")
        self.converter = TokenIDConverter(config["token_list"])

    def tokens2speech(
        self,
        tokens: List[str],
        speed_scale: float = 1,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
    ) -> Tuple[int, np.ndarray]:
        self.model.decode_conf.update(
            alpha=1 / speed_scale,
            noise_scale=noise_scale,
            noise_scale_dur=noise_scale_dur,
        )
        int_tensor = np.array(self.converter.tokens2ids(tokens))
        with torch.no_grad():
            tensor = self.model(int_tensor)["wav"]
        return self.model.fs, tensor.view(-1).cpu().numpy()

    def p2speech(
        self,
        p: str,
        speed_scale: float = 1,
        pitch_scale: float = 1,
        intonation_scale: float = 1,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
    ) -> Tuple[int, np.ndarray]:
        fs, wave = self.tokens2speech(
            p2tokens(p), speed_scale, noise_scale, noise_scale_dur
        )

        if pitch_scale == 1 and intonation_scale == 1:
            return fs, wave

        # pyworldでf0を加工して合成
        # pyworldよりもよいのがあるかもしれないが……

        wave = wave.astype(np.double)
        f0, t = pyworld.harvest(wave, fs)
        # 質が高そうだしとりあえずharvestにしておく
        # rmvpeが使えたらそれがいいのかも……？
        sp = pyworld.cheaptrick(wave, f0, t, fs)
        ap = pyworld.d4c(wave, f0, t, fs)

        non_zero_f0 = [f for f in f0 if f != 0]
        f0_mean = sum(non_zero_f0) / len(non_zero_f0)

        for i, f in enumerate(f0):
            if f == 0:
                continue
            f0[i] = pitch_scale * f0_mean + intonation_scale * (f - f0_mean)

        wave = pyworld.synthesize(f0, sp, ap, fs)
        return fs, wave
