# 学習のCLI

WebUIを使わずに学習を行うことができます。

以下全て、
```bash
venv\Scripts\activate
```
により仮想環境に入っていることを前提とします。

## 補足

- 音声データのうち5ファイルは学習データとして使われず、検証データとして使われます。
- どれだけの音声データがあれば質が良くなるか等は分かりません、実験してください。Tensorboardの`generator_mel_loss`がいい指標かもしれません。
- 学習を途中で中断したい場合は単にターミナルを閉じてください。学習を再開したいときは、最後のステップ以外を飛ばし、最後のステップを「同じモデル名」で実行すれば、`data/outputs/{model_name}/checkpoints`に保存されている最新エポック・最新状態から再開されます。
- 音声合成に使うには、`weights/{model_name}`フォルダを作って、`outputs/{model_name}/checkpoints`にある`{数字}epoch.pth`ファイルをコピーしてください。**学習中はグラボが競合しないように、音声合成はCPUモードを選んでください。**

## 1. データセットの準備

- `data/wavs/`フォルダ内に、学習に用いる音声wavファイルを入れてください。ファイル名は空白を含まない半角英数字にしてください。また**44.1kHzでモノラル**なことを前提とします。過程で自動的に44.1kHzに変換されますが、その際に音質が落ちる可能性があります。

- 既存コーパスなどでセリフ文章がすでにある場合は、`data/transcript_utf8.txt`ファイルに、音声ファイルのファイル名（拡張子以外）と、その音声のテキスト、半角コロン`:`区切りで書いてください。書き起こしが無い場合は、次のステップでの自動書き起こしを利用します。

例：

```txt
wav_file1:これは最初の音声です。
next_wav:これはもしかして、次のファイルの音声だったりする？
third:そうかもしれないにゃー。
...
```

## 2. 自動書き起こし

`data/transcript_utf8.txt`を作成済みの場合は、このステップは不要です。

[faster-whisper](https://github.com/guillaumekln/faster-whisper) を利用して、音声データから自動的に書き起こしを行います。

2つのオプションがあります。

1. 通常オプション: `data/wavs/`フォルダ内のwavファイル1つ1つをそのまま書き下します。
2. 分割オプション: `data/wavs/`フォルダ内のwavファイルそれぞれを、文が区切れている箇所（`。`等）で区切って、音声ファイルも分割し`data/split_wavs`に保存し、書き下しも分割します。音声ファイルの長さが長い場合に、学習の精度が上がるかもしれません。

結果は`data/transcript_utf8.txt`に保存されます。途中経過は開いているターミナルに表示されるはずです。

- 通常オプションの場合

```bash
python transcribe.py
```

- 分割オプションの場合

```bash
python transcribe_split.py
```

TIPS:

余裕があれば、手動で音声ファイルを聞きながら`data/transcript_utf8.txt`を修正しましょう：
- 不適切な音声（言葉にできない変な声・感情が激しすぎる声が入ってたり等）があれば、その行を削除する（wavファイルはそのままで構いません）
- 誤字脱字修正
- 語尾が上がる疑問口調は、その箇所にちゃんと`？`を入れる
- ポーズ位置に`、`や`。`を入れ、逆にポーズがないところには`、`等を削除する
- 読み方が複数あって曖昧なものは、ひらがな等にする（「何で→なんで」「行った→おこなった」）

が、こだわりすぎなくても大丈夫かもしれません、よく分かりません。

## 3. 事前準備
- `model_name`は、好きな名前（半角英数字）を指定してください。学習結果は`outputs/{model_name}/checkpoints/`フォルダ内に保存されます。
- `wavs_dir`は、自動書き起こしを使わなかった場合・通常オプションを使った場合は`data/wavs/`、分割オプションを使った場合は`data/split_wavs/`を指定してください。

```bash
python preprocess.py model_name wavs_dir
```

## 4. 学習
- `model_name`は上で指定したものと同じものを指定してください。
- `max_epochs`は、最大学習エポック数を指定します。デフォルトは200です。
- `batch_bins`は、学習時のバッチサイズのようなものを指定します。デフォルトは1000000です。グラボのVRAMと相談してください（12GBでは1500000ぐらいがギリでした）。

```bash
python train.py model_name [max_epochs=200] [batch_bins=1000000]
```

## Tensorboardでの可視化
```bash
tensorboard --logdir outputs/{model_name}/checkpoints/tensorboard/
```
`generator_mel_loss`に注目するとよい気がします（20を下回って18くらいだとよい感じ？）。