このフォルダには以下のようにファイルを準備してください。

- wavsフォルダに音声ファイル（wav形式）
- transcript_utf8.txt

transcript_utf8.txtの中身は以下の感じ（whisperを使う場合は自動で作られます。）

wav_filename1:ここに発言内容を書きます。
wav2:コロンの左側はファイル名のみで、拡張子は不要です。
…

また、wavファイルは以下のようにwavsフォルダに入れてください。
wavs
├── wav_filename1.wav
├── wav2.wav
└── …
