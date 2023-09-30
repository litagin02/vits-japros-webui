from typing import List
import jaconv
from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody


p2kata = {"cl": "ッ", "N": "ン", "_": "、", "#": " "}
kata2p = {v: k for k, v in p2kata.items()}


def g2tokens(text: str) -> List[str]:
    return pyopenjtalk_g2p_prosody(text)


def g2p(text: str) -> str:
    """
    与えられた文字列を、カタカナとアクセント記号からなる文字列に変換。
    記号の扱いは`p2kata`に準ずる：
    - `cl`は「`ッ`」
    - `N`は`ン`
    - `_`は`、`
    - `#`は` `（半角スペース）
    """
    tokens = g2tokens(text)
    # `tokens`の最初は'^'で始まり、通常文は'$'で終わり、疑問文は'?'で終わる
    tokens = tokens[1:]  # 最初の'^'を落とす
    if tokens[-1] == "$":
        tokens = tokens[:-1]  # 疑問文でないとき'$'を落とす
    tokens = [p2kata[token] if token in p2kata else token for token in tokens]
    # アルファベットを並べてjaconvでカタカナに変換
    return jaconv.alphabet2kata("".join(tokens))


k2a_with_spaces = {
    "キャ": " ky a",
    "キュ": " ky u",
    "キョ": " ky o",
    "ギャ": " gy a",
    "ギュ": " gy u",
    "ギョ": " gy o",
    "シャ": " sh a",
    "シュ": " sh u",
    "ショ": " sh o",
    "ジャ": " j a",
    "ジュ": " j u",
    "ジョ": " j o",
    "チャ": " ch a",
    "チュ": " ch u",
    "チョ": " ch o",
    "ニャ": " ny a",
    "ニュ": " ny u",
    "ニョ": " ny o",
    "ヒャ": " hy a",
    "ヒュ": " hy u",
    "ヒョ": " hy o",
    "ファ": " f a",
    "フィ": " f i",
    "フェ": " f e",
    "フォ": " f o",
    "ミャ": " my a",
    "ミュ": " my u",
    "ミョ": " my o",
    "リャ": " ry a",
    "リュ": " ry u",
    "リョ": " ry o",
    "ビャ": " by a",
    "ビュ": " by u",
    "ビョ": " by o",
    "ピャ": " py a",
    "ピュ": " py u",
    "ピョ": " py o",
    "ガ": " g a",
    "ギ": " g i",
    "グ": " g u",
    "ゲ": " g e",
    "ゴ": " g o",
    "ザ": " z a",
    "ジ": " j i",
    "ズ": " z u",
    "ゼ": " z e",
    "ゾ": " z o",
    "ダ": " d a",
    "ヂ": " j i",
    "ヅ": " z u",
    "デ": " d e",
    "ド": " d o",
    "バ": " b a",
    "ビ": " b i",
    "ブ": " b u",
    "ベ": " b e",
    "ボ": " b o",
    "パ": " p a",
    "ピ": " p i",
    "プ": " p u",
    "ペ": " p e",
    "ポ": " p o",
    "ア": " a",
    "イ": " i",
    "ウ": " u",
    "エ": " e",
    "オ": " o",
    "カ": " k a",
    "キ": " k i",
    "ク": " k u",
    "ケ": " k e",
    "コ": " k o",
    "サ": " s a",
    "シ": " sh i",
    "ス": " s u",
    "セ": " s e",
    "ソ": " s o",
    "タ": " t a",
    "チ": " ch i",
    "ツ": " ts u",
    "テ": " t e",
    "ト": " t o",
    "ナ": " n a",
    "ニ": " n i",
    "ヌ": " n u",
    "ネ": " n e",
    "ノ": " n o",
    "ハ": " h a",
    "ヒ": " h i",
    "フ": " f u",
    "ヘ": " h e",
    "ホ": " h o",
    "マ": " m a",
    "ミ": " m i",
    "ム": " m u",
    "メ": " m e",
    "モ": " m o",
    "ラ": " r a",
    "リ": " r i",
    "ル": " r u",
    "レ": " r e",
    "ロ": " r o",
    "ヤ": " y a",
    "ユ": " y u",
    "ヨ": " y o",
    "ワ": " w a",
    "ヰ": " w i",
    "ヲ": " w o",
    "ヱ": " w e",
}


def p2tokens(p: str) -> List[str]:
    """
    カタカナと記号で表記された音素列を、VITSモデルの入力となるトークン列に変換する。
    例：
    コ[ンニチワ → [^, k, o, [, N, n, i, ch, i, w, a, $]
    """
    p = p.replace("　", " ")
    p = p.replace("？", "?")
    if p[-1] != "?" and p[-1] != "$":  # 疑問でない場合、最後に"$"がなければ付ける
        p = p + "$"
    for k, v in kata2p.items():
        p = p.replace(k, v)
    p = "^" + p
    for k, v in k2a_with_spaces.items():
        p = p.replace(k, v)
    symbols = ["[", "]", "#", "_", "N", "cl", "?", "$"]
    for sym in symbols:
        p = p.replace(sym, " " + sym)
    return p.split(" ")
