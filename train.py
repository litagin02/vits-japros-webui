import argparse
import sys
import subprocess

python = sys.executable


def run_train(
    model_name: str,
    max_epoch: int = 200,
    batch_bins: int = 1000000,
    output_dir: str = "outputs",
) -> str:
    cmd = [python, "-m", "espnet2.bin.gan_tts_train"]
    from conf.train_args import train_args

    for i, arg in enumerate(train_args):
        # argの中に{model_name}があったら置換する
        if "{model_name}" in arg:
            train_args[i] = arg.format(model_name=model_name)
        if "{output_dir}" in arg:
            train_args[i] = arg.format(output_dir=output_dir)

    cmd.extend(train_args)
    cmd.extend(["--batch_bins", str(batch_bins)])
    cmd.extend(["--max_epoch", str(max_epoch)])

    print(" ".join(cmd))
    print("Submitted to subprocess.")
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)

    return "学習が開始されました。詳細はターミナルとTensorBoardを確認してください。"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--batch-bins", type=int, default=1000000)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    model_name = args.model_name
    max_epoch = args.max_epoch
    batch_bins = args.batch_bins
    output_dir = args.output_dir

    run_train(model_name, max_epoch, batch_bins, output_dir)
