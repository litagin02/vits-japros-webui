import sys
import subprocess

python = sys.executable


def run_train(model_name: str, max_epoch: int = 200, batch_bins: int = 1000000) -> str:
    cmd = [python, "-m", "espnet2.bin.gan_tts_train"]
    from conf.train_args import train_args

    for i, arg in enumerate(train_args):
        # argの中に{model_name}があったら置換する
        if "{model_name}" in arg:
            train_args[i] = arg.format(model_name=model_name)

    cmd.extend(train_args)
    cmd.extend(["--batch_bins", str(batch_bins)])
    cmd.extend(["--max_epoch", str(max_epoch)])

    print(" ".join(cmd))
    print("Submitted to subprocess.")
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)

    return "学習が開始されました。"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py model_name [max_epoch] [batch_bins]")
        sys.exit(1)
    model_name = sys.argv[1]
    max_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    batch_bins = int(sys.argv[3]) if len(sys.argv) > 3 else 1000000
    run_train(model_name, max_epoch, batch_bins)
