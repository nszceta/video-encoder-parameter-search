from tempfile import NamedTemporaryFile
from loguru import logger
import os.path
import re
import time
from subprocess import run
import pickle

import optuna
import optuna.visualization

# INPUT_FILE_PATH = "C:\\Users\\adam\\Videos\\Panasonic\\A003C074_230408_DJ0B.MOV"
INPUT_FILE_PATH = "C:\\Users\\adam\\Videos\\input.mkv"
OUTPUT_FILE_PATH = "C:\\Users\\adam\\Videos\\output.mkv"


def cmd_vmaf(output_pth):
    return [
        "ffmpeg",
        "-i",
        INPUT_FILE_PATH,
        "-i",
        output_pth,
        "-lavfi",
        "libvmaf=n_threads=12:model=version=vmaf_v0.6.1neg",  # :log_path=vmaf.json
        "-f",
        "null",
        "-",
    ]


def objective(trial):
    cmd_encode = [
        "ffmpeg",
        "-i",
        INPUT_FILE_PATH,
        "-pixel_format",
        "yuv444p16le",
        "-c:v",
        "hevc_nvenc",
        "-profile:v",
        "rext",
        "-b_ref_mode",
        "0",
        "-preset:v",
        "slow",
        "-b:v",
        "0M",
    ]
    suggested_qp = trial.suggest_int("qp", 20, 40)
    cmd_encode += ["-rc", "constqp", "-qp", str(suggested_qp)]
    cmd_encode += ["-y", OUTPUT_FILE_PATH]
    encoder_start = time.time()
    proc_encode = run(cmd_encode, capture_output=True)
    if proc_encode.returncode != 0:
        logger.warning(proc_encode.stderr.decode())
        raise optuna.TrialPruned
    encoder_time = time.time() - encoder_start
    proc_vmaf = run(cmd_vmaf(OUTPUT_FILE_PATH), capture_output=True)
    if proc_vmaf.returncode != 0:
        logger.warning(proc_vmaf.stderr.decode())
        raise optuna.TrialPruned
    file_size = os.path.getsize(OUTPUT_FILE_PATH) / 1_000_000  # bytes to megabytes
    quality = float(re.search("(?:VMAF score: )(\d+\.\d+)", proc_vmaf.stderr.decode(), re.MULTILINE).group(1))  # fmt: skip
    return quality, file_size, encoder_time


def main():
    try:
        with open("study.pkl", "rb") as fd:
            study = pickle.load(fd)
    except FileNotFoundError:
        study = optuna.create_study(
            directions=["maximize", "minimize", "minimize"]
            # ["quality", "file size", "encoder time"]
        )
    study.optimize(objective, n_trials=10)
    print(f"{study.best_trials=}")
    with open("study.pkl", "wb") as fd:
        pickle.dump(study, fd, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
