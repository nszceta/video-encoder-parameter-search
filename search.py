from tempfile import NamedTemporaryFile
from loguru import logger
import os.path
import re
import time
from subprocess import run
import pickle

import optuna
import optuna.visualization

INPUT_FILE_PATH = "C:\\Users\\adam\\Videos\\Panasonic\\A003C074_230408_DJ0B.MOV"
INPUT_FILE_PATH_SIZE = (
    os.path.getsize(INPUT_FILE_PATH) / 1_000_000
)  # bytes to megabytes
# INPUT_FILE_PATH = "C:\\Users\\adam\\Videos\\input.mkv"
# OUTPUT_FILE_PATH = "C:\\Users\\adam\\Videos\\output.mkv"
OUTPUT_FILE_PATH = "M:\\output.mkv"


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
    # < oracle >
    suggested_qp = trial.suggest_int("qp", 20, 40)
    cmd_encode += ["-rc", "constqp", "-qp", str(suggested_qp)]

    suggested_tier = trial.suggest_categorical("tier", ["main", "high"])
    cmd_encode += ["-tier", suggested_tier]

    suggested_rc_lookahead = trial.suggest_int("rc_lookahead", 0, 32)
    cmd_encode += ["-rc-lookahead", str(suggested_rc_lookahead)]

    suggested_spatial_aq = trial.suggest_categorical("spatial_aq", [0, 1])
    cmd_encode += ["-spatial-aq", str(suggested_spatial_aq)]

    suggested_nonref_p = trial.suggest_categorical("nonref_p", [0, 1])
    cmd_encode += ["-nonref_p", str(suggested_nonref_p)]

    suggested_strict_gop = trial.suggest_categorical("strict_gop", [0, 1])
    cmd_encode += ["-strict_gop", str(suggested_strict_gop)]

    suggested_aq_strength = trial.suggest_int("aq_strength", 1, 15)
    cmd_encode += ["-aq-strength", str(suggested_aq_strength)]

    suggested_init_qpP = trial.suggest_int("init_qpP", -1, 40)
    cmd_encode += ["-init_qpP", str(suggested_init_qpP)]

    suggested_init_qpI = trial.suggest_int("init_qpI", -1, 40)
    cmd_encode += ["-init_qpI", str(suggested_init_qpI)]

    suggested_init_qpB = trial.suggest_int("init_qpB", -1, 40)
    cmd_encode += ["-init_qpB", str(suggested_init_qpB)]

    suggested_weighted_pred = trial.suggest_categorical("weighted_pred", [0, 1])
    cmd_encode += ["-weighted_pred", str(suggested_weighted_pred)]
    # </ oracle >
    cmd_encode += ["-y", OUTPUT_FILE_PATH]
    logger.info(f"{trial.params=}")
    encoder_start = time.time()
    try:
        os.remove(OUTPUT_FILE_PATH)
    except FileNotFoundError:
        pass
    proc_encode = run(cmd_encode, capture_output=True)
    if proc_encode.returncode != 0:
        logger.warning(proc_encode.stderr.decode())
        raise optuna.exceptions.TrialPruned
    encoder_time = time.time() - encoder_start
    logger.success("Encode succeeded")
    file_size = os.path.getsize(OUTPUT_FILE_PATH) / 1_000_000  # bytes to megabytes
    if (
        file_size >= INPUT_FILE_PATH_SIZE * 0.50
    ):  # do not bother with encodes that don't save a good amount of space
        logger.warning(
            f"file_size ({round(file_size,2)} MB)  >= INPUT_FILE_PATH_SIZE ({round(INPUT_FILE_PATH_SIZE,2)} MB) * 0.50"
        )
        raise optuna.exceptions.TrialPruned
    proc_vmaf = run(cmd_vmaf(OUTPUT_FILE_PATH), capture_output=True)
    if proc_vmaf.returncode != 0:
        logger.warning(proc_vmaf.stderr.decode())
        raise optuna.exceptions.TrialPruned
    logger.success("VMAF measurement succeeded")
    quality = float(
        re.search(
            "(?:VMAF score: )(\d+\.\d+)", proc_vmaf.stderr.decode(), re.MULTILINE
        ).group(1)
    )
    logger.info(
        re.search(
            "(?:VMAF score: )(\d+\.\d+)", proc_vmaf.stderr.decode(), re.MULTILINE
        ).group(0)
    )
    with open("study.pkl", "wb") as fd:
        pickle.dump(trial.study, fd, protocol=pickle.HIGHEST_PROTOCOL)
    return quality, file_size, encoder_time


def main():
    try:
        with open("study.pkl", "rb") as fd:
            study = pickle.load(fd)
    except FileNotFoundError:
        study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
    study.optimize(objective, n_trials=1000000000)
    print(f"{study.best_trials=}")


if __name__ == "__main__":
    main()
