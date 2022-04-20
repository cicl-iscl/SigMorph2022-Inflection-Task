"""
This script evaluate your FST on the dev set.

Usage:
    python3 evaluate.py -d [directory for task's dev files] -l [language code]
Prerequisites:
    - your language FST in .bin format (save stack xxx.bin)
    - place this file in the same dir as your FST files or change the path to relevant files
"""

import subprocess
import argparse
from pathlib import Path
from logger import logger


def get_dev(data_dir=None, iso=None):
    if iso is not None:
        dev = [
            x
            for x in data_dir.iterdir()
            if x.stem.startswith(iso) and x.suffix == ".dev"
        ]
    else:
        logger.info("Please specify the language code.")
        return ""
    return dev[0]  # only one dev file per language


def generate_test_strings(dev_file, iso):
    targets = []
    with open(f"{iso}.txt", mode="w", encoding="utf8") as f:
        with open(dev_file, mode="r", encoding="utf-8") as df:
            for line in df:
                if line:
                    lemma, form, tag_str = line.rstrip().split("\t")
                    test_str = f"{lemma}+" + "+".join(tag_str.split(";"))
                    f.write(test_str + "\n")
                    targets.append(form)
    return targets


def make_predictions(iso):
    # cat hsi.txt| flookup -i hsi.bin > hsi_results.txt
    subprocess.getoutput(f"cat {iso}.txt| flookup -i {iso}.bin > {iso}_results.txt")
    predictions = []
    with open(f"{iso}_results.txt", mode="r", encoding="utf-8") as f:
        for line in f:
            if line[0].isalpha():
                test_str, _, pred = line.partition("\t")
                predictions.append(pred.rstrip())
    return predictions


def get_accuracy(targets, predictions):
    assert len(targets) == len(predictions)
    score_all = sum(1 for x, y in zip(targets, predictions) if x == y) / len(targets)
    score_predictions = sum(1 for x, y in zip(targets, predictions) if x == y) / len(
        [x for x in predictions if x != "+?"]  # unimplemented or out-of-vocab
    )
    return score_all, score_predictions


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "-d",
        "--directory",
        type=str,
        default="../../2022InflectionST/part1/development_languages",
        help="directory of dev file",
    )
    argp.add_argument(
        "-l",
        "--language",
        type=str,
        help="language code",
    )

    args = argp.parse_args()

    data_dir = Path(args.directory)
    language_code = args.language

    dev_file = get_dev(data_dir, language_code)
    if dev_file:
        logger.info(f"Found dev file for {language_code}:")
        logger.info(f"\t{dev_file.name}")

    targets = generate_test_strings(dev_file, language_code)
    predictions = make_predictions(language_code)
    acc_score, score_predictions = get_accuracy(targets, predictions)
    logger.info(f"Accuracy on dev: {acc_score:.3f}")
    logger.info(f"Accuracy for predicted items: {score_predictions:.3f}")


if __name__ == "__main__":
    main()
