"""
This script evaluates your FST on the dev set.

Usage:
    python3 evaluate.py -d [directory for task's dev files] -l [language code]
Prerequisites:
    - your language FST in .bin format (save stack xxx.bin)
    - place this file in the same dir as your FST files or change the path to relevant files
"""

import subprocess
import argparse
import pandas as pd
from pathlib import Path
from logger import logger


def get_train_sets(data_dir, iso):
    if iso is not None:
        files = [
            x
            for x in data_dir.iterdir()
            if x.stem.startswith(iso) and x.suffix == ".train"
        ]
    dfs = (pd.read_csv(f, sep="\t", names=["lemma", "form", "tag"]) for f in files)
    data = pd.concat(dfs)

    lemmas = list(data.lemma.unique())
    tags = list(data.tag.unique())

    return lemmas, tags


def get_dev(data_dir, iso=None):
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


def generate_test_strings(dev_file, iso, lemmas, tags):
    targets = []
    both_seen = []
    seen_lemma = []
    seen_feats = []
    unseen = []
    with open(f"{iso}.txt", mode="w", encoding="utf8") as f:
        with open(dev_file, mode="r", encoding="utf-8") as df:
            for i, line in enumerate(df):
                if line:
                    lemma, form, tag_str = line.rstrip().split("\t")
                    test_str = f"{lemma}+" + "+".join(tag_str.split(";"))
                    f.write(test_str + "\n")
                    targets.append(form)

                    if lemma in lemmas and tag_str not in tags:
                        seen_lemma.append(i)
                    if tag_str in tags and lemma not in lemmas:
                        seen_feats.append(i)
                    if lemma in lemmas and tag_str in tags:
                        both_seen.append(i)
                    if lemma not in lemmas and tag_str not in tags:
                        unseen.append(i)
    return targets, both_seen, seen_lemma, seen_feats, unseen


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


def get_accuracy(targets, predictions, both_seen, seen_lemma, seen_feats, unseen):
    assert len(targets) == len(predictions)
    score_all = sum(1 for x, y in zip(targets, predictions) if x == y) / len(targets)
    score_predictions = sum(1 for x, y in zip(targets, predictions) if x == y) / len(
        [x for x in predictions if x != "+?"]  # unimplemented or out-of-vocab
    )

    both_seen_pairs = [(targets[i], predictions[i]) for i in both_seen]
    score_both_seen = (
        sum(1 for x, y in both_seen_pairs if x == y) / len(both_seen_pairs)
        if len(both_seen_pairs) != 0
        else 0
    )

    seen_lemma_pairs = [(targets[i], predictions[i]) for i in seen_lemma]
    score_seen_lemma = (
        sum(1 for x, y in seen_lemma_pairs if x == y) / len(seen_lemma_pairs)
        if len(seen_lemma_pairs) != 0
        else 0
    )

    seen_feats_pairs = [(targets[i], predictions[i]) for i in seen_feats]
    score_seen_feats = (
        sum(1 for x, y in seen_feats_pairs if x == y) / len(seen_feats_pairs)
        if len(seen_feats_pairs) != 0
        else 0
    )

    unseen_pairs = [(targets[i], predictions[i]) for i in unseen]
    score_unseen = (
        sum(1 for x, y in unseen_pairs if x == y) / len(unseen_pairs)
        if len(unseen_pairs) != 0
        else 0
    )

    return (
        score_all,
        score_predictions,
        score_both_seen,
        score_seen_lemma,
        score_seen_feats,
        score_unseen,
    )


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

    lemmas, tags = get_train_sets(data_dir, language_code)

    targets, both_seen, seen_lemma, seen_feats, unseen = generate_test_strings(
        dev_file, language_code, lemmas, tags
    )
    predictions = make_predictions(language_code)
    (
        acc_score,
        score_predictions,
        score_both_seen,
        score_seen_lemma,
        score_seen_feats,
        score_unseen,
    ) = get_accuracy(targets, predictions, both_seen, seen_lemma, seen_feats, unseen)
    logger.info(f"Accuracy on dev: {acc_score:.3f}")
    logger.info(f"Accuracy for predicted items: {score_predictions:.3f}")

    logger.info(f"both:\t{score_both_seen:.3f}")
    logger.info(f"lemma:\t{score_seen_lemma:.3f}")
    logger.info(f"feats:\t{score_seen_feats:.3f}")
    logger.info(f"unseen:\t{score_unseen:.3f}")


if __name__ == "__main__":
    main()
