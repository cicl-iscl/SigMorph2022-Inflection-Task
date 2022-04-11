"""
"""

from pathlib import Path
import pandas as pd
from logger import logger

from paradigm import Paradigm

task = "part1"
# task = "part2"
# data_dir = Path("2021Task0", task, "development_languages")
data_dir = Path("..", "..", "2022InflectionST", task, "development_languages")


def get_files(data_dir=None, iso=None, test=False):
    if iso is not None:
        files = [x for x in data_dir.iterdir() if x.stem.startswith(iso)]
    else:
        files = data_dir.iterdir()
    if test is False:
        files = [x for x in files if x.suffix != ".test"]
    return files


def group_by_lemma(files):
    dfs = (pd.read_csv(f, sep="\t", names=["lemma", "form", "tag"]) for f in files)
    data = pd.concat(dfs)
    grouped = data.groupby("lemma")

    return grouped


def sort_tags(group):
    return group.sort_values(by=["tag"], ascending=True)


def decompose_group(group):
    return group["lemma"].iloc[0], group["form"].tolist(), group["tag"].tolist()


def main():
    # print(get_files(data_dir))
    language_code = "hsi"  # Kholosi

    # Needs transliteration:
    # language_code = "bra"  # Braj
    # language_code = "mag"  # Magahi

    files = get_files(data_dir, language_code)

    logger.info(f"Found {len(files)} files for {language_code}:")
    for f in files:
        logger.info(f"\t{f.name}")
    grouped = group_by_lemma(files)
    lemmata = grouped.groups.keys()
    groups = [grouped.get_group(x) for x in grouped.groups]
    logger.info(f"Collected {len(lemmata)} lemmas.")

    for grp in groups:
        lem = grp["lemma"].iloc[0]
        logger.debug(lem)
        sorted_grp = sort_tags(grouped.get_group(lem))
        logger.debug(f"\n{sorted_grp}")

        lemma, forms, tags = decompose_group(sorted_grp)
        logger.info(f"\n\n{lemma}")
        logger.info(forms)
        logger.info(tags)

        # # Use case from paradigm.py
        # paradigm = Paradigm(lemma, forms)
        # logger.debug("Paradigm:\n")
        # logger.info(f"\t{paradigm.alignment}")
        # logger.info(f"\t{paradigm.segments}\n\n")

        # logger.debug("Forms:\n")
        # for form in forms:
        #     logger.info(paradigm.form2segments(form))


if __name__ == "__main__":
    main()
