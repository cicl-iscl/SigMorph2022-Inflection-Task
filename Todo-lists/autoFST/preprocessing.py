"""
"""

from pathlib import Path
import pandas as pd

# from paradigm import Paradigm

task = "part1"
# task = "part2"
data_dir = Path("2021Task0", task, "development_languages")


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
    files = get_files(data_dir, "deu")
    print(files)
    print()
    print()
    grouped = group_by_lemma(files)
    lemmata = grouped.groups.keys()
    groups = [grouped.get_group(x) for x in grouped.groups]
    print(f"{len(lemmata)} lemmas")
    print(groups[0])
    print()
    print()
    umsteigen = grouped.get_group("umsteigen")
    print(umsteigen)
    print()
    print()
    umsteigen = sort_tags(umsteigen)
    print(umsteigen)
    trennbar = any(" " in x for x in umsteigen["form"].tolist())
    print(trennbar)  # potentially useful for data augmentation
    lemma, forms, tags = decompose_group(umsteigen)
    print(lemma)
    print(forms)
    print(tags)

    # To-Do:
    # - need LCS? generate variable stem representation like s t ei:{ei|ie} g, associate with tag???
    # - um: PART s t ei:TENSE g en:NUMBER+PERSON ???
    # - generate rules by comparing tags and forms within a paradigm
    # - or deduce affixes and morphophonemes
    # - (prioritize paradigms with more entries)
    # ==== Language-specific? ====
    # - define C, V
    # - if trennbar, strip particle, merge two paradigms, hallucinate more separable verbs
    # ==== Producing lexc files====
    # precedence rule?
    # tbc....

    # Use case from paradigm.py
    # paradigm = Paradigm(lemma, forms)

    # print()
    # print(paradigm.alignment)
    # print()
    # print(paradigm.segments)
    # print()

    # for form in forms:
    #     print(paradigm.form2segments(form))
    #     print()


if __name__ == "__main__":
    main()
