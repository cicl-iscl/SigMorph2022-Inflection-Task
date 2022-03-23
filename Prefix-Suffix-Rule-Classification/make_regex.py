import re
import pandas as pd

from logger import logger
from paradigm_align import BLANK
from typing import List, Dict
from tqdm.auto import tqdm
from paradigm_align import get_paradigms
from paradigm_align import paradigm_align
from collections import defaultdict

BASECHAR = "X"


def make_regexes_from_alignments(lemma: str, forms: List[str], form2alignment: Dict[str, List[str]]):
    alignments = [form2alignment[form] for form in [lemma] + forms]
    base_indices = []

    for idx in range(len(alignments[0])):
        if not any(alignment[idx] == BLANK for alignment in alignments):
            base_indices.append(idx)

    regex_templates = []
    for alignment in alignments:
        alignment = [(c if idx not in base_indices else BASECHAR) for idx, c in enumerate(alignment)]
        alignment = [c for c in alignment if c != BLANK]
        alignment = "".join(alignment)
        alignment = re.sub(re.compile(BASECHAR + '+'), BASECHAR, alignment)
        regex_templates.append(alignment)

    lemma_template = regex_templates[0]
    assert len(forms) == len(regex_templates[1:])
    return {
        form: (lemma_template, template) for form, template in zip(forms, regex_templates[1:])
        if template.count(BASECHAR) == lemma_template.count(BASECHAR)
    }


def get_regexes(lemmas: List[str], forms: List[str], paradigm_size_threshold: int = 1, regex_count_threshold: int = 1):
    logger.info(f"Data consists of {len(forms)} form-lemma pairs")

    paradigms = get_paradigms(lemmas, forms)
    logger.info(f"Collected {len(paradigms)} paradigms")

    sufficient_paradigms = {
        lemma: paradigm for lemma, paradigm in paradigms.items()
        if len(paradigm) >= paradigm_size_threshold
    }

    logger.info(f"Collected {len(sufficient_paradigms)} useful paradigms")
    num_useful_forms = sum([len(paradigm) for paradigm in sufficient_paradigms.values()])
    total_paradigm_size = sum([len(paradigm) for paradigm in paradigms.values()])
    paradigm_coverage = 100 * num_useful_forms / total_paradigm_size
    logger.info(f"Useful paradigms contain {num_useful_forms} ({paradigm_coverage:.1f}%) forms")

    all_regexes_counts = defaultdict(int)
    all_regexes = defaultdict(set)

    for lemma, lemma_forms in tqdm(sufficient_paradigms.items()):
        form2alignment = paradigm_align(lemma, lemma_forms)
        regexes = make_regexes_from_alignments(lemma, lemma_forms, form2alignment)

        for form, regex in regexes.items():
            all_regexes[form].add(regex)
            all_regexes_counts[regex] += 1

    all_regexes_filtered = defaultdict(set)
    all_regexes_raw = set()

    for form, regexes in all_regexes.items():
        for regex in regexes:
            if all_regexes_counts[regex] >= regex_count_threshold:
                all_regexes_filtered[form].add(regex)
                all_regexes_raw.add(regex)

    num_regex = len(all_regexes_raw)
    coverage = sum(map(all_regexes_counts.get, all_regexes_raw)) / sum(all_regexes_counts.values())
    logger.info(f"Collected {num_regex} ({100 * coverage:.2f}%) lemma -> form regex pairs")
    logger.info(f"Collected {len(set([regex[0] for regex in all_regexes_raw]))} lemma regexes")
    logger.info(f"Collected {len(set([regex[1] for regex in all_regexes_raw]))} form regexes")

    return list(sorted(all_regexes_raw, key=all_regexes_counts.get, reverse=True)), all_regexes_filtered


if __name__ == '__main__':
    language_code = "spa"
    path = f"../../inflection/data/part1/development_languages/{language_code}.train"
    data = pd.read_csv(path, sep='\t', names=["lemma", "form", "tag"])

    # Extract lemmas, forms, and tags
    lemmas = [str(lemma) for lemma in data["lemma"].tolist()]
    forms = [str(form) for form in data["form"].tolist()]
    tags = [str(tag) for tag in data["tag"].tolist()]

    regexes, form2regexes = get_regexes(lemmas, forms)

    for regex in regexes:
        print(regex)
