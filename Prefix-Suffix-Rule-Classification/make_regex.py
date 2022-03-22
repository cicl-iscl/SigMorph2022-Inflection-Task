import re
import pandas as pd

from paradigm_align import BLANK
from typing import List, Dict
from tqdm.auto import tqdm
from paradigm_align import get_paradigms
from paradigm_align import paradigm_align

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
    return [(lemma_template, template) for template in regex_templates[1:]
            if template.count(BASECHAR) == lemma_template.count(BASECHAR)]


if __name__ == '__main__':
    language_code = "kod"
    path = f"../../inflection/data/part1/development_languages/{language_code}.train"
    data = pd.read_csv(path, sep='\t', names=["lemma", "form", "tag"])

    # Extract lemmas, forms, and tags
    lemmas = [str(lemma) for lemma in data["lemma"].tolist()]
    forms = [str(form) for form in data["form"].tolist()]
    tags = [str(tag) for tag in data["tag"].tolist()]

    paradigms = get_paradigms(lemmas, forms)
    all_regexes = set()

    for lemma, lemma_forms in tqdm(paradigms.items()):
        if len(lemma_forms) >= 5 and all(" " not in form for form in lemma_forms):
            form2alignment = paradigm_align(lemma, lemma_forms)
            regexes = make_regexes_from_alignments(lemma, lemma_forms, form2alignment)
            all_regexes.update(set(regexes))

    all_regexes = list(sorted(all_regexes))
    print(len(all_regexes))
    print(len(set([regex[0] for regex in all_regexes])))
    print(len(set([regex[1] for regex in all_regexes])))

    for regex in all_regexes:
        print(regex)
