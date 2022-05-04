import os
import time
import pickle
import pandas as pd

from typing import List
from logger import logger
from make_regex import BASECHAR
from make_regex import parse_word
from make_regex import get_regexes
from collections import defaultdict
from make_fst import ProposalGenerator
from make_regex import filter_regexes
from make_regex import get_form_candidate_from_decomposition

DELIMITER = "#"


def get_target(lemma: str, form: str, lemma_regex: str, form_regex: str, states: List[int]):
    # Get analysis of lemma
    lemma_analysis = None

    lemma_parses = parse_word(lemma, lemma_regex)
    num_gaps = form_regex.count(BASECHAR)

    for decomposition in lemma_parses:
        if len(decomposition) != num_gaps:
            continue

        decomposition = list(decomposition)
        form_candidate = "".join(get_form_candidate_from_decomposition(decomposition, form_regex))

        if form_candidate == form:
            lemma_analysis = decomposition
            break

    assert lemma_analysis is not None

    # Build target
    target = []
    decomposition_counter = 0

    for form_regex_char, state in zip(form_regex, states):
        if form_regex_char == BASECHAR:
            target.extend(list(lemma_analysis[decomposition_counter]))
            decomposition_counter += 1

        else:
            target.append(f"S{state}:{form_regex_char}")

    return target


def make_state_dataset(language_code: str):
    # Read train data
    train_path = f"./development_languages/{language_code}.train"
    train_data = pd.read_csv(train_path, sep='\t', names=["lemma", "form", "tag"])

    # Extract lemmas, forms, and tags
    train_lemmas = [str(lemma) for lemma in train_data["lemma"].tolist()]
    train_forms = [str(form) for form in train_data["form"].tolist()]
    train_tags = [str(tag) for tag in train_data["tag"].tolist()]

    # Read dev data
    lang_prefix = language_code.split("_")[0]
    dev_path = f"./development_languages/{lang_prefix}.dev"
    dev_data = pd.read_csv(dev_path, sep='\t', names=["lemma", "form", "tag"])

    dev_lemmas = [str(lemma) for lemma in dev_data["lemma"].tolist()]
    dev_forms = [str(form) for form in dev_data["form"].tolist()]
    dev_tags = [str(tag) for tag in dev_data["tag"].tolist()]

    # Combine data
    lemmas = train_lemmas + dev_lemmas
    forms = train_forms + dev_forms
    tags = train_tags + dev_tags

    all_tags = set()
    for tag in tags:
        all_tags.update(set(tag.split(';')))

    all_regexes, form2regexes = get_regexes(lemmas, forms, paradigm_size_threshold=1, regex_count_threshold=1)
    time.sleep(0.01)
    # form2regexes = filter_regexes(lemmas, forms, form2regexes)
    # all_regexes = set(form2regexes.values())

    logger.info(f"After reduction: Kept {len(all_regexes)} regex pairs")
    logger.info(f"After reduction: Kept {len(set([regex[0] for regex in all_regexes]))} lemma regexes")
    logger.info(f"After reduction: Kept {len(set([regex[1] for regex in all_regexes]))} form regexes")

    regex2tags = defaultdict(set)
    for lemma, form, form_tags in zip(lemmas, forms, tags):
        form_regex = form2regexes[(lemma, form)]
        form_tags = form_tags.split(";")
        regex2tags[form_regex[1]].update(set(form_tags))

    regexes = list(set([regex[1] for regex in all_regexes]))
    regex_tags = [list(regex2tags[regex]) for regex in regexes]
    generator = ProposalGenerator(regexes, regex_tags, verbose=False)

    os.makedirs("state_data/", exist_ok=True)
    with open(os.path.join("state_data", f"{language_code}.train"), 'w') as df:
        for lemma, form, tags in zip(train_lemmas, train_forms, train_tags):
            lemma_regex, form_regex = form2regexes[(lemma, form)]
            try:
                target = get_target(lemma, form, lemma_regex, form_regex, generator.regex2states[form_regex])
            except AssertionError:
                continue

            target = DELIMITER.join(target)
            df.write(f"{lemma}\t{target}\t{tags}\n")

    with open(os.path.join("state_data", f"{language_code}.dev"), 'w') as df:
        for lemma, form, tags in zip(dev_lemmas, dev_forms, dev_tags):
            lemma_regex, form_regex = form2regexes[(lemma, form)]
            try:
                target = get_target(lemma, form, lemma_regex, form_regex, generator.regex2states[form_regex])
            except AssertionError:
                continue

            target = DELIMITER.join(target)
            df.write(f"{lemma}\t{target}\t{tags}\n")

    with open(os.path.join("state_data", f"{language_code}.graph"), 'wb') as gf:
        pickle.dump(generator, gf)


if __name__ == '__main__':
    language_codes = os.listdir("./development_languages/")
    language_codes = [file.split('.')[0] for file in language_codes if file.endswith(".train")]

    for code in language_codes:
        logger.info(f"Making data for {code=}")
        make_state_dataset(code)
