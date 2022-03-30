import time
import numpy as np
import pandas as pd

from tqdm.auto import trange
from logger import logger
from itertools import product
from make_regex import get_regexes
from collections import defaultdict
from make_fst import ProposalGenerator
from make_regex import filter_regexes
from make_regex import get_form_candidates


if __name__ == '__main__':
    language_code = "poma_small"
    path = f"./development_languages/{language_code}.train"
    data = pd.read_csv(path, sep='\t', names=["lemma", "form", "tag"])

    # Extract lemmas, forms, and tags
    lemmas = [str(lemma) for lemma in data["lemma"].tolist()]
    forms = [str(form) for form in data["form"].tolist()]
    tags = [str(tag) for tag in data["tag"].tolist()]

    all_tags = set()
    for tag in tags:
        all_tags.update(set(tag.split(';')))

    all_regexes, form2regexes = get_regexes(lemmas, forms, paradigm_size_threshold=1, regex_count_threshold=1)
    time.sleep(0.01)
    form2regexes = filter_regexes(lemmas, forms, form2regexes)
    all_regexes = set.union(*form2regexes.values())

    logger.info(f"After reduction: Kept {len(all_regexes)} regex pairs")
    logger.info(f"After reduction: Kept {len(set([regex[0] for regex in all_regexes]))} lemma regexes")
    logger.info(f"After reduction: Kept {len(set([regex[1] for regex in all_regexes]))} form regexes")

    regex2tags = defaultdict(set)
    for form, form_tags in zip(forms, tags):
        form_regexes = form2regexes[form]
        form_tags = form_tags.split(";")

        for regex in form_regexes:
            regex2tags[regex[1]].update(set(form_tags))

    regexes = list(set([regex[1] for regex in all_regexes]))
    # for regex in regexes:
    #     print(regex)
    regex_tags = [list(regex2tags[regex]) for regex in regexes]
    generator = ProposalGenerator(regexes, regex_tags, verbose=False)
    lemma_regexes = list(set([regex[0] for regex in all_regexes]))

    test_item = 78
    test_tags = tags[test_item].split(';')
    logger.info(f"Test tags: {test_tags}")
    templates = generator.propose_templates(test_tags)
    logger.info(f"Collected {len(templates)} templates")

    candidates = set()
    for lemma_regex, form_regex in product(lemma_regexes, templates):
        candidates.update(get_form_candidates(lemmas[test_item], lemma_regex, form_regex))

    candidates = list(sorted(candidates))

    # for candidate in candidates:
    #    print(candidate)

    logger.info(f"Collected {len(candidates)} candidates")
    logger.info(f"Correct form: {forms[test_item]}")
    logger.info(f"Correct form in candidates: {forms[test_item] in candidates}")

    # Evaluation

    path = f"./development_languages/{language_code.split('_')[0].strip()}.dev"
    eval_data = pd.read_csv(path, sep='\t', names=["lemma", "form", "tag"])

    eval_lemmas = [str(lemma) for lemma in eval_data["lemma"].tolist()]
    eval_forms = [str(form) for form in eval_data["form"].tolist()]
    eval_tags = [str(tag) for tag in eval_data["tag"].tolist()]

    covered = 0
    num_samples = min(1000, len(eval_lemmas))
    num_candidates = []

    for k in trange(num_samples):
        test_tags = eval_tags[k].split(';')
        templates = generator.propose_templates(test_tags)
        candidates = set()
        for lemma_regex, form_regex in product(lemma_regexes, templates):
            candidates.update(get_form_candidates(eval_lemmas[k], lemma_regex, form_regex))

        candidates = list(sorted(candidates))
        num_candidates.append(len(candidates))

        if eval_forms[k] in candidates:
            covered += 1

    logger.info(f"Covered {100 * covered / num_samples}% of forms")
    logger.info(f"Avg. #candidates: {np.mean(num_candidates)}")
