import time
import numpy as np
import pandas as pd

from tqdm.auto import trange
from logger import logger
from itertools import product
from make_regex import get_regexes
from collections import defaultdict
from make_regex import filter_regexes
from make_regex import get_form_candidates


if __name__ == '__main__':
    language_code = "ang_small"
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



