import re
import numpy as np
import pandas as pd

from make_regex import get_regexes
from paradigm_align import align_form
from make_regex import BASECHAR
from paradigm_align import BLANK
from logger import logger

from typing import List, Iterable
from tqdm.auto import tqdm
from collections import defaultdict


class ProposalGenerator:
    def __init__(self, form_regexes: List[str], tags: List[List[str]]):
        assert len(form_regexes) == len(tags)

        # Align generator regexes
        start_regex = form_regexes.pop(0)
        regex_alignment = [(c,) for c in start_regex]
        for regex in tqdm(form_regexes, desc="Aligning generator regexes"):
            regex_alignment = align_form(regex_alignment, regex)

        self.regex_alignments = np.array(regex_alignment).T
        logger.info(f"Generator FST has {self.regex_alignments.shape[1]} states")

        # Make states
        state_chars = [set(column) for column in self.regex_alignments.T]
        for column in state_chars:
            column.discard(BLANK)

        assert all(len(chars) == 1 for chars in state_chars)
        state_chars = [str(list(chars)[0]) for chars in state_chars]
        self.states = {k: char for k, char in enumerate(state_chars)}
        self.states[-1] = "<START>"
        self.states[-2] = "<END>"

        # Get tags for every state
        self.state_tags = {-1: set(), -2: set()}
        for k in self.states:
            if k < 0:
                continue

            column = self.regex_alignments[:, k].reshape(-1)
            non_blank_indices = [idx for idx, char in enumerate(column) if char not in (BLANK, BASECHAR)]
            column_tags = [set(tags[idx]) for idx in non_blank_indices]
            if not column_tags:
                self.state_tags[k] = set()
            else:
                self.state_tags[k] = set.union(*column_tags)

        for state, state_tags in self.state_tags.items():
            print(state, self.states[state], state_tags)

        print()

        # Make neighbours
        self.successors = {k: set() for k in self.states}
        non_blank_indices = []
        for regex_alignment in self.regex_alignments:
            non_blank_indices.append([idx for idx, char in enumerate(regex_alignment) if char != BLANK])

        for indices in non_blank_indices:
            self.successors[-1].add(indices[0])
            self.successors[indices[-1]].add(-2)

            for start, end in zip(indices[:-1], indices[1:]):
                self.successors[start].add(end)

        # print(self.successors)

    def propose_templates(self, tags: Iterable[str], start_state=-1):
        tags = set(tags)

        if start_state == -2:
            return [[]]

        state_char = self.states[start_state] if start_state >= 0 else ''
        successors = self.successors[start_state]

        templates = []
        for successor in successors:
            if (
                    successor != -2 and
                    self.states[successor] != BASECHAR and
                    not set.issubset(tags, self.state_tags[successor])
            ):
                continue

            recursive_templates = self.propose_templates(tags, start_state=successor)
            for template in recursive_templates:
                templates.append([state_char] + template)

        return templates


class LemmaAnalyser:
    def __init__(self, lemma_regexes: List[str]):
        self.lemma_regexes = [re.sub(re.compile(BASECHAR), '(.*?)', regex) for regex in lemma_regexes]
        self.lemma_regexes = [re.compile(regex) for regex in self.lemma_regexes]





if __name__ == '__main__':
    language_code = "deu"
    path = f"../../inflection/data/part1/development_languages/{language_code}.train"
    data = pd.read_csv(path, sep='\t', names=["lemma", "form", "tag"])

    # Extract lemmas, forms, and tags
    lemmas = [str(lemma) for lemma in data["lemma"].tolist()]
    forms = [str(form) for form in data["form"].tolist()]
    tags = [str(tag) for tag in data["tag"].tolist()]

    all_tags = set()
    for tag in tags:
        all_tags.update(set(tag.split(';')))

    all_regexes, form2regexes = get_regexes(lemmas, forms, paradigm_size_threshold=8, regex_count_threshold=50)
    regex2tags = defaultdict(set)
    for form, form_tags in zip(forms, tags):
        form_regexes = form2regexes[form]
        form_tags = form_tags.split(";")

        for regex in form_regexes:
            regex2tags[regex[1]].update(set(form_tags))

    regexes = list(set([regex[1] for regex in all_regexes]))
    regex_tags = [list(regex2tags[regex]) for regex in regexes]
    generator = ProposalGenerator(regexes, regex_tags)

    test_tags = tags[2000].split(';')
    logger.info(f"Test tags: {test_tags}")
    templates = generator.propose_templates(test_tags)

    logger.info(f"Collected {len(templates)} templates")

    for template in templates:
        print("".join(template))
