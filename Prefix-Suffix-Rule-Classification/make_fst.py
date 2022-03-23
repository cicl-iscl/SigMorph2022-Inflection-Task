import re
import numpy as np
import pandas as pd

from logger import logger
from make_regex import BASECHAR
from paradigm_align import BLANK
from tqdm.auto import tqdm, trange
from make_regex import get_regexes
from collections import defaultdict
from paradigm_align import align_form
from typing import List, Iterable, Tuple


class ProposalGenerator:
    """Class for generating (deterministically) templates for generating forms based on a set of tags"""
    def __init__(self, form_regexes: List[str], tags: List[List[str]]):
        assert len(form_regexes) == len(tags)

        # Align generator regexes
        # Warning! May be very costly if many regexes are used
        # TODO: make this more efficient, e.g. by divide-and-conquer
        start_regex = form_regexes.pop(0)
        regex_alignment = [(c,) for c in start_regex]
        for regex in tqdm(form_regexes, desc="Aligning generator regexes"):
            regex_alignment = align_form(regex_alignment, regex)

        self.regex_alignments = np.array(regex_alignment).T
        logger.info(f"Generator FST has {self.regex_alignments.shape[1]} states")

        # Make states
        # Each states represents 1 char in the aligned form regexes
        state_chars = [set(column) for column in self.regex_alignments.T]
        for column in state_chars:
            column.discard(BLANK)

        assert all(len(chars) == 1 for chars in state_chars)
        state_chars = [str(list(chars)[0]) for chars in state_chars]
        self.states = {k: char for k, char in enumerate(state_chars)}
        self.states[-1] = "<START>"
        self.states[-2] = "<END>"

        # Get tags for every state
        # We store with which tags each character has been observed in the training data
        # We can use this information later to constrain the form generation templates
        self.state_tags = {-1: set(), -2: set()}
        for state_idx in self.states:
            if state_idx < 0:
                continue

            # Extract forms for which the current state is not a gap
            column = self.regex_alignments[:, state_idx].reshape(-1)
            non_blank_indices = [idx for idx, char in enumerate(column) if char not in (BLANK, BASECHAR)]
            # Get tags of relevant forms
            column_tags = [set(tags[idx]) for idx in non_blank_indices]
            if not column_tags:
                self.state_tags[state_idx] = set()
            else:
                self.state_tags[state_idx] = set.union(*column_tags)

        # Make neighbours
        # For each state, store successors in graph
        self.successors = {k: set() for k in self.states}

        # For each form, find states != gap
        non_blank_indices = []
        for regex_alignment in self.regex_alignments:
            non_blank_indices.append([idx for idx, char in enumerate(regex_alignment) if char != BLANK])

        # Find & store successors
        for indices in non_blank_indices:
            # Connect start to all first states
            self.successors[-1].add(indices[0])
            # Connect all finishing states to end state
            self.successors[indices[-1]].add(-2)

            for start, end in zip(indices[:-1], indices[1:]):
                self.successors[start].add(end)

    def propose_templates(self, tags: Iterable[str], start_state=-1):
        """Recursively generate possible templates for given tags"""
        # End of recursion (end state reached)
        if start_state == -2:
            return [[]]

        tags, templates = set(tags), []
        state_char = self.states[start_state] if start_state >= 0 else ''
        successors = self.successors[start_state]

        # Recursively follow all possible successors
        for successor in successors:
            # Define criteria for excluding paths through template graph
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
    """Class for extracting possible subsequences of lemmas that can be inserted into form templates"""
    def __init__(self, lemma_regexes: List[str]):
        # Map BASECHAR to any contiguous subsequence
        # $ makes sure that we always match the entire lemma
        self.lemma_regexes = [re.sub(re.compile(BASECHAR), '(.+?)', regex) + '$' for regex in lemma_regexes]
        self.lemma_regexes = [re.compile(regex) for regex in self.lemma_regexes]

    def analyse_lemma(self, lemma: str):
        # TODO: Find way to get all possible analyses, not just the ones (greedily) calculated by re
        stems = []
        for regex in self.lemma_regexes:
            for groups in re.finditer(regex, lemma):
                stems.append(tuple([str(group) for group in groups.groups()]))

        return stems


def make_candidates(stem_decompositions: List[Tuple[str]], templates: List[str]) -> List[str]:
    """
    Given a list of stem subsequences and form templates, fill in stem subsequences into templates, where possible,
    and return exhaustive list of candidate forms.
    """
    candidates = []
    for stem_decomposition in stem_decompositions:
        for template in templates:
            # Can only fill in if we have the same number of stem subsequences and gaps in template
            if len(stem_decomposition) == template.count(BASECHAR):
                idx = 0
                candidate = []
                for char in template:
                    if char != BASECHAR:
                        candidate.append(char)
                    else:
                        stem = stem_decomposition[idx]
                        candidate.append(stem)
                        idx += 1

                candidates.append("".join(candidate))

    return candidates


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

    all_regexes, form2regexes = get_regexes(lemmas, forms, paradigm_size_threshold=3, regex_count_threshold=10)
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
    generator = ProposalGenerator(regexes, regex_tags)
    analyser = LemmaAnalyser(list(set([regex[0] for regex in all_regexes])))

    test_item = 10001
    test_tags = tags[test_item].split(';')
    logger.info(f"Test tags: {test_tags}")
    templates = generator.propose_templates(test_tags)
    logger.info(f"Collected {len(templates)} templates")
    # logger.info(f"{[''.join(template) for template in templates]}")

    stem_decompositions = analyser.analyse_lemma(lemmas[test_item])
    print(stem_decompositions)
    logger.info(f"Collected {len(stem_decompositions)} stem decompositions")

    candidates = make_candidates(stem_decompositions, templates)

    logger.info(f"Collected {len(candidates)} candidates")
    logger.info(f"Correct form: {forms[test_item]}")
    logger.info(f"Correct form in candidates: {forms[test_item] in candidates}")

    covered = 0
    num_samples = 10000
    for k in trange(num_samples):
        test_tags = tags[k].split(';')
        templates = generator.propose_templates(test_tags)
        # logger.info(f"Collected {len(templates)} templates")

        stem_decompositions = analyser.analyse_lemma(lemmas[k])
        # logger.info(f"Collected {len(stem_decompositions)} stem decompositions")
        candidates = make_candidates(stem_decompositions, templates)

        if forms[k] in candidates:
            covered += 1

    logger.info(f"Covered {100 * covered / num_samples}% of forms")
