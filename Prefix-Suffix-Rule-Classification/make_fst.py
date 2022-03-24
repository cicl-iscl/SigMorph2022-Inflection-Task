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

        # Separate regexes into prefix, base, and stem parts:
        # Prefix is everything before the first stem variable (BASECHAR)
        # Stem is everything between the first and last stem variables (inclusive)
        # Suffix is everything following the last stem variable
        regexes = form_regexes
        prefix_regexes, base_regexes, suffix_regexes = [], [], []

        for regex in regexes:
            # Ignore constant regexes
            if BASECHAR not in regex:
                continue

            # Get indices of stem variables
            gap_indices = [idx for idx, c in enumerate(regex) if c == BASECHAR]
            base_start_index = gap_indices[0]
            suffix_start_index = gap_indices[-1] + 1

            # Split regexes
            prefix = regex[:base_start_index]
            base = regex[base_start_index:suffix_start_index]
            suffix = regex[suffix_start_index:]

            prefix_regexes.append(prefix)
            base_regexes.append(base)
            suffix_regexes.append(suffix)

        # Compute alignments for prefix, base, suffix regexes separately
        # This is much faster than the combined step and hopefully makes some linguistic sense
        alignments = dict()
        for name, segments in [('prefix', prefix_regexes), ('base', base_regexes), ('suffix', suffix_regexes)]:
            # Filter empty segments
            segments = [regex.strip() for regex in set(segments) if regex.strip()]
            if len(segments) == 0:
                alignments[name] = None
                continue

            # Sort by length
            # (improves performance, because MSA is slower with longer sequences)
            segments = list(sorted(segments, key=len))

            # Sequencially compute MSAs
            start_regex = segments[0]
            regex_alignment = [(c,) for c in start_regex]
            for regex in tqdm(segments[1:], desc="Aligning generator regexes"):
                regex_alignment = align_form(regex_alignment, regex)

            alignments[name] = np.array(regex_alignment).T

        # Convert alignments to indices:
        # For each regex, we want the non-gap indices
        indices = dict()
        for name, field_alignments in alignments.items():
            if field_alignments is None:
                indices[name] = []
                continue

            field_indices = [[idx for idx, c in enumerate(alignment) if c != BLANK] for alignment in field_alignments]
            indices[name] = field_indices

        # Make graph:
        # Here, we generate the FST graph
        # States are characters that appear in the regexes
        self.states = dict()
        self.states[-1] = "<START>"
        self.states[-2] = "<END>"
        # Store FST transitions
        self.successors = {-1: set(), -2: set()}

        state_counter = 0
        state_counter_offsets = [0]

        for name in ["prefix", "base", "suffix"]:
            field_alignments = alignments[name]
            if field_alignments is None:
                continue

            for column in field_alignments.T:
                state_chars = list(set([char for char in column if char != BLANK]))
                assert len(state_chars) == 1
                state_char = state_chars[0]
                self.states[state_counter] = state_char
                self.successors[state_counter] = set()
                state_counter += 1

            state_counter_offsets.append(state_counter)
            offset = state_counter_offsets[-2]

            field_indices = indices[name]
            for regex_indices in field_indices:
                for start, end in zip(regex_indices[:-1], regex_indices[1:]):
                    start, end = start + offset, end + offset
                    self.successors[start].add(end)

                if name == "prefix":
                    self.successors[-1].add(offset + regex_indices[0])

                elif name == "base":
                    self.successors[-1].add(offset + regex_indices[0])
                    self.successors[offset + regex_indices[-1]].add(-2)

                    for prefix_regex_indices in indices["prefix"]:
                        self.successors[prefix_regex_indices[-1]].add(offset + regex_indices[0])

                elif name == "suffix":
                    self.successors[offset + regex_indices[-1]].add(-2)

                    for base_regex_indices in indices["base"]:
                        base_idx = state_counter_offsets[1] + base_regex_indices[-1]
                        self.successors[base_idx].add(offset + regex_indices[0])

        # Add tag constraints:
        # Currently, we only check for each state which tags are assigned to regexes
        # that use the state. This local checking is not very effective at reducing over-generation.
        self.state_tags = {state: set() for state in self.states}
        self.allowed_sequences = set()

        # For each regex, we collect all paths through the graph (by BFS)
        for regex, regex_tags in zip(form_regexes, tags):
            state_sequences = []
            # Initialise queue for BFS
            queue = [(-1, regex, [])]

            # Perform BFS
            while queue:
                state, remaining_regex, path = queue.pop(0)
                successors = self.successors[state]

                # If we have consumed the complete regex and can transition to final state, accept path
                if len(remaining_regex) == 0 and -2 in successors:
                    state_sequences.append(path)
                    continue

                # If we have consumed the complete regex but cannot transition to final state, reject path
                elif len(remaining_regex) == 0:
                    continue

                # Visit all successors whith matching states
                for successor in self.successors[state]:
                    if self.states[successor] == remaining_regex[0]:
                        queue.append((successor, remaining_regex[1:], path + [successor]))

            # Add tags to every visited state
            for state_sequence in state_sequences:
                # Also store allowed path prefixes (will be used as constrained for template generation)
                for stop in range(len(state_sequence)+1):
                    self.allowed_sequences.add(tuple(state_sequence[:stop]))

                for state in state_sequence:
                    if state < 0:
                        continue

                    self.state_tags[state].update(set(regex_tags))

        # Optional printing code
        """
        for state, successors in self.successors.items():
            succ = [(successor, self.states[successor]) for successor  in successors]
            print(f"{state}: {self.states[state]}: {succ}")

        for state, tags in self.state_tags.items():
            print(f"{state}: {self.states[state]}: {tags}")
        """

    def propose_templates(self, tags: Iterable[str]):
        """Generate possible templates for given tags by BFS"""
        tags, queue, templates = set(tags), [(-1, [])], []

        # Perform BFS
        while queue:
            state, path = queue.pop(0)
            successors = self.successors[state]

            # Discard illegal paths
            if len(path) > 0 and tuple(path) not in self.allowed_sequences:
                continue

            # If we can transition to final state, accept path
            if -2 in successors:
                templates.append(path)

            # Only transition to successors that allow given tag sequence
            for successor in successors:
                if set.issubset(tags, self.state_tags[successor]) or self.states[successor] == BASECHAR:
                    queue.append((successor, path + [successor]))

        # Decode regex chars from state sequences
        templates = [template for template in templates if tuple(template) in self.allowed_sequences]
        templates = [[self.states[state] for state in template] for template in templates]

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
    language_code = "spa"
    path = f"../../inflection/data/part1/development_languages/{language_code}.train"
    data = pd.read_csv(path, sep='\t', names=["lemma", "form", "tag"])

    # Extract lemmas, forms, and tags
    lemmas = [str(lemma) for lemma in data["lemma"].tolist()]
    forms = [str(form) for form in data["form"].tolist()]
    tags = [str(tag) for tag in data["tag"].tolist()]

    all_tags = set()
    for tag in tags:
        all_tags.update(set(tag.split(';')))

    all_regexes, form2regexes = get_regexes(lemmas, forms, paradigm_size_threshold=3, regex_count_threshold=1)
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
    num_samples = 1000
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
