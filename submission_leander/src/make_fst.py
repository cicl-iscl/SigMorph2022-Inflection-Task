import nltk
import time
import numpy as np

from logger import logger
from tqdm.auto import tqdm
from make_regex import BASECHAR
from paradigm_align import BLANK
from typing import List, Iterable
from collections import defaultdict
from paradigm_align import align_form


class ProposalGenerator:
    FIELDS = ["prefix", "base", "suffix"]
    START_STATE = -1
    FINAL_STATE = -2
    """Class for generating (deterministically) templates for generating forms based on a set of tags"""
    def __init__(self, form_regexes: List[str], tags: List[List[str]], verbose=True):
        assert len(form_regexes) == len(tags)

        self.regexes = form_regexes.copy()
        self.tags = tags.copy()

        self.regex2tags = {regex: set(tags) for regex, tags in zip(self.regexes, self.tags)}
        self.all_tags = set.union(*(set(t) for t in tags))
        self.ngram_constraint_order = 0

        logger.info("FST: Prepare fields")
        time.sleep(0.01)
        self._make_fields()
        time.sleep(0.01)
        logger.info("FST: Align fields")
        time.sleep(0.01)
        self._align_fields()
        time.sleep(0.01)
        logger.info("FST: Make transition graph")
        time.sleep(0.01)
        self._make_transition_graph()
        time.sleep(0.01)
        logger.info(f"Transition graph has {len(self.states)} states")
        logger.info("FST: Make constraints")
        time.sleep(0.01)
        self._make_constraints()

        if verbose:
            for state, successors in sorted(self.successors.items(), key=lambda s: s[0]):
                flattened_successors = [(successor, self.states[successor]) for successor in successors]
                print(f"{state}: {self.states[state]}: {flattened_successors}")

            for state, tags in sorted(self.state_allowed_tags.items(), key=lambda s: s[0]):
                print(f"{state}: {self.states[state]}: {tags}")

        time.sleep(0.1)
        # logger.info("FST: Check FST")
        time.sleep(0.01)
        # self._check_transition_graph()
        time.sleep(0.01)

    def _make_fields(self):
        # Separate regexes into prefix, base, and stem parts:
        # Prefix is everything before the first stem variable (BASECHAR)
        # Stem is everything between the first and last stem variables (inclusive)
        # Suffix is everything following the last stem variable
        complete_regexes = self.regexes.copy()
        prefix_regexes = defaultdict(set)
        base_regexes = defaultdict(set)
        suffix_regexes = defaultdict(set)

        for index, complete_regex in enumerate(complete_regexes):
            # Ignore constant regexes
            if BASECHAR not in complete_regex:
                continue

            # Get indices of stem variables
            gap_indices = [idx for idx, c in enumerate(complete_regex) if c == BASECHAR]
            base_start_index = gap_indices[0]
            suffix_start_index = gap_indices[-1] + 1

            # Split regexes
            prefix = complete_regex[:base_start_index]
            base = complete_regex[base_start_index:suffix_start_index]
            suffix = complete_regex[suffix_start_index:]

            if prefix.strip():
                prefix_regexes[prefix].add(index)

            if base.strip():
                base_regexes[base].add(index)

            if suffix.strip():
                suffix_regexes[suffix].add(index)

        self.field_regexes = {'prefix': prefix_regexes, 'base': base_regexes, 'suffix': suffix_regexes}

        # Sanity check
        assert all(field in self.field_regexes for field in self.FIELDS)
        for field_name in self.FIELDS:
            for segment, indices in self.field_regexes[field_name].items():
                for idx in indices:
                    assert segment in self.regexes[idx]

    def _align_fields(self):
        # Compute alignments for prefix, base, suffix regexes separately
        # This is much faster than the combined step and hopefully makes some linguistic sense
        self.field_alignments = dict()
        self.field_alignment_indices = dict()
        offset = 0

        for field_name in self.FIELDS:
            field_regexes = self.field_regexes[field_name].copy()

            # Sort by length (improves performance, because MSA is slower with longer sequences)
            sorted_field_regexes = list(sorted(field_regexes.keys(), key=len))
            sorted_field_regexes = list(map(str, sorted_field_regexes))

            # Sequentially compute MSAs
            if len(sorted_field_regexes) == 0:
                self.field_alignments[field_name] = None
                self.field_alignment_indices[field_name] = dict()
                continue

            start_regex, sorted_field_regexes = sorted_field_regexes[0], sorted_field_regexes[1:]
            field_alignment = [(c,) for c in start_regex]
            for segment in tqdm(sorted_field_regexes, desc="Aligning generator regexes"):
                field_alignment = align_form(field_alignment, segment)

            field_alignment = np.array(field_alignment).T
            self.field_alignments[field_name] = field_alignment

            # Convert alignments into indices
            field_alignment_indices = []
            for alignment in field_alignment:
                # Add offset to have different state indices for prefix, base, suffix
                field_alignment_indices.append([(offset + idx, c) for idx, c in enumerate(alignment) if c != BLANK])

            sorted_field_regexes = ["".join([idx[1] for idx in indices]) for indices in field_alignment_indices]
            field_tag_indices = [field_regexes[sorted_regex] for sorted_regex in sorted_field_regexes]

            self.field_alignment_indices[field_name] = {
                tuple(indices): tag_indices for indices, tag_indices in zip(field_alignment_indices, field_tag_indices)
            }

            # Sanity check
            for state_indices, tag_indices in self.field_alignment_indices[field_name].items():
                for _, char in state_indices:
                    for idx in tag_indices:
                        assert char in self.regexes[idx]

            # Increase offset
            offset += field_alignment.shape[1]

        # Sanity check
        assert all(field in self.field_alignment_indices for field in self.FIELDS)
        assert all(field in self.field_alignments for field in self.FIELDS)

    def _make_transition_graph(self):
        # Make graph:
        # Here, we generate the FST graph
        # States are characters that appear in the regexes
        self.states = dict()
        self.states[self.START_STATE] = "<START>"
        self.states[self.FINAL_STATE] = "<FINAL>"
        self.successors = defaultdict(set)
        self.state_regexes = defaultdict(set)
        self.state_allowed_tags = defaultdict(set)
        self.state_required_tags = defaultdict(lambda: self.all_tags.copy())

        self.regex2states = defaultdict(list)
        self.state2field = dict()

        for field_name in self.FIELDS:
            for state_indices, tag_indices in self.field_alignment_indices[field_name].items():
                first_state_index = state_indices[0][0]
                last_state_index = state_indices[-1][0]

                # Add state
                for state_index, state_char in state_indices:
                    assert self.states.get(state_index, state_char) == state_char
                    self.states[state_index] = state_char
                    self.state2field[state_index] = field_name

                    # Add regexes and tags
                    for tag_index in tag_indices:
                        state_tags = self.tags[tag_index]
                        self.state_allowed_tags[state_index].update(set(state_tags))
                        self.state_required_tags[state_index] = set.intersection(
                            self.state_required_tags[state_index], set(state_tags)
                        )

                        state_regex = self.regexes[tag_index]
                        # assert len(state_regex) == len(state_indices)
                        self.state_regexes[state_index].add(state_regex)
                        self.regex2states[state_regex].append(state_index)

                # Add successors
                for (start, _), (end, _) in zip(state_indices[:-1], state_indices[1:]):
                    self.successors[start].add(end)

                if field_name == "prefix":
                    self.successors[self.START_STATE].add(first_state_index)

                elif field_name == "base":
                    self.successors[self.START_STATE].add(first_state_index)
                    self.successors[last_state_index].add(self.FINAL_STATE)

                    for prefix_indices in self.field_alignment_indices["prefix"].keys():
                        self.successors[prefix_indices[-1][0]].add(first_state_index)

                elif field_name == "suffix":
                    self.successors[last_state_index].add(self.FINAL_STATE)

                    for base_indices in self.field_alignment_indices["base"].keys():
                        self.successors[base_indices[-1][0]].add(first_state_index)

        self.state_required_tags = {state: tags for state, tags in self.state_required_tags.items()}

    def _make_constraints(self):
        self.allowed_ngrams = set()
        self.allowed_ngram_tags = defaultdict(set)
        self.required_ngram_tags = defaultdict(lambda: self.all_tags.copy())

        for regex in self.regexes:
            if BASECHAR not in regex:
                continue

            regex_states = list(sorted(self.regex2states[regex]))
            try:
                assert "".join([self.states[state] for state in regex_states]) == regex
            except AssertionError as e:
                print(regex_states)
                print(regex)
                print("".join([self.states[state] for state in regex_states]) == regex)

                raise e

            regex_tags = self.regex2tags[regex]
            # print(regex, regex_states)

            for n in range(2, self.ngram_constraint_order + 1):
                for ngram in nltk.ngrams(regex_states, n):
                    ngram = tuple(ngram)
                    self.allowed_ngrams.add(ngram)
                    self.allowed_ngram_tags[ngram].update(regex_tags)
                    self.required_ngram_tags[ngram] = set.intersection(
                        self.required_ngram_tags[ngram], regex_tags
                    )

        self.required_ngram_tags = {ngram: tags for ngram, tags in self.required_ngram_tags.items()}

    def _check_transition_graph(self):
        fail = 0
        for regex, tags in zip(self.regexes, self.tags):
            if BASECHAR not in regex:
                continue

            templates = self.propose_templates(tags)

            if regex not in templates:
                fail += 1

        logger.warning(f"Failed to reconstruct {fail} of {len(self.regexes)} regexes")

    def propose_templates(self, condition_tags: Iterable[str]):
        """Generate possible templates for given tags by BFS"""
        condition_tags, queue, templates = set(condition_tags), [(self.START_STATE, [])], []

        # Perform BFS
        while queue:
            state, path = queue.pop(0)

            # Discard illegal paths
            illegal = False
            for n in range(2, self.ngram_constraint_order + 1):
                if n > len(path):
                    continue

                tail = tuple(path[-n:])
                if (
                        tail not in self.allowed_ngrams or
                        not set.issubset(condition_tags, self.allowed_ngram_tags[tail]) or
                        not set.issubset(self.required_ngram_tags[tail], condition_tags)
                ):
                    illegal = True
                    break

            if illegal:
                continue

            successors = self.successors[state]

            # If we can transition to final state, accept path
            if self.FINAL_STATE in successors:
                templates.append(path)

            # Only transition to successors that allow given tag sequence
            for successor in successors:
                if self.states[successor] == BASECHAR:
                    queue.append((successor, path + [successor]))

                elif set.issubset(condition_tags, self.state_allowed_tags[successor]):
                    if set.issubset(self.state_required_tags[successor], condition_tags):
                        queue.append((successor, path + [successor]))

        # Decode regex chars from state sequences
        templates = [[str(self.states[state]) for state in template] for template in templates]
        templates = ["".join(template) for template in templates]
        # templates = [template for template in templates if self._check_template(template, tags)]

        return templates

    def parse_form(self, form: str, tags: List[str] = None):
        best_analysis = None
        best_score = -1
        queue = [(self.START_STATE, form, [], 0)]

        while len(queue) > 0:
            state, suffix, candidate, state_count = queue.pop(0)

            if self.states[state] == BASECHAR and suffix:
                queue.append((state, suffix[1:], candidate + [suffix[0]], state_count))

            successors = self.successors[state]
            for successor in successors:
                if successor == self.FINAL_STATE and not suffix:
                    if state_count > best_score:
                        best_analysis = candidate
                        best_score = state_count

                elif self.states[successor] == BASECHAR and suffix:
                    queue.append((successor, suffix[1:], candidate + [suffix[0]], state_count))

                elif suffix and self.states[successor] == suffix[0]:
                    queue.append((successor, suffix[1:], candidate + [f"S{successor}:{suffix[0]}"], state_count + 1))

        return best_analysis



