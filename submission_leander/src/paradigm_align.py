import random
import numpy as np

from typing import List, Tuple
from collections import defaultdict

##################################
# Helper functions               #
##################################

BLANK = "#"
DIAGONAL = 0
HORIZONTAL = 1
VERTICAL = 2


def jaccard_index(s1: str, s2: str):
    """Calculates the jaccard score of 2 string: Character overlap divided by character union"""
    s1, s2 = set(s1), set(s2)
    return len(set.intersection(s1, s2)) / len(set.union(s1, s2))


def num_matches(column: Tuple[str], char: str):
    """Helper function to calculate the number of matching characters of a column in MSA"""
    total = 0

    for c in column:
        # Ignore blanks
        if c == BLANK:
            continue
        # If there's a different character, we can't align
        elif c != char:
            return -100000
        # If we find the same character, increase score
        else:
            total += 1

    return total


##################################
# Real function                  #
##################################

def get_paradigms(lemmas: List[str], forms: List[str]):
    """
    Cluster aligned lists of lemmas and strings into paradigms where each lemma is mapped to its corresponding forms.
    """
    paradigms = defaultdict(set)

    for lemma, form in zip(lemmas, forms):
        paradigms[lemma].add(form)

    paradigms = {lemma: list(sorted(forms)) for lemma, forms in paradigms.items()}
    return paradigms


def score_alignment(alignment: List[Tuple[str]]):
    """
    Scores alignments by the sum of the squared lengths of contiguous alignments.
    This favours alignment of longer contiguous subsequences, e.g.
    --erden-    vs.    --e-rden-
    geerde-t           geerde-t
    (the first one is preferred)
    """
    score = 0
    segment_length = 0

    for column in alignment:
        if BLANK not in column:
            segment_length += 1
        else:
            score += segment_length ** 2
            segment_length = 0

    # Add score for final segment
    score += segment_length ** 2

    return score


def convert_indices(prev_alignment, form, indices):
    alignment = []
    last_i, last_j = None, None

    for i, j in reversed(indices):
        i, j = i-1, j-1
        prev_alignment_column = tuple([BLANK] * len(prev_alignment[0])) if (j == last_j or j < 0) else prev_alignment[j]
        form_column = BLANK if (i == last_i or i < 0) else form[i]
        column = (*prev_alignment_column, form_column)
        alignment.append(column)
        last_i, last_j = i, j

    return alignment


def reconstruct_alignment(traceback: List[List[List[int]]], prev_alignment, form, max_candidates: int = 10000):
    best_alignment = None
    best_score = -np.inf
    candidate_counter = 0

    queue = [(len(traceback)-1, len(traceback[0])-1, [])]

    while len(queue) > 0 and candidate_counter < max_candidates:
        if len(queue) > max_candidates:
            queue = queue[-max_candidates:]
        i, j, candidate = queue.pop()

        # Check if candidate has been fully reconstructed
        if i == 0 and j == 0:
            candidate = convert_indices(prev_alignment, form, candidate)
            candidate_score = score_alignment(candidate)
            candidate_counter += 1

            if candidate_score > best_score:
                best_alignment = candidate.copy()
                best_score = candidate_score

        else:
            predecessor_directions = traceback[i][j]
            random.shuffle(predecessor_directions)
            for direction in predecessor_directions:
                if direction == DIAGONAL:
                    queue.append((i - 1, j - 1, candidate + [(i, j)]))

                elif direction == HORIZONTAL:
                    queue.append((i, j - 1, candidate + [(i, j)]))

                elif direction == VERTICAL:
                    queue.append((i - 1, j, candidate + [(i, j)]))

                else:
                    raise ValueError(f"Unknown direction: {direction}")

    return best_alignment


def align_form(prev_alignment: List[Tuple[str]], form: str, gap_cost: int = 0):
    """Align a new form to the previously computed MSA alignment of forms."""
    prev_alignment_len, form_len = len(prev_alignment), len(form)

    # Initialise score matrix
    score_matrix = np.zeros(shape=(form_len+1, prev_alignment_len+1))
    # Initialise traceback (=backpointer) matrix
    # We want to reconstruct all alignments with max. score, so we keep all relevant backpointers
    traceback = [[[] for _ in range(prev_alignment_len+1)] for _ in range(form_len+1)]

    # Initialise deletion of MSA columns
    for k in range(1, prev_alignment_len+1):
        score_matrix[0, k] = score_matrix[0, k-1] + gap_cost
        traceback[0][k].append(HORIZONTAL)

    # Initialise deletion of form characters
    for k in range(1, form_len+1):
        score_matrix[k, 0] = score_matrix[k-1, 0] + gap_cost
        traceback[k][0].append(VERTICAL)

    # Calculate edit distance
    for i in range(1, form_len+1):
        for j in range(1, prev_alignment_len+1):
            # Use column sum as metric for column-character matching
            diagonal_score = score_matrix[i-1, j-1] + num_matches(prev_alignment[j-1], form[i-1])
            # Gap in MSA columns
            horizontal_score = score_matrix[i, j-1] + gap_cost
            # Gap in form
            vertical_score = score_matrix[i-1, j] + gap_cost

            # Find best scores and add operations to traceback
            scores = [diagonal_score, horizontal_score, vertical_score]
            best_score = max(scores)
            score_matrix[i, j] = best_score
            for k, score in enumerate(scores):
                if score == best_score:
                    traceback[i][j].append(k)

    # Reconstruct all alignments with maximum score
    best_alignment = reconstruct_alignment(traceback, prev_alignment, form)
    return best_alignment


def msa(sequences: List[str], gap_cost: int = 0):
    assert len(sequences) >= 1

    pivot, sequences = sequences[0], sequences[1:]
    alignment = [(c,) for c in pivot]

    for seq in sequences:
        alignment = align_form(alignment, seq, gap_cost=gap_cost)

    alignment = np.array(alignment).T
    return alignment


def paradigm_align(lemma: str, forms: List[str], gap_cost: int = 0):
    """Compute MSA of paradigm (lemma + forms)"""
    # Discard lemma from forms
    forms = set(forms)
    forms.discard(lemma)

    # Sort forms (by jaccard index)
    forms = list(sorted(forms, key=lambda f: jaccard_index(lemma, f), reverse=True))

    # Sequentially align forms to lemma
    alignments = [(c,) for c in lemma]

    for form in forms:
        alignments = align_form(alignments, form, gap_cost=gap_cost)

    # Some reformatting
    alignments = np.array(alignments).T

    form2alignment = {lemma: alignments[0].tolist()}
    for form, alignment in zip(forms, alignments[1:]):
        form2alignment[form] = alignment.tolist()

    return form2alignment


if __name__ == '__main__':
    l = "tippen"
    f = ["tippst", "tippte", "getippt"]

    print(paradigm_align(l, f))