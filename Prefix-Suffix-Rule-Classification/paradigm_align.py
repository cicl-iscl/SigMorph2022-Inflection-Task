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
    s1, s2 = set(s1), set(s2)
    return len(set.intersection(s1, s2)) / len(set.union(s1, s2))


def num_matches(column: Tuple[str], char: str):
    total = 0

    for c in column:
        if c == BLANK:
            continue
        elif c != char:
            return -100000
        else:
            total += 1

    return total


##################################
# Real function                  #
##################################

def get_paradigms(lemmas: List[str], forms: List[str]):
    paradigms = defaultdict(set)

    for lemma, form in zip(lemmas, forms):
        paradigms[lemma].add(form)

    paradigms = {lemma: list(sorted(forms)) for lemma, forms in paradigms.items()}
    return paradigms


def score_alignment(alignment: List[Tuple[str]]):
    score = 0
    segment_length = 0

    for column in alignment:
        if BLANK not in column:
            segment_length += 1
        else:
            score += segment_length ** 2
            segment_length = 0

    return score


def reconstruct_alignments_from_traceback(traceback, partial_alignment, prev_alignment, form, num_prev_alignments):
    if len(prev_alignment) == 0 and len(form) == 0:
        return [partial_alignment]

    backpointers = traceback[-1][-1]
    reconstructed_alignments = []

    for backpointer in backpointers:
        if backpointer == DIAGONAL:
            column = (*prev_alignment[-1], form[-1])
            truncated_traceback = [row[:-1] for row in traceback[:-1]]
            local_reconstructed_alignments = reconstruct_alignments_from_traceback(
                truncated_traceback, partial_alignment + [column], prev_alignment[:-1], form[:-1], num_prev_alignments
            )
            reconstructed_alignments.extend(local_reconstructed_alignments)

        elif backpointer == HORIZONTAL:
            column = (*prev_alignment[-1], BLANK)
            truncated_traceback = [row[:-1] for row in traceback]
            local_reconstructed_alignments = reconstruct_alignments_from_traceback(
                truncated_traceback, partial_alignment + [column], prev_alignment[:-1], form, num_prev_alignments
            )
            reconstructed_alignments.extend(local_reconstructed_alignments)

        elif backpointer == VERTICAL:
            column = (*tuple([BLANK] * num_prev_alignments), form[-1])
            truncated_traceback = traceback[:-1]
            local_reconstructed_alignments = reconstruct_alignments_from_traceback(
                truncated_traceback, partial_alignment + [column], prev_alignment, form[:-1], num_prev_alignments
            )
            reconstructed_alignments.extend(local_reconstructed_alignments)

    return reconstructed_alignments


def align_form(prev_alignment: List[Tuple[str]], form: str, gap_cost: int = 0):
    prev_alignment_len, form_len = len(prev_alignment), len(form)

    score_matrix = np.zeros(shape=(form_len+1, prev_alignment_len+1))
    traceback = [[[] for _ in range(prev_alignment_len+1)] for _ in range(form_len+1)]

    for k in range(1, prev_alignment_len+1):
        score_matrix[0, k] = score_matrix[0, k-1] + gap_cost
        traceback[0][k].append(HORIZONTAL)

    for k in range(1, form_len+1):
        score_matrix[k, 0] = score_matrix[k-1, 0] + gap_cost
        traceback[k][0].append(VERTICAL)

    for i in range(1, form_len+1):
        for j in range(1, prev_alignment_len+1):
            diagonal_score = score_matrix[i-1, j-1] + num_matches(prev_alignment[j-1], form[i-1])
            horizontal_score = score_matrix[i, j-1] + gap_cost
            vertical_score = score_matrix[i-1, j] + gap_cost
            scores = [diagonal_score, horizontal_score, vertical_score]

            best_score = max(scores)
            score_matrix[i, j] = best_score
            for k, score in enumerate(scores):
                if score == best_score:
                    traceback[i][j].append(k)

    reconstructed_alignments = reconstruct_alignments_from_traceback(
        traceback, [], prev_alignment, form, len(prev_alignment[0])
    )
    best_alignment, best_alignment_score = None, -100

    for alignment in reconstructed_alignments:
        alignment_score = score_alignment(alignment)
        if alignment_score > best_alignment_score:
            best_alignment = alignment
            best_alignment_score = alignment_score

    best_alignment = list(reversed(best_alignment))
    return best_alignment


def paradigm_align(lemma: str, forms: List[str], gap_cost: int = 0):
    # Discard lemma from forms
    forms = set(forms)
    forms.discard(lemma)

    # Sort forms (by jaccard index)
    forms = list(sorted(forms, key=lambda f: jaccard_index(lemma, f), reverse=True))

    # Sequentially align forms to lemma
    alignments = [(c,) for c in lemma]

    for form in forms:
        alignments = align_form(alignments, form, gap_cost=gap_cost)

    alignments = np.array(alignments).T

    form2alignment = {lemma: alignments[0].tolist()}
    for form, alignment in zip(forms, alignments[1:]):
        form2alignment[form] = alignment.tolist()

    return form2alignment
