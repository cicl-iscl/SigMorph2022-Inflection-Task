import re

from logger import logger
from tqdm.auto import tqdm
from typing import List, Dict
from collections import Counter
from paradigm_align import BLANK
from collections import defaultdict
from paradigm_align import get_paradigms
from paradigm_align import paradigm_align

BASECHAR = "%"


def make_regexes_from_alignments(lemma: str, forms: List[str], form2alignment: Dict[str, List[str]]):
    """Collapse segments that are present in all forms in alignment"""
    # Get alignments for all forms
    alignments = [form2alignment[form] for form in [lemma] + forms]
    base_indices = []

    # Find indices of columns where no form has a gap (character present in all forms)
    for idx in range(len(alignments[0])):
        if not any(alignment[idx] == BLANK for alignment in alignments):
            base_indices.append(idx)

    # Collapse contiguous segments that are present in all forms into a single special character (BASECHAR)
    regex_templates = []
    for alignment in alignments:
        alignment = [(c if idx not in base_indices else BASECHAR) for idx, c in enumerate(alignment)]
        alignment = [c for c in alignment if c != BLANK]
        alignment = "".join(alignment)
        alignment = re.sub(re.compile(BASECHAR + '+'), BASECHAR, alignment)
        regex_templates.append(alignment)

    # Return lemma and form representations
    lemma_template = regex_templates[0]
    assert len(forms) == len(regex_templates[1:])
    return {
        form: (lemma_template, template) for form, template in zip(forms, regex_templates[1:])
        if template.count(BASECHAR) == lemma_template.count(BASECHAR)
    }


def get_regexes(lemmas: List[str], forms: List[str], paradigm_size_threshold: int = 1, regex_count_threshold: int = 1):
    """Generate regex-equivalent representations (collapsed segments that are present in all forms for all paradigms"""
    logger.info(f"Data consists of {len(forms)} form-lemma pairs")

    # Cluster lemmas and forms into lemma - paradigm mappings
    paradigms = get_paradigms(lemmas, forms)
    logger.info(f"Collected {len(paradigms)} paradigms")

    # Only consider paradigm with more forms than given threshold
    # paradigm with too few forms may be unreliable
    sufficient_paradigms = {
        lemma: paradigm for lemma, paradigm in paradigms.items()
        if len(paradigm) + 1 >= paradigm_size_threshold
    }
    logger.info(f"Collected {len(sufficient_paradigms)} useful paradigms")
    num_useful_forms = sum([len(paradigm) for paradigm in sufficient_paradigms.values()])
    total_paradigm_size = sum([len(paradigm) for paradigm in paradigms.values()])
    paradigm_coverage = 100 * num_useful_forms / total_paradigm_size
    logger.info(f"Useful paradigms contain {num_useful_forms} ({paradigm_coverage:.1f}%) forms")

    # Extract all regex-equivalent representations from paradigms
    all_regexes_counts = defaultdict(int)
    all_regexes = defaultdict(set)

    for lemma, lemma_forms in tqdm(sufficient_paradigms.items(), desc="Extract regexes"):
        # Align forms in paradigm by custom MSA aligner
        form2alignment = paradigm_align(lemma, lemma_forms)
        # Compute regex-equivalent representations from MSA
        regexes = make_regexes_from_alignments(lemma, lemma_forms, form2alignment)

        for form, regex in regexes.items():
            all_regexes[form].add(regex)
            all_regexes_counts[regex] += 1

    all_regexes_filtered = defaultdict(set)
    all_regexes_raw = set()

    # Only keep regexes that appear more frequently than a given threshold
    # Regexes that appear less frequently may be unreliable / noise
    for form, regexes in all_regexes.items():
        for regex in regexes:
            if all_regexes_counts[regex] >= regex_count_threshold:
                all_regexes_filtered[form].add(regex)
                all_regexes_raw.add(regex)

    num_regex = len(all_regexes_raw)
    coverage = sum(map(all_regexes_counts.get, all_regexes_raw)) / sum(all_regexes_counts.values())
    logger.info(f"Collected {num_regex} ({100 * coverage:.2f}%) lemma -> form regex pairs")
    logger.info(f"Collected {len(set([regex[0] for regex in all_regexes_raw]))} lemma regexes")
    logger.info(f"Collected {len(set([regex[1] for regex in all_regexes_raw]))} form regexes")

    # Return regexes and form -> regex mappings
    return list(sorted(all_regexes_raw, key=all_regexes_counts.get, reverse=True)), all_regexes_filtered


def parse_word(word: str, regex: str):
    if BASECHAR in word or BASECHAR not in regex:
        return []

    assert BASECHAR not in word
    assert BASECHAR in regex
    # Get all matches of word and regex
    queue = [(list(range(len(word))), list(range(len(regex))), [''])]
    decompositions = set()

    while len(queue) > 0:
        word_idx, regex_idx, decomposition = queue.pop(0)
        decomposition = decomposition.copy()

        # Save fully parsed variants
        if len(word_idx) == 0 and len(regex_idx) == 0:
            decomposition = [chunk for chunk in decomposition if chunk]
            decompositions.add(tuple(decomposition))

        # Read in next characters
        if len(word_idx) > 0 and len(regex_idx) > 0:
            # Match equal characters
            if word[word_idx[0]] == regex[regex_idx[0]]:
                queue.append((word_idx[1:], regex_idx[1:], decomposition))

            # In case we encounter gap in regex,
            # we can either continue with gap or end gap
            elif regex[regex_idx[0]] == BASECHAR:
                decomposition[-1] = decomposition[-1] + word[word_idx[0]]
                queue.append((word_idx[1:], regex_idx[1:], decomposition + ['']))
                queue.append((word_idx[1:], regex_idx, decomposition))

    return list(sorted(decompositions))


def get_form_candidates(lemma: str, lemma_regex: str, form_regex: str):
    lemma_parses = parse_word(lemma, lemma_regex)
    num_gaps = form_regex.count(BASECHAR)
    candidates = set()

    for decomposition in lemma_parses:
        if len(decomposition) != num_gaps:
            continue

        candidate, decomposition_counter = [], 0
        for char in form_regex:
            if char == BASECHAR:
                candidate.append(decomposition[decomposition_counter])
                decomposition_counter += 1
            else:
                candidate.append(char)

        candidates.add("".join(candidate))

    return list(sorted(candidates))


def filter_regexes(lemmas: List[str], forms: List[str], form2regex):
    lemma_regexes, form_regexes = [], []

    for form in forms:
        regex_pairs = form2regex.get(form, [])
        for lemma_regex, form_regex in regex_pairs:
            lemma_regexes.append(lemma_regex)
            form_regexes.append(form_regex)

    lemma_regex_counts = Counter(lemma_regexes)
    form_regex_counts = Counter(form_regexes)

    lemma_regexes = list(sorted(set(lemma_regexes), key=lemma_regex_counts.get, reverse=True))
    form_regexes = list(sorted(set(form_regexes), key=form_regex_counts.get, reverse=True))

    filtered_form2regex = defaultdict(set)
    not_found_counter = 0

    for lemma, form in tqdm(zip(lemmas, forms), total=len(forms), desc="Reducing regexes"):
        found = False
        for lemma_regex in lemma_regexes:
            if found:
                continue
            for form_regex in form_regexes:
                candidates = get_form_candidates(lemma, lemma_regex, form_regex)
                if form in candidates:
                    filtered_form2regex[form].add((lemma_regex, form_regex))
                    found = True
                    break

        if not found:
            not_found_counter += 1

    logger.info(f"Not found regex for: {not_found_counter} of {len(forms)} forms")
    lemma_regexes, form_regexes = [], []

    for form in forms:
        regex_pairs = filtered_form2regex.get(form, [])
        for lemma_regex, form_regex in regex_pairs:
            lemma_regexes.append(lemma_regex)
            form_regexes.append(form_regex)

    return filtered_form2regex
