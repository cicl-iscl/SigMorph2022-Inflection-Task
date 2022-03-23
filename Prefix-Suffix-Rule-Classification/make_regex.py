import re

from logger import logger
from tqdm.auto import tqdm
from typing import List, Dict
from paradigm_align import BLANK
from collections import defaultdict
from paradigm_align import get_paradigms
from paradigm_align import paradigm_align

BASECHAR = "X"


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
        if len(paradigm) >= paradigm_size_threshold
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
