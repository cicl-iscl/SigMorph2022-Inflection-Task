# Paradigm learning and paradigm prediction

This is a modified copy of the paradigmextract files from https://github.com/marfors/paradigmextract.

[Forsberg, M; Hulden, M. (2016). Learning Transducer Models for
Morphological Analysis from Example Inflections. In Proceedings of
StatFSM. Association for Computational Linguistics.] (http://anthology.aclweb.org/W16-2405)

## Quick reference

### Paradigm learning: `pextract.py`

#### Description

Extract paradigmatic representations from input inflection tables. See
Section 2 in Forsberg and Hulden (2016) for details.

#### Example

Run the following line in Python 2:
`$ python src/pextract.py < ddninsharedtaskformat/english.tsv > text.p`
