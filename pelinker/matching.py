import re


def match_pattern(pattern, text, suffix_length=1, buffer_length=10):
    """

        given the pattern, split it into words, match ordered groups of words truncating each of at size n-1
            if its size > 5. Allow for pieces of text, spaces and dashes in between of size at most 10.
    :param pattern:
    :param text:
    :param suffix_length: remove word ending `induces` -> `induce`
    :param buffer_length: allow for words (and dashes) in between
    :return:
    """
    pattern_words = pattern.split()
    pattern_words_limit = [
        rf"\b{re.escape(word[:-suffix_length] if len(word) > 5 else word)}\w*"
        for word in pattern_words
    ]
    buffer = rf"\s+[\w\s-]{{0,{buffer_length - 1}}}"
    regex_pattern = buffer.join(pattern_words_limit) + r"\b"
    matches = [(m.start(), m.end()) for m in re.finditer(regex_pattern, text)]
    return matches


def match_pieces(pattern, text, suffix_length=1, matches=None):
    """
        given the pattern and the match of constituent words in the sense of match_pattern,
        for each match find the boundaries of matching words
    :param pattern:
    :param text:
    :param suffix_length:
    :param matches:
    :return:
    """
    if matches is None:
        matches = []
    pattern_words = pattern.split()

    r = {k: [] for k in matches}
    if matches:
        pats = [
            rf"\b{re.escape(word[:-suffix_length] if len(word) > 5 else word)}\w*\b"
            for word in pattern_words
        ]
        for pat in pats:
            word_matches = [(m.start(), m.end()) for m in re.finditer(pat, text)]

            for x, y in matches:
                r[(x, y)] += [
                    (a, b) for a, b in word_matches if min([y, b]) - max([x, a]) > 0
                ]
    return r
