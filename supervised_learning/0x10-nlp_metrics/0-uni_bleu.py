#!/usr/bin/env python3
"""Calculate unigram bleu score"""


import numpy as np


def uni_bleu(references, sentence):
    """Calculate unigram bleu score"""
    sent_dict = {}
    for word in sentence:
        sent_dict[word] = sent_dict.get(word, 0) + 1
    max_dict = {}
    for reference in references:
        this_ref = {}
        for word in reference:
            this_ref[word] = this_ref.get(word, 0) + 1
        for word in this_ref:
            max_dict[word] = max(max_dict.get(word, 0), this_ref[word])
    in_ref = 0
    for word in sent_dict:
        in_ref += min(max_dict.get(word, 0), sent_dict[word])
    closest = np.argmin(np.abs([len(ref) - len(sentence)
                                for ref in references]))
    closest = len(references[closest])
    if len(sentence) >= closest:
        brevity = 1
    else:
        brevity = np.exp(1 - closest / len(sentence))
    return brevity * in_ref / len(sentence)
