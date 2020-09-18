#!/usr/bin/env python3
"""Calculate n-gram bleu score"""


import numpy as np


def ngramify(corpus, n):
    """Convert a corpus of 1-grams to n-grams"""
    unlist = 0
    if type(corpus[0]) is not list:
        corpus = [corpus]
        unlist = 1
    new_corpus = []
    for line in corpus:
        new_line = []
        for gram in range(len(line) - n + 1):
            new_gram = ""
            for i in range(n):
                if i != 0:
                    new_gram += " "
                new_gram += line[gram + i]
            new_line.append(new_gram)
        new_corpus.append(new_line)
    if unlist:
        return new_corpus[0]
    return new_corpus


def ngram_bleu(references, sentence, n):
    """Calculate unigram bleu score"""
    references = ngramify(references, n)
    sentence = ngramify(sentence, n)
    sent_dict = {}
    for gram in sentence:
        sent_dict[gram] = sent_dict.get(gram, 0) + 1
    max_dict = {}
    for reference in references:
        this_ref = {}
        for gram in reference:
            this_ref[gram] = this_ref.get(gram, 0) + 1
        for gram in this_ref:
            max_dict[gram] = max(max_dict.get(gram, 0), this_ref[gram])
    in_ref = 0
    for gram in sent_dict:
        in_ref += min(max_dict.get(gram, 0), sent_dict[gram])
    closest = np.argmin(np.abs([len(ref) - len(sentence)
                        for ref in references]))
    closest = len(references[closest])
    if len(sentence) >= closest:
        brevity = 1
    else:
        brevity = np.exp(1 - (closest + n - 1) / (len(sentence) + n - 1))
    return brevity * in_ref / len(sentence)
