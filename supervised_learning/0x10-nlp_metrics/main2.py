#!/usr/bin/env python3

cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

#references = [["hi", "there", "me", "me"]]
#sentence = ["hi", "there", "you", "me"]

print(cumulative_bleu(references, sentence, 4))
