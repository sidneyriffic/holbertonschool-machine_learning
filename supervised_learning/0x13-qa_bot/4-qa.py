#!/usr/bin/env python3
"""Create a basic Q/A input loop"""


qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def qa_bot(corpus_path):
    """Create a basic Q/A input loop"""
    in_q = ''
    while (in_q is not None):
        """Create a basic Q/A input loop"""
        in_q = input("Q: ")
        if in_q.lower() in ['bye', 'exit', 'quit', 'goodbye']:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, in_q)
        answer = qa(in_q, reference)
        if answer == '':
            print("A: Sorry, I do not understand your question.")
            continue
        print("A: {}".format(answer))
