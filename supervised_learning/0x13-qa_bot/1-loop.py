#!/usr/bin/env python3
"""Create a basic Q/A input loop"""

in_q = ''
while (in_q is not None):
    """Create a basic Q/A input loop"""
    in_q = input("Q: ")
    if in_q.lower() in ['bye', 'exit', 'quit', 'goodbye']:
        print("A: Goodbye")
        break
    print("A: ")
