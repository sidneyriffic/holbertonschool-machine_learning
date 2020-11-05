#!/usr/bin/env python3
"""Update a school with topics"""


import pymongo


def update_topics(mongo_collection, name, topics):
    """Update a school with topics"""
    return mongo_collection.update({'name': name},
                                   {'$set': {'topics': topics}},
                                   multi=True)
