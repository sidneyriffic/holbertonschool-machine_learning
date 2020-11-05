#!/usr/bin/env python3
"""List all documents in a collection"""


import pymongo


def insert_school(mongo_collection, **kwargs):
    """List all documents in a collection"""
    return mongo_collection.insert(kwargs)
