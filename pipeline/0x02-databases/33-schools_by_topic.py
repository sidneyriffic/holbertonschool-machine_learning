#!/usr/bin/env python3
"""Find a school with topics"""


import pymongo


def schools_by_topic(mongo_collection, topic):
    """Find a school with topics"""
    return mongo_collection.find({'topics': topic})
