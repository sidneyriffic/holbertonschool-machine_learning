#!/usr/bin/env python3
"""Print location of a user via github api"""


import requests
import sys
import time


if __name__ == '__main__':
    data = requests.get(sys.argv[1])
    response = data.status_code
    if response == 200:
        print(data.json()['location'])
    elif response == 403:
        reset = data.headers['X-Ratelimit-Reset']
        reset = int(reset) - int(time.time())
        print("Reset in {} min".format(int(reset / 60)))
    else:
        print("Not found")
