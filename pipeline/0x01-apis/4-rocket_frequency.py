#!/usr/bin/env python3
"""Display number of launches per rocket"""


import requests


if __name__ == '__main__':
    data = requests.get('https://api.spacexdata.com/v4/launches',
                        headers={'pagination': 'false'})
    rockets = {}
    for launch in data.json():
        rocket = launch['rocket']
        rockets[rocket] = rockets.get(rocket, 0) + 1
    rocket_sort = []
    launches = []
    for rocket in rockets:
        name = requests.get('https://api.spacexdata.com/v4/rockets/'
                            + rocket).json()['name']
        launches.append(rockets[rocket])
        rocket_sort.append(name)
    zipped = zip(rocket_sort, launches)
    zipped = list(zipped)
    zipped.sort(key=lambda x: x[0])
    zipped.sort(key=lambda x: x[1], reverse=True)
    for pair in zipped:
        print("{}: {}".format(pair[0], pair[1]))
