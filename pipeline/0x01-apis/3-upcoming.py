#!/usr/bin/env python3
"""Display soonest upcoming launch"""


import requests


if __name__ == '__main__':
    data = requests.get('https://api.spacexdata.com/v4/launches/upcoming',
                        headers={'pagination': 'false'})
    data = data.json()
    time = 99999999999
    next = None
    for launch in data:
        thistime = int(launch['date_unix'])
        if thistime < time:
            time = thistime
            next = launch
    if next is not None:
        rocket = requests.get('https://api.spacexdata.com/v4/rockets/'
                              + next['rocket'])
        rocket = rocket.json()['name']
        lpad = requests.get('https://api.spacexdata.com/v4/launchpads/'
                            + next['launchpad'])
        lpad = lpad.json()
        locale = lpad['locality']
        lpad = lpad['name']
        print('{} ({}) {} - {} ({})'.format(next['name'], next['date_local'],
                                            rocket, lpad, locale))
