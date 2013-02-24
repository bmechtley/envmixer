'''
studies/crowdflower/data.py
natural-mixer

2013 Brandon Mechtley
Arizona State University

Create CSV files for CrowdFlower studies, including gold tests.
Use gold.py to modify Gold Reports.

Usage: python data.py mapping.txt
'''

import sys
import argparse
import random
import pprint
import csv
import numpy as np
pp = pprint.PrettyPrinter(indent=4)

def unique_rows(a):
    '''Return only unique rows from an array.
        a: np.array
            input array'''

    return np.array([np.array(x) for x in set(tuple(x) for x in a)])

def makedict(fn):
    '''Create a nested dictionary of type:position:filename for every file
    in a given mapping file. See studies/makemapping.sh for more information.
        fn: str
            mapping filename'''

    mapping = open(fn, 'r')
    
    sounds = {}

    for line in mapping:
        (filename, filehash) = line.rstrip('\n').split(', ')
        tokens = filename.split('.wav')[0].split('-')
        stype = tokens[1]
        pos = '-'.join(tokens[2:5])
        iteration = tokens[0] if stype == 'source' else tokens[5]
     
        if stype not in sounds:
            sounds[stype] = {}
    
        if pos not in sounds[stype]:
            sounds[stype][pos] = {}

        sounds[stype][pos][iteration] = filehash
    
    return sounds

def maketrials(sounds):
    '''Create an array of random permutations for each test sound and every
    possible gold (i.e. trick) question.
        sound: dict
            hash output from makedict.'''

    realunits, goldunits = [], []

    for stype in sounds:
        if stype != 'source':
            for pos in sounds[stype]:
                for iteration in sounds[stype][pos]:
                    filehash = sounds[stype][pos][iteration]
                
                    sourcehashes = [
                        sounds['source'][pos][loc] 
                        for loc in sounds['source'][pos]
                    ]
                    
                    random.shuffle(sourcehashes)
                    fakehash = sourcehashes[random.randint(0, 2)]
                    
                    realunits.append([f for f in [filehash] + sourcehashes])
                    goldunits.append([f for f in [fakehash] + sourcehashes])
                    goldunits.append([f for f in [fakehash] + list(np.roll(sourcehashes, 1))])
                    goldunits.append([f for f in [fakehash] + list(np.roll(sourcehashes, 2))])

    random.shuffle(realunits)
    random.shuffle(goldunits)
    
    goldunits = unique_rows(goldunits)

    return realunits, goldunits

def printdata(realunits, goldunits):
    '''Print CSV output for all real and golden units for initial data upload
    to CrowdFlower.
        realunits: list
            list of real unit sets from maketrials.
        goldunits: list
            list of gold unit sets from maketrials.'''

    realunits, goldunits = np.array(realunits), np.array(goldunits)
    nreal, ngold = len(realunits), len(goldunits)
    nunits = nreal + ngold

    # np.string_ indexing is weird, so we'll just use np.object_.
    csv = np.array([[""] * 5] * (nunits + 1), dtype=np.object_)
    
    # Header.
    csv[0,0:5] = [
        "s0",
        "s1",
        "s2",
        "s3",
        "_Golden"
    ]

    # Fill in the units.
    csv[1:nreal+1,0:4] = [['"%s"' % r for r in unit] for unit in realunits]
    csv[nreal+1:,0:4] = [['"%s"' % r for r in unit] for unit in goldunits]

    # All gold units need _Golden=True.
    csv[nreal+1:,4] = '"True"'
    
    # Print CSV output.
    print '\n'.join([','.join(row) for row in csv])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a CSV file for a CrowdFlower batch.')
    parser.add_argument('infile', metavar='file', default='mapping.txt', type=str, help='CSV input file. See makemapping.sh for more information.')
    args = parser.parse_args()
    
    sounds = makedict(args.infile)
    printdata(*maketrials(sounds))
