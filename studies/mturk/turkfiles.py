'''
mturk/turkfiles.py
envmixer

2013 Brandon Mechtley
Arizona State University

Create CSV files for Mechanical Turk batches, given the number of groups and
the frequency of fake CAPTCHA trials.

Usage: python makehit.py ngroups
    ngroups (int): number of test/source trials per HIT.
'''

import sys
import argparse
import random
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Create a hash of all test/source sounds.
def makedict(fn):
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
    # Create array of random permutations for each test sound.
    realtests, faketests = [], []

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
                    realtests.append([f for f in [filehash] + sourcehashes])
                
                    fakehash = sourcehashes[random.randint(0, 2)]
                    faketests.append([f for f in [fakehash] + sourcehashes])

    random.shuffle(realtests)
    random.shuffle(faketests)
    
    return realtests, faketests

def grouptrials(realtests, faketests, groups):
    # Group them into ngroups per row.
    hits, hit = [], []

    while len(realtests):
        if len(hit) >= groups - 1:
            # Add a CAPTCHA somewhere random for each group.
            hit.append(faketests.pop())
            random.shuffle(hit)
        
            hits.append([s for test in hit for s in test])
            hit = []
    
        hit.append(realtests.pop())

    return hits

def printhits(hits, groups):
    # Print CSV header.
    print ','.join([
        ','.join(['g%ds%d' % (i, j) for j in range(4)])
        for i in range(groups)
    ])

    # Print rows.
    print '\n'.join([','.join(hit) for hit in hits])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a CSV file for a\
        Mechanical Turk batch.')
    parser.add_argument('mapping', metavar='file', default='mapping.txt',
        type=str, help='mapping CSV file. See makemapping.sh for more\
        information.')
    parser.add_argument('-g', '--groups', metavar='int', default=5, type=int, 
        help='Number of test/source trials per HIT, including CAPTCHAs.')
    args = parser.parse_args()
    
    sounds = makedict(args.mapping)
    realtrials, faketrials = maketrials(sounds)
    hits = grouptrials(realtrials, faketrials, args.groups)
    printhits(hits, args.groups)
