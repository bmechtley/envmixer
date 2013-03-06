"""
mturk/turkfiles.py
envmixer

2013 Brandon Mechtley
Arizona State University

Create CSV files for Mechanical Turk batches, given the number of groups and the frequency of fake CAPTCHA trials.

Usage: python makehit.py ngroups
    ngroups (int): number of test/source trials per HIT.
"""

import argparse
import random
import pprint
pp = pprint.PrettyPrinter(indent=4)

def makedict(fn):
    """
    Create a hash of all test/source sounds.
    
    :type fn: str
    :param fn: filename of the mapping file that maps source filenames with coordinates to their hashes. 
    :rtype: dict
    :return: dictionary that maps filenames with coordinates to their hashes. 
    """
    
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
    """
    Create array of random permutations for each test sound.
    
    :type sounds: dict
    :param sounds: dictionary formatted as in makedict. {stype: {pos: {iteration: []}}}
    :rtype: (list, list)
    :return: list of "real tests" and list of "fake tests" (i.e. trick questions). Each test is a list of four sounds,
        where the first is the test sound (usually synthesized) and the other three are the source sounds nearest to
        where it was synthesized. Fake tests are formed by randomly using one of the source sounds as the test sound.
    """
    
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
    """
    Group trials from maketrials into groups, ngroups per HIT.
    
    :type realtests: list
    :param realtests: list of real tests.
    :type faketests: list
    :param faketests: list of fake tests (trick questions where the test sound is the same as one of the sources.)
    :type groups: int
    :param groups: number of tests per HIT.
    :rtype: list
    :return: a list of lists of tests, ngroup per row.
    """
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
    """
    Print out a CSV of the HITs for uploading to Amazon Mechanical Turk.
    
    :type hits: list
    :param hits: list of lists of tests, as returned by grouptrials.
    :type groups: int
    :param groups: number of tests per HIT. 
    """
    
    # TODO: Grab groups from the size of hits, rather than having the user specify.
    
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
