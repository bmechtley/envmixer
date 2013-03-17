"""
studies/crowdflower/data.py
natural-mixer

2013 Brandon Mechtley
Arizona State University

Create CSV files for CrowdFlower studies, including gold tests.
Use gold.py to modify Gold Reports.

Usage: python data.py mapping.txt
"""

import argparse
import random
import pprint
import numpy as np
from numpy.random import randint

pp = pprint.PrettyPrinter(indent=4)

def unique_rows(a):
    """
    Return only unique rows from an array.
    
    :type a: numpy.ndarray
    :param a: input array
    :rtype: numpy.ndarray
    :return: array with one row per unique row of a.
    """
    
    return np.array([np.array(x) for x in set(tuple(x) for x in a)])

def makedict(fn):
    """
    Create a nested dictionary of type:position:filename for every file in a given mapping file. See
    studies/makemapping.sh for more information.
    
    :type fn: str
    :param fn: mapping filename
    :rtype: dict
    :return: dictionary of format {type: {position: {iteration: filehash}}}}
    """
    
    mapping = open(fn, 'r')
    
    sounds = {}
    mapdict = {}
    
    for line in mapping:
        filename, filehash = line.rstrip('\n').split(', ')
        mapdict[filehash] = filename
        tokens = filename.split('.wav')[0].split('-')
        
        if tokens[0] not in ['tone', 'numstr']:
            stype = tokens[1]
            pos = '-'.join(tokens[2:5]) 
            iteration = tokens[0] if stype == 'source' else tokens[5]
            
            if stype not in sounds:
                sounds[stype] = {}
            
            if pos not in sounds[stype]:
                sounds[stype][pos] = {}
            
            sounds[stype][pos][iteration] = filehash
        else:
            if tokens[0] not in sounds:
                sounds[tokens[0]] = {}
            
            sounds[tokens[0]][tokens[1]] = filehash
    
    return mapdict, sounds

def maketrials(sounds, iterations):
    """
    Create an array of random permutations for each test sound and every possible gold (i.e. trick)
    question.
    
    :type sounds: dict
    :param sounds: hash output from makedict.
    :rtype: (list, list)
    :return: real and gold units, where each unit is a list of filehashes.
    """
    
    realunits, goldunits = [], []
    tones = sounds['tone'].values()
    
    for stype in sounds:
        if stype not in ['source', 'tone', 'numstr']:
            for pos in sounds[stype]:
                for iteration in sounds[stype][pos]:
                    if iterations[1] == 0 or (iterations[0] <= int(iteration) < iterations[1]):
                        filehash = sounds[stype][pos][iteration]
                        
                        sourcehashes = [
                            sounds['source'][pos][loc] 
                            for loc in sounds['source'][pos]
                        ]
                        
                        random.shuffle(sourcehashes)
                        fakehash = sourcehashes[randint(0, 2)]
                        
                        realunits.append([f for f in [filehash] + sourcehashes])
                        
                        for sources in [
                            sourcehashes, list(np.roll(sourcehashes, 1)), list(np.roll(sourcehashes, 2))
                        ]:
                            goldunits.append([tones[randint(0, len(tones) - 1)]] + sources)
                            goldunits.append([f for f in [fakehash] + sourcehashes])
        
    random.shuffle(realunits)
    random.shuffle(goldunits)
    
    goldunits = unique_rows(goldunits)
    
    return realunits, goldunits

def printdata(realunits, goldunits, mapdict, numstrs, filename):
    """
    Print CSV output for all real and golden units for initial data upload to CrowdFlower.
    
    :type realunits: list
    :param realunits: list of real unit sets from maketrials.
    :type goldunits: list
    :param goldunits: list of gold unit sets from maketrials.
    """
    
    realunits, goldunits = np.array(realunits), np.array(goldunits)
    nreal, ngold = len(realunits), len(goldunits)
    nunits = nreal + ngold
    
    # np.string_ indexing is weird, so we'll just use np.object_.
    csv = np.array([[""] * 15] * (nunits + 1), dtype=np.object_)
    
    # Header.
    csv[0,0:15] = [
        "s0", "s1", "s2", "s3", 'ns_nums', 'ns', "_Golden", 
        'pc', 's1sim', 's2sim', 's3sim', 
        'pc_reason', 's1sim_reason', 's2sim_reason', 's3sim_reason'
    ]
    
    cols = {csv[0,i] : i for i in range(len(csv[0]))}
    
    # Fill in the units.
    csv[1:nreal+1,0:4] = [['"%s"' % r for r in unit] for unit in realunits]
    csv[nreal+1:,0:4] = [['"%s"' % r for r in unit] for unit in goldunits]
    
    csv[1:,4:6] = np.array([numstrs.items()[randint(0, len(numstrs.values()) - 1)] for i in range(len(csv) - 1)])
    
    # All gold units need _Golden=True.
    csv[nreal+1:,6] = '"True"'
    
    f = open(filename, 'w')
    
    for row in csv[1:]:
        s0, s1, s2, s3 = [row[cols['s%d' % i]].strip('"') for i in range(4)]
        s0tokens = mapdict[s0].split('-')
        
        if s0 in [s1, s2, s3]:
            if s0 == s1:
                row[cols['s1sim']] = '"5"'
                row[cols['s1sim_reason']] = '"This is the same as the test clip."'
            elif s0 == s2:
                row[cols['s2sim']] = '"5"'
                row[cols['s2sim_reason']] = '"This is the same as the test clip."'
            elif s0 == s3:
                row[cols['s3sim']] = '"5"'
                row[cols['s3sim_reason']] = '"This is the same as the test clip."'
            
            row[cols['pc']] = '"3\n4\n5"'
            row[cols['pc_reason']] = '"This is a real recording."'
        elif s0tokens[0] == 'tone':
            row[cols['s1sim']] = '"1"'
            row[cols['s1sim_reason']] = '"The test sound is fake, and this sound is a real recording."'
            row[cols['s2sim']] = '"1"'
            row[cols['s2sim_reason']] = '"The test sound is fake, and this sound is a real recording."'
            row[cols['s3sim']] = '"1"'
            row[cols['s3sim_reason']] = '"The test sound is fake, and this sound is a real recording."'
            row[cols['pc']] = '"1"'
            row[cols['pc_reason']] = '"This is a series of synthesized tones and is not a recording."'
    
    # Print CSV output.
    f.write('\n'.join([','.join(row) for row in csv]))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a CSV file for a CrowdFlower batch.')
    parser.add_argument('infile', metavar='file', default='mapping.txt', type=str, 
        help='CSV input file. See makemapping.sh for more information.')
    parser.add_argument('-i', '--iterations', nargs=2, default=[0, 0], type=int,
        help='Starting and ending iteration, exclusive.')
    parser.add_argument('-o', '--output', type=str, default='out.csv', help='output CSV file')
    args = parser.parse_args()
    
    mapdict, sounds = makedict(args.infile)
    realunits, goldunits = maketrials(sounds, args.iterations)
    printdata(realunits, goldunits, mapdict, sounds['numstr'], args.output)
