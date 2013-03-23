"""
studies/crowdflower/data.py
natural-mixer

2013 Brandon Mechtley
Arizona State University

Create CSV files for CrowdFlower studies, including gold tests. Use gold.py to modify Gold Reports.

Usage: python data.py mapping.txt
"""

from itertools import permutations
import argparse

import numpy as np

from mapping import makedict

def unique_rows(a):
    return np.array([np.array(x) for x in set(tuple(x) for x in a)])

def maketrials(sounds, indices):
    """
    Create CrowdFlower units, including two types of gold units (trick questions):
        a) tone units: test sound is a 15s clip of 15 1s sine tones. (expected similarity 1/5, low realism)
        b) copy units: test sound is the same as one of the source sounds (expected similarity 5/5, high realism)
    
    Also ensure that there is an equal number of real and gold units (some gold will go unused, but this is fine.)
    There isn't any methodological reason for this, except that we don't actually need much gold and this makes it
    easier to manage units. CrowdFlower will ensure that a single user never sees the same gold more than once, and not
    all gold will be used anyway.
    
    No need to randomize this list, as users will be provided with a random grouping of units per page anyway.
    
    Args:
        sounds (dict): hash output from makedict.
        indices (list): list of indices of synthesized versions to use. If None, use all versions.

    Returns:
        Dictionary of form {'real': realunits, 'gold': goldunits}
    """
    
    realunits = []  # Real.
    toneunits = []  # Gold (sine tones).
    copyunits = []  # Gold (use a source as the test).
    
    # Use sine tones in order.
    tones = sounds['tone'].values()
    tonenum = 0
    
    for stype in [s for s in sounds if s not in ['source', 'tone', 'numstr']]:
        for pos in sounds[stype]:
            for i in [i for i in sounds[stype][pos] if indices is None or int(i) in indices]:
                filehash = sounds[stype][pos][i]
                
                # 1. Hashed filenames for source sounds closest to the test sound's position in the triangle.
                sources = [
                    sounds['source'][pos][loc] 
                    for loc in sounds['source'][pos]
                ]
                
                # 2. Add actual unit to be tested.
                realunits.append([f for f in [filehash] + sources])
                
                # 3. Add gold units where a) the test is the same as a source and b) the test is 15s of sine tones.
                for slist in permutations(sources, 3):
                    toneunits.append([tones[tonenum % len(tones)]] + list(slist))
                    
                    for fakehash in slist:
                        copyunits.append([fakehash] + list(slist))
                    
                    tonenum += 1
    
    copyunits = list(unique_rows(copyunits))
    toneunits = list(unique_rows(toneunits))
    copyunits = copyunits[:min(len(copyunits), len(toneunits))]
    toneunits = toneunits[:min(len(copyunits), len(toneunits))]
    goldunits = [unit for pair in zip(copyunits, toneunits) for unit in pair]
    goldunits = goldunits[:min(len(goldunits), len(realunits))]
    
    return {'real': realunits, 'gold': goldunits}

def printdata(units, mapdict, numstrs, filename):
    """
    Print CSV output for all real and golden units for initial data upload to CrowdFlower.
    
    Args:
        realunits (list): list of real unit sets from maketrials.
        goldunits (list): list of gold unit sets from maketrials.
    """
    
    realunits, goldunits = np.array(units['real']), np.array(units['gold'])
    nreal, ngold, nunits = len(realunits), len(goldunits), len(realunits) + len(goldunits)
    
    # np.string_ indexing is weird, so I'll just use np.object_.
    csv = np.array([[""] * 15] * (nunits + 1), dtype=np.object_)
    
    # 1. Header.
    csv[0,0:15] = [
        "s0", "s1", "s2", "s3", 'ns_nums', 'ns', "_Golden", 
        'pc', 's1sim', 's2sim', 's3sim', 
        'pc_reason', 's1sim_reason', 's2sim_reason', 's3sim_reason'
    ]
    
    cols = {csv[0,i] : i for i in range(len(csv[0]))}
    
    # 2. Fill in the units.
    csv[1:nreal+1,0:4] = [['"%s"' % r for r in unit] for unit in realunits]
    csv[nreal+1:,0:4] = [['"%s"' % r for r in unit] for unit in goldunits]
    
    csv[1:,4:6] = np.array([numstrs.items()[i % len(numstrs.items())] for i in range(len(csv) - 1)])
    
    # 3. All gold units need _Golden=True.
    csv[nreal+1:,6] = '"True"'
    
    ntones, ncopies = 0, 0
    
    # 4. Set answers / explanations for gold units.
    for row in csv[1:]:
        if mapdict[row[cols['s0']].strip('"')].split('-')[0] == 'tone':
            for scol in ['s1', 's2', 's3']:
                row[cols[scol + 'sim']] = '"1"'
                row[cols[scol + 'sim_reason']] = '"The test sound is fake, and this sound is a real recording."'
            
            row[cols['pc']] = '"1"'
            row[cols['pc_reason']] = '"This is a series of synthesized tones and is not a recording."'
            
            ntones += 1
        else:
            for scol in ['s%d' % i for i in range(1, 4) if row[cols['s%d' % i]] == row[cols['s0']]]:
                row[cols[scol + 'sim']] = '"5"'
                row[cols[scol + 'sim_reason']] = '"This is the same as the test clip."'
                row[cols['pc']] = '"3\n4\n5"'
                row[cols['pc_reason']] = '"This is a real recording."'
                
                ncopies += 1
    
    # 5. Write CSV output to file.
    f = open(filename, 'w')
    f.write('\n'.join([','.join(row) for row in csv]))
    f.close()
    
    print '\tOutput CrowdFlower units/gold units to %s.' % filename
    print '\t%d units: %d real, %d gold (%d tones, %d copies).' % (nunits, nreal, ngold, ntones, ncopies)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a CSV file for a CrowdFlower batch.')
    parser.add_argument('infile', metavar='file', default='mapping.txt', type=str, 
        help='CSV input file. See makemapping.sh for more information.')
    parser.add_argument('-i', '--indices', nargs=2, default=[0, 0], type=int,
        help='Starting and ending version index, exclusive.')
    parser.add_argument('-o', '--output', type=str, default='out.csv', help='output CSV file')
    args = parser.parse_args()
    
    mapping, sounds = makedict(args.infile)
    units = maketrials(sounds, None if args.indices[1] == 0 else range(args.indices[0], args.indices[1]))
    printdata(units, mapping, sounds['numstr'], args.output)
