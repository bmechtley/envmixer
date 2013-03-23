"""
mturk/mapping.py
envmixer

2013 Brandon Mechtley
Arizona State University

This script moves files based on mapping.txt. See makemapping.sh for more information.

Usage: python mapping.py {do,undo}
    do: move original wavfiles to [md5].wav files.
    undo: move the [md5].wav files to their original names.
"""

import sys
import subprocess

def makedict(fn):
    """
    Create a nested dictionary of type/position/iteration: filename for every file in a given mapping file. See
    studies/makemapping.sh for more information.
    
    Args:
        fn (str): mapping filename

    Returns:
        (sounds, mapdict) where sounds is a dictionary of format {type: {position: {iteration: filehash}}}} and mapdict
        is a dictionary of format {filehash: original filename}
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

def undomapping(fn):
    """
    Move file hashes back to their original filenames.
    """

    f = open(fn, 'rb')

    for line in f:
        s = line.rstrip('\n').split(', ')
        print 'mv', s[1], s[0]
        subprocess.call(['mv', s[1], s[0]])
    
    exit()

def domapping(fn):
    """
    Move all mapped files to their hash.
    """
    
    f = open(fn, 'rb')

    for line in f:
        s = line.rstrip('\n').split(', ')
        print 'mv', s[0], s[1]
        subprocess.call(['mv', s[0], s[1]])
    
    exit()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'do':
            domapping(sys.argv[2])
        elif sys.argv[1] == 'undo':
            undomapping(sys.argv[2])
    
    print 'usage: python mapping.py {do,undo} mapping.txt'
