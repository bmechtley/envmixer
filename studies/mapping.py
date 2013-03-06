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

f = open('mapping.txt', 'rb')

def undomapping():
    """
    Move file hashes back to their original filenames.
    """
    
    for line in f:
        s = line.rstrip('\n').split(', ')
        print 'mv', s[1], s[0]
        subprocess.call(['mv', s[1], s[0]])
    
    exit()

def domapping():
    """
    Move all mapped files to their hash.
    """
    
    for line in f:
        s = line.rstrip('\n').split(', ')
        print 'mv', s[0], s[1]
        subprocess.call(['mv', s[0], s[1]])
    
    exit()

if len(sys.argv) > 1:
    if sys.argv[1] == 'do':
        domapping()
    elif sys.argv[1] == 'undo':
        undomapping()

print 'usage: python mapping.py {do,undo}'
