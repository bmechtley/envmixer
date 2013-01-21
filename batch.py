'''
batch.py
natural-mixer

2012 Brandon Mechtley
Arizona State University

Batch resynthesis for generating soundfiles for use with an Amazon Mechanical Turk user study.
'''

import os, argparse, multiprocessing, subprocess
from itertools import izip, product
import numpy as np
import matplotlib.pyplot as pp
from barycentric import *

def work(command):
    '''Wrapper function to call a subprocess. For use with multiprocessing.Pool in order to spawn
    subprocesses in a pool.
        command: list
            list of elements in the command to call.'''
    
    print '\t', ' '.join([str(par) for par in command])
    return subprocess.call([str(p) for p in command], shell=False)

def main():
    parser = argparse.ArgumentParser(
        description='Batch resynthesis for generating soundfiles with mixer.py')
    parser.add_argument('inputs', metavar='wav', nargs='+', type=str,
        help='wav or sv files to mix.')
    parser.add_argument('-l', '--length', metavar='s', default=10, type=float,
        help='length of output in seconds')
    parser.add_argument('-g', '--graindur', nargs=2, metavar='ms', default=[100, 500], type=float,
        help='duration of grains in milliseconds')
    parser.add_argument('-j', '--jumpdev', metavar='s', default=60, type=float,
        help='standard deviation of random jumps in seconds')
    parser.add_argument('-i', '--instances', metavar='int', default=1, type=int,
        help='number of random instances to generate')
    parser.add_argument('-o', '--output', metavar='file', default='./', type=str,
        help='base path for output')
    args = parser.parse_args()
    
    # 1. Calculate coordinates in test lattice.
    bary = lattice(len(args.inputs))
    corners = polycorners(len(args.inputs))
    cart = bary2cart(bary, corners)
	
    # 2. Save unmodified clips centered around the coordinates for all three source sounds.
    print 'Writing source comparisons.'
    
    multiprocessing.Pool(processes = multiprocessing.cpu_count()).map(work, [
        ['python', 'sources.py'] + ['-c'] + list(b) + 
        [
            '-l', args.length,
            '-o', args.output,
            '-s', 'source-' + '-'.join('%.2f' % p for p in b)
        ] + args.inputs
     for b, c in izip(list(bary), list(cart))])
    
    # 3. Save mixed versions at the specified coordinates.
    print '\nWriting mixed versions.'
    
    multiprocessing.Pool(processes = multiprocessing.cpu_count()).map(work, [
        ['python', 'mixer.py'] + args.inputs + ['-c'] + list(b) + 
        [
            '-l', args.length,
            '-g', args.graindur[0], args.graindur[1],
            '-j', args.jumpdev,
            '-o', os.path.join(
                args.output, 
                'mix-naive-' + '-'.join(['%.2f' % p for p in b]) + '-%03d.wav' % r
            )
        ]
    for r, (b, c) in product(range(args.instances), zip(list(bary), list(cart)))])

if __name__ == '__main__':
    main()

