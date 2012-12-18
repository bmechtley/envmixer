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

def bary2cart(bary, corners):
    '''Convert barycentric coordinates to cartesian coordinates given the cartesian coordinates of 
    the corners. 
        bary: np.ndarray
            barycentric coordinates to convert. If this matrix has multiple rows, each row is 
            interpreted as an individual coordinate to convert.
        corners: np.ndarray
            cartesian coordinates of the corners.'''
    
    if len(bary.shape) > 1:
        return np.array([np.sum(b * corners.T, axis=1) for b in bary])
    else:
        return np.sum(bary * corners.T, axis=1)

def lattice(ncorners=3, sides=False):
    '''Create a lattice of linear combinations of barycentric coordinates with ncorners corners. 
    This lattice is constructed from the corners, the center point between them, points between the 
    corners and the center, and pairwise combinations of the corners and the center.
        ncorners: int, optional 
            number of corners of the boundary polygon (default 3).
        sides: bool, optional
            whether or not to include pairwise combinations of the corners (i.e. sides) in the
            lattice (default False).'''
    
    # 1. Corners.
    coords = list(np.identity(ncorners))
    
    # 2. Center.
    center = np.array([1. / ncorners] * ncorners)
    coords.append(center)
    
    # 3. Corner - Center.
    for i in range(ncorners):
        for j in range(i + 1, ncorners):
            coords.append((coords[i] + coords[j] + center) / 3)
    
    # 4. Corner - Corner - Center.
    for i in range(ncorners):
        coords.append((coords[i] + center) / 2)
    
    # 5. Corner - Corner (Sides)
    if sides:
        for i in range(ncorners):
            for j in range(i + 1, ncorners):
                coords.append((coords[i] + coords[j]) / 2)
    
    # 5. Return unique coordinates (some duplicates using this method with e.g. ncorners=2)
    return np.array(list(set(tuple(c) for c in coords)))

def polycorners(ncorners=3):
    '''Return 2D cartesian coordinates of a regular convex polygon of a specified number of corners.
        ncorners: int, optional
            number of corners for the polygon (default 3).'''
    
    center = np.array([0.5, 0.5])
    points = []
    
    for i in range(ncorners):
        angle = (float(i) / ncorners) * (np.pi * 2) + (np.pi / 2)
        x = center[0] + np.cos(angle) * 0.5
        y = center[1] + np.sin(angle) * 0.5
        points.append(np.array([x, y]))
    
    return np.array(points)

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
        ['python', 'sources.py'] + args.inputs +
        [
            '-c', c[0], c[1],
            '-l', args.length,
            '-o', args.output,
            '-s', 'source-' + '-'.join('%.2f' % p for p in b)
        ]
     for b, c in izip(bary, cart)])
    
    # 3. Save mixed versions at the specified coordinates.
    print '\nWriting mixed versions.'
    multiprocessing.Pool(processes = multiprocessing.cpu_count()).map(work, [
        ['python', 'mixer.py'] + args.inputs +
        [
            '-c', c[0], c[1],
            '-l', args.length,
            '-g', args.graindur[0], args.graindur[1],
            '-j', args.jumpdev,
            '-o', os.path.join(
                args.output, 
                'mix-naive-' + '-'.join(['%.2f' % p for p in b]) + '-%03d.wav' % r
            )
        ]
    for r, (b, c) in list(product(range(args.instances), zip(bary, cart)))])

if __name__ == '__main__':
    main()
