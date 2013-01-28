'''
mturk/completion.py
natural-mixer

2013 Brandon Mechtley
Arizona State University

Create an image of Mechanical Turk study completion percentage for each input coordinate.

Usage: python completion.py mapping.txt results.csv
    mapping.txt: mapping file for mapping md5 -> original filename
    results.csv: the results from mechanical turk.
'''

import argparse
import numpy as np
import matplotlib.pyplot as pp

from turkfiles import *
from barycentric import *

def circumcircle(a, b, c):
    dotABAB = np.dot(b - a, b - a)
    dotABAC = np.dot(b - a, c - a)
    dotACAC = np.dot(c - a, c - a)
    d = 2. * (dotABAB * dotACAC - dotABAC * dotABAC)
    
    if np.abs(d) < np.finfo(type(a[0])).eps:
        c = np.array([(b[0] + c[0]) / 2., (a[1] + b[1]) / 2.])
    else:
        s = (dotABAB * dotACAC - dotACAC * dotABAC) / d
        t = (dotACAC * dotABAB - dotABAB * dotABAC) / d
        
        if s < 0:
            c = (a + c) / 2
        elif t < 0:
            c = (a + b) / 2
        elif (s + t) > 1:
            c = (b + c) / 2
        else:
            c = a + s * (b - a) + t * (c - a)
    
    return c

def voronoi(x, y):
    #p = np.array([x, y]).transpose()
    
    p = np.zeros((x.size + 4, 2))
    p[:x.size, 0], p[:y.size, 1] = x, y
    
    mx, my = np.mean(x), np.mean(y)
    
    minx, maxx = (np.amin(x) - np.mean(x)) * 10 + mx, (np.amax(x) - np.mean(x)) * 10 + mx
    miny, maxy = (np.mean(y) - np.amin(y)) * 10 + my, (np.amax(y) - np.mean(y)) * 10 + my
    p[x.size:, 0] = minx, minx, maxx, maxx
    p[y.size:, 1] = miny, maxy, miny, maxy
    
    d = matplotlib.tri.Triangulation(p[:,0], p[:,1])
    t = d.triangles         # Delaunay triangles.
    n = t.shape[0]          # # of triangles.
    c = np.zeros((n,2))     # Circle centers.
    
    for i in range(n):
        c[i] = circumcircle(p[t[i, 0]], p[t[i, 1]], p[t[i, 2]])
    
    x, y = c[:,0], c[:,1]
    
    segments = []
    
    for i in range(n):
        for k in d.neighbors[i]:
            if k != -1:
                segments.append([(x[i], y[i]), (x[k], y[k])])
    
    return segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an image of Mechanical Turk study\
        completion percentage for each input coordinate.')
    parser.add_argument('mapping', metavar='txt', type=str, default='mapping.txt',
        help='mapping CSV file. See makemapping.sh for more information.')
    parser.add_argument('results', metavar='csv', type=str, default='results.csv',
        help='results CSV from Mechanical Turk.')
    args = parser.parse_args()
     
    sounds = makedict(args.mapping)
    results = np.genfromtxt(args.results, delimiter='","', skip_header=1, dtype=str)
    results = np.genfromtxt(args.results, delimiter='","', usecols=range(results.shape[1]), dtype=str)
    sourcecol = np.where(results[0] == 'Input.g0s0')[0][0]
    realismcol = np.where(results[0] == 'Answer.g0s0')[0][0]
    
    points = lattice(3)
    
    counts = {}
    realism = {}
    
    for coord in sounds['naive']:
        if coord not in counts: counts[coord] = 0
        if coord not in realism: realism[coord] = 0
        
        for v in sounds['naive'][coord].values():
            relevant = results[:,sourcecol] == v
            count = np.sum(relevant)
            counts[coord] += count
            print relevant, count
            if count:
                print results[relevant, realismcol]
                realism[coord] += np.sum([int(r) for r in results[relevant, realismcol]])
    
    for c in realism:
        if counts[c]:
            realism[c] /= counts[c]
    
    labels = [counts['-'.join(['%.2f' % c for c in p])] for p in points]
    colors = ['%.2f' % (realism['-'.join(['%.2f' % c for c in p])] / 9.) for p in points]
    labels = ['%s, %s' % items for items in zip(labels, colors)]
    polyshow(lattice(3), label=labels, color=colors)
    
    pp.show()