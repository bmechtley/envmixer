'''
studies/crowdflower/baryplot.py
natural-mixer

2013 Brandon Mechtley
Arizona State University

Plot results from CrowdFlower study reports.

Usage: python baryplot.py
'''

import argparse

import numpy as np

import matplotlib.pyplot as pp
import matplotlib.tri as tri

from barycentric import *

def circumcircle(p1, p2, p3): 
    ''' 
    Return center of the circle containing P1, P2 and P3 

    If P1, P2 and P3 are colinear, return None 

    Adapted from: 
    http://local.wasp.uwa.edu.au/~pbourke/geometry/circlefrom3/Circle.cpp
    ''' 
    delta_a = p2 - p1 
    delta_b = p3 - p2 

    if np.abs(delta_a[0]) <= 0.000000001 and np.abs(delta_b[1]) <= 0.000000001: 
        center_x = 0.5*(p2[0] + p3[0]) 
        center_y = 0.5*(p1[1] + p2[1]) 
    else: 
        slope_a = delta_a[1]/delta_a[0] 
        slope_b = delta_b[1]/delta_b[0] 

        if np.abs(slope_a - slope_b) <= 0.000000001: 
            return None

        center_x = (slope_a * slope_b * (p1[1] - p3[1]) + \
        	slope_b * (p1[0] + p2[0]) - \
            slope_a * (p2[0] + p3[0])) / (2.* (slope_b - slope_a)) 

        center_y = -(center_x - (p1[0] + p2[0]) / 2.) / aSlope + \
        	(p1[1] + p2[1]) / 2. 

    return center_x, center_y 

def voronoi(x, y): 
    ''' Return line segments describing the voronoi diagram of X and Y ''' 
    
    p = np.zeros((x.size + 4, 2))
    p[:x.size, 0], p[:y.size, 1] = x, y
    
    m = max(np.abs(x).max(), np.abs(y).max()) * 1e5
    
    p[x.size:, 0] = -m, -m, +m, +m
    p[p.size:, 1] = -m, +m, -m, +m
    
    d = tri.Triangulation(p[:,0], p[:,1])
    t = d.triangles
    n = t.shape[0]
    c = np.zeros((n,2)) 

    for i in range(n): 
        c[i] = circumcircle(p[t[i,0]], p[t[i,1]], p[t[i,2]]) 
    
    x, y = c[:,0], c[:,1] 

    segments = [] 
    for i in range(n): 
        for k in d.neighbors[i]: 
            if k != -1: 
                segments.append([(x[i], y[i]), (x[k], y[k])])

    return segments 


#if __name__ == '__main__': 
#    P = np.random.random((2,256)) 
#    X,Y = P[0],P[1] 
#    fig = plt.figure(figsize=(10,10)) 
#    axes = plt.subplot(1,1,1) 
#    plt.scatter(X,Y, s=5) 
#    segments = voronoi(X,Y) 
#    lines = matplotlib.collections.LineCollection(segments, color='0.75') 
#    axes.add_collection(lines) 
#    plt.axis([0,1,0,1]) 
#    plt.show() 

if __name__ == '__main__':
	points = bary2cart(lattice(3))
	lines = voronoi(points[:,0], points[:,1])
	vpoints = lines[:,0]
	print vpoints

	tris = tri.Triangulation(points[:,0], points[:,1])
	values = np.random.randn(len(points))

	pp.figure()
	pp.gca().set_aspect('equal')
	pp.tripcolor(points[:,0], points[:,1], values, shading='gouraud', edgecolors=False)
	pp.title('Triangles!')
	pp.show()