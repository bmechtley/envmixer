"""
studies/crowdflower/baryplot.py
envmixer

2013 Brandon Mechtley
Arizona State University

Plot results from CrowdFlower study reports.

Usage: python baryplot.py
"""

import argparse
import json

import numpy as np
import matplotlib as mpl
import scipy.stats as stats
mpl.use('pdf')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble='\usepackage[usenames,dvipsnames]{xcolor}')
mpl.rc('font', family='Times serif', style='normal', weight='medium')

import matplotlib.pyplot as pp

from pybatchdict import *
from barycentric import *

def makedict(fn):
    """
    Create a dictionary that maps sound type->position->iteration to its filehash, given an input mapping CSV file.
    See envmixer/studies/makemapping.sh for more information on how this CSV file is formatted.
    
    :type fn: str
    :param fn: filename of the mapping CSV file.
    :rtype: dict
    :return: nested dictionary of format {'soundtype': {'pos': {'iteration': filehash}}} 
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

def circumcircle(a, b, c):
    """
    Return the center coordinates of a circle that circumscribes an input triangle defined by 2D vertices a, b, and c.
     
    :type a: (number, number) or np.ndarray
    :param a: first vertex of the input triangle.
    :type b: (number, number) or np.ndarray
    :param b: second vertex of the input triangle.
    :type c: (number, number) or np.ndarray
    :param c: third vertex of the input triangle.
    :rtype: (number, number)
    :return: center coordinates of circumscribing circle.
    """
    
    ax, ay = a
    bx, by = b
    cx, cy = c
    ax2, ay2, bx2, by2, cx2, cy2 = [d ** 2 for d in [ax, ay, bx, by, cx, cy]]
    
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    
    ux = (
        (ax2 + ay2) * (by - cy) +
        (bx2 + by2) * (cy - ay) +
        (cx2 + cy2) * (ay - by)
    ) / d
    
    uy = (
        (ax2 + ay2) * (cx - bx) +
        (bx2 + by2) * (ax - cx) +
        (cx2 + cy2) * (bx - ax)
    ) / d
    
    return ux, uy

def voronoi(x, y):
    """
    :type x: list or np.ndarray
    :param x: list of coordinates' x components.
    :type y: list or np.ndarray
    :param y: list of coordinates' y components.
    :rtype: (list, list)
    :return: (cells, triangles), where cells is a list of voronoi cells, each once containing a list of two-dimensional
        points; and triangles is a list of the triangles from a Delaunay triangulation.
    """
    
    p = np.array(zip(x, y))
    d = mpl.tri.Triangulation(x, y)
    t = d.triangles
    n = t.shape[0]
    
    # Get circle for each triangle, center will be a voronoi cell point. 
    cells = [[] for i in range(x.size)]                     # [[]] * x.size will have the same object at each index.
    
    for i in range(n):
        v = [p[t[i,j]] for j in range(3)]
        pt = circumcircle(v[0], v[1], v[2])
        
        cells[t[i,0]].append(pt)
        cells[t[i,1]].append(pt)
        cells[t[i,2]].append(pt)
    
    # Reordering cell p in trigonometric way
    for i, cell in enumerate(cells):
        xy = np.array(cell)
        order = np.argsort(np.arctan2(xy[:,1] - y[i], xy[:,0] - x[i]))
        
        cell = xy[order].tolist()
        cell.append(cell[0])
        
        cells[i] = cell
    
    return cells

def unique_rows(a):
    """
    Return an array containing unique rows from a.
    :type a: np.ndarray
    :param a: input array.
    :return: smaller array containing unique rows from a.
    """
    
    return np.array([np.array(x) for x in set(tuple(x) for x in a)])

def baryplot(values, points=None, labels='abc', cmap=mpl.cm.BrBG, clabel='', vmin=None, vmax=None):
    """
    Create a triangular voronoi cell pseudocolor plot. Create a voronoi diagram for each coordinate (points) within the
    triangle and color each cell according to its value (values).
    :type values: list or np.ndarray
    :param values: list of scalar values for each barycentric point.
    :type points: list or np.ndarray
    :param points: list of three-parameter barycentric points.
    :type labels: list
    :param labels: list of three label strings, one for each side of the triangle.
    :type cmap: matplotlib.colors.Colormap
    :param cmap: colormap for pseudocolor plot.
    :type clabel: str
    :param clabel: colorbar label for values
    :type vmin: number
    :param vmin: minimum value for coloring. If None, use minimum of the input values.
    :type vmax: number
    :param vmax: maximum value for coloring. If None, use maximum of the input values.
    """
    
    if points is None: points = []
    
    vmin = vmin if vmin is not None else np.amin(values)
    vmax = vmax if vmax is not None else np.amax(values)
    
    p = bary2cart(points) if len(points) else bary2cart(lattice(3))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(values)
    
    #values = (values - np.amin(values)) / (np.amax(values) - np.amin(values))
    cells = voronoi(p[:,0], p[:,1])
    
    xmin, xmax, xavg = np.amin(p[:,0]), np.amax(p[:,0]), np.mean(p[:,0])
    ymin, ymax, yavg = np.amin(p[:,1]), np.amax(p[:,1]), np.mean(p[:,1])
    
    s60, c60 = np.sin(np.pi / 3.), np.cos(np.pi / 3.)
    s30, c30 = np.sin(np.pi / 6.), np.cos(np.pi / 6.)
    
    # Start drawing.
    ax = pp.gca()
    
    # Clipping triangle for the voronoi patches.
    clip = mpl.patches.Polygon([
        (xmin, ymin), (xmax, ymin), (xavg, ymax),
    ], transform=ax.transData)
    
    # Draw voronoi patches.
    for i, cell in enumerate(cells):
        codes = [mpl.path.Path.MOVETO] \
            + [mpl.path.Path.LINETO] * (len(cell) - 2) \
            + [mpl.path.Path.CLOSEPOLY]
        
        pth = mpl.path.Path(cell, codes)
        
        patch = mpl.patches.PathPatch(
            pth, 
            zorder=-1, 
            facecolor=colors[i],
            clip_path=clip,
            edgecolor='none'
        )
        
        ax.add_patch(patch)
    
    # Add barycentric labels for vertices.
    ax.text(xmin - .0125, ymin - .02, '$(0,1,0)$', ha='right', va='center')
    ax.text(xmax + .0125, ymin - .02, '$(0,0,1)$', ha='left', va='center')
    ax.text(xavg, ymax + .035, '$(1,0,0)$', ha='center', va='bottom')
    
    # Labels.
    ax.text(
        xavg + c30 * .35, yavg + s30 * .35, 
        labels[2], ha='center', va='center', rotation=-60
    )
    
    ax.text(
        xavg, ymin - .05, 
        labels[1], ha='center', va='top'
    )
    
    ax.text(
        xavg - c30 * .35, yavg + s30 * .35, 
        labels[0], ha='center', va='center', rotation=60
    )
    
    arrowopts = dict(
        width=.00125,
        frac=.0125,
        headwidth=.01,
        transform=ax.transData
    )
    
    fig = pp.gcf()
    
    # Arrows along edges.
    ax.add_patch(mpl.patches.YAArrow(
        fig,
        (xmin - c60 * .025, ymin + s60 * .025),
        (xavg - c60 * .025, ymax + s60 * .025),
        **arrowopts
    ))
    
    ax.add_patch(mpl.patches.YAArrow(
        fig,
        (xmax, ymin - .025),
        (xmin, ymin - .025),
        **arrowopts
    ))
    
    ax.add_patch(mpl.patches.YAArrow(
        fig,
        (xavg + c60 * .025, ymax + s60 * .025),
        (xmax + c60 * .025, ymin + s60 * .025),
        **arrowopts
    ))
    
    # Make axes equal, get rid of border.
    pp.axis('equal')
    ax.axis([
        xmin - c60 * .2, xmax + c60 * .2, 
        ymin - s60 * .2, ymax + s60 * .2
    ])
    pp.axis('off')
    
    cax, kw = mpl.colorbar.make_axes(ax, orientation='vertical', shrink=0.7)
    cb = mpl.colorbar.ColorbarBase(
        cax, 
        cmap=cmap,
        norm=norm,
        orientation='vertical',
        ticks=np.linspace(vmin, vmax, 5)
    )
    
    cb.set_label(clabel)

def aggregate_result(json, coordmapping, resultpath):
    """
    Convert a JSON CrowdFlower results dictionary to a dictionary that pairs barycentric coordinates with a list of
    values for a given key in the results dictionary.
    
    :type json: list
    :param json: CrowdFlower-formatted results JSON list.
    :type coordmapping: dict
    :param coordmapping: dictionary that maps sound file hashes to coordinates.
    :type resultpath: str
    :param resultpath: keypath for the desired property to aggregate. For more information on keypaths, see the
        pybatchdict package.
    :rtype: dict
    :return: dict of format {'u,v,w': [values]}, where 'u,v,w' is a
        string of serialized barycentric coordinates and [values] is a list of values for the resultpath nested dictionary
        key aggregated for the given position.
    """
    
    aggdict = {}
    
    for result in json:
        jsondict = coordmapping[result['data']['s0']]
        
        if jsondict['type'] != 'source':
            posstr = '%.2f,%.2f,%.2f' % jsondict['pos']
            aggdict.setdefault(posstr, [])
            
            for judgment in result['results']['judgments']:
                if not judgment['rejected']:
                    value = float(getkeypath(judgment, resultpath))
                    aggdict[posstr].append(value)
    
    return aggdict

if __name__ == '__main__':
    # 
    # 1. Command-line arguments.
    # 
    
    parser = argparse.ArgumentParser(description='Make plots for a CrowdFlower results batch.')
    
    parser.add_argument(
        'mapping', metavar='mapping', default='mapping.txt', type=str, 
        help='CSV input file. See makemapping.sh for more information.'
    )
    
    parser.add_argument(
        'json', metavar='json', default='results.json', 
        type=str, help='JSON results file from CrowdFlower.'
    )
    
    args = parser.parse_args()
    
    #
    # 2. Load/manipulate JSON.    
    #
    
    jsonfile = open(args.json)
    jsondata = json.loads('[' + ','.join([line for line in jsonfile]) + ']')
    jsonfile.close()
    
    # 2a. Map coords to file hashes.
    mapping = makedict(args.mapping)
    
    # 2b. Convert to {filehash: {'pos': (a, b, c), 'type': t}} where type is 'source' or 'naive'
    coordmapping = {}
    for stype in mapping:
        for sloc in mapping[stype]:
            a, b, c = [float(d) for d in sloc.split('-')]
            
            for source in mapping[stype][sloc]:
                coordmapping[mapping[stype][sloc][source]] = {
                    'pos': (a, b, c),
                    'type': stype
                }
    
    # 2c. Aggregate data.
    agg = aggregate_result(
        jsondata, coordmapping, 
        '/data/test_clip_perceptual_convincingness'
    )
    
    #
    # 3. Plots.
    #
    
    points = lattice(3)
    labels = [r'$S_{DB}$', r'$S_{SG}$', r'$S_{DC}$']
    pp.figure(figsize=(5, 9))
     
    valuefunc = lambda func: [func(agg[','.join(['%.2f' % a for a in p])]) for p in points]
    data = [valuefunc(f) for f in [np.mean, lambda v: stats.scoreatpercentile(v, 25), np.median, lambda v: stats.scoreatpercentile(v, 75)]]
    names = ['$\overline{c}$', '$Q1(c)$', '$Q2(c)$', '$Q3(c)$']
    vmin = np.amin(data)
    vmax = np.amax(data)
    
    for i, plotset in enumerate(zip(data, names)):
        pp.subplot(4, 1, i)
        baryplot(plotset[0], points=points, labels=labels, clabel=plotset[1], vmin=vmin, vmax=vmax)
    
    # 3c. Save figure.
    pp.savefig('convincingness.pdf')
    
    pp.figure(figsize=(6, 14))
    pp.subplots_adjust(hspace=.8,top=.95,bottom=.05)
    
    for i, p in enumerate(points):
        key = ','.join(['%.2f' % a for a in p])
        pp.subplot(len(points), 1, i + 1)
        print agg[key]
        pp.hist(agg[key], bins=5, range=(1, 5))
        pp.xlabel('$(%s)$' % key)
        pp.xlim((1, 5))
        pp.yticks([0, 40, 80, 120])
        
        if i == 0:
            pp.title('Distribution of convincingness ratings, $n=%d$' % len(agg['1.00,0.00,0.00']))
    
    pp.savefig('distributions.pdf')
