from itertools import izip
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

def verttext(pt, txt, center=[.5,.5], dist=1./15, color='red'):
    '''Display a text label for a vertex with respect to the center. The text will be a certain 
    distance from the specified vertex in the direction of the vector extending from the center 
    point.
        pt: np.ndarray
            two-dimensional array of cartesian coordinates of the point to label.
        txt: str
            text to display. Text will be horizontally and vertically centered around pt.
        center: np.ndarray, optional
            reference center point used for arranging the text around pt (default [.5, .5]).
        dist: float, optional
            distance between point and the text (default 1./15).
        color: str, optional
            matplotlib color of the text (default 'red').'''
    
    vert = pt - center
    vert /= sum(abs(vert))
    vert *= dist
    
    text(
        pt[0] + vert[0],
        pt[1] + vert[1],
        txt,
        horizontalalignment='center',
        verticalalignment='center',
        color=color
    )

def polyshow(coords, color=None):
    '''Plot a regular convex polygon surrounding one or more barycentric coordinates within the 
    it. Vertices and corners will be labeled sequentially starting at 0.
        coords: np.ndarray or list
            one or more barycentric coordinates of equal arbitrary dimension. The dimensionality of 
            the coordinates will correspond to the number of vertices of the polygon that is drawn.
        color: str or list, optional
             color in which to draw the coords. If color is a list of the same length as coords, 
             each entry will correspond to the respective coordinates.'''
    
    coords = np.array(coords)
    if len(coords.shape) < 2: coords = [coords]
    dim = len(coords[0])
    
    if color == None: color = ['blue']
    elif type(color) == str: color = [color] * len(coords)
    
    f = pp.figure(figsize=(4,4), frameon=False)
    ax = pp.axes(frameon=False)
    corners = polycorners(dim)
    
    ax.add_patch(pp.Polygon(corners, closed=True, fill=False))
    ax.scatter(corners[:,0], corners[:,1], color='red', s=50, alpha=0.5)
    map(lambda i: verttext(corners[i], i), range(len(corners)))
    
    for i, coord in enumerate(coords):
        s = sum(coord)
        if s > 0: coord /= s
        cart = sum([c * cnr for c, cnr in izip(coord, corners)], axis=0)
        ax.scatter(cart[0], cart[1], color=color[i], s=100, alpha=0.5)
        verttext(cart, i, div=8, color=color[i])
    
    pp.xticks([])
    pp.yticks([])
    
    return f
