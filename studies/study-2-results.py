from os.path import splitext
from sys import argv
import itertools
import sqlite3
import pprint
import yaml
import csv

from matplotlib import rc
import matplotlib.cm
from pylab import *

from pybatchdict import *
from barycentric import *

def circumcircle(a, b, c):
    """
    Return the center coordinates of a circle that circumscribes an input triangle defined by 2D vertices a, b, and c.
    
    Args:
        a ((number, number) or np.ndarray): first vertex of the input triangle.
        b ((number, number) or np.ndarray): second vertex of the input triangle.
        c ((number, number) or np.ndarray): third vertex of the input triangle.
        
    Returns:
        Center coordinates of circumscribing circle in form (x, y).
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
    Return a list of voronoi cells for a collection of points.

    Args:
        x (list or np.ndarray): list of coordinates' x components.
        y (list or np.ndarray): list of coordinates' y components.
    
    Returns:
        (cells, triangles), where cells is a list of voronoi cells, each once containing a list of two-dimensional
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
    return np.array([np.array(x) for x in set(tuple(x) for x in a)])

def baryplot(values, points=None, labels='abc', cmap=mpl.cm.BrBG, clabel='', vmin=None, vmax=None):
    """
    Create a triangular voronoi cell pseudocolor plot. Create a voronoi diagram for each coordinate (points) within the
    triangle and color each cell according to its value (values).
    
    Args:
        values (list or np.ndarray): list of scalar values for each barycentric point.
        points (list or np.ndarray): list of three-parameter barycentric points.
        labels (list): list of three label strings, one for each side of the triangle.
        cmap (matplotlib.colors.Colormap): colormap for pseudocolor plot.
        clabel (str): colorbar label for values
        vmin (number): minimum value for coloring. If None, use minimum of the input values.
        vmax (number): maximum value for coloring. If None, use maximum of the input values.
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

def named_product(d, keys=None):
	keys = d.keys() if keys==None else keys
	
	return [
		dict(zip(keys, prod))
		for prod in itertools.product(*[d[k] for k in keys])
	]

def db_distinct(db, colname, allownone=False):
	return [
		str(r[0]) for r in db.execute(
			'select distinct %(c)s from sounds%(an)s' % {
				'c': colname,
				'an': [' where %s != "None"' % colname, ''][allownone]
			}
		)
	]

def make_db(csvs):
	'''Make the database from a CSV file.'''
	
	with open(csvs[0], 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		data = np.array([r for r in reader])
	
	names = {data[0,i]: i for i in range(len(data[0]))}
	
	db = sqlite3.connect('.temp.db')
	db.row_factory = sqlite3.Row
	db.execute('drop table if exists sounds')
	db.execute('create table sounds(%s)' % ','.join(['`%s`' % n for n in names]))
	
	for csvfn in csvs:
		with open(csvfn, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			data = np.array([r for r in reader])
			
			for row in data[1:]:
				db.execute('insert into sounds(%s) values(%s)' % (
					','.join([c for c in names.keys()]),
					','.join(['"%s"' % c.replace('\n', '\\n') for c in row[names.values()]])
				))
	
	db.commit()
	
	return db

def aggregate_preferences(db, joincols, compcols):
	'''
	Return a nested dictionary of combinations, coordinates and their preference values.
		joincols: column combinations that will be aggregate responses into groups.
		compcols: column combinations that will compete for preference from the aggregated groups.
	'''
	
	# 1. Gather distinct values for each of the column names in columns.
	join_distinct = {c: sorted(db_distinct(db, 's0_' + c)) for c in joincols}
	comp_distinct = {c: sorted(db_distinct(db, 's0_' + c)) for c in compcols}
	
	# 2. Come up with combinations of values that these columns can take.
	join_prod = sorted(named_product(join_distinct, joincols), key=lambda j: '.'.join(j.values()))
	comp_prod = sorted(named_product(comp_distinct, compcols), key=lambda c: '.'.join(c.values())) 
	
	ncomp = len(comp_prod)
	njoin = len(join_prod)
	
	# 3. Construct template for all queries.
	prefcol = 'which_sound_was_more_realistic'
	qtemplate = ' AND '.join([
		'select * from sounds where %s',
		'_tainted="false"',
		'_missed=""',
		'_golden="false"'
	])
	
	# 4. Results is a list of results, keyed by the joined parameters.
	# Parameter keys:
	#	'params': joined parameters.
	#	'best': best comp_prod (e.g. iteration) for the given joined set of parameters.
	# Array keys:
	# 	'votes: ncomp x ncomp array of the number of times the pair's been compared (sanity check).
	#	'for': ncomp x ncomp array of votes FOR the param combination.
	#	'against': ncomp x ncomp array of votes AGAINST the param combination.
	#	'same': ncomp x ncomp array of votes where the param combinations were the SAME.
	#	'null': ncomp x ncomp array of votes where the user couldn't decide.
	#	'pref': ncomp x ncomp array of calculate preference values for each pair of combinations.
	
	results = []
	
	for j in range(len(join_prod)):
		rset = {
			k: np.zeros((len(comp_prod),) * 2) 
			for k in ['for', 'against', 'same', 'null', 'pref', 'votes']
		}
		
		jcond = '(%s)' % ' AND '.join([
			'(%s)' % ' AND '.join(['s%s_%s = "%s"' % (s, k, v) for k, v in join_prod[j].items()])
			for s in [0, 1]
		])
		
		for ci in range(len(comp_prod)):
			for cj in range(ci+1, len(comp_prod)):
				cidx = [ci, cj]
				
				# Just want this to be done with . . . 
				s0_ci = ' AND '.join([
					's0_%s="%s"' % (k, v) for k, v in comp_prod[ci].items()
				])
				
				s1_ci = ' AND '.join([
					's1_%s="%s"' % (k, v) for k, v in comp_prod[ci].items()
				])
				
				s0_cj = ' AND '.join([
					's0_%s="%s"' % (k, v) for k, v in comp_prod[cj].items()
				])
				
				s1_cj = ' AND '.join([
					's1_%s="%s"' % (k, v) for k, v in comp_prod[cj].items()
				])
				
				for s in [0, 1]:
					ccond = '((%s) AND (%s))' % ([s0_ci, s1_ci][s], [s1_cj, s0_cj][s])
					query = qtemplate % ' AND '.join([jcond, ccond])
					
					for row in db.execute(query):
						rset['votes'][cidx[s], cidx[1 - s]] += 1
						rset['votes'][cidx[1 - s], cidx[s]] += 1
						
						if row[prefcol] == 'same':
							rset['same'][cidx[s], cidx[1 - s]] += 1
							rset['same'][cidx[1 - s], cidx[s]] += 1
						elif row[prefcol] == 'null':
							rset['null'][cidx[s], cidx[1 - s]] += 1
							rset['null'][cidx[1 - s], cidx[s]] += 1
						elif row[prefcol] == 's0':
							rset['against'][cidx[1 - s], cidx[s]] += 1
							rset['for'][cidx[s], cidx[1 - s]] += 1
						elif row[prefcol] == 's1':
							rset['for'][cidx[1 - s], cidx[s]] += 1
							rset['against'][cidx[s], cidx[1 - s]] += 1
		
		rset['params'] = join_prod[j]
		matsame = rset['same'] + rset['null']
		rset['pref'] = (rset['for'] + (0.5 * matsame)) / (matsame + rset['for'] + rset['against'])
		rset['best'] = comp_prod[argmax(nansum(rset['pref'], 0))]
		
		results.append(rset)
	
	return results, comp_prod
	
if __name__=='__main__':
	db = make_db(argv[1:])
	
	preferences, params = aggregate_preferences(db, [
		'coordinates'
	], [
		'simplegrainlength',
		'simplemaxdist'
	])
	
	for i in range(len(params)):
		prefdict = {
			p['params']['coordinates']: nansum(p['pref'][i,:])
			for p in preferences
		}
		
		points = np.array([[float(t) for t in k.split('_')] for k in prefdict.keys()])
		values = np.array(prefdict.values())
		
		subplot(3,3,i)
		title('\n'.join(['%s=%s' % (k,v) for k,v in params[i].items()] + ['avgpref=%.2f' % mean(values)]))
		baryplot(values, points, ['s0', 's1', 's2'], clabel='preference')
		
	show()