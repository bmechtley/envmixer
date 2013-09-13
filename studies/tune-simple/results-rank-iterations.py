'''
studies/study1.py
envmixer

2013 Brandon Mechtley
Arizona State University

Find out which iteration is best for every combination of:
	simple.grainlength: 3 versions
	simple.maxdist: 3 versions
	coordinates: 10 points

Creates a YAML configuration for index.py to seed Study 2, finding out which parameter set is best.

Usage: python simple-study1.py 1.csv 2.csv ... n.csv
'''

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
	
	# 5. Output YAML.
	print yaml.dump([dict(r['params'].items() + r['best'].items()) for r in results])

if __name__=='__main__':
	db = make_db(argv[1:])
	
	aggregate_preferences(db, [
		'simplegrainlength',
		'simplemaxdist',
		'coordinates'
	], [
		'iteration'
	])
	
	#doplot()
