import sqlite3, argparse, os, os.path, subprocess
from itertools import izip, product, combinations
from pprint import PrettyPrinter
from random import randint
from hashlib import md5

import numpy as np
import yaml

def db_exec(db, command):
	print command
	db.execute(command)

def do_hash(db, hashpath):
	for sndname, hashname in db.execute('select name, hash from sounds'):
		hashdest = os.path.join(hashpath, hashname + os.path.splitext(sndname)[1]) 
		subprocess.call(['cp', sndname, hashdest])
		print 'cp', sndname, hashdest

def do_unhash(db, hashpath):
	for sndname, hashname in db.execute('select name, hash from sounds'):
		hashsrc = os.path.join(hashpath, hashname + os.path.splitext(sndname)[1])
		subprocess.call(['cp', hashsrc, sndname])
		print 'cp', hashsrc, sndname

def make_csv(db, inyaml):
	config = yaml.load(open(inyaml))
	
	config.setdefault('exclude', [])
	config.setdefault('combine', [])
	config.setdefault('limit', [])
	config.setdefault('jobname', 'job')
	
	units = []
	sounds = []
	
	sounds = [dict(s) for s in db.execute('select * from sounds')]
	columns = sounds[0].keys()
	
	tones = [s for s in sounds if s['mix']=='tones']
	sounds = [
		s for s in sounds 
		if s['mix']!='tones'
		and (not len(config['limit'])) or any([
			all([
				s.get(k) == v 
				for k, v in l.items()
			])
			for l in config['limit']
		])
	]
	
	print len(sounds)
	groups = []
	
	if len(config['combine']):
		uniques = {}
		
		# Get list of columns that are used to pivot subgroups.
		for colname in config['combine']:
			uniques[colname] = list(np.unique([
				str(s[colname]) 
				for s in sounds 
				if s[colname] != None
			]))
			
			prod = list(product(*[uniques[key] for key in uniques]))			
			names = [uniques.keys()] * len(prod)
			
			# Get a list of subgroups that have homogenous values for fields in "uniques."
			groups = [
				[
					s for s in sounds
					if all([s[k] == v for k, v in zip(keys, values)])
				]
				for keys, values in zip(names, prod)
			]
	else:
		groups = [[s for s in sounds]]
	
	# Add all combinations of sounds within each subgroup.
	pairs = []
	
	for g in groups:
		pairs.extend(list(combinations(g, 2)))
	
	print len(pairs), 'normal pairs.'
	
	# Add tone gold.
	for s in sounds:
		pairs.append((s, tones[randint(0, len(tones) - 1)]))
		pairs.append((s, s))
	
	print len(pairs), 'total pairs.'
	
	# Create list of dicts for each unit.
	units = []
	
	for p in pairs:
		order = [0, 1] if randint(0, 1) else [1, 0]
				
		unit = {
			's%d_%s' % (i, k): p[order[i]][k] 
			for k, i in product(columns, [0, 1])
		}
		
		unit['jobname'] = config['jobname']
		
		if unit['s0_hash'] == unit['s1_hash']: 
			unit['_golden'] = 'True'
			unit['morepc_gold'] = 'same\nnull'
			unit['morepc_gold_reason'] = 'Sounds are the same.'
		elif unit['s0_mix'] == 'tones':
			unit['_golden'] = 'True'
			unit['morepc_gold'] = 's1'
			unit['morepc_gold_reason'] = 'Sound 1 is obviously a series of synthesized tones.'
		elif unit['s1_mix'] == 'tones':
			unit['_golden'] = 'True'
			unit['morepc_gold'] = 's0'
			unit['morepc_gold_reason'] = 'Sound 2 is obviously a series of synthesized tones.'
		
		# pprint.PrettyPrinter().pprint([
		# any([
		# 			all([('%s_%s' % (s,k), v) in unit.items() for k,v in l.items()])
		# 			for l in config['limit']
		# 		])
		# 		for s in['s0', 's1']
		# 		
		# 	])
		
		units.append(unit)
	
	# Add row number for easy hashing of pairs.
	for i, unit in enumerate(units): 
		unit['rownum'] = i
	
	# Get column headers.
	fields = []
	
	for unit in units:
		for key in unit.keys():
			if key not in fields:
				fields.append(key)
	
	# Set empty values for fields not in each unit.
	for unit in units:
		for key in fields:
			unit.setdefault(key, '')
	
	fields.sort()
	
	print '%d normal units.' % len([u for u in units if u['_golden']==''])
	print '%d gold units.' % len([u for u in units if u['_golden']=='True'])
	
	# Add column headers.
	units.insert(0, {f: f for f in fields})
	
	# Convert list of dicts to CSV and save out.
	csv = [[str(unit[f]) for f in fields] for unit in units]
	np.savetxt(config['jobname'] + '.csv', csv, delimiter=",", fmt='"%s"')

def make_db(db, soundpath):
	fields = ['hash', 'name']
	sounds = []
	
	sndfiles = os.listdir(soundpath)
	
	for sndfile in sndfiles:
		basename = os.path.basename(sndfile)
		tokens = os.path.splitext(basename)[0].split('-')
		
		for token in tokens[::2]:
			if token not in fields:
				fields.append(token)
		
		sound = {f: v for f, v in izip(tokens[::2], tokens[1::2])}
		
		hash = md5()
		hash.update(basename)
		
		sound['hash'] = hash.hexdigest()
		sound['name'] = os.path.join(soundpath, sndfile)
		
		sounds.append(sound)
	
	print fields
	
	db_exec(db, 'drop table if exists sounds')
	db_exec(db, 'create table sounds (%s)' % ', '.join(['`%s` text' % f for f in fields]))
	
	for sound in sounds:
		db_exec(db, 'INSERT INTO sounds (%s) VALUES (%s)' % (
			', '.join(['`%s`' % s for s in sound.keys()]),
			', '.join(['"%s"' % v for v in sound.values()])
		))
	
	db.commit()
	db.close()

if __name__=='__main__':
	parser = argparse.ArgumentParser(
		description='Creates an index of generated wav files for study purposes.'
	)
	
	parser.add_argument(
		'command', 
		type=str, 
		default='create', 
		help='Command to perform.', 
		choices=['create', 'hash', 'unhash', 'csv']
	)
	
	parser.add_argument(
		'database', 
		type=str, 
		default='study.db', 
		help='sqlite3 database name to load/create.'
	)
	
	parser.add_argument(
		'-s', '--soundpath', 
		required=False, 
		type=str, 
		default='./', 
		help='{create, hash, unhash}: path to input/output sound files\
			 from which to create an index, hash, or unhash.'
	)
	
	parser.add_argument(
		'-m', '--hashpath', 
		required=False, 
		type=str, 
		default='./', 
		help='{hash, unhash}: path to save/load un/hashed wav files.'
	)
	
	parser.add_argument(
		'-y', '--yaml', 
		required=False, 
		type=str, 
		default='config.yaml',
		help='csv: filename for user study configuration.'
	)
	
	args = parser.parse_args()
	db = sqlite3.connect(args.database)
	db.row_factory = sqlite3.Row
	
	if args.command == 'create':
		make_db(db, args.soundpath)
	elif args.command == 'hash':
		do_hash(db, args.hashpath)
	elif args.command == 'unhash':
		do_unhash(db, args.hashpath)
	elif args.command == 'csv':
		make_csv(db, args.yaml)
	
	db.close()
