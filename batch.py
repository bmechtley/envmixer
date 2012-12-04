from sys import argv
from subprocess import call
from pylab import *
import os

s1 = argv[1]
s2 = argv[2]
s3 = argv[3]
length = argv[4]
grainmin, grainmax = argv[5], argv[6]
jumpdev = argv[7]
reps = int(argv[8])
basedir = argv[9]
				
coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1./3, 1./3, 1./3)]
for i in range(len(coords)):
	coords[i] = array(coords[i])
	
coords.append((coords[0] + coords[1] + coords[3]) / 3)
coords.append((coords[0] + coords[2] + coords[3]) / 3)
coords.append((coords[1] + coords[2] + coords[3]) / 3)
coords.append((coords[0] + coords[3]) / 2)
coords.append((coords[1] + coords[3]) / 2)
coords.append((coords[2] + coords[3]) / 2)

p = array([(0, 0), (1, 0), (.5, sin(pi / 3))])

for i in range(len(coords)):
	c = array(coords[i])
	cart = [sum(c * p[:,0]), sum(c * p[:,1])]
	
	for r in range(reps):
		print i, r, c, cart
		params = [
			'python', 'mixer.py', 
			s1, s2, s3, 
			str(cart[0]), str(cart[1]), 
			length, grainmin, grainmax, jumpdev, 
			os.path.join(basedir, 'mix-naive-%.2f-%.2f-%.2f-%d.wav' % (c[0], c[1], c[2], r))
		]
		
		print '\t', ' '.join(params)
		
		call(params)
