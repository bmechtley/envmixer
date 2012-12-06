'''
batch.py
natural-mixer

2012 Brandon Mechtley
Arizona State University

Batch resynthesis for generating soundfiles for use with an Amazon Mechanical Turk user study.
'''

import os, argparse, itertools, multiprocessing, subprocess
import numpy as np

def work(command):
    print ' '.join([str(par) for par in command])
    return subprocess.call([str(p) for p in command], shell=False)

parser = argparse.ArgumentParser(
    description='Batch resynthesis for generating soundfiles with mixer.py')
parser.add_argument('inputs', metavar='wav', nargs=3, type=str,
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

coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1./3, 1./3, 1./3)]
for i in range(len(coords)):
    coords[i] = np.array(coords[i])

coords.append((coords[0] + coords[1] + coords[3]) / 3)
coords.append((coords[0] + coords[2] + coords[3]) / 3)
coords.append((coords[1] + coords[2] + coords[3]) / 3)
coords.append((coords[0] + coords[3]) / 2)
coords.append((coords[1] + coords[3]) / 2)
coords.append((coords[2] + coords[3]) / 2)

p = np.array([(0, 0), (1, 0), (.5, np.sin(np.pi / 3))])

multiprocessing.Pool(processes = multiprocessing.cpu_count()).map(work, [[
    'python', 'mixer.py', args.inputs[0], args.inputs[1], args.inputs[2],
    '-c', sum(c * p[:,0]), sum(c * p[:,1]),
    '-l', args.length,
    '-g', args.graindur[0], args.graindur[1],
    '-j', args.jumpdev,
    '-o', os.path.join(args.output, 'mix-naive-%.2f-%.2f-%.2f-%03d.wav' % (c[0], c[1], c[2], r))
] for r, c in list(itertools.product(range(args.instances), coords))])
