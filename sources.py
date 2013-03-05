"""
sources.py
envmixer

2012 Brandon Mechtley
Arizona State University

This utility takes in a list of source sounds and a list of barycentric
coordinates and outputs the \"closest\" portion of each source sound of a
specified length, where each source sound is the side of the regular convex
polygon in which the coordinates sit. . . . see mixer.py for more information.
"""

import argparse
from os.path import splitext, basename, join
from scipy.io import wavfile
from itertools import izip
from barycentric import *
from soundwalks import *


parser = argparse.ArgumentParser(description='Output closest sound texture segments in three source sound textures to\
    a given set of coordinates in a triangle.')
parser.add_argument('inputs', metavar='wav', nargs=3, type=str, help='wav or sv files to mix.')
parser.add_argument('-c', '--coords', nargs=3, metavar='float', type=float, default=[1, 0, 0],
    help='cartesian coordinates within the triangle.')
parser.add_argument('-l', '--length', metavar='s', type=float, default=10, help='length of output in seconds.')
parser.add_argument('-o', '--output', metavar='path', type=str, default='./', help='path prefix for output.')
parser.add_argument('-s', '--suffix', metavar='str', type=str, default='source', 
    help='suffix to add after base filename for each source sound file.')
args = parser.parse_args()

# Load commandline arguments.
sounds = [Soundwalk(w) for w in args.inputs]

# Make sure barycentric coordinates are normalized.
coords = np.array(args.coords / sum(args.coords))

# Percentage completion for each soundwalk.
percs = baryedges(coords, sidecoords=True)[:,1]
frames = [int(p * sw.len) for p, sw in izip(percs, sounds)]
fs = zip(frames, sounds)

# Get start frames and end frames. If we are too close to the start point,
# clip it to the beginning of the sound and adjust the end point.
# noinspection PyTypeChecker,PyTypeChecker
starts = np.array([max(0, int(f - (args.length * s.rate) / 2)) for f, s in izip(frames, sounds)])
# noinspection PyTypeChecker
ends = np.array([min(s.len, int(f + (args.length * s.rate) / 2)) for f, s in izip(frames, sounds)])

# If we are too close to the end point, clip it to the end of the sound and
# adjust the start point.
for s, e, sw in izip(starts, ends, sounds):
    seglen = int(sw.rate * args.length)
    
    if e - s < seglen:
        s = max(0, e - seglen)

starts, ends = np.array(starts), np.array(ends)

# Write the source wav files.
for s, e, sw in izip(starts, ends, sounds):
    name = '%s-%s.wav' % (splitext(basename(sw.wavfile))[0], args.suffix)
    wavfile.write(join(args.output, name), sw.rate, sw.frames[s:e])


