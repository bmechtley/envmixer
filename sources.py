'''
sources.py
natural-mixer

2012 Brandon Mechtley
Arizona State University

This utility takes in a list of source sounds and a list of barycentric coordinates and outputs the 
"closest" portion of each source sound of a specified length, where each source sound is the side 
of the regular convex polygon in which the coordinates sit. . . . see mixer.py for more information.
'''

import argparse
from os.path import splitext, basename, join
from scipy.io import wavfile
from itertools import izip

from soundwalks import *

def main():
    parser = argparse.ArgumentParser(description='Output closest sound texture segments in three\
        source sound textures to a given set of coordinates in a triangle.')
    parser.add_argument('inputs', metavar='wav', nargs=3, type=str,
        help='wav or sv files to mix.')
    parser.add_argument('-c', '--coords', nargs=2, metavar='float', default=[0.5, 0.5], type=float,
        help='cartesian coordinates within the triangle.')
    parser.add_argument('-l', '--length', metavar='s', default=10, type=float,
        help='length of output in seconds.')
    parser.add_argument('-o', '--output', metavar='path', default='./', type=str,
        help='path prefix for output.')
    parser.add_argument('-s', '--suffix', metavar='str', default='source', type=str,
        help='suffix to add after base filename for each source sound file.')
    args = parser.parse_args()
    
    # Load commandline arguments.
    sounds = [Soundwalk(w) for w in args.inputs]
    
    # List of start and end cartesian coordinates for each walk.
    starts = array([(0, 0), (0.5, sin(pi / 3)), (1, 0)])
    ends = array([(0.5, sin(pi / 3)), (1, 0), (0, 0)])
    
    # List of closest points to the coordinates for each walk.
    cp = array([closepoint(s, e, args.coords) for s,e in izip(starts, ends)])
    percs = [percline(c, s, e) for c,s,e in izip(cp, starts, ends)]
    frames = [int(p * sw.len) for p, sw in izip(percs, sounds)]
    
    starts = [max(0, int(f - (args.length * sw.rate) / 2)) for f, sw in izip(frames, sounds)]
    ends = [min(sw.len, int(f + (args.length * sw.rate) / 2)) for f, sw in izip(frames, sounds)]
    
    for s, e, sw in izip(starts, ends, sounds):
        seglen = int(sw.rate * args.length)
        
        if e - s < seglen:
            s = max(0, e - seglen)
    
    starts, ends = np.array(starts), np.array(ends)
    
    for s, e, sw in izip(starts, ends, sounds):
        filename = '%s-%s.wav' % (splitext(basename(sw.wavfile))[0], args.suffix)
        wavfile.write(join(args.output, filename), sw.rate, sw.frames[s:e])

if __name__=='__main__':
    main()
