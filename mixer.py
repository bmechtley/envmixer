"""
mixer.py
envmixer

2012 Brandon Mechtley
Arizona State University

Creates a mixed environmental sound recording from 3 source recordings. Source 
recordings are assumed to be of equal duration. The paths that define along 
which they were recorded are fake, organized along on equilateral triangle:

             p1 (.5, sin(pi / 3))
           /    \
    p0 (0,0) -- p2 (1, 0)

Soundwalk 0: p0->p1
Soundwalk 1: p1->p2
Soundwalk 2: p2->p0

The mixed synthesized version will be fixed at coordinates (x, y) and last for
the specified duration (in seconds).

Various methods are used for resynthesis. See the appropriate functions for
details. The basic rule of thumb is that the closer to a given walk the cursor
is, the more similar the synthesized version will be to it. Additionally, the
output should be most similar to the region of each soundwalk that is closest
to the cursor.

Usage: python mixer.py soundwalk0.sv soundwalk1.sv soundwalk2.sv x y duration
"""

import os, bz2, argparse
from scipy.io import wavfile
from soundwalks import *
from barycentric import *

class Grain:
    def __init__(self):
        """
        Initialize properties.
        """
        
        self.outpos, self.srcpos, self.dur, self.src = -1, -1, -1, -1

    def __str__(self):
        """
        Convert to string.
        
        :return: string representation of grain.
        """
        
        return '%d %d %d %d' % (self.outpos, self.srcpos, self.dur, self.src)

def output_grain_annotations(grains, rate, wavname, filename):
    """
    
    :param grains: 
    :param rate: 
    :param wavname: 
    :param filename: 
    """
    outstr = open('template.xml', 'r').read() % (
        wavname,
        rate,
        grains[-1].outpos + grains[-1].dur,
        os.path.join(os.getcwd(), wavname),
        rate,
        grains[0].outpos,
        grains[-1].outpos + grains[-1].dur,
        len(grains),
        '\n'.join(['<point frame="%d" value="%d" duration="%d" label=""/>' % \
            (g.outpos, i, g.dur) for i, g in enumerate(grains)
        ]),
        (grains[-1].outpos + grains[-1].dur) / 2
    )
    
    f = open(filename, 'wb')
    f.write(bz2.compress(outstr))
    f.close()

def simple_grain_train(coords, sounds, length=10, graindur=(500, 2000), jumpdev=60):
    """
    Simplest synthesis algorithm. Creates a sequence of overlapping grains, each selected from a different source
    recording. The source recording is randomly selected, weighted according to which recording is closest to the input
    coordinates. Each grain has a random duration, sampled from a beta distribution (a = 2, b = 5) on the interval 
    [100, 2000] milliseconds. Each grain is copied from that point in the selected source recording that is closest to 
    the input coordinates with a random offset, selected from a normal distribution with mean = 0, variance = 60 
    seconds.
    
    :param coords: 
    :param sounds: 
    :param length: 
    :param graindur: 
    :param jumpdev: 
    """
    
    coords = np.array(coords)
    
    # Percentage completion along each soundwalk (side).
    sideproj = baryedges(coords)
    percs =  baryedges(coords, sidecoords=True)[:,1]
    
    # Prior probability of playing each soundwalk (side).
    sidecart = bary2cart(sideproj)
    cart = bary2cart(coords)
    
    prob = np.array([np.linalg.norm(sc - cart) for sc in sidecart])
    prob = np.log(prob + np.finfo(float).eps)
    prob /= np.sum(prob)
    
    # Create list of sound grains. Eeach grain is a tuple of frame numbers:
    # (output frame offset, duration in frames, source recording frame offset, 
    # source # (0-2))
    
    grains = []
    rate = min([s.rate for s in sounds])
    pos = 0
    
    while pos < rate * length:
        g = Grain()
        
        # Random source.
        g.src = choice(range(3), p = prob)[0]
        g.dur = np.random.randint(graindur[0] / 1000. * rate, graindur[1] / 1000. * rate)
        
        # Random offset from closest source point.
        jump = int(np.random.randn() * rate * jumpdev)
        frame = int(percs[g.src] * sounds[g.src].len + sounds[g.src].start)
        g.srcpos = frame + jump
        g.srcpos = min(g.srcpos, sounds[g.src].end - g.dur * rate)
        g.srcpos = max(g.srcpos, sounds[g.src].start)
        
        g.outpos = 0
        grains.append(g)
        
        # If this isn't the first grain, overlap the grains to crossfade.
        if len(grains) > 1:
            fadedur = int(min(grains[-1].dur, grains[-2].dur) / 2)
            grains[-1].outpos = grains[-2].outpos + grains[-2].dur - fadedur
        
        pos = grains[-1].outpos + grains[-1].dur
    
    grains[-1].dur = min(grains[-1].dur, rate * length - grains[-1].outpos)
    
    return grains

def output_grain_train(sounds, grains, filename):
    """
    Write a synthesized version from a sequence of grains from different sources, with optional overlap.
    
    :param sounds: 
    :param grains: 
    :param filename: 
    """
    
    rate = min([s.rate for s in sounds])
    length = max([g.outpos + g.dur for g in grains])
    sound = zeros(length, dtype=np.int16)

    for i, g in enumerate(grains):
        g.data = array(sounds[g.src].frames[g.srcpos:g.srcpos + g.dur])
        
        # Apply window halves exiting previous grain, entering current grain.
        if i > 0:
            p = grains[i - 1]
            fadedur = p.outpos + p.dur - g.outpos
            
            half = 1 - (cos(linspace(0, pi, fadedur)) + 1) / 2
            
            if len(p.data[-fadedur:]):
                p.data[-fadedur:] = p.data[-fadedur:] * sqrt(1 - half)
            
            if len(g.data[:fadedur]):
                g.data[:fadedur] = g.data[:fadedur] * sqrt(half)
    
    for g in grains:
        if len(g.data):
            sound[g.outpos:g.outpos + g.dur] += g.data
    
    wavfile.write(filename, rate, sound)

def main():
    parser = argparse.ArgumentParser(
        description='Create a mixture of two or more sound textures.')
    parser.add_argument('inputs', metavar='wav', nargs='+', 
        type=str, help='wav or sv files to mix.')
    parser.add_argument('-c', '--coords', nargs='+', metavar='float', 
        type=float, default=[0.5, 0.5], 
        help='barycentric coordinates of the observation point.')
    parser.add_argument('-l', '--length', metavar='s', default=10, 
        type=float, help='length of output in seconds')
    parser.add_argument('-g', '--graindur', nargs=2, metavar='ms', 
        type=float, default=[100, 500], 
        help='shortest / longest duration of grains in ms')
    parser.add_argument('-j', '--jumpdev', metavar='s', default=60, 
        type=float, help='standard deviation of normally distributed jumps\
        between grains in seconds')
    parser.add_argument('-o', '--output', metavar='file', default='output.wav',
        type=str, help='wav file for output')
    parser.add_argument('-s', '--svl', metavar='file', default='output.svl',
        type=str, help='sonic visualiser layer output')
    args = parser.parse_args()
    
    # Load commandline arguments.
    sounds = [Soundwalk(w) for w in args.inputs]
    
    grains = simple_grain_train(
        args.coords, 
        sounds, 
        args.length, 
        args.graindur, 
        args.jumpdev
    )

    output_grain_train(sounds, grains, args.output)
    output_grain_annotations(grains, sounds[0].rate, args.output, args.svl)

if __name__ == '__main__':
    main()
