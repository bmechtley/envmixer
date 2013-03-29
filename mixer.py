"""
mixer.py
envmixer

2012 Brandon Mechtley
Arizona State University

Creates a mixed environmental sound recording from 3 source recordings. Source recordings are assumed to be of equal
duration. The paths that define along which they were recorded are fake, organized along on equilateral triangle:

::
             p1 (.5, sin(pi / 3))
           /    \
    p0 (0,0) -- p2 (1, 0)

Soundwalk 0: p0->p1
Soundwalk 1: p1->p2
Soundwalk 2: p2->p0

The mixed synthesized version will be fixed at coordinates (x, y) and last for the specified duration (in seconds).

Various methods are used for resynthesis. See the appropriate functions for details. The basic rule of thumb is that
the closer to a given walk the cursor is, the more similar the synthesized version will be to it. Additionally, the
output should be most similar to the region of each soundwalk that is closest to the cursor.

Usage: python mixer.py soundwalk0.sv soundwalk1.sv soundwalk2.sv x y duration
"""

# Standard.
from time import time
import os, bz2, argparse
from itertools import izip
import multiprocessing as mp

# Pylab.
import numpy as np
import scikits.audiolab as al
from scipy.stats import scoreatpercentile
from scipy.io import wavfile

# External.
import yaml
import matplotlib
matplotlib.use('pdf')

import scikits.audiolab as audiolab
import matplotlib.pyplot as pp
import matplotlib.cm as cm

# Personal.
from barycentric import lattice, baryedges, bary2cart
from Recording import Recording, plot_waveform
import pybatchdict

class Grain:
    def __init__(self):
        self.outpos, self.srcpos, self.dur, self.src = -1, -1, -1, -1
        self.data = None
        self.env = None

    def __str__(self):
        return 'outpos: %d, srcpos: %d, dur: %d, src: %d' % (self.outpos, self.srcpos, self.dur, self.src)

def annotate_grain_train(grains, rate, wavname, filename):
    """
    Save a SonicVisualiser/Annotator annotations file, treating each grain as a segment.
    
    Args:
        grains (list): list of grain objects to output
        rate (int): sampling rate of grains
        wavname (str): original filename of wav from which grains are sourced.
        filename (str): filename for the SonicVisualiser annotations.
    """
    
    outstr = open('data/template.xml', 'r').read() % (
        wavname,
        rate,
        grains[-1].outpos + grains[-1].dur,
        os.path.join(os.getcwd(), wavname),
        rate,
        grains[0].outpos,
        grains[-1].outpos + grains[-1].dur,
        len(grains),
        '\n'.join([
            '<point frame="%d" value="%d" duration="%d" label=""/>' % (g.outpos, i, g.dur) 
            for i, g in enumerate(grains)
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

    Args:
        coords (list or numpy.ndarray): barycentric coordinates to sample synthesis.
        sounds (list): list of instances of soundwalk, one per source edge.
        length (number): duration of granular synthesis in seconds.
        graindur ((number, number)): range of duration of grains in milliseconds, (low, high)
        jumpdev (number): standard deviation of grain-to-grain start time differences in seconds. 

    Returns:
        list of grains
    """
    
    coords = np.array(coords)
    
    # Percentage completion along each soundwalk (side).
    sideproj = baryedges(coords)
    percs = baryedges(coords, sidecoords=True)[:,1]
    
    # Prior probability of playing each soundwalk (side).
    sidecart = bary2cart(sideproj)
    cart = bary2cart(coords)
    prob = np.array([np.linalg.norm(sc - cart) for sc in sidecart])
    prob = np.log(prob + np.finfo(float).eps)
    prob /= np.sum(prob)
    
    # Create list of sound grains. Eeach grain is a tuple of frame numbers: (output frame offset, duration in frames,
    # source recording frame offset, source # (0-2))
    grains = []
    rate = min([s.rate for s in sounds])
    pos = 0
    
    while pos < rate * length:
        g = Grain()
        
        # Random source.
        g.src = int(np.random.choice(range(len(sounds)), p=prob))
        g.dur = int(np.random.randint(graindur[0] / 1000. * rate, graindur[1] / 1000. * rate))
        
        # Random offset from closest source point.
        jump = int(np.random.randn() * rate * jumpdev)
        frame = int(percs[g.src] * sounds[g.src].len + sounds[g.src].start)

        g.srcpos = frame + jump
        g.srcpos = int(min(g.srcpos, sounds[g.src].end - g.dur * rate))
        g.srcpos = int(max(g.srcpos, sounds[g.src].start))
        
        g.outpos = 0
        grains.append(g)
        
        # If this isn't the first grain, overlap the grains to crossfade.
        if len(grains) > 1:
            fadedur = int(min(grains[-1].dur, grains[-2].dur) / 2)
            grains[-1].outpos = grains[-2].outpos + grains[-2].dur - fadedur
        
        pos = grains[-1].outpos + grains[-1].dur
        
    grains[-1].dur = int(min(grains[-1].dur, rate * length - grains[-1].outpos))

    return grains

def mix_grain_train(grains, rate, sounds, envtype='cosine'):
    sound = Recording()
    sound.rate = rate
    
    for i, g in enumerate(grains):
        g.data = np.array(sounds[g.src].wav[g.srcpos:g.srcpos + g.dur])
        g.env = np.ones(g.data.shape)
        
        # Apply window halves exiting previous grain, entering current grain.
        if i > 0:
            p = grains[i - 1]
            fadedur = p.outpos + p.dur - g.outpos
            
            envelope = None
            
            if envtype == 'cosine':
                envelope = 1 - (np.cos(np.linspace(0, np.pi, fadedur)) + 1) / 2
            elif envtype == 'linear':
                envelope = np.linspace(0, 1, fadedur)
            
            assert envelope is not None, 'Invalid mixing envelope: %s' % envtype
            
            if len(p.data[-fadedur:]):
                p.env[-fadedur:] *= 1 - envelope
            
            if len(g.data[:fadedur]):
                g.env[:fadedur] *= envelope
    
    length = max([g.outpos + g.dur for g in grains])
    sound.wav = np.zeros(length, dtype=grains[0].data.dtype)
    
    for g in [g for g in grains if len(g.data)]:
        sound.wav[g.outpos:g.outpos + g.dur] += g.data * g.env
    
    sound.len = len(sound.wav)
    
    return sound

def plot_grain_train(grains, rate, framesize=512, hopsize=256, cmap=cm.gist_rainbow, npoints=3):
    """
    Plot three subplots:
        a) An illustration of placement of grain waveforms and their envelopes before mixing.
        b) The final mixed waveform.
        c) The final mixed spectrogram.
    
    Args:
        grains (list): the list of Grain objects in the grain train.
        rate (int): the sampling rate of the resulting mixed waveform.
        framesize (int): number of samples per frame in the waveform plots.
        hopsize (int): number of samples between frames in the waveform plots.
        cmap (matplotlib.colors.Colormap): color map to use for coloring waveforms.
        npoints (int): number of percentile points to use for waveforms.
    """
        
    sources = np.unique([g.src for g in grains])
    nsrc = len(sources)
    maxsrc = float(np.amax(sources))
    minsrc = float(np.amin(sources))
    
    oldpos = 0
    layer = False

    for i, g in enumerate(grains):
        # If two grains overlap, draw them stacked amongst two layers.
        if oldpos > g.outpos:
            layer = not layer
        
        oldpos = g.outpos + len(g.data)

        ymin = 0 if not layer else 0.5
        xmin, xmax = g.outpos / rate, (g.outpos + len(g.data)) / rate
        
        # Resample envelope.
        envx = np.linspace(xmin, xmax, len(g.env) / framesize)
        envmax = np.interp(envx, np.linspace(xmin, xmax, len(g.env)), g.env)
        envmax = np.interp(envmax, [0, 1], [ymin+.25, ymin+.5])

        # Draw envelope on top of waveform.
        pp.fill_between(
            envx, 
            envmax, 
            np.zeros(envx.shape) + ymin + .25, 
            color=cmap(g.src / maxsrc), 
            alpha=.5
        )
        
        # Draw waveform under envelope.
        plot_waveform(
            g.data * g.env, 
            framesize, hopsize,
            npoints=npoints,
            xmin=xmin, xmax=xmax,
            ymin=ymin, ymax=ymin+.25,
            color=cmap(g.src / maxsrc),
            alpha=.25
        )
    
    # Tighten axes.
    tmin = np.amin([g.outpos for g in grains]) / rate
    tmax = np.amax([g.outpos + len(g.data) for g in grains]) / rate
    
    pp.ylim(-0.05, 1.05)
    pp.xlim(tmin, tmax)

def write_grain_train(config, sounds, outname, verbosity=0):
    """
    Full process for making a simple grain train. Called by main function in a multiprocessing.Pool.
    
    Args:
        config (dict): dictionary of configuration values.
        sounds (list): list of instances of Soundwalk
        outname (str): path + basename for output files (no extension).
        verbosity (float): debug verbosity (default 0).
    """
    
    grains = simple_grain_train(
        config['coordinates'], 
        sounds, 
        config['trainlength'], 
        config['grainlength'], 
        config['jumpdev'],
    )
    
    sound = mix_grain_train(grains, sounds[0].rate, sounds, config['envelope'])

    # 4. Save wavfile.
    if config['save']['wav']:
        sound.filename = outname + '.wav'
        
        if config['endnumbers'] > 0:
            for i in range(confing['endnumbers']):
                num = np.random.randint(9) + 1
                
                numsnd = al.Sndfile(
                    os.path.join(confing['numpath'], '%d.wav' % num),
                    mode='r',
                    format=al.Format(),
                    channels=1,
                    samplerate=sound.rate
                )
                
                sound.append_frames(numsnd.read_frames(numsnd.nframes))
        
        sound.save()

    # 5. Save Sonic Visualiser annotation layer.
    if config['save']['svl']:
        annotate_grain_train(grains, sounds[0].rate, outname + '.wav', outname + '.svl')
    
    if config['save']['plot']:
        pp.figure(figsize=(16,8))
        
        # 1. Plot grain train.
        pp.subplot(311)
        
        pp.title('simple grain train')
        plot_grain_train(grains, rate=sounds[0].rate)
        pp.xlim(0, sound.len / sounds[0].rate)
        pp.xticks([])
        pp.yticks([])
        pp.ylabel('enveloped grains')
        pp.gca().yaxis.set_label_coords(-.04, 0.5)

        lines = [
            matplotlib.lines.Line2D(
                range(1), 
                range(1), 
                color=cm.gist_rainbow(float(i) / (len(sounds) - 1)), 
                markerfacecolor=cm.gist_rainbow(float(i) / (len(sounds) - 1)),
                markeredgecolor=cm.gist_rainbow(float(i) / (len(sounds) - 1)),
                alpha=0.5,
                marker='s'
            )
            for i in range(len(sounds))
        ]

        pp.legend(
            lines,
            [os.path.splitext(os.path.split(s.filename)[1])[0] for s in sounds], 
            numpoints=1, 
            loc='upper right',
            bbox_to_anchor=(1.0, 1.2125),
            markerscale=1.5,
            ncol=len(sounds),
            handlelength=0,
            frameon=False
        )
        
        # 2. Plot mixed waveform.
        pp.subplot(312)
        plot_waveform(
            sound.wav,
            512, 256,
            xmin=0, xmax=sound.len / sound.rate,
            ymin=-1, ymax=1,
            npoints=5,
        )
        
        pp.xlim(0, sound.len / sounds[0].rate)
        pp.xticks([])
        pp.ylabel('mixed waveform')
        pp.gca().yaxis.set_label_coords(-.04, 0.5)

        # 3. Plot mixed spectrogram.
        pp.subplot(313)
        pxx, freqs, bins, im = pp.specgram(sound.wav, Fs=sound.rate)
        pp.xlim(0, sound.len / sounds[0].rate)
        pp.ylim(freqs[0], freqs[-1])
        pp.yticks(range(0, 20001, 5000), range(0, 21, 5))
        pp.ylabel('frequency (KHz)')
        pp.xlabel('time (s)')
        pp.gca().yaxis.set_label_coords(-.04, 0.5)

        pp.savefig(outname + '.pdf')

def write_source_sounds(config, sounds, outname):
    """
    Save portions of the source Recordings nearest to the config's sampling coordinates.
    
    Args:
        config (dict): configuration dictionary.
        sounds (list): list of source Recordings.
        outname (str): base path / name for output. Filenames will be formatted as follows:
            [head]/[fn]-[tail].wav, where outname is [head]/[tail] for each source Recording named [fn].wav.
    """
    
    percs = baryedges(np.array(config['coordinates']), sidecoords=True)[:,1]
    frames = [int(p * s.len) for p, s in izip(percs, sounds)]
    fs = zip(frames, sounds)
    
    # Get start frames and end frames. If we are too close to the start point, clip it to the beginning of the sound
    # and adjust the end point.
    starts = np.array([int(f - (config['trainlength'] * s.rate) / 2) for f, s in izip(frames, sounds)])
    ends = np.array([int(f + (config['trainlength'] * s.rate) / 2) for f, s in izip(frames, sounds)])
    
    # If we are too close to the end point, clip it to the end of the sound and adjust the start point.
    for i in range(len(sounds)):
        seglen = int(sounds[i].rate * config['trainlength'])
    
        starts[i] = max(0, starts[i])
        ends[i] = min(starts[i] + seglen, sounds[i].len)
        starts[i] = max(0, ends[i] - seglen)
    
    starts, ends = np.array(starts), np.array(ends)
    
    # Write the source wav files.
    for s, e, sound in izip(starts, ends, sounds):
        soundpart = Recording()
        
        outparts = os.path.split(outname)
        sndbase = os.path.splitext(os.path.split(sound.filename)[1])[0]
        
        soundpart.filename = '%s/%s-source-%s.wav' % (outparts[0], sndbase, outparts[1])
        soundpart.rate = sound.rate
        soundpart.wav = sound.wav[s:e]
        soundpart.save()

def process_group(intuple):
    """
    Process grain trains / source sounds in groups, rather than pooling all together, to save memory.
    
    Args:
        intuple (tuple): tupled arguments for write_grain_train / write_source_sounds. Components are:
            configs (list): list of configuration dictionaries, one for each call.
            sounds (list): list of source Recordings to use for mixes / sources.
            outnames (list): list of output file basenames, one for each call.
            verbosity (int): debug verbosity.
    """
        
    configs, sounds, outnames, verbosity, func = intuple
    
    # Need to randomize seed, as the seed is copied to the new process in everything but Windows.
    np.random.seed(int((time() + mp.current_process().pid * 1000)))
    
    for config, outname in izip(configs, outnames):
        if verbosity > 0:
            print mp.current_process().name, outname
        
        func(config, sounds, outname)

def main():
    """Just here for PyCharm structure listing :)"""
    
    parser = argparse.ArgumentParser(description='Create a mixture of two or more sound textures.')
    parser.add_argument('inputs', metavar='wav', nargs='+', type=str, help='wav or sv files to mix.')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='YAML config file.')
    parser.add_argument('-o', '--output', metavar='file', default='output.wav', type=str, help='wav file for output')
    args = parser.parse_args()
    
    # 1. YAML config. Enumerate over combinations of list values for studies.
    config = yaml.load(open(args.config))
    config.setdefault('coordinates', {'@': 'lattice'})
    config.setdefault('processes', mp.cpu_count())
    config.setdefault('trainlength', 15.0)
    config.setdefault('grainlength', [500, 2000])
    config.setdefault('envelope', 'cosine')
    config.setdefault('jumpdev', 5.0)
    config.setdefault('endnumbers', 2)
    config.setdefault('save', {})
    config['save'].setdefault('plot', False)
    config['save'].setdefault('wav', False)
    config['save'].setdefault('svl', False)
    
    if type(config['coordinates']) == dict:
        if config['coordinates'][config['coordinates'].keys()[0]] == 'lattice':
            config['coordinates'] = {config['coordinates'].keys()[0]: lattice(len(args.inputs))}
    
    batchdict = pybatchdict.BatchDict(config)
    outnames = [os.path.join(args.output, outname) for outname in batchdict.hyphenate_changes()]
    
    # 2. Load soundwalklss.
    sounds = [Recording(s) for s in args.inputs]
    
    # 3. Create grain trains in parallel.
    cpus = min(config['processes'], mp.cpu_count())
    outnames = [outnames[i::cpus] for i in range(min(cpus, len(outnames)))]
    combos = [batchdict.combos[i::cpus] for i in range(min(cpus, len(batchdict.combos)))]
    
    if cpus > 1:
        mp.Pool(processes=cpus).map(
            process_group, [(c, sounds, on, 1, write_grain_train) for c, on in izip(combos, outnames)]
        )
    else:
        map(process_group, [(c, sounds, on, 1, write_grain_train) for c, on in izip(combos, outnames)])
    
    # 4. Create source sounds nearest to the mixing coordinates.
    del config['mix']
    del config['iteration']
    
    batchdict = pybatchdict.BatchDict(config)
    outnames = [os.path.join(args.output, outname) for outname in batchdict.hyphenate_changes()]
    
    outnames = [outnames[i::cpus] for i in range(min(cpus, len(outnames)))]
    combos = [batchdict.combos[i::cpus] for i in range(min(cpus, len(batchdict.combos)))]
    
    mp.Pool(processes=cpus).map(
        process_group, [(c, sounds, on, 1, write_source_sounds) for c, on in izip(combos, outnames)]
    )
        
if __name__ == '__main__':
    main()
