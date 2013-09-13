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
import os, argparse
from time import time
from itertools import izip
import multiprocessing as mp

# Pylab.
import numpy as np
import matplotlib
matplotlib.use('pdf')

# External.
import yaml

# Local.
import pybatchdict
import train as gt
import recordings as rec
import barycentric as bary

def make_tones(freqs, duration=1.0, amplitude=1.0, rate=44100):
    tonelen = duration * rate

    tones = [rec.Recording() for i in range(len(freqs))]

    tonefunc =lambda x: np.sin(x * 2) #* np.fmod(x, 2 * np.pi) / (2 * np.pi)

    for i in range(len(tones)):
        tones[i].wav = tonefunc(np.linspace(0, tonelen * 2 - 1, tonelen * 2) * 2 * np.pi * freqs[i] / rate) * amplitude
        tones[i].rate = rate

    return tones

def append_nums(recording, outname, wait=0, npath='./', ncount=0, namp=1.0):
    nums = []
    soundmax = np.amax(np.abs(recording.wav))
    
    for i in range(ncount):
        num = np.random.randint(9) + 1
        nums.append(num)
        
        numsnd = rec.Recording(os.path.join(npath, '%d.wav' % num))
        nummax = np.amax(np.abs(numsnd.wav))
        
        recording.wav[-(recording.rate / 4):] *= np.linspace(1, 0, recording.rate / 4)
        
        recording.append_frames(np.zeros(wait * numsnd.rate))
        recording.append_frames(numsnd.wav * (soundmax / nummax) * namp)
    
    recording.filename = outname + '-nums-' + ''.join(str(n) for n in nums) + '.wav'

def write_simple_mix(config, sounds, name):
    """
    Full process for making a mix of three simple grain trains. Called by main function in a multiprocessing.Pool.
    
    Args:
        config (dict): dictionary of configuration values.
        sounds (list): list of instances of Soundwalk
        name (str): path + basename for output files (no extension).
    """
    
    coords = np.array(config['coordinates'])

    # Percentage completion along each soundwalk (side).
    sideproj = bary.baryedges(coords)
    percs = bary.baryedges(coords, sidecoords=True)[:,1]
    
    # Prior probability of playing each soundwalk (side).
    sidecart = bary.bary2cart(sideproj)
    cart = bary.bary2cart(coords)
    prob = np.array([np.linalg.norm(sc - cart) for sc in sidecart])
    prob = np.log(prob + np.finfo(float).eps)
    prob /= np.sum(prob)

    trains = gt.make_simple_mix_trains(
        config['coordinates'],
        sounds,
        config['trainlength'],
        config['simplemix']['grainlength'],
        config['simplemix']['maxdist']
    )
    
    for t in range(len(trains)):
        trains[t].fillgrains(envtype=config['simplemix']['envelope'])
        trains[t].mixdown()
        trains[t].sound.wav = trains[t].sound.wav[:int(config['trainlength'] * trains[t].sound.rate)]
        print t, len(trains[t].sound.wav), len(trains[t].sound.wav) / trains[t].sound.rate

    mixed = rec.Recording()
    mixed.rate = trains[0].sound.rate
    mixed.wav = np.zeros(len(trains[0].sound.wav))
    
    for t in range(len(trains)):
        mixed.wav += trains[t].sound.wav * prob[t]
    
    mixed.filename = name + '.wav'
    mixed.save()

def write_simple(config, sounds, name):
    """
    Full process for making a simple grain train. Called by main function in a multiprocessing.Pool.
    
    Args:
        config (dict): dictionary of configuration values.
        sounds (list): list of instances of Soundwalk
        name (str): path + basename for output files (no extension).
    """
    
    train = gt.make_simple_train(
        config['coordinates'], 
        sounds, 
        config['trainlength'], 
        config['simple']['grainlength'], 
        config['simple']['maxdist']
    )
    
    train.basename = name
    train.fillgrains(envtype=config['simple']['envelope'])
    train.mixdown()#envtype=config['simple']['envelope'])
    train.sound.wav = train.sound.wav[:int(config['trainlength'] * train.sound.rate)]

    # 5. Save Sonic Visualiser annotation layer.
    if config['simple']['svl']:
        train.save_svl()

    if config['simple']['plot']:
        train.save_plot()
    
    # 4. Save wavfile.
    train.sound.filename = train.basename + '.wav'
    
    train.sound.save()

def write_sources(config, sounds, name):
    """
    Save portions of the source Recordings nearest to the config's sampling coordinates.
    
    Args:
        config (dict): configuration dictionary.
        sounds (list): list of source Recordings.
        name (str): base path / name for output. Filenames will be formatted as follows:
            [head]/[fn]-[tail].wav, where outname is [head]/[tail] for each source Recording named [fn].wav.
    """
    
    percs = bary.baryedges(np.array(config['coordinates']), sidecoords=True)[:,1]
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
        soundpart = rec.Recording()
        
        srcbase = os.path.splitext(os.path.split(sound.filename)[1])[0]
        
        soundpart.rate = sound.rate
        soundpart.wav = sound.wav[s:e]
        
        outpath = '%s-source-%s' % (name, srcbase)
        soundpart.filename = outpath + '.wav'
        soundpart.save()

def write_tones(config, sounds, name):    
    train = gt.make_tone_train(
        sounds,
        traindur=config['trainlength'],
        tonedur=config['tones']['length'],
        amplitude=config['tones']['amplitude'],
        fadedur=config['tones']['overlap']
    )
    
    train.basename = name
    sound = train.mixdown(envtype=config['tones']['envelope'])
    sound.wav = sound.wav[:int(config['trainlength'] * config['tones']['rate'])]
    sound.save()

def process_group(intuple):
    """
    Process grain trains / source sounds in groups, rather than pooling all together, to save memory.
    
    Args:
        intuple (tuple): tupled arguments for write_grain_train / write_source_sounds. Components are:
            configs (list): list of configuration dictionaries, one for each call.
            sounds (list): list of source Recordings to use for mixes / sources.
            names (list): list of output file basenames, one for each call.
            verbosity (int): debug verbosity.
    """
        
    configs, sounds, names = intuple
    
    # Need to randomize seed, as the seed is copied to the new process in everything but Windows.
    np.random.seed(int((time() + mp.current_process().pid * 1000)))
    
    for config, name in izip(configs, names):
        if config['verbosity'] > 0:
            print mp.current_process().name, name
               
        if config['mix'] == 'simple':
            write_simple(config, sounds, name)
        elif config['mix'] == 'simplemix':
            write_simple_mix(config, sounds, name)
        elif config['mix'] == 'sources':
            write_sources(config, sounds, name)
        elif config['mix'] == 'tones':
            write_tones(config, sounds, name)

def parallel_pool(func, cpus, args):
    if cpus > 1:
        mp.Pool(processes=cpus).map(func, args)
    else:
        map(func, args)

def main():
    parser = argparse.ArgumentParser(description='Create a mixture of two or more sound textures.')
    parser.add_argument('config', type=str, default='config.yaml', help='YAML config file.')
    args = parser.parse_args()

    # 1. YAML config. Enumerate over combinations of list values for studies.
    config = yaml.load(open(args.config))

    config.setdefault('outpath', '')
    config.setdefault('sources', [])
    config.setdefault('mix', 'simple')
    config.setdefault('coordinates', [1, 0, 0])
    config.setdefault('processes', mp.cpu_count())
    config.setdefault('interactive', False)

    config.setdefault('simple', {})
    config['simple'].setdefault('svl', False)
    config['simple'].setdefault('plot', False)
    config['simple'].setdefault('maxdist', 60.0)
    config['simple'].setdefault('trainlength', 15.0)
    config['simple'].setdefault('envelope', 'cosine')
    config['simple'].setdefault('grainlength', [500, 2000])

    config.setdefault('tones', {})
    config['tones'].setdefault('count', 0)
    config['tones'].setdefault('rate', 44100)
    config['tones'].setdefault('minfreq', 2000)
    config['tones'].setdefault('maxfreq', 5000)
    config['tones'].setdefault('amplitude', 1.0)
    config['tones'].setdefault('envelope', 'cosine')

    if type(config['coordinates']) == dict:
        if config['coordinates'][config['coordinates'].keys()[0]] == 'lattice':
            config['coordinates'] = {config['coordinates'].keys()[0]: bary.lattice(len(config['sources']))}

        config['coordinates'][config['coordinates'].keys()[0]] = [
            np.array(coord,dtype=float) / sum(coord) for coord in
            config['coordinates'][config['coordinates'].keys()[0]]
        ]

    print config['coordinates']

    # 2. Load soundwalks.
    sounds = []

    if config['mix'] == 'tones':
        sounds = make_tones(
            config['tones']['freqs'],
            duration=config['trainlength'],
            amplitude=config['tones']['amplitude'],
            rate=config['tones']['rate']
        )
    else:
        sounds = [rec.Recording(s) for s in config['sources']]

        # 3. Create groups of processes to do in parallellsl.
        # Split into group of number of CPUs to avoid copying pickled recordings for every single execution. I think.
        cpus = min(config['processes'], mp.cpu_count())

        batch = pybatchdict.BatchDict(config)

        names = [os.path.join(config['outpath'], 'mix-' + config['mix'] + '-' + name) for name in batch.hyphenate_changes()]
        names = [names[i::cpus] for i in range(min(cpus, len(names)))]

        combos = [batch.combos[i::cpus] for i in range(min(cpus, len(batch.combos)))]
        groups = [(c, sounds, on) for c, on in izip(combos, names)]

        parallel_pool(process_group, cpus, groups)

if __name__ == '__main__':
    main()
