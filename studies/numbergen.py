"""
studies/numbergen.py
envmixer

2013 Brandon Mechtley
Arizona State University

1. Generate 100 4-digit numbers read aloud, sourced from files with paths numbers/[1-9].wav, saved as
numstrs/numstr-[1111-9999].wav. Used for audio captchas.

2. Also generate 100 15s clips with chained 15 1s sinusoids with random frequencies in the range 100-1100Hz. Used as
trick questions (similarity to any source clip should be 1/5.)
"""

import scikits.audiolab as al
import numpy as np
import argparse

def append_numbers(fn, nums):
    numbers = np.random.randint(low=1, high=10, size=nums)
    wavfile = al.Sndfile(fn, mode='r')
    
    format = wavfile.format
    channels = wavfile.channels
    samplerate = wavfile.samplerate
    nframes = wavfile.nframes

    wavfile = al.Sndfile(fn, mode='rw', format=format, channels=channels, samplerate=samplerate)
    
    for num in numbers:
        numfile = al.Sndfile('numbers/%d.wav' % num, 'r')
        numdata = numfile.read_frames(numfile.nframes)
        
        if numfile.channels > wavfile.channels:
            numdata = np.mean(numdata, 1)
        
        wavfile.write_frames(numdata)
        numfile.close()
        
    
    wavfile.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Append random numbers to the end of wav files.')
    parser.add_argument('inputs', metavar='wav', nargs='+', type=str, help='wav files to modify.')
    parser.add_argument('-n', '--numbers', metavar='int', type=int, help='number of numbers to append.')
    args = parser.parse_args()

    for fn in args.inputs:
        append_numbers(fn, args.numbers)
