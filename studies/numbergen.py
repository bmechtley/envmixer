"""
studies/numbergen.py
envmixer

2013 Brandon Mechtley
Arizona State University

1. Generate 100 4-digit numbers read aloud, sourced from files with paths
numbers/[1-9].wav, saved as numstrs/numstr-[1111-9999].wav. Used for audio
captchas.

2. Also generate 100 15s clips with chained 15 1s sinusoids with random frequencies
in the range 100-1100Hz. Used as trick questions (similarity to any source clip
should be 1/5.)
"""

import scikits.audiolab as al
import numpy as np

numbers = np.array([np.random.randint(low=1, high=10, size=100) for i in range(4)]).transpose()
format = al.Format('wav')

# 1. Generate number strings.
for i, numstr in enumerate(numbers):
    print i, numstr
    
    f = al.Sndfile('numstrs/numstr-%s.wav' % ''.join(str(num) for num in numstr), 'w', format, 2, 44100)
    
    for num in numstr:
        sd, fs, enc = al.wavread('numbers/%d.wav' % num)
        f.write_frames(sd)
    
    f.close()

# 2. Generate random tones.
for i in range(100):
    print i
    
    f = al.Sndfile('tones/tone-%d.wav' % i, 'w', format, 1, 44100)
    
    for j in range(15):
        freq = np.random.ranf() * 1000 + 100
        data = np.sin(np.linspace(0, 1, 44100) * 2 * np.pi * freq) 
        f.write_frames(data)
    
    f.close()

print numbers
