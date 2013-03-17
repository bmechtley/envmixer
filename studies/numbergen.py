import scikits.audiolab as al
import numpy as np

numbers = np.array([np.random.randint(low=1, high=10, size=100) for i in range(4)]).transpose()
format = al.Format('wav')

# Generate number strings.
for i, numstr in enumerate(numbers):
    print i, numstr
    
    f = al.Sndfile('numstrs/numstr-%s.wav' % ''.join(str(num) for num in numstr), 'w', format, 2, 44100)
    
    for num in numstr:
        sd, fs, enc = al.wavread('numbers/%d.wav' % num)
        f.write_frames(sd)
    
    f.close()

# Generate random tones.
for i in range(100):
    print i
    
    f = al.Sndfile('tones/tone-%d.wav' % i, 'w', format, 1, 44100)
    
    for j in range(15):
        freq = np.random.ranf() * 1000 + 100
        data = np.sin(np.linspace(0, 1, 44100) * 2 * np.pi * freq) 
        f.write_frames(data)
    
    f.close()

print numbers
