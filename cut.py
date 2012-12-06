'''
cut.py
natural-mixer

2012 Brandon Mechtley
Arizona State University

Uses information embedded in Sonic Visualizer export file to remove segments on a layer called "cut", crossfading the adjacent sections with an equal-power logarithmic crossfade with length specified by the second command-line argument, in millesconds.

usage: python cut.py input.wav length output.wav
'''

from sys import argv
from scipy.io import wavfile
from soundwalks import *

sw = Soundwalk(argv[1])
env = int(argv[2]) * sw.rate / 1000
of = argv[3]

# Make a list of [begin, end] frames for "cut" segments over which to crossfade.
cut = array(sw.cut)
cut = cut[argsort(cut[:,0]),:]
cut[:,1] += cut[:,0]

# Merge segments.
mergedcut = []
begin, end = cut[0]

for i in range(1, len(cut)):
	if end + env > cut[i, 0]:
		end = cut[i, 1]
	else:
		mergedcut.append([begin, end])
		begin, end = cut[i]

# Add final merged segment.
mergedcut.append([begin, end])
mergedcut = array(mergedcut)

# Clip segments to playable boundaries.
for i in range(0, len(mergedcut)):
	mergedcut[i, 0] = min(max(0, mergedcut[i, 0]), sw.len - 1)
	mergedcut[i, 1] = min(max(0, mergedcut[i, 1]), sw.len - 1)
	
cut = mergedcut

# Crossfade windows. Equal-power logarithmic crossfade.
windowon = sqrt((1 - cos(linspace(0, pi, env))) / 2)
windowoff = sqrt(1 - windowon)

# Window and overlap-add the segments.
newframes = zeros(sw.len, dtype=sw.frames.dtype)
pointer = 0

for i in range(len(cut) + 1):
	# Beginning/ending/duration of the uncut segment.
	begin = cut[i - 1][1] + 1 if i > 0 else 0
	end = cut[i][0] - 1 if i < len(cut) else sw.len - 1
	dur = end - begin
	
	# Window the uncut segment if it's long enough (should be after merge).
	if end - begin >= env:
		sw.frames[begin:begin + env] *= windowon
		sw.frames[end - env:end] *= windowoff
		
		newframes[pointer:pointer + dur] += sw.frames[begin:end]
		pointer += sw.frames[begin:end].shape[0] - env

# Clip crossfaded output and write it to disk.
newframes = newframes[:pointer]
nf = array(newframes, dtype=np.float)
nf /= max(nf)

wavfile.write(of, sw.rate, newframes)
