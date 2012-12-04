'''mixer.py
	Usage: python mixer.py soundwalk0.sv soundwalk1.sv soundwalk2.sv x y duration
	
	Creates a mixed environmental sound recording from 3 source recordings. Source recordings are 
	assumed to be of equal duration. The paths that define along which they were recorded are fake, 
	organized along on equilateral triangle:
	
				 p1 (.5, sin(pi / 3))
			   /	\
		p0 (0,0) --	p2 (1, 0)
	
	Soundwalk 0: p0->p1
	Soundwalk 1: p1->p2
	Soundwalk 2: p2->p0
	
	The mixed synthesized version will be fixed at coordinates (x, y) and last for the specified 
	duration (in seconds).
	
	Various methods are used for resynthesis. See the appropriate functions for details. The basic 
	rule of thumb is that the closer to a given walk the cursor is, the more similar the 
	synthesized version will be to it. Additionally, the output should be most similar to the 
	region of each soundwalk that is closest to the cursor.'''

from sys import argv
from scipy.io import wavfile
from soundwalks import *
import os
import bz2

class Grain: 
	def __str__(self):
		return '%d %d %d %d' % (self.outpos, self.srcpos, self.dur, self.src)

def output_grain_annotations(grains, rate, wavname, filename):
	outstr = '<?xml version="1.0" encoding="UTF-8"?>\n\
<!DOCTYPE sonic-visualiser>\n\
<sv>\n\
<data>\n\
	<model id="0" name="%s" sampleRate="%d" start="0" end="%d" type="wavefile" file="%s" mainModel="true"/>\n\
	<playparameters mute="false" pan="0" gain="1" pluginId="" model="0"/>\n\
	<model id="2" name="" sampleRate="%d" start="%d" end="%d" type="sparse" dimensions="3" resolution="1" notifyOnAdd="true" dataset="1"  subtype="region" valueQuantization="0" minimum="0" maximum="%d" units=""/>\n\
	<dataset id="1" dimensions="3">\n\
		%s\n\
	</dataset>\n\
	<layer id="3" type="timeruler" name="Ruler" model="0" colourName="Black" colour="#000000" darkBackground="false" />\
	<layer id="4" type="waveform" name="Waveform" model="0" gain="1" showMeans="1" greyscale="0" channelMode="0" channels="-1" scale="0" aggressive="0" autoNormalize="0" colourName="Black" colour="#000000" darkBackground="false" />\
	<layer id="5" type="spectrogram" name="Spectrogram" model="0" channel="-1" windowSize="1024" windowHopLevel="2" gain="1" threshold="0" minFrequency="10" maxFrequency="0" colourScale="3" colourScheme="0" colourRotation="0" frequencyScale="0" binDisplay="0" normalizeColumns="false" normalizeVisibleArea="false"/>\
	<layer id="6" type="regions" name="Regions" model="2" verticalScale="1" plotStyle="0" colourName="Bright Blue" colour="#1e96ff" darkBackground="true" />\
</data>\n\
<display>\
	<window width="1680" height="1002"/>\
	<view centre="%d" zoom="256" followPan="1" followZoom="1" tracking="page" type="pane" centreLineVisible="1" height="884" >\
		<layer id="3" type="timeruler" name="Ruler" model="0" visible="true"/>\
		<layer id="4" type="waveform" name="Waveform" model="0" visible="true"/>\
		<layer id="5" type="spectrogram" name="Spectrogram" model="0" visible="true"/>\
		<layer id="6" type="regions" name="Regions" model="2" visible="true"/>\
	</view>\
</display>\
<selections>\
</selections>\
</sv>\n' % (
		wavname, 
		rate, 
		grains[-1].outpos + grains[-1].dur,
		os.path.join(os.getcwd(), wavname), 
		rate,
		grains[0].outpos,
		grains[-1].outpos + grains[-1].dur,
		len(grains),
		'\n\t\t\t\t'.join([
			'<point frame="%d" value="%d" duration="%d" label=""/>' % (g.outpos, i, g.dur) 
			for i, g in enumerate(grains)
		]),
		(grains[-1].outpos + grains[-1].dur) / 2
	)
	
	f = open(filename, 'wb')
	f.write(bz2.compress(outstr))
	f.close()
	
def simple_grain_train(coords, sounds, length = 10, graindur = [500, 2000], jumpdev=60):
	'''Simplest synthesis algorithm. Creates a sequence of overlapping grains, each selected from a 
	different source recording. The source recording is randomly selected, weighted according to 
	which recording is closest to the input coordinates.
	
	Each grain has a random duration, sampled from a beta distribution (a = 2, b = 5) on the 
	interval [100, 2000] milliseconds. Each grain is copied from that point in the selected source 
	recording that is closest to the input coordinates with a random offset, selected from a normal 
	distribution with mean = 0, variance = 60 seconds.'''
	
	# List of grains, each grain is a tuple of frame numbers: (output frame offset, duration in 
	# frames, source recording frame offset, source # (1-3))
	grains = []
	
	# List of start and end points for each walk.
	p0 = array([(0, 0), (0.5, sin(pi / 3)), (1, 0)])
	p1 = array([(0.5, sin(pi / 3)), (1, 0), (0, 0)])
	
	# List of closest points to the coordinates for each walk.
	cp = array([closepoint(p0[i], p1[i], coords) for i in range(3)])
	
	# List of percentage along segment for each walk.
	perc = array([percline(cp[i], p0[i], p1[i]) for i in range(3)])
	
	# Probability of playing each walk / mixing %.
	prob = array([norm(cp[i] - coords) for i in range(3)])
	prob = max(prob) - prob
	prob /= sum(prob)
	
	# Create list of sound grains.
	rate = min([s.rate for s in sounds])
	pos = ol = 0
	
	while pos < rate * length:
		g = Grain()
		
		# Random source.
		g.src = choice(3, p = prob)[0]
		g.dur = randint(graindur[0] * rate, graindur[1] * rate)
		
		# Random offset from closest source point.
		jump = int(randn() * rate * jumpdev)
		frame = int(perc[g.src] * sounds[g.src].len + sounds[g.src].start)
		g.srcpos = frame + jump
		g.srcpos = min(g.srcpos, sounds[g.src].end - (g.dur / 1000.) * rate)
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
	'''Write a synthesized version from a sequence of grains from different sources, with optional 
	overlap.'''
	
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
			
			p.data[-fadedur:] *= sqrt(1 - half)
			g.data[:fadedur] *= sqrt(half)
	
	for g in grains:
		sound[g.outpos:g.outpos + g.dur] += g.data
	
	wavfile.write(filename, rate, sound)
	
def main():
	# Load commandline arguments.
	sounds = [Soundwalk(argv[i]) for i in [1, 2, 3]]
	coords = array([float(argv[i]) for i in [4, 5]])
	length = float(argv[6])
	graindur = [float(argv[7]), float(argv[8])]
	jumpdev = float(argv[9])
	outputwav = argv[10].replace('\ ', ' ')
	
	grains = simple_grain_train(coords, sounds, length, graindur, jumpdev)
	
	output_grain_train(sounds, grains, outputwav)
	
	if len(argv) > 11:
		outputsvl = argv[11].replace('\ ', ' ')
		output_grain_annotations(grains, sounds[0].rate, outputwav, outputsvl)
	
if __name__=='__main__':
	main()
	