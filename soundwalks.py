'''
soundwalks.py
natural-mixer

Defines a class to describe a Soundwalk, annotated with Sonic Visualizer, along with a number of helpful mathematical functions.

Two layers are interpreted at the moment, one titled "cut," which will specify segments to cut from the recording in resynthesis, such as speech. Another layer, "jasa-el" specifies portions of the recordings to use for resynthesis in an upcoming JASA-EL letter.
'''

from xml.dom.minidom import parse, parseString
from pylab import *
from os.path import splitext
import wave
import bz2

def percline(p0, p1, p2):
	'''Calculate the percentage of distance of point p0 along line segment p1->p2.'''
	
	p1p0 = norm(p0 - p1)
	p1p2 = norm(p2 - p1)
	
	return p1p0 / p1p2

def closepoint(p0, p1, p2):
	'''Return the closest point on the line segment p0->p1 to point p2.'''
	
	x1, y1 = p0
	x2, y2 = p1
	x3, y3 = p2
	
	u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
	x = x1 + u * (x2 - x1)
	y = y1 + u * (y2 - y1)
	
	return (x, y)


def choice(a, size=1, replace=True, p=None):
	'''Copy of NumPy's choice implementation. Choose an element of a at random. If a is a number, 
	treat as arange(a). p defines a discrete probably distribution over the element indices.'''
	
	# Format and Verify input
	if isinstance(a, int):
		if a > 0:
			pop_size = a #population size
		else:
			raise ValueError("a must be greater than 0")
	else:
		a = np.asarray(a)
		if len(a.shape) != 1:
			raise ValueError("a must be 1-dimensional")
		pop_size = a.size
		if pop_size is 0:
			raise ValueError("a must be non-empty")

	if None != p:
		p = np.asarray(p)
		if len(p.shape) != 1:
			raise ValueError("p must be 1-dimensional")
		if p.size != pop_size:
			raise ValueError("a and p must have same size")
		if any(p < 0):
			raise ValueError("probabilities are not non-negative")
		if not np.allclose(p.sum(), 1):
			raise ValueError("probabilities do not sum to 1")

	# Actual sampling
	if replace:
		if None != p:
			cdf = p.cumsum()
			uniform_samples = np.random.random(size)
			idx = cdf.searchsorted(uniform_samples, side='right')
		else:
			idx = self.randint(0, pop_size, size=size)
	else:
		if size > pop_size:
			raise ValueError(''.join(["Cannot take a larger sample than ",
									  "population when 'replace=False'"]))
			
		if None != p:
			if np.sum(p>0) < size:
				raise ValueError("Fewer non-zero entries in p than size")
			n_uniq = 0
			p = p.copy()
			found = np.zeros(size, dtype=np.int)
			while n_uniq < size:
				x = self.rand(size-n_uniq)
				if n_uniq > 0:
					p[found[0:n_uniq]] = 0
				p = p/p.sum()
				cdf = np.cumsum(p)
				new = cdf.searchsorted(x, side='right')
				new = np.unique(new)
				found[n_uniq:n_uniq+new.size] = new
				n_uniq += new.size
			idx = found
		else:
			idx = self.permutation(pop_size)[:size]

	#Use samples as indices for a if a is array-like
	if isinstance(a, int):
		return idx
	else:
		return a.take(idx)

def first_element(n, tag, attr, value):
	'''Helper function that returns the first DOM child node with a specific tag type, tag, and an 
	attribute, attr, that matches a specified value.'''
	
	return [c for c in n.getElementsByTagName(tag) if c.getAttribute(attr) == value][0]

def dataset_from_layer(d, name):
	'''Helper function that returns the associated dataset for a given layer name in a 
	SonicVisualiser XML file.'''
	
	l = first_element(d, 'layer', 'presentationName', name)
	m = first_element(d, 'model', 'id', l.getAttribute('model'))
	
	return first_element(d, 'dataset', 'id', m.getAttribute('dataset'))
		
class Soundwalk:
	'''Class that represents a SonicVisualiser analysis file for a soundwalk.
	These have specific layers:
		cut: Regions layer that defines regions to avoid--speech, coughs, clicks, etc. 
		jasa-el: Region layer containing one region to use for JASA-EL user  study.'''
	
	def __init__(self, s, useseg = True):
		'''Constructor. s is the path to the Sonic Visualiser .sv file, and useseg determines 
		whether or not only the portion of the sound annotated in the jasa-el layer should be 
		used.'''
		
		if (splitext(s)[1]) == '.wav':
			self.wavStart = self.start = 0
			self.wavfile = s
			wav = wave.open(self.wavfile)
			self.nchannels, sw, self.rate, self.wavEnd, ct, cn = wav.getparams()
			self.end = self.wavEnd / self.nchannels
			self.cut = []
			self.len = self.wavEnd / self.nchannels
			dts = {1: np.int8, 2: np.int16, 4: np.int32}
			self.frames = fromstring(wav.readframes(self.wavEnd), dtype=dts[sw])
			
			# Deinterlace.
			if self.nchannels > 1:
				self.frames = array([self.frames[0::2], self.frames[1::2]]).transpose()
		elif splitext(s)[1] == '.sv':
			doc = parseString(bz2.decompress(open(s).read()))
		
			# Wave information.
			w = first_element(doc, 'model', 'type', 'wavefile')
			self.rate = int(w.getAttribute('sampleRate'))
			self.wavStart = int(w.getAttribute('start'))
			self.wavEnd = int(w.getAttribute('end'))
			self.wavfile = w.getAttribute('file')
		
			# Region used for synthesis.
			seg_data = dataset_from_layer(doc, 'jasa-el')
			p = seg_data.getElementsByTagName('point')[0]
		
			if useseg:
				self.start = int(p.getAttribute('frame')) if useseg else 0
				self.len = int(p.getAttribute('duration')) if useseg else self.wavEnd
				self.end = self.start + self.len if useseg else self.wavEnd
			
			# Open only the requested segment of wave file.
			wav = wave.open(self.wavfile)
			self.nchannels, sw, fr, nf, ct, cn = wav.getparams()
		
			dts = {1: np.int8, 2: np.int16, 4: np.int32}		
			wav.setpos(self.start * self.nchannels)
			#print 'reading', self.len * self.nchannels, self.len, self.nchannels
			self.frames = fromstring(wav.readframes(self.len * self.nchannels), dtype=dts[sw])
		
			# Deinterlace.
			if self.nchannels > 1:
				self.frames = array([self.frames[0::2], self.frames[1::2]]).transpose()
		
			# Regions to cut from synthesis.
			cut_data = dataset_from_layer(doc, 'cut')
			self.cut = [
				[int(p.getAttribute('frame')) - self.start, int(p.getAttribute('duration'))] 
				for p in cut_data.getElementsByTagName('point')
			]
