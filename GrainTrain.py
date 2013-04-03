# Standard.
import os, bz2

# Pylab.
import numpy as np
import matplotlib as mpl

# Local.
from barycentric import lattice, baryedges, bary2cart
from Recording import Recording, plot_waveform

def make_tone_train(tones, traindur=10.0, tonedur=1.0, amplitude=1.0):
    """
    Construct a grain train from a number of source tones. Each train is created by a number of half-overlap clips
    of length tonedur.
    
    Args:
        tones (Recording.Recording): source recordings.
        traindur (number): length of the entire train in seconds.
        tonedur (number): length of each tone in seconds. The actual length of the grains be this multiplied by two for
            half-overlap.
        amplitude (number): amplitude multiplier for the tones. 1.0 = 0dB.
    
    Returns:
        A new instance of GrainTrain.GrainTrain.
    """
    
    trainlen = traindur * tones[0].rate
    tonelen = tonedur * tones[0].rate
    
    train = GrainTrain()
    train.set_sources(tones)
    train.grains = [Grain() for i in range(int((float(trainlen) / tonelen) * 2))]
    
    for i in range(len(train.grains)):
        train.grains[i].dur = tonelen * 2
        train.grains[i].src = np.random.randint(len(train.sources))
        train.grains[i].srcpos = 0
        train.grains[i].outpos = 0 if i == 0 else train.grains[i - 1].outpos + train.grains[i - 1].dur / 2
    
    return train

def make_simple_train(coords, sounds, length=10, graindur=(500, 2000), jumpdev=60):
    """
    Simplest mixing algorithm. Creates a new grain train, each grain selected from one of the source recording. The
    source recording is randomly selected, weighted according to which recording is closest to the input coordinates. 
    Each grain has a random duration, sampled from a uniform distribution on the specified interval. Each grain is
    copied from that point in the selected source recording that is closest to  the input coordinates with a random
    offset, selected from a normal distribution with mean = 0, variance = jumpdev seconds.

    Args:
        coords (numpy.ndarray): barycentric coordinates of the mix in the uniform polygon formed by the sources as 
            edges. 
        sounds (list): list of instances of Recording.Recording, one per source edge.
        length (number): duration of grain train in seconds.
        graindur ((number, number)): range of duration of grains in milliseconds, (low, high)
        jumpdev (number): standard deviation of grain-to-grain source offset in seconds. 
    
    Returns:
        A new instance of GrainTrain.GrainTrain.
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
    
    train = GrainTrain()
    train.set_sources(sounds)
    
    # Create list of sound grains. Eeach grain is a tuple of frame numbers: (output frame offset, duration in frames,
    # source recording frame offset, source # (0-2))
    pos = 0
    
    while pos < train.rate * length:
        g = Grain()
        
        # Random source.
        g.src = int(np.random.choice(range(len(train.sources)), p=prob))
        g.dur = int(np.random.randint(graindur[0] / 1000. * train.rate, graindur[1] / 1000. * train.rate))
        
        # Random offset from closest source point.
        jump = int(np.random.randn() * train.rate * jumpdev)
        frame = int(percs[g.src] * train.sources[g.src].len + train.sources[g.src].start)
        
        g.srcpos = frame + jump
        g.srcpos = int(min(g.srcpos, train.sources[g.src].end - g.dur * train.rate))
        g.srcpos = int(max(g.srcpos, train.sources[g.src].start))
        
        g.outpos = 0
        train.grains.append(g)
        
        # If this isn't the first grain, overlap the grains to crossfade.
        if len(train.grains) > 1:
            fadedur = int(min(train.grains[-1].dur, train.grains[-2].dur) / 2)
            train.grains[-1].outpos = train.grains[-2].outpos + train.grains[-2].dur - fadedur
        
        pos = train.grains[-1].outpos + train.grains[-1].dur
    
    train.grains[-1].dur = int(min(train.grains[-1].dur, train.rate * length - train.grains[-1].outpos))
    
    return train

class Grain:
    """
    Simple structure to store information for each grain.
    
    Properties:
        outpos (int): position in the grain train in samples.
        srcpos (int): position in the source recording in samples.
        dur (int): duration in samples.
        src (int): source index from the grain train's list of sources.
        data (numpy.ndarray): samples for the grain.
        env (numpy.ndarray): amplitude envelope for the grain. Same shape as data.
    """
    
    def __init__(self):
        self.outpos, self.srcpos, self.dur, self.src = -1, -1, -1, -1
        self.data = None
        self.env = None
    
    def __str__(self):
        return 'outpos: %d, srcpos: %d, dur: %d, src: %d' % (self.outpos, self.srcpos, self.dur, self.src)

class GrainTrain:
    """
    Class that represents a sequence of grains with samples sourced from a number of source recordings.
    
    Properties:
        grains (list): list of instances of Grain. These are the grains that are sequenced in the grain tarin.
        sources (list): list of instances of Recording.Recording. These are the sources from which each grain obtains
            its samples.
        sound (audiolab.Sndfile): the final mixed sound to be output (via playing or writing to disk).
        basename (str): base path of the grain train minus extension. This tells the saving methods where to save.
    """
    
    def __init__(self):
        self.grains = []
        self.sources = []
        self.sound = None
        self.rate = None
        self.basename = ''
    
    def set_sources(self, sources):
        """
        Set the source list. Using this method is preferred to setting self.sources directly, as it ensures that every
        source has the same sampling rate and updates the grain train's sampling rate accordingly.
        
        Args:
            sources (list): list of instances of Recording.Recording.
        """
        
        for i, s in enumerate(sources):
            assert s.rate == sources[0].rate, "Source %d's rate (%f) must equal Source 0's rate (%f)." % (
                i, s.rate, sources[0].rate
            )
        
        self.rate = sources[0].rate
        self.sources = sources
    
    def save_svl(self):
        """
        Save a SonicVisualiser/Annotator annotations file, treating each grain as a segment. The filename will be
        self.basename + '.svl', so make sure self.basename is set first.
        """
        
        assert len(self.basename), '.basename must be set before calling .save_svl()'
        
        wavname = self.basename + '.wav'
        
        outstr = open('data/template.xml', 'r').read() % (
            wavname,
            self.rate,
            self.grains[-1].outpos + self.grains[-1].dur,
            os.path.join(os.getcwd(), wavname),
            self.rate,
            self.grains[0].outpos,
            self.grains[-1].outpos + self.grains[-1].dur,
            len(self.grains),
            '\n'.join([
                '<point frame="%d" value="%d" duration="%d" label=""/>' % (g.outpos, i, g.dur)
                for i, g in enumerate(self.grains)
            ]),
            (self.grains[-1].outpos + self.grains[-1].dur) / 2
        )
        
        f = open(self.basename + '.svl', 'wb')
        f.write(bz2.compress(outstr))
        f.close()
    
    def mixdown(self, envtype='cosine'):
        """
        Set self.sound to an instance of Recording.Recording, the final mixed sound file for the grain train.
        
        Args:
            envtype (str): Type of envelope to use for overlapping grains. Envelope types are as follows:
                'cosine': cosine / equal-power crossfade.
                'linear': linear crossfade.
        
        Returns:
            self.sound, an instance of Recording.Recording, the final mixed recording.
        """
        
        self.sound = Recording()
        
        if len(self.basename):
            self.sound.filename = self.basename + '.wav'
        
        self.sound.rate = self.rate
        
        for i, g in enumerate(self.grains):
            g.data = np.array(self.sources[g.src].wav[g.srcpos:g.srcpos + g.dur])
            g.env = np.ones(g.data.shape)
            
            # Apply window halves exiting previous grain, entering current grain.
            if i > 0:
                p = self.grains[i - 1]
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
        
        length = max([g.outpos + g.dur for g in self.grains])
        self.sound.wav = np.zeros(length, dtype=self.grains[0].data.dtype)
        
        for g in [g for g in self.grains if len(g.data)]:
            self.sound.wav[g.outpos:g.outpos + g.dur] += g.data * g.env
        
        self.sound.len = len(self.sound.wav)
        
        return self.sound
    
    def save_plot(self):
        """
        Save a set of plots to a PDF. The filename will be self.basename + '.pdf', so ensure that self.basename is set
        before calling this.
        
        Three subplots are included:
            1. An illustration of the grain train, with envelopes and waveforms for each grain, colored according to
                their sources.
            2. The waveform of the final mixed recording.
            3. The spectrogram of the final mixed recording.
        """
        
        mpl.pyplot.figure(figsize=(16,8))
        
        # 1. Plot grain train.
        mpl.pyplot.subplot(311)
        
        mpl.pyplot.title('simple grain train')
        self.plot_train()
        mpl.pyplot.xlim(0, self.sound.len / self.sound.rate)
        mpl.pyplot.xticks([])
        mpl.pyplot.yticks([])
        mpl.pyplot.ylabel('enveloped grains')
        mpl.pyplot.gca().yaxis.set_label_coords(-.04, 0.5)
        
        lines = [
            mpl.lines.Line2D(
                range(1),
                range(1),
                color=mpl.cm.gist_rainbow(float(i) / (len(self.sources) - 1)),
                markerfacecolor=mpl.cm.gist_rainbow(float(i) / (len(self.sources) - 1)),
                markeredgecolor=mpl.cm.gist_rainbow(float(i) / (len(self.sources) - 1)),
                alpha=0.5,
                marker='s'
            )
            for i in range(len(self.sources))
        ]
        
        mpl.pyplot.legend(
            lines,
            [os.path.splitext(os.path.split(s.filename)[1])[0] for s in self.sources],
            numpoints=1,
            loc='upper right',
            bbox_to_anchor=(1.0, 1.2125),
            markerscale=1.5,
            ncol=len(self.sources),
            handlelength=0,
            frameon=False
        )
        
        # 2. Plot mixed waveform.
        mpl.pyplot.subplot(312)
        plot_waveform(
            self.sound.wav,
            512, 256,
            xmin=0, xmax=self.sound.len / self.sound.rate,
            ymin=-1, ymax=1,
            npoints=5,
        )
        
        mpl.pyplot.xlim(0, self.sound.len / self.sound.rate)
        mpl.pyplot.xticks([])
        mpl.pyplot.ylabel('mixed waveform')
        mpl.pyplot.gca().yaxis.set_label_coords(-.04, 0.5)
        
        # 3. Plot mixed spectrogram.
        mpl.pyplot.subplot(313)
        pxx, freqs, bins, im = mpl.pyplot.specgram(self.sound.wav, Fs=self.sound.rate)
        mpl.pyplot.xlim(0, self.sound.len / self.sound.rate)
        mpl.pyplot.ylim(freqs[0], freqs[-1])
        mpl.pyplot.yticks(range(0, 20001, 5000), range(0, 21, 5))
        mpl.pyplot.ylabel('frequency (KHz)')
        mpl.pyplot.xlabel('time (s)')
        mpl.pyplot.gca().yaxis.set_label_coords(-.04, 0.5)

        mpl.pyplot.savefig(self.basename + '.pdf')
    
    def plot_train(self, framesize=512, hopsize=256, cmap=mpl.cm.gist_rainbow, npoints=3):
        """
        Plot an illustration of the grain train to the current axes. The plot includes envelopes and waveforms for each
        grain, colored according to their sources.
        
        Args:
            framesize (int): number of samples per frame in the waveform plots.
            hopsize (int): number of samples between frames in the waveform plots.
            cmap (matplotlib.colors.Colormap): color map to use for coloring waveforms.
            npoints (int): number of percentile points to use for shading the waveforms.
        """
    
        sources = np.arange(len(self.sources))
        nsrc = len(sources)
        maxsrc = float(np.amax(sources))
        minsrc = float(np.amin(sources))
        
        oldpos = 0
        layer = False
        
        for i, g in enumerate(self.grains):
            # If two grains overlap, draw them stacked amongst two layers.
            if oldpos > g.outpos:
                layer = not layer
    
            oldpos = g.outpos + len(g.data)
    
            ymin = 0 if not layer else 0.5
            xmin, xmax = g.outpos / self.rate, (g.outpos + len(g.data)) / self.rate
    
            # Resample envelope.
            envx = np.linspace(xmin, xmax, len(g.env) / framesize)
            envmax = np.interp(envx, np.linspace(xmin, xmax, len(g.env)), g.env)
            envmax = np.interp(envmax, [0, 1], [ymin+.25, ymin+.5])
    
            # Draw envelope on top of waveform.
            mpl.pyplot.fill_between(
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
        tmin = np.amin([g.outpos for g in self.grains]) / self.rate
        tmax = np.amax([g.outpos + len(g.data) for g in self.grains]) / self.rate

        mpl.pyplot.ylim(-0.05, 1.05)
        mpl.pyplot.xlim(tmin, tmax)
