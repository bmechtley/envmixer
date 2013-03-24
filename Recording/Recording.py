"""
Recording/Recording.py
envmixer

2012 Brandon Mechtley
Arizona State University
"""

from os import makedirs
from itertools import izip, count
from os.path import exists

import numpy as np
import scikits.audiolab as al
import matplotlib
import matplotlib.pyplot as pp
from scipy.stats import scoreatpercentile

def plot_waveform(
        wf, 
        framesize=1024, 
        hopsize=512, 
        shades=3, 
        xmin=0, 
        xmax=1, 
        ymin=0, 
        ymax=1,
        clip_ends=False, 
        **kwargs
):
    """
    Plot a waveform in several regions, based on windowed percentiles.

    Args:
        wf (np.array): input waveform
        framesize (int): number of samples per frame (default 1024).
        hopsize (int): number of samples between frame (generally 1/2 framesize) (default 512).
        shades (int): number of regions to draw (tip: to see them, use alpha) (default 3).
        xmin (float): minimum x coordinate to start the waveform (default 0).
        xmax (float): maximum x coordinate to stretch the waveform (default 1).
        ymin (float): minimum y coordinate to start the waveform (default 0).
        ymax (float): maximum y coordinate to stretch the waveform (default 1).
        clip_ends (bool): whether or not to force every region to begin/end with the beginning/ending sample of the
            orgiginal waveform. Use this to force enveloped waveforms to start/end drawing at 0 (default False).
        **kwargs: Keyword arguments to send to matplotlib.pyplot.fill_between.
    """

    perc = np.linspace(0., 100., shades * 2)
    
    percentiles = np.array([
        [
            scoreatpercentile(wf[i:i+framesize], p) 
            for i in range(0, len(wf), hopsize)
        ] 
        for p in perc
    ])
    
    absmax = np.amax(np.abs(percentiles))
    smin, smax = np.amin(percentiles), np.amax(percentiles)
    percentiles = np.interp(percentiles, [-absmax, absmax], [ymin, ymax])
    
    # Simple way to ensure enveloped waveforms begin/end at 0.
    if clip_ends:
        percentiles[:,0] = np.interp(wf[0], [-absmax, absmax], [ymin, ymax])
        percentiles[:,-1] = np.interp(wf[-1], [-absmax, absmax], [ymin, ymax])
    
    #cmap = matplotlib.cm.get_cmap()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100, clip=True)
    mappable = matplotlib.cm.ScalarMappable(norm=norm)
    
    for i in range(len(percentiles) / 2):
        pp.fill_between(
            np.linspace(xmin, xmax, len(percentiles[i])),
            percentiles[i],
            percentiles[len(percentiles) - 1 - i],
            color=mappable.to_rgba(perc[i]),
            **kwargs
        )
    
    return mappable
    
class Recording(object):
    """
    Base recording class that holds a waveform and its various statistics/properties. Inherited 
    by a few classes that attempt to split figure and ground.
    """
    
    def __init__(self, filename=None):
        if filename:
            self.filename = filename
            self.snd = al.Sndfile(self.filename)
            
            self.init_after_load()
        else:
            self.snd = None
    
    def init_after_load(self):
        self.rate = float(self.snd.samplerate)
        self.wav = self.snd.read_frames(self.snd.nframes)
        self.len = len(self.wav)
        self.start = 0
        self.end = self.len
        
        # Mixdown to mono.
        if len(self.wav.shape) > 1:
            self.wav = np.mean(self.wav, 1)
        
    def resample(self, newrate):
        """
        Resample a loaded audio file at a new sampling rate.
        
        Args:
            newrate (int): new sampling rate in Hz.
        """
        
        if newrate != self.rate:
            if self.snd:
                self.snd = al.Sndfile(self.filename, samplerate=newrate)
                self.init_after_load()
            else:
                self.rate = newrate
      
    def save(self):
        """
        Save the recording. If the recording filename does not yet exist, create it. Recording.rate, .filename, and
        .frames must already be set. If the recording was loaded from a file and new created anew, resave it with
        .frames as data.
        
        Returns:
            the filename saved (Recording.filename).
        """
    
        # TODO: Make sure this works with resampling loaded audio.
       
        assert len(self.filename) != 0, 'Recording must have a filename (.filename).'
        assert len(self.wav), 'Recording must have frames (.wav).'
        assert self.rate != 0, 'Recording must have valid samplerate (.rate != 0).'
        
        if not self.snd:
            self.snd = al.Sndfile(
                self.filename, 
                mode='rw', 
                format=al.Format(), 
                channels=1, 
                samplerate=self.rate
            )
        
        self.snd.write_frames(self.wav)
        self.snd = al.Sndfile(
            self.filename,
            mode='r'
        )
        
        self.init_after_load()
        
        return self.filename
