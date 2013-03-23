"""
Recording/Recording.py
envground

2012 Brandon Mechtley
Arizona State University
"""

from os import makedirs
from itertools import izip, count
from os.path import exists

import numpy as np
import scikits.audiolab as audiolab

class Recording(object):
    """
    Base recording class that holds a waveform and its various statistics/properties. Inherited 
    by a few classes that attempt to split figure and ground.
    """
    
    def __init__(self, filename=None):
        if filename:
            self.filename = filename
            self.snd = audiolab.Sndfile(self.filename)
            
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
                self.snd = audiolab.Sndfile(self.filename, samplerate=newrate)
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
            self.snd = audiolab.Sndfile(
                self.filename, 
                mode='rw', 
                format=audiolab.Format(), 
                channels=1, 
                samplerate=self.rate
            )
        
        self.snd.write_frames(self.wav)
        self.snd = audiolab.Sndfile(
            self.filename,
            mode='r'
        )
        
        self.init_after_load()
        
        return self.filename
