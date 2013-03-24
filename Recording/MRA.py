"""
Recording/MRA.py
envmixer

2012 Brandon Mechtley
Arizona State University
"""

import pywt
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as pp

from .Recording import Recording

class MRA(Recording):
    """
    MRA tree creation/manipulation functions for a Recording.
    """
    
    def __init__(self, filename):
        super(Recording, self).__init__(filename)
        
    def calculate_mra(self, wavelet='db10', mode='per'):
        """
        Creates an MRA wavelet tree on the recording.
        
        Args:
            wavelet (str): wavelet to use. Any string supported by PyWavelets will work.
            mode (str): method for handling overrun. Default "per," start over at the beginning of the waveform
                (periodic).
        """
        
        self.wavelet, self.mode = wavelet, mode
        self.dwt = pywt.wavedec(self.wav, wavelet, mode=mode, level=int(np.log2(len(self.wav))) + 1)
        
        for l in range(22):
            self.dwt[l] = self.dwt[l] * 0.5 + np.median(self.dwt[l]) * 0.5
    
    def reconstruct_mra(self):
        """
        Bake an MRA into a one-dimensional waveform.
        """
        
        wav = pywt.waverec(self.dwt, self.wavelet, mode=self.mode)
        max16 = np.iinfo(np.int16).max
        wf.write('blah.wav', self.rate, np.int16((self.wav - wav[:len(self.wav)]) * max16))
        pp.specgram(wav)
        pp.show()
