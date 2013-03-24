"""
Recording/FigureGround.py
envmixer

2013 Brandon Mechtley
Arizona State University

Base abstract class for Figure/Ground separation classes. Do not instatiate directly.
"""

from os.path import exists, basename, splitext
from os import makedirs

import numpy as np
import scikits.audiolab as al

from .Recording import Recording

class FigureGround(Recording):
    def __init__(self, filename):
        super(FigureGround, self).__init__(filename)
        
        self.figwav = None
        self.gndwav = None
    
    def save(self, prefix='', suffix='', progress=None, ground=True, figure=True, composite=True):
        """
        Save the computed figure, ground, and composite waveforms to disk. The output filenames will be of the form
        [input]-figure.wav, [input]-ground.wav, and [input]-composite.wav, including the prefix and suffix parameters.
        Must have already used relevant figure/ground separation methods before calling this, such that self.gndwav
        and self.figwav exist.
        
        Args:
            prefix (str): path prefix for all output files. If the empty string, files will be saved in the current
                working directory (default '').
            suffix (str): path suffix for all output files. If the empty string, there will be no suffix (default '').
            progress (progressbar.ProgressBar or None): optional progress bar for debug output (default None).
            ground (bool): whether or not to save the ground waveform (default True).
            figure (bool): whether or not to save the figure waveform (default True).
            composite (bool): whether or not to save the composite waveform (default True).
            
        Returns:
            List of filenames saved to disk.
        """
        
        if not exists(prefix):
            makedirs(prefix)
        
        base = splitext(basename(self.filename))[0]
        
        if len(suffix):
            suffix = '-' + suffix
        
        use = np.array([ground, figure, composite])
        names = np.array([prefix + base + suffix + '-' + nm + '.wav' for nm in ['ground', 'figure', 'composite']])[use]
        wavs = np.array([self.gndwav, self.figwav, self.gndwav + self.figwav])[use]
        
        if progress:
            progress.maxval = 3
            progress.start()
        
        for i, (fn, wav) in enumerate(zip(names, wavs)):
            snd = al.Sndfile(fn, mode='w', format=al.Format(), channels=1, samplerate=self.rate)
            snd.write_frames(wav)
            snd.close()
            
            if progress:
                progress.update(i + 1)

        if progress:
            progress.finish()
        
        return names
