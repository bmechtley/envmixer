"""
Recording/REPET.py
envground

2012 Brandon Mechtley
Arizona State University

REPET-SIM algorithm used with permission by:
Zafar Rafii and Bryan Pardo
Northwestern University
http://music.cs.northwestern.edu/
"""

import numpy as np
import scipy.spatial.distance as dist

from .FigureGround import FigureGround
from .STFT import STFT, ISTFT

def sigmoid(x, t, h):
    """
    :type x: numpy.ndarray
    :param x: input array to be thresholded with values in [0, 1]
    :type t: number
    :param t: threshold in location [0, 1].
    :type h: number
    :param h: hardness in [0, 1], such that h=0 corresponds to no change and h=1 corresponds to a binary step.
    
    Sigmoid function as in REPET-SIM.
    """
    
    y = np.array(x)
    
    if h == 0:
        return y
    elif h == 1:
        return y > t
    
    gt = x > t
    lt = x < t
    
    y[gt] = 1 - ((1 - y[gt]) / (1 - t) ** h) ** (1 / (1 - h))
    y[lt] = (y[lt] / (t ** h)) ** (1 / (1 - h))
    
    return y

class REPET(FigureGround):
    """
    Recording subclass for REPET-SIM-inspired figure/ground separation. REPET-SIM compares STFT frames and creates a
    ground spectrogram by taking the median of similar frames. See Z. Rafii and B. Pardo, "Music/Voice Separation Using
    the Similarity Matrix," ISMIR 2012, pp. 583-588
    """
    
    def __init__(self, filename):
        """
        Initialization.
        """
        
        super(REPET, self).__init__(filename)
        
    def compute_stft(self, nfft=4096, nhop=2048, window=np.hanning, progress=None):
        """
        :type nfft: int
        :param nfft: optional size of the STFT analysis window in samples (default 4096).
        :type nhop: int
        :param nfft: optional number of samples between consecutive frames (default 2048).
        :type window: function
        :param window: optional windowing function, e.g. from numpy (default numpy.hanning).
        :type progress: progress.ProgressBar
        :param progress: optional progress bar for debug output (default None).
        
        Compute the Short-Time Fourier Transform of the waveform.
        """
        
        self.nfft = nfft
        self.nhop = nhop
        self.window = window
        
        self.stft = STFT(self.wav, nfft, nhop, window, progress=progress)
        self.pstft = np.abs(np.array(self.stft[:,:self.nfft / 2 + 1])) ** 2.0
        self.nframes = len(self.pstft)
        self.specsize = self.pstft.shape[1]

    def comparedict(self, bands=1, threshold=0, hardness=0, minsim=0, maxframes=0, windowlen=None, progress=None):
        """
        :type bands: int
        :param bands: optional number of evenly distributed frequency bands over which to separate the similarity /
            separation process (default 1).
        :type threshold: number
        :param threshold: optional center threshold in [0, 1] for sigmoid applied to similarity matrices (default 0).
        :type hardness: number
        :param hardness: optional hardness of sigmoid applied to the similarity matrices. hardness=0 will make no
            change; hardness=1 will binarize (default 0).
        :type minsim: number or "eps"
        :param minsim: if nonzero, similarity values will be linearly interpolated to make this value the minimum. If
            "eps," the smallest possible value for the similarity matrix datatype will be used.
        :type progress: progress.ProgressBar
        :param progress: optional progress bar for debug output (default None).
        
        Compute frame-by-frame similarity matrix according to the cosine similarity between 
        normalized spectrogram frames. Similarity is scaled from 0 to 1.
        """

        self.nbands = bands

        # 1. Calculate lower and upper bin indices for each band.
        bw = self.specsize / bands
        self.bandindices = [
            (b * bw, self.specsize if b == self.nbands - 1 else b * bw + bw)
            for b in range(bands)
        ]
        
        self.bandsim = {band: {} for band in range(bands)}
        
        if type(minsim) == str:
            minsim = np.finfo(self.pstft.dtype).eps

        if progress:
            progress.maxval = bands * self.pstft.shape[0]
            progress.start()

        for band in range(bands):
            l, r = self.bandindices[band]
            
            for frame in range(self.pstft.shape[0]):
                self.bandsim[band][frame] = {
                    f: np.interp(
                        dist.cosine(self.pstft[f,l:r], self.pstft[frame,l:r]),
                        [0, 1],
                        [minsim, 1]
                    )
                    for f in range(max(0, frame - windowlen / 2), min(frame + windowlen / 2, self.pstft.shape[0]))
                }
                
                if progress: progress.update(band * self.pstft.shape[0] + frame)

        if progress: progress.finish()
        
    def compare(self, bands=1, threshold=0, hardness=0, minsim=0, maxframes=0, windowlen=None, progress=None):
        """
        :type bands: int
        :param bands: optional number of evenly distributed frequency bands over which to separate the similarity /
            separation process (default 1).
        :type threshold: number
        :param threshold: optional center threshold in [0, 1] for sigmoid applied to similarity matrices (default 0).
        :type hardness: number
        :param hardness: optional hardness of sigmoid applied to the similarity matrices. hardness=0 will make no
            change; hardness=1 will binarize (default 0).
        :type minsim: number or "eps"
        :param minsim: if nonzero, similarity values will be linearly interpolated to make this value the minimum. If
            "eps," the smallest possible value for the similarity matrix datatype will be used.
        :type progress: progress.ProgressBar
        :param progress: optional progress bar for debug output (default None).
        
        Compute frame-by-frame similarity matrix according to the cosine similarity between 
        normalized spectrogram frames. Similarity is scaled from 0 to 1.
        """
        
        self.nbands = bands
        
        # 1. Calculate lower and upper bin indices for each band.
        bw = self.specsize / bands
        self.bandindices = [
            (b * bw, self.specsize if b == self.nbands - 1 else b * bw + bw) 
            for b in range(bands)
        ]
        
        # 2. Compute similarity for each band.
        self.bandsim = [np.zeros((self.nframes, self.nframes)) for a in range(bands)]
        
        if type(minsim) == str:
            minsim = np.finfo(self.bandsim[0].dtype).eps

        if progress:
            progress.maxval = bands
            progress.start()
        
        for band in range(bands):
            l, r = self.bandindices[band]
                
            self.bandsim[band] = sigmoid(
                1.0 - dist.squareform(dist.pdist(self.pstft[:,l:r], metric='cosine')),
                threshold,
                hardness
            )
            
            self.bandsim[band] = np.interp(self.bandsim[band], [0, 1], [minsim, 1])
            
            if progress: progress.update(band + 1)
            
        if progress: progress.finish()
    
    def separatedict(self, minlag=0, maxlag=0, maxframes=0, minsim=0, threshold=0, hardness=0, progress=None):
        """
        :type minlag: int
        :param minlag: optional minimum distance between consecutive frames to consider as similar (default 0).
        :type maxlag: int
        :param maxlag: optional maximum distance between consecutive frames to consider as similar (default ).
        :type maxframes: int
        :param maxframes: optional maximum number of frames to use in median and/or weighted mean (default 0).
        :type minsim: number
        :param minsim: optional minimum similarity to consider. Higher values can make sorting faster (default 0).
        :type threshold: number
        :param threshold: optional center threshold in [0, 1] for sigmoid function applied to TF mask (default 0).
        :type hardness: number
        :param hardness: optional hardness of sigmoid function applied to TF mask. hardness=0 will make no change,
            hardess=1 will create a binary mask (default 0).
        :type progress: progressbar.ProgressBar
        :param progress: optional progress bar for debug output (default None).
        
        Separate figure from ground using precomputed band similarity matrices by either taking a weighted mean of
        similar frames or the median of similar frames.
        """
        
        self.repeating = np.zeros(self.pstft.shape)
        
        # 1. Convert minimum and maximum distance between similar frames from seconds to frames.
        minlag = int(minlag * self.rate / self.nhop)
        maxlag = int(maxlag * self.rate / self.nhop) if maxlag > 0 else self.nframes
        #repairlen = int(repairlen * self.rate / self.nhop)
        
        if minsim == 'eps': minsim = np.finfo(self.pstft.dtype).eps
        
        if progress:
            progress.maxval = self.nbands * self.nframes
            progress.start()
        
        # 3. Loop through each band, filling the repeating ground spectrogram.
        for band in range(self.nbands):
            sim =self.bandsim[band]
            l, r = self.bandindices[band]
            
            # 3.a. Compute repeating mixture frame for each frame.
            for i in range(len(sim)):
                indices = np.array(sim[i].keys()) 
                values = np.array(sim[i].values())#[indices]
                
                # 3.a.i. Remove indices with too low of similarity values.
                gt = values >= minsim
                indices = indices[gt]
                values = values[gt]
                
                # 3.a.ii. Remove our index.
                noti = indices != i
                indices = indices[noti]
                values = values[noti]
                
                # 3.a.iii. Sort.
                sortind = np.argsort(values)[::-1]
                indices = indices[sortind]
                
                # 3.a.iv. Erase indices that are too close together.
                cleaned = 0
                for j in range(len(indices)):
                    if indices[j] >= 0:
                        distances = abs(indices - indices[j])
                        indices[np.bitwise_and(distances < minlag, indices != indices[j])] = -1
                        cleaned += 1
                    
                    if 0 < maxframes <= cleaned:
                        break
                
                indices = indices[indices >= 0]
                
                # 3.a.v. Limit the number of frames over which to take median/mean.
                if maxframes:
                    indices = indices[:maxframes]
                
                # 3.a.vi. Create repeating ground spectrogram.
                self.repeating[i, l:r] = np.median(self.pstft[indices,l:r], 0)
                
                if progress: progress.update(band * self.nframes + i + 1)
        
        if progress: progress.finish()
        
        # 4. Create repeating background mask, enforcing nonnegativity.
        self.mask = np.amin((self.repeating, self.pstft), 0)
        self.mask /= self.pstft
        self.mask = sigmoid(self.mask, threshold, hardness)
        self.mask = np.concatenate((self.mask, np.fliplr(self.mask[:,1:-1])), axis=1)
        
        # 5. Create ground and figure waveforms.
        gndstft = self.mask * self.stft
        
        self.gndwav = ISTFT(gndstft, self.nhop, len(self.wav))
        self.figwav = self.wav - self.gndwav
    
    def separate(self, minlag=0, maxlag=0, maxframes=0, minsim=0, threshold=0, hardness=0, progress=None):
        """
        :type minlag: int
        :param minlag: optional minimum distance between consecutive frames to consider as similar (default 0).
        :type maxlag: int
        :param maxlag: optional maximum distance between consecutive frames to consider as similar (default ).
        :type maxframes: int
        :param maxframes: optional maximum number of frames to use in median and/or weighted mean (default 0).
        :type minsim: number
        :param minsim: optional minimum similarity to consider. Higher values can make sorting faster (default 0).
        :type threshold: number
        :param threshold: optional center threshold in [0, 1] for sigmoid function applied to TF mask (default 0).
        :type hardness: number
        :param hardness: optional hardness of sigmoid function applied to TF mask. hardness=0 will make no change,
            hardess=1 will create a binary mask (default 0).
        :type progress: progressbar.ProgressBar
        :param progress: optional progress bar for debug output (default None).
        
        Separate figure from ground using precomputed band similarity matrices by either taking a weighted mean of
        similar frames or the median of similar frames.
        """
        
        self.repeating = np.zeros(self.pstft.shape)
        
        # 1. Convert minimum and maximum distance between similar frames from seconds to frames.
        minlag = int(minlag * self.rate / self.nhop)
        maxlag = int(maxlag * self.rate / self.nhop) if maxlag > 0 else self.nframes
        #repairlen = int(repairlen * self.rate / self.nhop)
        
        if minsim == 'eps': minsim = np.finfo(self.bandsim[0].dtype).eps
        
        if progress: 
            progress.maxval = self.nbands * self.nframes
            progress.start()
        
        # 3. Loop through each band, filling the repeating ground spectrogram.
        for band in range(self.nbands):
            sim = self.bandsim[band]
            l, r = self.bandindices[band]
            
            # 3.a. Compute repeating mixture frame for each frame.
            for i in range(len(sim)):
                indices = np.arange(self.nframes)
                values = sim[i, indices]
                
                # 3.a.i. Remove indices with too low of similarity values.
                gt = values >= minsim
                indices = indices[gt]
                values = values[gt]
                
                # 3.a.ii. Remove our index.
                noti = indices != i
                indices = indices[noti]
                values = values[noti]
                
                # 3.a.iii. Sort.
                sortind = np.argsort(values)[::-1]
                indices = indices[sortind]
                
                # 3.a.iv. Erase indices that are too close together.
                cleaned = 0
                for j in range(len(indices)):
                    if indices[j] >= 0:
                        distances = abs(indices - indices[j])
                        indices[np.bitwise_and(distances < minlag, indices != indices[j])] = -1
                        cleaned += 1
                        
                    if 0 < maxframes <= cleaned:
                        break
                
                indices = indices[indices >= 0]
                
                # 3.a.v. Limit the number of frames over which to take median/mean.
                if maxframes:
                    indices = indices[:maxframes]
                
                # 3.a.vi. Create repeating ground spectrogram.
                self.repeating[i, l:r] = np.median(self.pstft[indices,l:r], 0)
                
                if progress: progress.update(band * self.nframes + i + 1)
        
        if progress: progress.finish()
        
        # 4. Create repeating background mask, enforcing nonnegativity.
        self.mask = np.amin((self.repeating, self.pstft), 0)
        self.mask /= self.pstft
        self.mask = sigmoid(self.mask, threshold, hardness)
        self.mask = np.concatenate((self.mask, np.fliplr(self.mask[:,1:-1])), axis=1)
        
        # 5. Create ground and figure waveforms.
        gndstft = self.mask * self.stft

        self.gndwav = ISTFT(gndstft, self.nhop, len(self.wav))
        self.figwav = self.wav - self.gndwav
