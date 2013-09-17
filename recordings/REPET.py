"""
Recording/REPET.py
envmixer

2012 Brandon Mechtley
Arizona State University

REPET-SIM algorithm used with permission by:
Zafar Rafii and Bryan Pardo
Northwestern University
http://music.cs.northwestern.edu/
"""

import numpy as np
import scipy.spatial.distance as dist

from .Recording import plot_waveform
from .FigureGround import FigureGround
from .STFT import STFT, ISTFT

from os import makedirs
from itertools import izip, count
from os.path import join, splitext, basename, exists

import numpy as np
import matplotlib
import matplotlib.mlab as ml
import matplotlib.pyplot as pp

def axis_fontsize(fs, ax=None):
    """
    :type fs: int
    :param fs: desired font size in points.
    :type ax: matplotlib.Axes
    :param ax: axes for which to change font size. Current axes by default.
    
    Change the font size for axis labels. I prefer this to setting rc parameters, as that could  affect future plots.
    """

    if not ax:
        ax = pp.gca()

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)

def sigmoid(x, t, h):
    """
    Sigmoid function as in REPET-SIM.
    
    Args:
        x (numpy.ndarray): input array to be thresholded with values in [0, 1]
        t (number): threshold in location [0, 1].
        h (number): hardness in [0, 1], such that h=0 corresponds to no change and h=1 corresponds to a binary step.
        
    Returns:
        sigmoid-adjusted input array with shape x.shape.
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
        Compute the Short-Time Fourier Transform of the waveform.
        
        Args:
            nfft (int): optional size of the STFT analysis window in samples (default 4096).
            nhop (int): optional number of samples between consecutive frames (default 2048).
            window (function): optional windowing function, e.g. from numpy (default numpy.hanning).
            progress (progress.ProgressBar): optional progress bar for debug output (default None).
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
        Compute frame-by-frame similarity matrix according to the cosine similarity between normalized spectrogram
        frames. Similarity is scaled from 0 to 1.
        
        Args:
            bands (int): optional number of evenly distributed frequency bands over which to separate the similarity /
                separation process (default 1).
            threshold (number): optional center threshold in [0, 1] for sigmoid applied to similarity matrices (default 0).
            hardness (number): optional hardness of sigmoid applied to the similarity matrices. hardness=0 will make no
                change; hardness=1 will binarize (default 0).
            minsim (number or "eps"): if nonzero, similarity values will be linearly interpolated to make this value the
                minimum. If "eps," the smallest possible value for the similarity matrix datatype will be used.
            progress (progress.ProgressBar): optional progress bar for debug output (default None).
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
        Compute frame-by-frame similarity matrix according to the cosine similarity between 
        normalized spectrogram frames. Similarity is scaled from 0 to 1.
        
        Args:
            bands (int): optional number of evenly distributed frequency bands over which to separate the similarity /
                separation process (default 1).
            threshold (number): optional center threshold in [0, 1] for sigmoid applied to similarity matrices
                (default 0).
            hardness (number): optional hardness of sigmoid applied to the similarity matrices. hardness=0 will make no
                change; hardness=1 will binarize (default 0).
            minsim (number or "eps"): if nonzero, similarity values will be linearly interpolated to make this value
                the minimum. If "eps," the smallest possible value for the similarity matrix datatype will be used.
            progress (progress.ProgressBar): optional progress bar for debug output (default None).
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
        Separate figure from ground using precomputed band similarity matrices by either taking a weighted mean of
        similar frames or the median of similar frames.
        
        Args:
            minlag (int): optional minimum distance between consecutive frames to consider as similar (default 0).
            maxlag (int): optional maximum distance between consecutive frames to consider as similar (default 0).
            maxframes (int): optional maximum number of frames to use in median and/or weighted mean (default 0).
            minsim (number): optional minimum similarity to consider. Higher values can make sorting faster
                (default 0).
            threshold (number): optional center threshold in [0, 1] for sigmoid function applied to TF mask
                (default 0).
            hardness (number): optional hardness of sigmoid function applied to TF mask. hardness=0 will make no
                change, hardess=1 will create a binary mask (default 0).
            progress (progressbar.ProgressBar): optional progress bar for debug output (default None).
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
        Separate figure from ground using precomputed band similarity matrices by either taking a weighted mean of
        similar frames or the median of similar frames.
        
        Args:
            minlag (int): optional minimum distance between consecutive frames to consider as similar (default 0).
            maxlag (int): optional maximum distance between consecutive frames to consider as similar (default 0).
            maxframes (int): optional maximum number of frames to use in median and/or weighted mean (default 0).
            minsim (number): optional minimum similarity to consider. Higher values can make sorting faster
                (default 0).
            threshold (number): optional center threshold in [0, 1] for sigmoid function applied to TF mask
                (default 0).
            hardness (number): optional hardness of sigmoid function applied to TF mask. hardness=0 will make no
                change, hardess=1 will create a binary mask (default 0).
            progress (progressbar.ProgressBar): optional progress bar for debug output (default None).
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
    
    def plot(self, prefix='', suffix='', progress=None):
        """
        Create a plot of similarity matrices and spectrograms and save it as an image. One image will be saved showing
        the similarity matrices for each band, and one image will be produced showing spectrograms for the figure,
        ground, and composite waveforms and the repeating background mask. Filenames will be of the form
        [input]-similarities.pdf and [input]-spectrograms.pdf, where [input] is the name of the original input file
        without its extension.
            prefix (str): optional path prefix for output image. If the empty string, files will be saved in the
            current working directory (default '').
            suffix (str): optional path suffix for output image.
            progress (progressbar.ProgressBar or None): optional progress bar for debug output.
        """
        
        if not exists(prefix):
            makedirs(prefix)
        
        if len(suffix):
            suffix = '-' + suffix
        
        simname = prefix + splitext(basename(self.filename))[0] + suffix + '-similarities.pdf'
        specname = prefix + splitext(basename(self.filename))[0] + suffix + '-spectrograms.pdf'

        # 1. Plot each band's similarity matrix.
        cols = np.ceil(np.sqrt(self.nbands))
        rows = cols if self.nbands - cols ** 2 <= 0 else cols + 1

        pp.figure(figsize=(4 * cols, 4 * rows))
        if progress:
            progress.maxval = self.nbands + 5
            progress.start()
        
        for band in range(self.nbands):
            pp.subplot(rows, cols, band + 1)
            l, r = self.bandindices[band]
            lhz = float(l) / self.specsize * self.rate / 2
            rhz = float(r) / self.specsize * self.rate / 2
            
            pp.title('Band %d (%d-%d Hz)' % (band + 1, lhz, rhz))
            
            if type(self.bandsim[band]) != dict:
                pp.imshow(self.bandsim[band], aspect='equal', origin='lower')
            else:
                bandmat = np.zeros((len(self.bandsim[band]), len(self.bandsim[band])))
                winlen = len(self.bandsim[band][0])
                                
                for f1 in self.bandsim[band]:
                    for f2 in self.bandsim[band][f1]:
                        bandmat[f1, f2] = self.bandsim[band][f1][f2]
                        bandmat[f2, f1] = self.bandsim[band][f1][f2]

                pp.imshow(bandmat, aspect='equal', origin='lower')
            if progress:
                progress.update(band + 1)

        pp.savefig(simname)

        # 2. Plot each spectrogram, including the time-frequency mask.
        pp.figure(figsize=(20, 25))
        pp.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3, wspace=0)

        names = [
            'Figure Spectrogram',
            'Ground Spectrogram',
            'Composite Spectrogram',
            'Repeating Spectrogram',
            'Time-Frequency Mask'
        ]
    
        wavnames = [
            'Figure Waveform',
            'Ground Waveform',
            'Composite Waveform',
            'Repeating Waveform'
        ]
        
        nplots = len(names) + len(wavnames)
        
        scales = ['dBFS'] * 4 + ['Ground/Composite']

        repeating = 10 * np.log10(self.repeating.T)
        wavs = [self.figwav, self.gndwav, self.figwav + self.gndwav]
    
        specs = [10 * np.log10(
            ml.specgram(
                wav,
                NFFT=self.nfft,
                noverlap=self.nhop)[0])
        for wav in wavs]
        
        vmin = min(np.amin(repeating), np.amin(np.array(specs)))
        vmax = max(np.amax(repeating), np.amax(np.array(specs)))
        vmins = [vmin] * 4 + [0]
        vmaxs = [vmax] * 4 + [1]
        specs += [repeating, self.mask[:,:self.specsize].T]
        
        plotnum = 1
        
        # 3. Plot all the spectrograms/the mask.
        for i, (name, scale, spec, vmin, vmax) in enumerate(izip(names, scales, specs, vmins, vmaxs)):
            # Plot waveform.
            if i < len(wavs):
                pp.subplot(nplots, 1, plotnum)
                plotnum += 1
                
                wf, ticks, labels = plot_waveform(
                    wavs[i], 
                    framesize=len(self.wav) / 512, 
                    hopsize=len(self.wav) / 1024, 
                    npoints=9, 
                    xmin=0,
                    xmax=len(self.wav) / float(self.rate), 
                    ymin=np.amin(self.wav),
                    ymax=np.amax(self.wav), 
                    clip_ends=False,
                    cmap=matplotlib.cm.Blues,
                    pmin=-25,
                    pmax=50
                )

                pp.title(wavnames[i], fontsize=10)
                pp.xlim(0, len(self.wav) / float(self.rate))              
                pp.ylabel('Amplitude', fontsize=10)
                pp.gca().yaxis.set_label_coords(-.04, 0.5)

                boundaries=sum([
                    [ticks[0]],
                    [
                        (ticks[a - 1] + ticks[a]) / 2
                        for a in range(1, len(ticks))
                    ],
                    [ticks[-1]]
                ], [])
                
                tickpositions = [
                    (boundaries[a - 1] + boundaries[a]) / 2
                    for a in range(1, len(boundaries))
                ]
                
                cb = pp.colorbar(
                    mappable=wf, 
                    ticks=tickpositions, 
                    pad=0.01, 
                    aspect=10, 
                    boundaries=boundaries,
                    values=ticks
                )
                
                cb.set_ticklabels(labels)
                cl = pp.getp(cb.ax, 'ymajorticklabels')
                pp.setp(cl, fontsize=10)
                cb.ax.set_ylabel('Percentile Range', fontsize=10)
                axis_fontsize(10, ax=cb.ax)
                
                axis_fontsize(10)

            # Plot spectrogram.
            pp.subplot(nplots, 1, plotnum)
            plotnum += 1
            
            pp.imshow(
                spec,
                origin='lower',
                extent=[0, len(self.wav) / self.rate, 0, self.rate / 2],
                vmin=vmin,
                vmax=vmax
            )
            
            pp.title(name, fontsize=10)
            pp.axis('auto')
            
            pp.xlim((0, len(self.wav) / self.rate))
            pp.ylim((0, self.rate / 2))
            pp.gca().set_yticklabels(np.array(pp.gca().get_yticks() / 1000, dtype=np.int))
            pp.ylabel('Frequency (KHz)', fontsize=10)
            pp.gca().yaxis.set_label_coords(-.04, 0.5)
            
            ticks = np.floor(np.linspace(vmin, vmax, 5)) if vmax > 1.0 else np.linspace(vmin, vmax, 5)
            fmt = '%d' if vmax > 1 else '%.2f'
            
            cb = pp.colorbar(aspect=10, format=fmt, ticks=ticks, pad=0.01)
            cl = pp.getp(cb.ax, 'ymajorticklabels')
            pp.setp(cl, fontsize=10)
            cb.ax.set_ylabel(scale, fontsize=10)
            
            axis_fontsize(10, ax=cb.ax)
            axis_fontsize(10)
            
            if progress:
                progress.update(self.nbands + i + 1)
        
        pp.xlabel('Time (s)', fontsize=10)
        pp.savefig(specname, bbox_inches='tight')
        
        if progress:
            progress.finish()
        
        return simname, specname
