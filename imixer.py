# System.
import os, sys

# External.
import argparse
import yaml

# SCIENCE.
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.cm as cm
import matplotlib.pyplot as pp
import matplotlib.widgets as widgets
import scikits.audiolab as al

# Custom.
import pybatchdict
import barycentric as bary

# Internal.
import train
import recordings as rec

def make_tones(freqs, duration=1.0, amplitude=1.0, rate=44100):
    tonelen = duration * rate

    tones = [rec.Recording() for i in range(len(freqs))]

    tonefunc =lambda x: np.sin(x * 2) #* np.fmod(x, 2 * np.pi) / (2 * np.pi)

    for i in range(len(tones)):
        tones[i].wav = tonefunc(np.linspace(0, tonelen * 2 - 1, tonelen * 2) * 2 * np.pi * freqs[i] / rate) * amplitude
        tones[i].rate = rate

    return tones

class InteractivePlot:
    def __init__(self, sounds, config):
        self.figure = None
        self.spec_axes = None
        self.preview_axes = None
        self.control_axes = None

        self.sounds = sounds
        self.config = config

        if config['mix'] == 'tones':
            self.train = train.make_tone_train(
                sounds,
                traindur=config['trainlength'],
                tonedur=config['tones']['length'],
                amplitude=config['tones']['amplitude'],
                fadedur=config['tones']['overlap']
            )
        elif config['mix'] == 'simple':
            self.train = train.make_simple_train(
                config['coordinates'],
                sounds,
                config['trainlength'],
                config['simple']['grainlength'],
                config['simple']['maxdist']
            )
        else:
            self.train = None

    def update(self, val):
        pp.ion()

        # Remake grain train.
        self.train = train.make_simple_train(
            self.config['coordinates'],
            self.sounds,
            self.tl_slider.val,
            [self.glmin_slider.val, self.glmax_slider.val],
            self.config['simple']['maxdist'],
            train=self.train
        )

        # Preview grain envelopes.
        pp.sca(self.preview_axes)
        self.preview_axes.cla()
        self.train.fillgrains()
        self.train.plot_train(512, 256, cmap=mpl.cm.gist_rainbow, npoints=3)
        pp.draw()

        pp.ioff()


    def bake(self, event):
        pp.ion()

        self.train.mixdown()

        pxx, freqs, t = mpl.mlab.specgram(self.train.sound.wav, Fs=self.train.sound.rate, NFFT=512, noverlap=256)
        pxx = np.flipud(10. * np.log10(pxx))
        self.pspec_axes.cla()
        self.pspec_axes.imshow(pxx, extent=(0, np.amax(t), freqs[0], freqs[-1]))
        self.pspec_axes.text(0, 0, self.train.sound.filename)
        self.pspec_axes.set_aspect('auto')
        self.pspec_axes.set_xlim((0, np.amax(t)))
        self.pspec_axes.set_ylim((np.amin(freqs), np.amax(freqs)))
        pp.draw()

        pp.ioff()

    def play(self, index):
        if index < 0:
            self.train.mixdown()
            al.play(self.train.sound.wav, self.train.sound.rate)
        else:
            al.play(self.sounds[index].wav, self.sounds[index].rate)

    def load(self, event):
        pp.ion()

        # Draw source spectrograms only once.
        for i in range(len(self.sounds)):
            pp.sca(self.spec_axes[i])

            basecolor = cm.gist_rainbow(float(i) / (len(self.sounds) - 1))[0:3]
            cmap = mplc.LinearSegmentedColormap('colors%d' % i, {
                'red': [
                    (0.0, 1.0, 1.0),
                    (1.0, basecolor[0], basecolor[0])
                ],
                'green': [
                    (0.0, 1.0, 1.0),
                    (1.0, basecolor[1], basecolor[1])
                ],
                'blue': [
                    (0.0, 1.0, 1.0),
                    (1.0, basecolor[2], basecolor[2])
                ]
            })

            pxx, freqs, t = mpl.mlab.specgram(self.sounds[i].wav, Fs=self.sounds[i].rate, NFFT=1024, noverlap=512)
            pxx = np.flipud(10. * np.log10(pxx))
            pxx = (pxx - np.amin(pxx)) / (np.amax(pxx) - np.amin(pxx))
            pxx = (np.tanh((pxx * 2 - 1)) + 1) / 2.0
            pxx = pxx ** 6

            self.spec_axes[i].imshow(pxx, cmap, extent=(0, np.amax(t), freqs[0], freqs[-1]))
            self.spec_axes[i].text(0, 0, self.sounds[i].filename)
            self.spec_axes[i].set_aspect('auto')
            self.spec_axes[i].set_xlim((0, np.amax(t)))
            self.spec_axes[i].get_xaxis().set_visible(False)
            self.spec_axes[i].set_ylim((np.amin(freqs), np.amax(freqs)))
            self.spec_axes[i].set_ylabel('Frequency (Hz)')

            pp.draw()

        self.preview_axes.set_xlabel('Time (s)')
        self.preview_axes.get_yaxis().set_visible(False)

        self.update(None)

        spp = self.gs.get_subplot_params()
        spp.left = 0.05
        spp.right = 0.95
        spp.hspace = 0.5

        pp.ioff()

    def show(self):
        if self.figure == None:
            self.figure = pp.figure()
            self.gs = mpl.gridspec.GridSpec(8, 3, height_ratios=[10,10,10,10,10,1,1,1], width_ratios=[1,14,1])

            self.spec_axes = [pp.subplot(self.gs[i,0:2]) for i in range(len(self.sounds))]
            self.preview_axes = pp.subplot(self.gs[3,0:2])
            self.pspec_axes = pp.subplot(self.gs[4,0:2])

            # Sliders.
            tl_axes = pp.subplot(self.gs[5,1:])
            self.tl_slider = widgets.Slider(tl_axes, '', 1.0, 10.0, valinit=self.config['trainlength'])
            tl_axes.text(0, 0, 'Train length')

            glmin_axes = pp.subplot(self.gs[6,1:])
            self.glmin_slider = widgets.Slider(glmin_axes, '', 500, 5000, valinit=self.config['simple']['grainlength'][0])
            glmin_axes.text(0, 0, 'Minimum grain length')

            glmax_axes = pp.subplot(self.gs[7,1:])
            self.glmax_slider = widgets.Slider(glmax_axes, '', 500, 5000, valinit=self.config['simple']['grainlength'][1])
            glmax_axes.text(0, 0, 'Maximum grain length')

            self.tl_slider.on_changed(self.update)
            self.glmin_slider.on_changed(self.update)
            self.glmax_slider.on_changed(self.update)

            # Buttons.
            self.play_buttons = []

            for i in range(len(self.sounds)):
                self.play_buttons.append(widgets.Button(pp.subplot(self.gs[i,2]), 'Play'))
                self.play_buttons[-1].on_clicked(lambda event: self.play(i))

            self.play_buttons.append(widgets.Button(pp.subplot(self.gs[4,2]), 'Play'))
            self.play_buttons[-1].on_clicked(lambda event: self.play(-1))

            self.bake_button = widgets.Button(pp.subplot(self.gs[3,2]), 'Bake')
            self.bake_button.on_clicked(self.bake)

            self.load_button = widgets.Button(pp.subplot(self.gs[5:,0]), 'Load')
            self.load_button.on_clicked(self.load)

        pp.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05, wspace=0.05)
        pp.show()

def main():
    parser = argparse.ArgumentParser(description='Create a mixture of two or more sound textures.')
    parser.add_argument('config', type=str, default='config.yaml', help='YAML config file.')
    args = parser.parse_args()
    
    # 1. YAML config. Enumerate over combinations of list values for studies.
    config = yaml.load(open(args.config))
    
    config.setdefault('outpath', '')
    config['outpath'] = os.path.expanduser(config['outpath'])
    config.setdefault('sources', [])
    for i in range(len(config['sources'])):
        config['sources'][i] = os.path.expanduser(config['sources'][i])

    config.setdefault('mix', 'simple')
    config.setdefault('coordinates', [1, 0, 0])

    config.setdefault('simple', {})
    config['simple'].setdefault('svl', False)
    config['simple'].setdefault('plot', False)
    config['simple'].setdefault('maxdist', 60.0)
    config['simple'].setdefault('trainlength', 15.0)
    config['simple'].setdefault('envelope', 'cosine')
    config['simple'].setdefault('grainlength', [500, 2000])
    
    config.setdefault('tones', {})
    config['tones'].setdefault('count', 0)
    config['tones'].setdefault('rate', 44100)
    config['tones'].setdefault('minfreq', 2000)
    config['tones'].setdefault('maxfreq', 5000)
    config['tones'].setdefault('amplitude', 1.0)
    config['tones'].setdefault('envelope', 'cosine')
    
    if type(config['coordinates']) == dict:
        if config['coordinates'][config['coordinates'].keys()[0]] == 'lattice':
            config['coordinates'] = {config['coordinates'].keys()[0]: bary.lattice(len(config['sources']))}
        
        config['coordinates'][config['coordinates'].keys()[0]] = [
            np.array(coord,dtype=float) / sum(coord) for coord in
            config['coordinates'][config['coordinates'].keys()[0]]
        ]

    batch = pybatchdict.BatchDict(config)
    config = batch.combos[0]

    # 2. Load soundwalks.
    sounds = []
    
    if config['mix'] == 'tones':
        sounds = make_tones(
            config['tones']['freqs'], 
            duration=config['trainlength'], 
            amplitude=config['tones']['amplitude'], 
            rate=config['tones']['rate']
        )
    else:
        sounds = [rec.Recording(s) for s in config['sources']]

    itrain = InteractivePlot(sounds, config)
    itrain.sounds = sounds
    itrain.config = config
    itrain.show()

if __name__ == '__main__':
    main()
