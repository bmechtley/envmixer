"""
separator.py
envmixer

2012 Brandon Mechtley
Arizona State University

Make use of Recording.* to separate figure/ground from an environmental sound file.

See 'python separator.py --help' for more information.

YAML options are passed directly as keyword arguments for each routine (see relevant classes for 
argument explanations):
    stft: Recording.REPET.compute_stft
    similarity: Recording.REPET.compare
    separate: Recording.REPET.separate
    save: Recording.save
    plot: [True/False] (whether or not to save result plots)
"""

import yaml
import argparse
from progressbar import *

import matplotlib
matplotlib.use('pdf')

import pybatchdict
from Recording import REPET, MRA

def main():
    # Command-line arguments.
    parser = argparse.ArgumentParser(
        description='Make use of Recording to separate figure/ground from an environmental sound file.'
    )
    parser.add_argument('input', metavar='wav', nargs='+', help='wav files to separate.')
    parser.add_argument('-c', '--config', metavar='file', default='config.yaml',
        help='envrepet configuration YAML file. See config.yaml for an example.')
    parser.add_argument('-o', '--output', metavar='path', default='',
        help='path prefix for output files.')
    parser.add_argument('-v', '--verbosity', metavar='int', default=1, type=int,
        help='debug output verbosity. 0 for no output, 1 for progress, 2 for the rest of the details.')
    args = parser.parse_args()

    # YAML configs. Enumerate over combinations of list values for studies.
    config = yaml.load(open(args.config))
    config.setdefault('type', 'repet')
    config.setdefault('stft', {})
    config.setdefault('similarity', {})
    config.setdefault('separate', {})
    config.setdefault('save', {})
    config.setdefault('plot', False)

    batchdict = pybatchdict.BatchDict(config)
    outnames = batchdict.hyphenate_changes()

    # Process each file according to config YAML.
    for filename in args.input:
        for i, (combo, outname) in enumerate(zip(batchdict.combos, outnames)):
            if args.verbosity > 0:
                print 'Config %d / %d: %s' % (i + 1, len(batchdict.combos), outname)

            # Options are to use REPET-SIM or an MRA tree-based approach.
            if combo['type'] == 'repet':
                # 1. Load the file. Print file and YAML configuration details.
                r = REPET(filename)

                if args.verbosity > 1:
                    print yaml.dump(
                        {'%s: %.2fs, %d Hz' % (filename, len(r.wav) / r.rate, r.rate): combo},
                        indent=4,
                        default_flow_style=False
                    )

                # 3. Creaprogressbars.
                progress = dict(zip(
                    ['STFT', 'Comparing', 'Separating', 'Saving', 'Plotting'],
                    [None] * 5
                ))

                if args.verbosity > 0:
                    for k, v in progress.items():
                        progress[k] = ProgressBar(
                            widgets=['    %s: ' % k, Percentage(), ' ', Bar(), ' ', ETA()]
                        )

                # 4. Compute everything.
                r.compute_stft(progress=progress['STFT'], **combo['stft'])
                r.compare(progress=progress['Comparing'], **combo['similarity'])
                r.separate(progress=progress['Separating'], **combo['separate'])

                # 5. Print result details.
                if args.verbosity > 1:
                    print '    STFT info:\n       ', '\n        '.join([
                        '%d frames' % r.nframes,
                        '%f ms/frame' % (1000 * r.nfft / r.rate),
                        '%f ms/hop' % (1000 * r.nhop / r.rate),
                        '%d bins/frame\n' % r.specsize
                    ])

                # Save figure/ground/composite wavfiles.
                wavs = r.save(progress=progress['Saving'], prefix=args.output, suffix=outname, **combo['save'])
                if args.verbosity > 1:
                    print '    WAVs saved:\n       ', '\n        '.join(wavs), '\n'

                # Save plot PDFs.
                if combo['plot']:
                    plots = r.plot(progress=progress['Plotting'], prefix=args.output, suffix=outname)
                    if args.verbosity > 1:
                        print '    Plots saved:\n       ', '\n        '.join(plots)
            elif combo['type'] == 'mra':
                r = MRA(filename)
                r.calculate_mra()
                r.reconstruct_mra()
    
if __name__ == '__main__':
    main()
