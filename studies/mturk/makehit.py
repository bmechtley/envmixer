'''
mturk/makehit.py
envmixer

2012 Brandon Mechtley
Arizona State University

This is a really messy script for generating Mechanical Turk HIT templates.
Should probably use an actual .html template.

usage: makehit.py [-h] [-g int] [-p url] [-o file]

Create a Mechanical Turk HIT template.

optional arguments:
  -h, --help             show this help message and exit
  -g int, --groups int   number of trial groups per HIT.
  -p url, --prefix url   url prefix for wav files.
  -o file, --output file output HTML file.
'''

import argparse
from xml.dom.minidom import parseString

def audioelem(group, num, prefix):
    return '\
        <audio controls="controls" style="display: inline-block;\
        vertical-align: middle">\
            <source src="%s/${g%ds%d}"></source>\
            Your browser does not support the audio element. Please download a\
            browser that supports HTML5 to complete this HIT.\
        </audio>' % (prefix, group, num)

def slider(prompt, group, num):
    return '\
        <p>%(prompt)s</p>\
        <p>\
            1\
            <input\
                style="vertical-align: middle"\
                type="range"\
                name="g%(group)ds%(num)d"\
                id="g%(group)ds%(num)d"\
                min="1"\
                max="9"\
                value="1"\
                onchange="showValue(\'g%(group)ds%(num)d\')"/>\
            9 <span id="sg%(group)ds%(num)d">(You have selected: 1)</span>\
        </p>' % {
            'prompt': prompt, 'group': group, 'num': num
        }

def textarea(group, num):
    return '\
        <p>Please provide a few words that describe the sound:</p>\
        <textarea\
            style="width:30em; height: 5em; margin-bottom: 1em"\
            id="dg%(group)ds%(num)d"\
            name="dg%(group)ds%(num)d"\
        ></textarea>\
    ' % {'group': group, 'num': num}

def makehit(groups, prefix):
    html = '\
        <script type="text/javascript">\
            function showValue(sid) {\
                document.getElementById("s" + sid).innerHTML = \
                    "(You have selected: " +\
                        document.getElementById(sid).value +\
                        ")";\
            }\
        </script>\
        <h2>Compare Sound Clips</h2>\
        <h3>Instructions</h3>\
        <p>\
            In the following %d tasks, you will first be given a 15-second\
            <b>test</b> sound and then three 15-second <b>source</b> sounds.\
            The source sounds are <b>real recordings</b>, each of a different\
            location. The test sound is either a real recording or a synthetic\
            mixture of recordings.\
        </p>\
        <p>\
            For each test sound, you will first be asked to rate how\
            <b>perceptually convincing</b> the sound is on a scale from 1 to\
            9. Here is a guide for your rating:\
        </p>\
        <ul>\
            <li>1: Completely unrealistic.</li>\
            <li>3: Barely realistic.</li>\
            <li>5: Somewhat realistic.</li>\
            <li>7: Very realistic.</li>\
            <li>9: This is a real recording.</li>\
        </ul>\
        <p>\
            You will then be given the three source recordings. For each\
            source recording, corresponding to three different recording\
            locations, please rate how similar it is to the test sound on a\
            scale from 1 to 9. Here is a guide for how to rate similarity:\
        </p>\
        <ul>\
            <li>1: Completely different from the test sound.</li>\
            <li>3: Barely similar to the test sound.</li>\
            <li>5: Somewhat similar to the test sound.</li>\
            <li>7: Very similar to the test sound.</li>\
            <li>9: Exactly the same as the test sound.</li>\
        </ul>\
        <p>\
            For every sound, you will also be asked to provide a short\
            description of what you hear.\
        </p>' % groups
    
    for i in range(groups):
        html += '\
            <hr />\
            <h3>Group %d</h3>\
            <ol>\
                <li>\
                    <h4>Test sound.</h4>\
                    <ol type="a">\
                        <li>\
                            <p>\
                                Please listen to the following sound as many\
                                times as you like.\
                            </p>\
                            <p>' % (i + 1) + \
                                audioelem(i, 0, prefix) + '\
                            </p>\
                        </li>\
                        <li>' + \
                            slider(
                                'Please rate how <b>perceptually\
                                convincing</b> the test sound is:',
                                i, 0
                            ) + '\
                        </li>\
                        <li>' + \
                            textarea(i, 0) + '\
                        </li>\
                    </ol>\
                </li>\
                <li>\
                    <h4>Source sounds.</h4>\
                    <ol type="a">'
        
        for j in range(1, 4):
            html += '\
                        <li>Source %d' % j + '\
                            <ol type="i">\
                                <li>\
                                    <p>\
                                        Please listen to the following sound\
                                        as many times as you like.\
                                    </p>\
                                    <p>' + \
                                        audioelem(i, j, prefix) + '\
                                    </p>\
                                </li>\
                                <li>' + \
                                    slider(
                                        'Please rate how <b>similar</b> the\
                                        sound is to the test sound:',
                                        i, j
                                    ) + '\
                                </li>\
                                <li>' + \
                                    textarea(i, j) + '\
                                </li>\
                            </ol>\
                        </li>'
        
        html += '\
                    </ol>\
                </li>\
            </ol>'
    
    return parseString('<body>' + html + '</body>').toprettyxml()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Create a Mechanical Turk HIT template.'
    )

    parser.add_argument('-g', '--groups', metavar='int', default=5, type=int,
        help='number of trial groups per HIT.')
    parser.add_argument('-p', '--prefix', metavar='url', 
        default='https://s3.amazonaws.com/naturalmixer',
        help='url prefix for wav files.')
    parser.add_argument('-o', '--output', metavar='file', default='hit.html', 
        type=str, help='output HTML file.')
    args = parser.parse_args()
    
    text = makehit(args.groups, args.prefix)
    text = re.sub('<[/]*body>', '', text)
    text = text.replace('<?xml version="1.0" ?>', '')
    
    f = open(args.output, 'w')
    f.write(text)
    f.close()

