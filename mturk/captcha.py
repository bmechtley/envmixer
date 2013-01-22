'''
mturk/captcha.py
natural-mixer

2013 Brandon Mechtley
Arizona State University

Look through a Mechanical Turk results CSV file to verify that users passed the captcha. Every HIT
should have one or more test groups where the test sound is the same file as one of the source
sounds, so the similarity to the test sound should be maximum in these cases.
'''

import argparse
import numpy as np

def verify(resultsfile):
    results = np.genfromtxt(resultsfile, delimiter='","', dtype=str)
    ids = results[1:,0]
    inputcols, soundids, answercols, desccols = [], [], [], []
    
    for i, c in enumerate(results[0]):
        tokens = c.replace('"', '').split('.')
        
        if len(tokens) == 2:
            if len(tokens[1]) == 4 and tokens[1][0] == 'g' and tokens[1][2] == 's':
                if tokens[0] == 'Input':
                    soundids.append(tokens[1])
                    inputcols.append(i)
                elif tokens[0] == 'Answer':
                    answercols.append(i)
            elif len(tokens[1]) == 5 and tokens[1][0:2] == 'dg' and tokens[1][3] == 's':
                if tokens[0] == 'Answer':
                    desccols.append(i)
    
    inputcols = np.array(inputcols).reshape((-1, 4))
    answercols = np.array(answercols).reshape((-1, 4))
    desccols = np.array(desccols).reshape((-1, 4))
    soundids = np.array(soundids).reshape((-1, 4))
    
    for i, row in enumerate(results[1:]):
        for ginput, ganswer, gdesc, gid in zip(inputcols, answercols, desccols, soundids):
            inputs = row[ginput]
            answers = row[ganswer]
            desc = row[gdesc]
            
            duplicates = inputs == inputs[0]
            
            if np.sum(duplicates) > 1:
                dupanswers = answers[duplicates]
                print ids[i].strip('"'), \
                    gid[duplicates], \
                    np.array([a.strip('"') for a in dupanswers], dtype=int), \
                    desc[duplicates]
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Verify that Mechanical Turk workers have passed the captcha test.')
    parser.add_argument('results', metavar='csv', type=str, help='mturk results .csv file')
    args = parser.parse_args()
    
    verify(args.results)