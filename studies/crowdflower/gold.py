'''
studies/crowdflower/gold.py
natural-mixer

2013 Brandon Mechtley
Arizona State University

Modify a gold report from CrowdFlower to have the expected user input/reasons
for gold marking.

Usage: python gold.py report.csv
'''

import csv
import argparse
import numpy as np

def altergolden(report):
    '''Alter a downloaded golden report from CrowdFlower, setting the expected
    gold answers.
        csv: np.array
            array of CSV data from CrowdFlower report.'''

    teststr = 'test_clip_perceptual_convincingness_gold'
    simstr = "source_%d_similarity_to_test_clip_gold"

    cols = {
        col: np.where(report[0] == col)[0] for col in [
            's0',
            's1',
            's2',
            's3',
            '_difficulty',
            "_id",
            "_golden",
            teststr,
            teststr + '_reason',
            simstr % 1,
            simstr % 1 + '_reason',
            simstr % 2,
            simstr % 2 + '_reason',
            simstr % 3,
            simstr % 3 + '_reason'
        ]
    }

    for row in report[1:]:
        if len(row[cols['_golden']]):
            sounds = [row[cols[si]][0] for si in ['s0', 's1', 's2', 's3']]
            
            row[cols['_difficulty']] = '"Easy"'
            row[cols[teststr]] = '"4\n5"'
            row[cols[teststr + '_reason']] = '"This is a real sound."'
            
            for i in range(1, 4):                
                if sounds[i] == sounds[0]:
                    row[cols[simstr % i]] = '"5"'
                    row[cols[simstr % i + '_reason']] = '"This is the same as the test clip."'
    
    print '\n'.join([','.join(row) for row in report])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modify a golden report from CrowdFlower to have the expected user input/reasons for gold marking.')
    parser.add_argument('report', metavar='file', default='mapping.txt', type=str, help='CSV gold report from CrowdFlower.')
    args = parser.parse_args()
    
    with open(args.report, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        csvarr = np.array([row for row in reader])
        
        altergolden(csvarr)
