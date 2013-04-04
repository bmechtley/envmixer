import sqlite3, argparse, os, os.path, hashlib, subprocess, pprint, random
from itertools import izip
import numpy as np

def do_hash(db, hashpath):
    for sndname, hashname in db.execute('SELECT name, hash FROM sounds'):
        hashdest = os.path.join(hashpath, hashname + os.path.splitext(sndname)[1]) 
        subprocess.call(['cp', sndname, hashdest])
        print 'cp', sndname, hashdest

def do_unhash(db, hashpath):
    for sndname, hashname in db.execute('SELECT name, hash FROM sounds'):
        hashsrc = os.path.join(hashpath, hashname + os.path.splitext(sndname)[1])
        subprocess.call(['cp', hashsrc, sndname])
        print 'cp', hashsrc, sndname

def make_csv(db, outname):
    # 1. Create real units and gold source units.  
    testunits = []
    
    for test_hash, test_mix, test_coordinates, test_nums, test_iteration, test_name in db.execute(
        'SELECT hash, mix, coordinates, nums, iteration, name FROM sounds WHERE (mix="simple" OR mix="sources")'
    ):
        sources = []
        
        # 1.a. Gather sources.
        for source_hash, source_mix, source_nums, source_coordinates, source_name in db.execute(
            'SELECT hash, mix, nums, coordinates, name FROM sounds WHERE (coordinates="%s" AND mix="sources")' % test_coordinates
        ):
            sources.append({
                'hash': source_hash,
                'nums': source_nums,
                'name': source_name
            })
        
        # 1.b. Shuffle sources and get their gold numbers.
        random.shuffle(sources)
        
        # 1.c. Build the unit. 
        testunit = {
            'test_hash': test_hash, 
            'test_nums': test_nums, 
            'coordinates': test_coordinates, 
            'test_name': test_name
        }
        
        testunit.update({'s%d_hash' % (i+1): v['hash'] for i, v in enumerate(sources)})
        testunit.update({'s%d_nums' % (i+1): v['nums'] for i, v in enumerate(sources)})
        testunit.update({'s%d_name' % (i+1): v['name'] for i, v in enumerate(sources)})
        
        # 1.d. Iteration for test units, gold convincingness/similarity for source units.
        if test_mix == 'simple':
            testunit['iteration'] = test_iteration
        elif test_mix == 'sources':
            testunit.update({'test_pc': '4\n5', '_golden': True})
            
            for i, source in enumerate(sources):
                if source['hash'] == test_hash:
                    testunit['s%d_sim' % (i + 1)] = '4\n5'
        
        # 1.e. Add the unit to the list.
        testunits.append(testunit)
    
    # 2. Add gold tone units.
    for tone_hash, tone_name in db.execute('SELECT hash, name FROM sounds WHERE mix="tones"'):
        copyunit = testunits[np.random.randint(len(testunits))].copy()
        copyunit.update({
            'test_hash': tone_hash, 
            'test_name': tone_name, 
            's1_sim': 1, 
            's2_sim': 1, 
            's3_sim': 1, 
            'test_pc': 1, 
            'coordinates': '',
            'iteration': '',
            '_golden': True
        })
        
        testunits.append(copyunit)
    
    # 3. Bake it all down into a CSV.
    fields = []
    
    for testunit in testunits:
        for key in testunit.keys():
            if key not in fields:
                fields.append(key)

    for testunit in testunits:
        for key in fields:
            testunit.setdefault(key, '')

    fields.sort()
    
    testunits.insert(0, {f: f for f in fields})
    
    # 4. Save the CSV.
    csv = [[str(testunit[f]) for f in fields] for testunit in testunits]
    np.savetxt(outname, csv, delimiter=",", fmt='"%s"')

def make_db(db, soundpath):
    fields = ['hash', 'name']
    sounds = []
    
    sndfiles = os.listdir(soundpath)
    
    for sndfile in sndfiles:
        basename = os.path.basename(sndfile)
        tokens = os.path.splitext(basename)[0].split('-')

        for token in tokens[::2]:
            if token not in fields:
                fields.append(token)
        
        sound = {f: v for f, v in izip(tokens[::2], tokens[1::2])}
        
        hash = hashlib.md5()
        hash.update(basename)
        
        sound['hash'] = hash.hexdigest()
        sound['name'] = os.path.join(soundpath, sndfile)
        
        sounds.append(sound) 
    
    db.execute('DROP TABLE IF EXISTS sounds')
    db.execute('CREATE TABLE sounds (%s)' % ', '.join(['%s text' % f for f in fields]))

    for sound in sounds:
        db.execute('INSERT INTO sounds (%s) VALUES (%s)' % (
            ', '.join(sound.keys()),
            ', '.join(['"%s"' % v for v in sound.values()])
        ))

        print 'Inserting: ', sound

    db.commit()
    db.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Creates an index of generated wav files for study purposes.')
    parser.add_argument('command', type=str, default='create', help='Command to perform.', 
        choices=['create', 'hash', 'unhash', 'csv']
    )
    parser.add_argument('database', type=str, default='study.db', help='SQLite database name to load/create.')
    parser.add_argument('-s', '--soundpath', required=False, type=str, default='./', 
        help='{create, hash, unhash}: path to input/output sound files from which to create an index, hash, or unhash.')
    parser.add_argument('-m', '--hashpath', required=False, type=str, default='./', 
        help='{hash, unhash}: path to save/load un/hashed wav files.'
    )
    parser.add_argument('-c', '--csvpath', required=False, type=str, default='sounds.csv',
        help='csv: filename for output CSV file.'
    )
    
    args = parser.parse_args()
    db = sqlite3.connect(args.database)

    if args.command == 'create':
        make_db(db, args.soundpath)
    elif args.command == 'hash':
        do_hash(db, args.hashpath)
    elif args.command == 'unhash':
        do_unhash(db, args.hashpath)
    elif args.command == 'csv':
        make_csv(db, args.csvpath)
    
    db.close()
