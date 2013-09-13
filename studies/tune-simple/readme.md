tune-simple refers to the tuning stage for the "simple" synthesis method,
which uses concatenative synthesis on a single stream to chain together
a bunch of sound grains. Sound grains from the source recording that is
nearest the coordinates are more likely, and sound grains from portions of
those recordings that are nearest the coordinates are more likely.

# Step 1: Generate everything.

1. `python mixer.py generate-sounds.yaml`
	- Generates the actual sounds we want as well as clips from the recordings
		that are nearest to the various coordinates.
2. `python mixer.py generate-tones.yaml`
	- Generates a number of concatenated sinusoids for trick questions.
3. `python studies/index.py create sounds.db -s wavpath`
	- Creates sound.db, which will be used to link up our hashed filenames
		to their originals and get coordinates from filenames used in studies.
	- `wavpath` is path to all the generated .wav files (including tones).
4. `python studies/index.py hash sounds.db -s wavpath -m hashpath`
	- Copies all the sound files to another directory, replacing their names
		with md5 hashes of their parameters so that users don't know their
		contents.
	- `hashpath` is where we want the hashed copies to end up.
5. `cd haspath; sh studies/mp3ify.sh`
	- Convert all the sounds to mp3s for web playback.


# Step 2: Choose best random versions for each tuning / coordinate.

1. `python index.py csv sounds.db -y 1-rank-iterations.yaml`
	- Create CSV for a CrowdFlower study. rank-iterations chooses the best out 
		of 5 different random versions for each (coordinate, grain length, 
		max distance) tuple.
	- 3 (grainlen) * 3 (maxdist) * 10 (coordinates) * 5 (versions) = 450 sounds.
	- 3 (grainlen) * 3 (maxdist) * 10 (coordinates) * 10 (version combinations) 
		= 900 real judgments needed.
	- 450 (sounds vs. themselves) + 450 (sounds vs. tones) = 900 gold judgments
		needed.
	- 900 (real judgments) + 900 (gold judgments) = 1800 total judgments.
	- Total: 1800 (total judgments) * 5 (users per judgment) = 9000 total 
		judgments needed max.
2. Now run the study.

# Step 3: Choose best tuning for each coordinate.

1. Download the CSV of the results from CrowdFlower.
	- `results/rank-iterations-f191477.csv` (x 1 user response)
	- `results/rank-iterations-f193529.csv` (x 4 user responses)
2. `python results-rank-iterations.py [results csv files] > 2-rank-tunings.yaml`
	- Create YAML to configure the second CrowdFlower study. Multiple CSV files
		will be concatenated.
3. `python index.py csv sounds.db -y 2-rank-tunings.yaml`
	- Create CSV for a CrowdFlower study. rank-tunings chooses the best tuning 
		for each coordinate from among the best random versions of each tuning, 
		determined from the previous study.
	- 3 (grainlen) * 3 (maxdist) * 10 (coordinates) = 90 sounds.
	- 10 (coordinates) * 36 (tuning combinations) = 360 real judgments needed.
	- 90 (sounds vs. themselves) + 90 (sounds vs. tones) = 180 gold judgments 
		needed.
	- 360 (real judgments) + 180 (gold judgments) = 540 total judgments.
	- 540 (total judgments) * 5 (user per judgment) = 2700 total judgements 
		needed max.
4. Now run the study.


# Step 4: Visualize some data.

Note: This needs to be replaced with a YAML generator that gives the best tuning for each coordinate, as we don't need to visualize anything yet.

1. Download the CSV of the results from CrowdFlower.
	- `results/rank-tunings-f196643.csv` (x 5 user responses)
2. `python results-rank-tunings.py [results csv file]`
	- Makes some plots of results.
