for f in *.wav
do 
	echo $f;
    lame -V2 -h --silent $f `basename -s .wav $f`.mp3;
done
