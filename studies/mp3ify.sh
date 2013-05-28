for f in *.wav; do echo $f && lame -V2 -h --silent $f; done
