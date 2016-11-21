Removing the non IMPAIRED subtitles (that ones which were downloaded mistakenly)

for x in $(ls | grep -v IMPAIRED); do rm -v $x; done
