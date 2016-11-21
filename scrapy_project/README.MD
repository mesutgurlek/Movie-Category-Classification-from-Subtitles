- Removing the non IMPAIRED subtitles (that ones which were downloaded mistakenly) (run in a category folder)

for x in $(ls | grep -v IMPAIRED); do rm -v $x; done

- Counting number of subtitles in each category (run this in Subtitles folder)

for x in $(ls); do echo $x $(ls $x | wc -l); done
