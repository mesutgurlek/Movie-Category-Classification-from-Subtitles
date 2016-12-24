cd NonImpairedSubtitles/
for x in $(ls)
do
    if [ -d "$x" ]
    then

    echo "Processing directory:"$x
    cd $x
(IFS='
'; for sub in $(ls | grep IMPAIRED); do mv $sub ${sub:0:-15}.srt; done;  )
    cd ..

    fi

done
