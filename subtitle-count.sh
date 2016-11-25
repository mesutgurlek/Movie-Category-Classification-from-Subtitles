cd Subtitles/
total=0
for x in $(ls)
do
    count=$(ls $x | wc -l)
    total=$(( $total + $count ))
    echo $x $count
done
echo
echo "Total:" $total
