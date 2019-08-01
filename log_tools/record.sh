#!/bin/bash
echo "Start Record!"
for j in 1 10 100 600:
do
	echo $j
	for i in {1..30}:
	do
		echo $i	
		touch ./logs/log_"$1"_$j
		../adabox ../data/"$1" ../data/out.csv $j >> ./logs/log_"$1"_$j
	done
done
echo "Finish Record!"
