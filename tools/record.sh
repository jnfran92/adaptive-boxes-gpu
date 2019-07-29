#!/bin/bash
touch log
for i in {1..10}:
do	
	./adabox 600 >> ./log
done
echo "Finish test!"
