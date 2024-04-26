#!/bin/bash 
num_runs=2
for ((i=1; i<=$num_runs; i++))
do 
    echo "Run $i"
    python3 experiment.py
    echo "Sleeping for 5 seconds"
    sleep 5
done
