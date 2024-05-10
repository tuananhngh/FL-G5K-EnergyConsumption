#!/bin/bash 
num_runs=3
# strategy="random"
for ((i=1; i<=$num_runs; i++))
do 
    echo "Run $i"
    python3 experiment.py
    echo "Sleeping for 5 seconds"
    sleep 5
done


# List of strategies
# strategies=("fedadam" "fedadagrad" "fedyogi" "fedavg")

# # Loop over strategies
# for strategy in ${strategies[@]}
# do
#     # Call python script with the current strategy
#     python experiment.py --strategy $strategy
#     sleep 5
# done