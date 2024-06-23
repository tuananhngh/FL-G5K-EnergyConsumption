#!/bin/bash 
num_runs=4
# # strategy="random"
for ((i=1; i<=$num_runs; i++))
do 
    echo "Run $i"
    python3 experiment.py
    echo "Sleeping for 120 seconds"
    sleep 120
done


# List of strategies
#strategies=("fedadam" "fedadagrad" "fedyogi" "fedavg")
# nb_clients=(30)
# # Loop over strategies
# for client in ${nb_clients[@]}
# do
#     # Call python script with the current strategy
#     #python experiment.py --strategy $strategy
#     echo "Running with $client clients"
#     python3 experiment.py --nb_clients $client
#     sleep 120
# done 