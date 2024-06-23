#!/bin/bash 
#Script to launch server and client on terminal
# Define a function to kill all background jobs
kill_background_jobs() {
    jobs -p | xargs kill
}
trap kill_background_jobs SIGINT
set -e
for i in $(seq 1 20)
do
    python3 client.py &
done
wait 