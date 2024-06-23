#!/bin/bash 
#Script to launch server and client on terminal
kill_background_jobs() {
    jobs -p | xargs kill
}
trap kill_background_jobs SIGINT
set -e
python3 server.py 
echo "Server started"
wait
