#!/bin/bash
#server_ip="$1"

for cid in $(seq 0 4); do
    echo "Starting client $cid"
    python3 client.py client.cid=$cid & #comm.host=$server_ip &
done

rm -rf /data/ # Remove old data
# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait