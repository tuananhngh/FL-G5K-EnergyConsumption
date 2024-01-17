#!/bin/bash
set -e
cd "$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"/

#server_ip="$1"

echo "Starting server"
python3 main_server.py #omm.host=$server_ip &
sleep 3  # Sleep for 3s to give the server enough time to start

# for cid in $(seq 0 1); do
#     echo "Starting client $cid"
#     python3 client.py client.cid=$cid comm.host=$server_ip &
# done

rm -rf /data/ # Remove old data
# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

