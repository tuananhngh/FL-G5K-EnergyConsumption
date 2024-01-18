#!/bin/bash
set -e
cd "$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
server_ip=$(hostname -I)
echo "Server IP: $server_ip"
ip4=$(echo $server_ip | awk '{print $1}')
echo "Server IP4: $ip4"
model=$1
python3 main_server.py comm.host=$(echo $ip4) neuralnet=$model&
sleep 3  # Sleep for 3s to give the server enough time to start

# for cid in $(seq 0 1); do
#     echo "Starting client $cid"
#     python3 client.py client_params.client_id=$cid &
# done

rm -rf /data/ # Remove old data
# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

# Run CMD : bash_run_server.sh Net