#!/bin/bash
set -e
cd "$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"/

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <server_ip> <client_id> <model>"
  exit 1
fi

server_ip=$1
cid=$2
model=$3

# for cid in $(seq 0 1); do
#     echo "Starting client $cid"
#     python3 client.py client.cid=$cid neuralnet=Net &
# done

echo "Starting Client $cid"
python3 client.py comm.host=$server_ip client.cid=$cid neuralnet=$model &

# RUN CMD : bash run_client.sh $SERVER_IP $CLIENT_ID $MODEL