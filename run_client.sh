#!/bin/bash
set -e
cd "$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"/

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <server_ip> <client_id>"
  exit 1
fi

server_ip=$1
cid=$2

echo "Starting Client $cid"
python3 client.py comm.host=$server_ip client_params.client_id=$cid &