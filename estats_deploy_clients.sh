#!/bin/bash

# This script is running on the frontend

# Reserve deployment resources
#oarsub -t deploy -t exotic -q testing -p estats -l host=1,walltime=1 -I

# Deploy the environment on all reserved nodes
# kadeploy3 -a ~/public/ubuntu-estats.dsc 
# sleep 10

# Get IP address and hostname of deployed nodes
echo "Get Hostname and IP address of deployed nodes"
hostnames=$(oarprint host)
ips=$(oarprint ip)

IFS=$'\n' hosts_array=($(echo "$hostnames" | tr -s '[:space:]'))

declare -p -a hosts_array

server_host=${hosts_array[0]}
# server_ip=$(host $server_host | awk '{print $4}')
clients_host=(${hosts_array[@]:1})

echo "Clients: ${clients_host[@]}"


docker_image="tuanngh/fl-jetson:latest"
server_ip="$1"
echo "Server IP: $server_ip"

# SSH connect to the CLIENTS in parallel
# Array to store background PIDs
bg_pids=()
for key in "${!clients_host[@]}"
do
    (   
        c_host=${clients_host[$key]}
        echo "---CONNECTING TO CLIENT $c_host---"
        ssh root@"$c_host" <<HERE
            echo "---CONFIGURATE DOCKER CLIENT $c_host---"
            echo "Pull Docker image on CLIENT"
            docker pull $docker_image
HERE
    ) &

    # Store the background PID
    bg_pids+=($!)
done

# Wait for all background processes to finish
wait

echo "Background PIDs: ${bg_pids[@]}"


# RUN CLIENTS
for key in "${!clients_host[@]}";
do
    (   c_host=${clients_host[$key]}
        #echo "---CONNECTING TO CLIENT $c_host---"
        ssh root@"$c_host" <<HERE2
            echo "---EXECUTE DOCKER ON CLIENT $c_host---"
            docker run -d --runtime nvidia --rm --network host --name client_"$key" -it $docker_image &&
            docker exec client_"$key" bash -c "python3 client.py comm.host=$server_ip client_params.client_id=$key params.num_rounds=10 params.num_clients=2"  
HERE2
    ) &
done


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait


#docker context create server --docker "host=ssh://root@$server_host"
# docker --context server run --runtime nvidia --rm --network host -it $docker_image
# docker --context server exec 

# for key in "${!clients_host[@]}"
# do  
#     echo "$key : ${clients_host[$key]}"
    
#     #docker context create "client_$c_host" --docker "host=ssh://root@$c_host"
# done


# for host in $hostnames; do
#     ssh root@$host << EOF
#       echo "Installing Docker on $host"
#       docker pull $docker_image && 
#       docker run --runtime nvidia --rm --network host -it $docker_image
# EOF
# done





