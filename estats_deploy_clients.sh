#!/bin/bash

# This script is running on the frontend

# Reserve deployment resources
#oarsub -t deploy -t exotic -q testing -p estats -l host=1,walltime=1 -I

# Deploy the environment on all reserved nodes
# kadeploy3 -a ~/public/ubuntu-estats.dsc 
# sleep 60

# Get IP address and hostname of deployed nodes
echo "Get Hostname and IP address of deployed nodes"
hostnames=$(oarprint host)
# job_id=$(oarstat --array | grep "2" | awk '{print $1}')
ips=$(oarprint ip)

IFS=$'\n' hosts_array=($(echo "$hostnames" | tr -s '[:space:]'))

declare -p -a hosts_array

server_host=${hosts_array[0]}
# server_ip=$(host $server_host | awk '{print $4}')
clients_host=(${hosts_array[@]})
echo "JOB ID: $job_id"
echo "Clients: ${clients_host[@]}"



docker_image="tuanngh/fl-jetson:latest"
server_ip="$1"
num_clients="$2"
#job_id="$2"
echo "Server IP: $server_ip"

# FIREWALL RULES FOR CONNECTION WITH SERVER OUTSIDE GRID5000
# COMMENT THESE LINES IF YOU DON'T NEED TO CONNECT TO SERVER OUTSIDE GRID5000
# SERVER SHOULD HAVE IPv6 ADDRESS
# for key in "${!clients_host[@]}";
# do  
#     echo "Add firewall rule for client $key"
#     c_host=${clients_host[$key]}
#     ipv6_host=$(echo "$c_host" | sed 's/\./-ipv6./')
#     echo "IPv6: $ipv6_host"
#     curl -i "https://api.grid5000.fr/stable/sites/toulouse/firewall/$job_id" -d '[{"addr":"'$ipv6_host'", "port":22}]'
# done


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
            docker exec client_"$key" bash -c "python3 client.py comm.host=$server_ip client.cid=$key client.local_epochs=10 params.num_rounds=10 params.num_clients=$num_clients"  
HERE2
    ) &
done

wait

for key in "${!clients_host[@]}";
do
    (   c_host=${clients_host[$key]}
        #echo "---CONNECTING TO CLIENT $c_host---"
        ssh root@"$c_host" <<HERE3
            echo "---REMOVE CONTAINER ON CLIENT $c_host---"
            docker ps -aq | xargs docker stop
            echo "---REMOVED---"
HERE3
    ) &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait



# RUN CMD : bash estats_deploy_clients.sh $SERVER_IP $num_clients