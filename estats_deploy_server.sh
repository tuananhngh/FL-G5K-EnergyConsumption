# Deploy the environment on all reserved nodes
# kadeploy3 -a ~/public/ubuntu-estats.dsc 
# sleep 20

# Get IP address and hostname of deployed nodes
echo "Get Hostname of server"
hostnames=$(oarprint host)

IFS=$'\n' hosts_array=($(echo "$hostnames" | tr -s '[:space:]'))

declare -p -a hosts_array

server_host=${hosts_array[0]}
server_ip=$(host $server_host | awk '{print $4}')
echo "Server: $server_host, IP_SERVER: $server_ip"

docker_image="tuanngh/fl-jetson:latest"

# SSH connect to the SERVER
echo "---CONNECTING TO SERVER $server_host---"
ssh root@$server_host <<EOF 
    echo "---CONFIGURATE DOCKER SERVER $server_host---"
    echo "Pull Docker image on SERVER"
    docker pull $docker_image 
EOF

# RUN SERVER
echo "RUNNING SERVER"
ssh root@$server_host << EOF2
    echo "EXECUTE DOCKER ON SERVER"
    docker run -d --runtime nvidia --rm --network host --name server -it $docker_image &&
    docker exec server bash -c "python3 main_server.py comm.host=$server_ip params.num_rounds=10 params.num_clients=2"
EOF2


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait