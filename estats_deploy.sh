#!/bin/bash

# This script is running on the frontend

# Reserve deployment resources
#oarsub -t deploy -t exotic -q testing -p estats -l host=1,walltime=1 -I

# Deploy the environment on all reserved nodes
#kadeploy3 -a ~/public/ubuntu-estats.dsc 

# Get IP address and hostname of deployed nodes
echo "Get Hostname and IP address of deployed nodes"
hostnames=$(oarprint host -P host)
ips=$(oarprint host -P ip)

echo $hostnames
echo $ips

for host in $hostnames; do
  {
    echo "Deploying to host $host"
    ssh root@$host << EOF
      echo "Installing Docker on $host"
      docker pull slatonnguyen410/flower-client-jetson:1.0 &&\
      docker run -d --runtime nvidia --rm --network host -it slatonnguyen410/flower-client-jetson:1.0
EOF
  } &
done



# Assuming there's only one node, extract the first element
# echo "Extract the first element of the list"
# hostname=$(echo $hostnames | awk '{print $1}')
# ip=$(echo $ips | awk '{print $1}')

#   # SSH connect to the node
#   ssh root@$ip <<EOF
#     # Install Docker
#     apt-get update
#     apt-get install -y docker.io

#     # Clone the repository
#     docker pull slatonnguyen410/flower-client-server:1.0

#     # Run the Docker container with the --rm option to remove it after exit
#     docker run --runtime nvidia --rm --network host -it slatonnguyen410/flower-client-server:1.0
# EOF

# docker exec -it container_name /bin/bash