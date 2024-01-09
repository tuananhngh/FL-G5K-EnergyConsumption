# Studying the energy consumption of Federate Learning
The goal if this project is to study the energy consumption of Federated Learning.

## FL on Grid'5000
We use the French experimental platform wall Grid'5000 which contains a variety of machines from A100 clusters to Jetson Xavier AGX.
Below are the steps required to execute the FL algorithm on Grid'5000 using the Flower framework.

### Book as many nodes as needed
```
flyon$ oarsub -l host=3 -p chifflot -I
```

### Create a conda environment (no need if using docker)
```
flyon$ module load conda
flyon$ conda env create -f environment.yml
```
### Lanch server and clients on same node
```
flyon$ ssh node-server (for exemple chifflot-2)
node-server$ module load conda
node-server$ conda activate fl
node-server$ bash run_server.sh
```

### Launch server and clients on different nodes
```
flyon$ ssh node-server (for exemple chifflot-2)
node-server$ module load conda
node-server$ conda activate fl
node-server$ hostname -I (ou IP_SERVER=`hostname -I` puis echo $IP_SERVER)
```
This returns the server ID. 
Copy it into the host parameter in [[/home/mjay/FL-G5K-Test/config/config_file.yaml]].
```
node-server$ python main_server.py comm.host=$IP_SERVER
```
For every client to be launched:
```
flyon$ ssh node-client (for exemple chifflot-4)
node-client$ module load conda
node-client$ conda activate fl
node-client$ cd FL-G5K-Test
node-client$ python client.py comm.host=$IP_SERVER client_params.client_id=0
```
The server will start when enough client are connected (as defined in the config file).

### Using docker
Install docker on your node:
```
g5k-setup-docker -t
```
To build and run the docker image on machine similar to chifflot:
```
bash run_docker.sh -b pt
```
To build and run the docker image on machine similar to a Jetson:
```
bash run_docker.sh -b jetson
```